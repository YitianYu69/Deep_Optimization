"""Two-rank Gloo acceptance of synchronized/local histories and exact resume."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import Trainer, FP8Config, DelayedScalingConfig, initialize_distributed, shutdown_distributed
from deepopt.engine.checkpointing.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from deepopt.opt.scaling_strategy.delayed import delayed_states


def build(options):
    torch.manual_seed(171)
    model = nn.Sequential(nn.Linear(5, 3))
    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=.001, momentum=.9)
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, fp8=options, device="cpu", checks=False)
    assert tuple(trainer.model.parameters()) == parameters
    return trainer


context = initialize_distributed("cpu")
torch.set_num_threads(1)
try:
    for synchronize in (True, False):
        options = FP8Config(enabled=True, scaling_recipe="delayed", intercept_linear=False,
                            delayed_scaling=DelayedScalingConfig(history_length=3, reduce_amax=synchronize))
        trainer = build(options)
        batch = (torch.full((7, 5), float(2 + context.global_rank * 6)), torch.zeros(7, 3))
        history = []
        trainer.fit([batch], epochs=1, progress_bar=False,
                    callbacks=[lambda trainer, row: history.append(deepcopy(row))])
        state = delayed_states(trainer.model)[0]
        expected = 8. if synchronize else float(2 + context.global_rank * 6)
        torch.testing.assert_close(state.history[0, 0], torch.tensor(expected), rtol=0, atol=0)
        assert state.updates.tolist() == [1, 1, 1]
        ignored = trainer.runtime.core.wrapper.parameters_to_ignore
        assert any("fp8_scaling.history" in name for name in ignored)
        # Validation-time explicit buffer synchronization must also exclude the
        # numerical state, even when the recipe normally synchronizes amax.
        before = state.history.clone()
        state.history[0, 2] = 20 + context.global_rank
        trainer.runtime.core.sync_buffers()
        assert state.history[0, 2].item() == 20 + context.global_rank
        state.history.copy_(before)
        path = Path(sys.argv[1]) / ("synchronized.pt" if synchronize else "local.pt")
        save_checkpoint(path, trainer, generators={}, history=history)
        expected_state = deepcopy(trainer.model.state_dict())
        restored = build(options)
        restore_checkpoint(read_checkpoint(path), restored, generators={}, batch_size_mode="manual")
        for name, value in restored.model.state_dict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)
        trainer.fit([batch], epochs=2, progress_bar=False)
        restored.fit([batch], epochs=2, progress_bar=False)
        for a, b in zip(trainer.model.parameters(), restored.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(delayed_states(trainer.model), delayed_states(restored.model)):
            torch.testing.assert_close(a.scales, b.scales, rtol=0, atol=0)
            torch.testing.assert_close(a.history, b.history, rtol=0, atol=0)
        if context.global_rank == 0:
            print(json.dumps({"synchronize": synchronize, "rank_local_checkpoint": True, "exact_next_update": True}), flush=True)
finally:
    shutdown_distributed()
