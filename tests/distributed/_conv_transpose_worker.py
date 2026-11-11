"""Two-rank transposed decoder training with exact rank-local state restore."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import torch
from torch import distributed as dist, nn
from torch.nn import functional as F

from deepopt import Trainer, FP8Config, DelayedScalingConfig, initialize_distributed, shutdown_distributed
from deepopt.engine.checkpointing.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from deepopt.opt.scaling.delayed import delayed_states


def build(automatic, synchronize):
    torch.manual_seed(971)
    model = nn.Sequential(nn.Conv2d(4, 4, 1), nn.GroupNorm(2, 4),
                          nn.ConvTranspose2d(4, 4, 4, stride=2, padding=1), nn.SiLU(), nn.Conv2d(4, 2, 1))
    options = FP8Config(enabled=True, backend="reference", trace_mode="modules", scaling_recipe="delayed",
                        intercept_linear=False, backward_precision="auto" if automatic else "native",
                        delayed_scaling=DelayedScalingConfig(history_length=3, reduce_amax=synchronize))
    return Trainer(model, torch.optim.SGD(model.parameters(), lr=.001, momentum=.9), loss_fn=F.mse_loss,
                   fp8=options, device="cpu", checks=False, accumulation_steps=2,
                   config={"runtime": {"activation_compression": {"enabled": True, "group_size": 17}}})


context = initialize_distributed("cpu")
torch.set_num_threads(1)
try:
    for automatic in (False, True):
        for synchronize in (False, True):
            trainer = build(automatic, synchronize)
            x = torch.arange(72, dtype=torch.float32).reshape(2, 4, 3, 3) / 72 + context.global_rank
            batches = [(x, torch.zeros(2, 2, 6, 6)), (x * 1.5, torch.zeros(2, 2, 6, 6))]
            trainer.fit(batches, epochs=1, progress_bar=False)
            if synchronize:
                for state in delayed_states(trainer.model):
                    gathered = [torch.empty_like(state.history) for _ in range(context.world_size)]
                    dist.all_gather(gathered, state.history)
                    for other in gathered:
                        torch.testing.assert_close(state.history, other, rtol=0, atol=0)
            # Explicit DDP buffer synchronization must preserve numerical state.
            state = trainer.model[2].fp8_scaling
            before = state.history.clone()
            state.history[0, 0] = context.global_rank + 31
            trainer.runtime.core.sync_buffers()
            assert state.history[0, 0].item() == context.global_rank + 31
            state.history.copy_(before)
            path = Path(sys.argv[1]) / f"transpose-{automatic}-{synchronize}.pt"
            save_checkpoint(path, trainer, generators={}, history=[{}])
            expected = deepcopy(trainer.model.state_dict())
            restored = build(automatic, synchronize)
            restore_checkpoint(read_checkpoint(path), restored, generators={}, batch_size_mode="manual")
            for name, value in restored.model.state_dict().items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)
                else:
                    assert value == expected[name], name
            trainer.fit(batches, epochs=2, progress_bar=False)
            restored.fit(batches, epochs=2, progress_bar=False)
            for a, b in zip(trainer.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for a, b in zip(delayed_states(trainer.model), delayed_states(restored.model)):
                torch.testing.assert_close(a.scales, b.scales, rtol=0, atol=0)
                torch.testing.assert_close(a.history, b.history, rtol=0, atol=0)
            if context.global_rank == 0:
                print(json.dumps({"backward_auto": automatic, "synchronize": synchronize, "exact_resume": True}), flush=True)
finally:
    shutdown_distributed()
