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
from deepopt.opt.scaling.delayed import delayed_states
from deepopt.opt.nn import functional as D


class PoolUnpool(nn.Module):
    def forward(self, x):
        value, indices = D.do_max_pool1d(x, 2, return_indices=True)
        return D.do_max_unpool1d(value, indices, 2)


def build(options):
    torch.manual_seed(171)
    activation = {"sigmoid": nn.Sigmoid, "silu": nn.SiLU, "gelu": nn.GELU,
                  "tanh": nn.Tanh, "relu6": nn.ReLU6, "relu": nn.ReLU,
                  "layer_norm": lambda: nn.LayerNorm(8),
                  "average_pool": lambda: nn.AdaptiveAvgPool1d(1),
                  "maximum_pool": lambda: nn.AdaptiveMaxPool1d(1),
                  "power_pool": lambda: nn.LPPool1d(2, 8), "pool_unpool": PoolUnpool}[activation_kind]
    last_features = 1 if activation_kind in ("average_pool", "maximum_pool", "power_pool") else 8
    model = (nn.Sequential(nn.Linear(5, 8), activation(), nn.Linear(last_features, 3))
             if options.backward_precision == "auto" else nn.Sequential(nn.Linear(5, 3)))
    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=.001, momentum=.9)
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, fp8=options, device="cpu", checks=False,
                      config={"runtime": {"activation_compression": {"enabled": True, "norm_xhat_recenter": recenter}}} if activation_kind != "relu" else None)
    assert tuple(trainer.model.parameters()) == parameters
    return trainer


context = initialize_distributed("cpu")
automatic = len(sys.argv) > 2 and sys.argv[2] == "auto"
activation_kind = sys.argv[3] if len(sys.argv) > 3 else "relu"
recenter = len(sys.argv) > 4 and sys.argv[4] == "recenter"
torch.set_num_threads(1)
try:
    for synchronize in (True, False):
        options = FP8Config(enabled=True, scaling_recipe="delayed", intercept_linear=False,
                            backward_precision="auto" if automatic else "native",
                            grad_dtype="e5m2" if automatic else "e4m3fn",
                            delayed_scaling=DelayedScalingConfig(history_length=3, reduce_amax=synchronize))
        trainer = build(options)
        batch = (torch.full((7, 5), float(2 + context.global_rank * 6)), torch.zeros(7, 3))
        history = []
        trainer.fit([batch], epochs=1, progress_bar=False,
                    callbacks=[lambda trainer, row: history.append(deepcopy(row))])
        state = delayed_states(trainer.model)[0]
        expected = 8. if synchronize else float(2 + context.global_rank * 6)
        torch.testing.assert_close(state.history[0, 0], torch.tensor(expected), rtol=0, atol=0)
        assert state.updates.tolist() == ([1, 1, 0] if automatic else [1, 1, 1])
        if automatic:
            from deepopt.opt.precision import precision_report

            expected_edges = 3 if activation_kind == "pool_unpool" else 2
            assert sum(info is not None for info in precision_report(trainer.model)["backward_execution"].values()) == expected_edges
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
            row = {"synchronize": synchronize, "rank_local_checkpoint": True, "exact_next_update": True}
            if automatic:
                row["backward_auto"] = True
            if activation_kind != "relu":
                row["activation"] = activation_kind
            if recenter:
                row["norm_xhat_recenter"] = True
            print(json.dumps(row), flush=True)
finally:
    shutdown_distributed()
