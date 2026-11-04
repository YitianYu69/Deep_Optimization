"""Real Gloo worker for automatic FP8 core/runtime/trainer acceptance tests."""

from copy import deepcopy
import gc
import json
import sys
import traceback
from unittest.mock import patch

import torch
from torch import distributed as dist, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from deepopt import DDPCore, Trainer, TrainingRuntime, initialize_distributed, shutdown_distributed
from deepopt.logger import log_output
from deepopt.opt import transform_model
from deepopt.opt.policy import fp8_plan


class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8)
        self.alias = self.projection
        self.head = nn.Linear(8, 4)

    def forward(self, inputs):
        hidden = torch.tanh(self.projection(inputs) + self.alias(inputs))
        return F.linear(hidden, self.head.weight, self.head.bias)


def batch(rank):
    generator = torch.Generator().manual_seed(178 + rank)
    return torch.randn(3, 8, generator=generator), torch.randn(3, 4, generator=generator)


def train_variant(context, api):
    torch.manual_seed(71)
    model = SharedModel()
    parameters = tuple(model.parameters())
    keys = tuple(model.state_dict())
    baseline, _ = transform_model(deepcopy(model), plan=fp8_plan(True))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.03)
    checked_gradients = []

    def check_gradients(current_optimizer, args, kwargs):
        gradients = [p.grad for group in current_optimizer.param_groups for p in group["params"]]
        assert all(value is not None and value.dtype == torch.float32 for value in gradients)
        flat = torch.cat([value.flatten() for value in gradients])
        assert torch.isfinite(flat).all()
        gathered = [torch.empty_like(flat) for _ in range(context.world_size)]
        dist.all_gather(gathered, flat)
        for value in gathered:
            torch.testing.assert_close(value, flat, rtol=0, atol=0)
        checked_gradients.append(True)

    optimizer.register_step_pre_hook(check_gradients)
    inputs, targets = batch(context.global_rank)
    if api == "trainer":
        trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, fp8=True)
        core = trainer.runtime.core
        trainer.train_epoch([(inputs, targets)])
        assert trainer.state.optimizer_steps == 1
    elif api == "runtime":
        runtime = TrainingRuntime(model, optimizer, fp8=True)
        core = runtime.core
        # This closure intentionally retains the caller's original composite root.
        loss = runtime.compute((inputs, targets), lambda b: F.mse_loss(model(b[0]), b[1]))
        runtime.backward(loss)
        runtime.step()
    else:
        core = DDPCore(model, context.device, fp8=True)
        loss = core.compute((inputs, targets), lambda b: F.mse_loss(core.model(b[0]), b[1]))
        core.backward(loss)
        optimizer.step()

    assert isinstance(core, DDPCore)
    assert core.model is model
    assert model.projection is model.alias
    assert tuple(model.state_dict()) == keys
    assert all(a is b for a, b in zip(parameters, model.parameters()))
    assert {id(p) for p in core.wrapper.parameters()} == {id(p) for p in parameters}
    assert checked_gradients == [True]
    # An independently quantized loss for each rank gives the native DDP mean.
    for rank in range(context.world_size):
        x, y = batch(rank)
        F.mse_loss(baseline(x), y).backward()
    for parameter in baseline.parameters():
        parameter.grad.div_(context.world_size)
    baseline_optimizer.step()
    for actual, expected in zip(model.parameters(), baseline.parameters()):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    return {"api": api, "core": type(core).__name__, "world_size": context.world_size,
            "synchronized_gradients": True, "baseline_matches": True}


def reject_mismatched_policy(context):
    rejected = []
    for api in ("runtime", "core"):
        model = nn.Linear(8, 4)
        settings = {"enabled": True, "grad_dtype": "e4m3fn" if context.global_rank == 0 else "e5m2"}
        # Only spy on construction: the policy agreement uses a real Gloo group.
        with patch.object(DistributedDataParallel, "__init__",
                          side_effect=AssertionError("DDP was wrapped before FP8 policy agreement")):
            try:
                if api == "runtime":
                    TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.03), fp8=settings)
                else:
                    DDPCore(model, context.device, fp8=settings)
            except ValueError as exc:
                assert "matching" in str(exc), str(exc)
                rejected.append(api)
            else:
                raise AssertionError("Rank-inconsistent FP8 policies were accepted")
    return {"mismatch_rejected": rejected, "world_size": context.world_size}


def main():
    torch.set_num_threads(1)
    context = initialize_distributed("cpu", timeout=45)
    try:
        if sys.argv[1] == "mismatch":
            log_output(json.dumps(reject_mismatched_policy(context)))
        else:
            for api in ("trainer", "runtime", "core"):
                log_output(json.dumps(train_variant(context, api)))
                gc.collect()  # Release native wrappers while their process group is alive.
    except BaseException as exc:
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        gc.collect()
        shutdown_distributed()


if __name__ == "__main__":
    main()
