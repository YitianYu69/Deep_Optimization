"""Real two-rank FP8 runtime checks with bounded worker lifetime."""

from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import time

import torch
from torch import distributed as dist, nn
import torch.multiprocessing as mp
from torch.nn import functional as F

from deepopt import Checkpoint, DDPCore, Trainer
from deepopt.opt import transform_model
from tests.opt.test_training_black_box import InterruptedTraining, fp8_plan, make_training


class SharedProjectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8)
        self.alias = self.projection
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        hidden = torch.tanh(self.projection(x) + self.alias(x))
        return F.linear(hidden, self.head.weight, self.head.bias)


def batch(rank, index, device="cpu"):
    rng = torch.Generator().manual_seed(100 + 10 * rank + index)
    return torch.randn(3, 8, generator=rng).to(device), torch.randn(3, 4, generator=rng).to(device)


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_equal(a, b)
    else:
        assert actual == expected, (actual, expected)


def _worker(rank, directory, white_box, device_type):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if device_type == "cuda" else "cpu")
    backend = "triton" if device_type == "cuda" else "reference"
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo",
                            init_method=Path(directory, "rendezvous").as_uri(),
                            rank=rank, world_size=2,
                            timeout=timedelta(seconds=180 if device_type == "cuda" else 45))
    try:
        torch.manual_seed(31)
        model, _ = transform_model(SharedProjectionModel().to(device), plan=fp8_plan(backend))
        state_keys = tuple(model.state_dict())
        expected = deepcopy(model)
        parameter_ids = {id(p) for p in model.parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, device=device, accumulation_steps=2)
        assert isinstance(trainer.runtime.core, DDPCore)
        assert model.projection is model.alias

        if white_box:
            core = trainer.runtime.core
            assert core.wrapper.module.model is model
            assert {id(p) for p in core.wrapper.parameters()} == parameter_ids
            assert not core.forward_pending
            with core.accumulation_context(synchronize=False):
                assert not core.wrapper.require_backward_grad_sync
                inputs, targets = batch(rank, 0, device)
                loss = trainer.runtime.compute((inputs, targets),
                                               lambda values: F.mse_loss(model(values[0]), values[1]))
                assert core.forward_pending
                trainer.runtime.backward(loss)
                assert not core.forward_pending
            assert core.wrapper.require_backward_grad_sync
            inputs, targets = batch(rank, 1, device)
            loss = trainer.runtime.compute((inputs, targets),
                                           lambda values: F.mse_loss(model(values[0]), values[1]))
            trainer.runtime.backward(loss)
            for parameter in model.parameters():
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                assert parameter.grad.dtype == torch.float32
                gathered = [torch.empty_like(parameter.grad) for _ in range(2)]
                dist.all_gather(gathered, parameter.grad)
                _assert_equal(gathered[0], gathered[1])
            assert tuple(model.state_dict()) == state_keys
        else:
            # Average independently quantized per-rank/microbatch gradients.
            # Quantizing one concatenated batch would use different FP8 scales.
            expected_optimizer = torch.optim.SGD(expected.parameters(), lr=.01, momentum=.9)
            for step in range(2):
                expected_optimizer.zero_grad(set_to_none=True)
                for other_rank in range(2):
                    for micro in range(2):
                        inputs, targets = batch(other_rank, 2 * step + micro, device)
                        F.mse_loss(expected(inputs), targets).backward()
                for parameter in expected.parameters():
                    parameter.grad.div_(4)
                expected_optimizer.step()
            result = trainer.train_epoch([batch(rank, index) for index in range(4)])
            assert result["samples"] == 24
            for actual, reference in zip(model.parameters(), expected.parameters()):
                torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-7)
                gathered = [torch.empty_like(actual) for _ in range(2)]
                dist.all_gather(gathered, actual.detach())
                _assert_equal(gathered[0], gathered[1])

            reference, loader = make_training(device=device, backend=backend)
            reference.fit(loader, epochs=2, monitor=None, progress_bar=False)
            for boundary in ("step", "epoch"):
                interrupted, loader = make_training(device=device, backend=backend)
                path = Path(directory, boundary)
                callback = Checkpoint(path, every_steps=1 if boundary == "step" else 0)

                def stop_step(current, event):
                    if current.state.optimizer_steps == 1:
                        raise InterruptedTraining()

                def stop_epoch(current, row):
                    if current.state.completed_epochs == 1:
                        raise InterruptedTraining()

                try:
                    interrupted.fit(loader, epochs=2, monitor=None, progress_bar=False,
                                    callbacks=[callback] + ([stop_epoch] if boundary == "epoch" else []),
                                    step_callbacks=[stop_step] if boundary == "step" else [])
                except InterruptedTraining:
                    pass
                else:
                    raise AssertionError("The checkpoint interruption did not execute")
                resumed, loader = make_training(device=device, backend=backend)
                callback = Checkpoint(path, every_steps=1 if boundary == "step" else 0)
                callback.restore(resumed, loader)
                resumed.fit(loader, epochs=2, callbacks=[callback], monitor=None, progress_bar=False)
                _assert_equal(resumed.model.state_dict(), reference.model.state_dict())
                _assert_equal(resumed.optimizer.state_dict(), reference.optimizer.state_dict())
                _assert_equal(resumed.scheduler.state_dict(), reference.scheduler.state_dict())
                assert resumed.state.optimizer_steps == reference.state.optimizer_steps
        Path(directory, f"rank-{rank}.json").write_text(json.dumps({"ok": True}))
    finally:
        dist.destroy_process_group()


def run_workers(test, *, white_box, device_type="cpu"):
    with tempfile.TemporaryDirectory() as directory:
        context = mp.spawn(_worker, args=(directory, white_box, device_type), nprocs=2, join=False)
        deadline = time.monotonic() + (300 if device_type == "cuda" else 120)
        try:
            while not context.join(timeout=1):
                if time.monotonic() > deadline:
                    test.fail("FP8 DDP workers exceeded their deadline")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
        for rank in range(2):
            test.assertTrue(json.loads(Path(directory, f"rank-{rank}.json").read_text())["ok"])
