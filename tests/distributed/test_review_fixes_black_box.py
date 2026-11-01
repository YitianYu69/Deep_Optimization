"""Real-process acceptance tests for the review's distributed correctness fixes."""

from collections import Counter
from copy import deepcopy
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from torch import distributed as dist, nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from deepopt import Checkpoint, LossResult, Trainer
from deepopt.data import ResumableDataLoader, SeededTransformDataset
from deepopt.engine.strategies import ExplicitObjective


def _comm_hook(state, bucket):
    state["calls"] += 1
    value = bucket.buffer()
    dist.all_reduce(value)
    value.div_(dist.get_world_size())
    future = torch.futures.Future()
    future.set_result(value)
    return future


def _weighted_step(model, batch):
    value = model(batch).square().mean()
    return {"loss": LossResult.from_mean(value, weight=batch.numel()), "sample_count": len(batch)}


class Images(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index, torch.arange(16).reshape(4, 4).double() / 16 + index / 100


def crop_flip(sample, *, generator):
    index, image = sample
    row, column = torch.randint(0, 3, (2,), generator=generator).tolist()
    crop = image[row:row + 2, column:column + 2]
    if torch.rand((), generator=generator).item() < .5:
        crop = crop.flip(-1)
    return index, crop.flatten().tolist()


def _tensorize(samples):
    return {"indices": [index for index, _ in samples],
            "x": torch.tensor([values for _, values in samples], dtype=torch.double)}


class PortableBatch:
    def __init__(self, samples):
        self.samples = samples

    def __reduce__(self):
        return _tensorize, (self.samples,)


def _close(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def _validation_cases(rank, device):
    for failing_rank in (0, 1):
        for location in ("step", "observer", "loader", "setup"):
            model = nn.Sequential(nn.Linear(1, 1), nn.Dropout()).to(device)
            model[1].eval()
            modes = [module.training for module in model.modules()]
            message = f"validation-{location}-rank-{failing_rank}"

            def validation(model, batch):
                if location == "step" and rank == failing_rank:
                    raise ValueError(message)
                return model(batch).square().mean()

            def observer(event):
                if location == "observer" and rank == failing_rank:
                    raise LookupError(message)

            def factory():
                from deepopt.engine.reporting.metrics import MetricAccumulator
                if location == "setup" and rank == failing_rank:
                    raise ValueError(message)
                return MetricAccumulator()

            class ValidationBatches:
                def __iter__(self):
                    if location == "loader" and rank == failing_rank:
                        raise ValueError(message)
                    yield from [torch.ones(1, 1)] * (2 if rank == failing_rank else 0)

            trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), device=device,
                              step_fn=validation, checks=[observer], metric_factory=factory)
            with patch.object(trainer._epochs.metrics, "compute", side_effect=AssertionError("metrics must not reduce after failure")):
                try:
                    trainer.evaluate(ValidationBatches())
                except (ValueError, LookupError, RuntimeError) as error:
                    assert message in str(error), str(error)
                    assert "Connection closed" not in str(error)
                else:
                    raise AssertionError("Validation failure was swallowed")
            assert trainer.runtime._failed
            assert modes == [module.training for module in model.modules()]

    model = nn.Linear(1, 1).to(device)
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), device=device,
                      loss_fn=nn.functional.mse_loss, execution_mode="max_speed")
    counts = []
    for length in (1, rank + 3):
        calls = Counter()
        originals = {name: getattr(dist, name) for name in ("all_reduce", "all_gather", "all_gather_object")}

        def wrapped(name):
            def call(*args, **kwargs):
                calls[name] += 1
                return originals[name](*args, **kwargs)
            return call

        with patch.object(dist, "all_reduce", side_effect=wrapped("all_reduce")), \
                patch.object(dist, "all_gather", side_effect=wrapped("all_gather")), \
                patch.object(dist, "all_gather_object", side_effect=wrapped("all_gather_object")):
            result = trainer.evaluate([(torch.ones(1, 1), torch.zeros(1, 1))] * length)
        assert result["samples"] > 0
        counts.append(calls)
    assert counts[0] == counts[1], counts  # No collective per validation batch.


def _accumulation_cases(rank, world, device):
    for factor in (1, 2, 4, 8):
        expected = torch.tensor(1., dtype=torch.double, device=device)
        for start in range(0, 10, factor):
            inputs = torch.cat([torch.full((other + 1,), (index + other + 1) / 10,
                                            dtype=torch.double, device=device)
                                for index in range(start, min(start + factor, 10)) for other in range(world)])
            expected -= .01 * 2 * expected * inputs.square().mean()
        for mode in ("max_speed", "balanced", "analyze"):
            model = nn.Linear(1, 1, bias=False).double().to(device)
            nn.init.ones_(model.weight)
            trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), device=device,
                              step_fn=_weighted_step, accumulation_steps=factor, execution_mode=mode)
            communication = {"calls": 0}
            trainer.runtime.core.wrapper.register_comm_hook(communication, _comm_hook)
            batches = [torch.full((rank + 1, 1), (index + rank + 1) / 10, dtype=torch.double) for index in range(10)]
            trainer.train_epoch(batches)
            torch.testing.assert_close(model.weight.squeeze(), expected, rtol=1e-12, atol=1e-12)
            assert communication["calls"] == math.ceil(10 / factor), (factor, mode, communication)

    # Unknown-length streams intentionally retain synchronization per microbatch.
    model = nn.Linear(1, 1, bias=False).double().to(device)
    trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), device=device,
                      step_fn=_weighted_step, accumulation_steps=4, execution_mode="max_speed")
    communication = {"calls": 0}
    trainer.runtime.core.wrapper.register_comm_hook(communication, _comm_hook)
    trainer.train_epoch(torch.ones(1, 1).double() for _ in range(10))
    assert communication["calls"] == 10

    for reduction in ("legacy", "rank_mean", "global_mean", "global_sum"):
        model = nn.Linear(1, 1, bias=False).double().to(device)
        nn.init.ones_(model.weight)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1), device=device,
                          step_fn=_weighted_step, execution_mode="max_speed",
                          loss_strategy=ExplicitObjective(distributed_reduction=reduction))
        trainer.train_epoch([torch.full((rank + 1, 1), rank + 1., dtype=torch.double)])
        local_terms = [(other + 1.) ** 2 for other in range(world)]
        if reduction in ("legacy", "global_mean"):
            gradient = 2 * sum(value * (other + 1) for other, value in enumerate(local_terms)) / sum(range(1, world + 1))
        elif reduction == "rank_mean":
            gradient = 2 * sum(local_terms) / world
        else:
            gradient = 2 * sum(local_terms)
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 1 - .1 * gradient), rtol=1e-12, atol=1e-12)

    original_mode = trainer.execution_mode
    try:
        trainer.set_execution_mode("analyze" if rank == 0 else "max_speed")
    except ValueError as error:
        assert "matching" in str(error)
    else:
        raise AssertionError("Ranks accepted different mode transitions")
    assert trainer.execution_mode == original_mode
    bad_model = nn.Linear(1, 1).to(device)
    try:
        Trainer(bad_model, torch.optim.SGD(bad_model.parameters(), lr=.01), device=device,
                loss_fn=nn.functional.mse_loss, execution_mode="invalid-mode" if rank == 0 else "max_speed")
    except (ValueError, RuntimeError) as error:
        assert "invalid-mode" in str(error)
    else:
        raise AssertionError("An invalid rank-local policy was accepted")


def _compiled_mode_cases(rank, device):
    precisions = ("fp32", "bf16") if device.type == "cuda" and torch.cuda.is_bf16_supported() else ("fp32",)
    for precision in precisions:
        expected = None
        for schedule in (None, {2: "analyze", 3: "max_speed"}):
            torch._dynamo.reset()
            raw = nn.Linear(1, 1, bias=False).to(device)
            nn.init.ones_(raw.weight)
            model = torch.compile(raw, backend="inductor" if device.type == "cuda" else "eager")
            trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), device=device,
                              precision=precision, loss_fn=nn.functional.mse_loss, accumulation_steps=2,
                              execution_mode="max_speed", mode_schedule=schedule)
            trainer.train_epoch([(torch.full((3, 1), (rank + 1) / 10), torch.zeros(3, 1))] * 5)
            actual = raw.weight.detach().clone()
            if expected is None:
                expected = actual
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch._dynamo.reset()


def _seeded_resume(rank, world, device, directory, *, fixed_buckets=False):
    def make(records):
        torch.manual_seed(13)
        model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Dropout(.1), nn.Linear(4, 1)).double().to(device)

        def step(model, batch):
            records.append((batch["indices"], batch["x"].detach().cpu().tolist()))
            loss = model(batch["x"]).square().mean()
            return {"loss": LossResult.from_mean(loss, weight=len(batch["indices"])), "sample_count": len(batch["indices"])}

        trainer = Trainer(model, torch.optim.AdamW(model.parameters(), lr=.001), device=device,
                          step_fn=step, accumulation_steps=4, execution_mode="max_speed",
                          ddp_options={"find_unused_parameters": True} if fixed_buckets else None)
        loader = ResumableDataLoader(SeededTransformDataset(Images(world * 11), crop_flip, seed=123, transform_id="crop-flip-v1"),
                                     batch_size=2, generator=torch.Generator().manual_seed(17), num_workers=2,
                                     prefetch_factor=3, persistent_workers=True, multiprocessing_context="spawn",
                                     timeout=30, collate_fn=PortableBatch)
        torch.manual_seed(31 + rank)
        return trainer, loader

    reference_records = []
    reference, loader = make(reference_records)
    try:
        reference.fit(loader, epochs=2, progress_bar=False)
        expected = deepcopy(reference.model.state_dict())
        expected_optimizer = deepcopy(reference.optimizer.state_dict())
    finally:
        _close(loader)
    del reference, loader
    gc.collect()

    records = []
    interrupted, loader = make(records)
    checkpoint_directory = Path(directory, "seeded-fixed" if fixed_buckets else "seeded")
    checkpoint = Checkpoint(checkpoint_directory, every_steps=1)

    class Interrupted(Exception):
        pass

    def stop(current, event):
        if current.state.optimizer_steps == 1:
            raise Interrupted()

    try:
        try:
            interrupted.fit(loader, epochs=2, callbacks=[checkpoint], step_callbacks=[stop], progress_bar=False)
        except Interrupted:
            pass
    finally:
        _close(loader)
    del interrupted, loader, checkpoint
    gc.collect()
    resumed, loader = make(records)
    checkpoint = Checkpoint(checkpoint_directory, every_steps=1)
    try:
        checkpoint.restore(resumed, loader)
        resumed.fit(loader, epochs=2, callbacks=[checkpoint], progress_bar=False)
        assert records == reference_records
        # Four-rank reductions can change summation order when native DDP
        # rebuilds buckets. Input replay remains exact; test bitwise state with
        # fixed buckets separately rather than claiming it for every reducer.
        tolerance = {"rtol": 1e-13, "atol": 1e-15} if world > 2 and not fixed_buckets else {"rtol": 0, "atol": 0}
        for name, tensor in resumed.model.state_dict().items():
            torch.testing.assert_close(tensor, expected[name], **tolerance)
        for key, state in resumed.optimizer.state_dict()["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, expected_optimizer["state"][key][name], **tolerance)
    finally:
        _close(loader)


def _worker(rank, world, device_type, directory):
    os.environ["DEEPOPT_LOGGING_LEVEL"] = "ERROR"
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if device_type == "cuda" else "cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo", rank=rank, world_size=world,
                            init_method=Path(directory, "group").as_uri(), timeout=timedelta(seconds=60))
    try:
        _validation_cases(rank, device)
        _accumulation_cases(rank, world, device)
        _compiled_mode_cases(rank, device)
        _seeded_resume(rank, world, device, directory)
        if world > 2:
            _seeded_resume(rank, world, device, directory, fixed_buckets=True)
        Path(directory, f"rank-{rank}.json").write_text(json.dumps({"passed": True}))
    finally:
        dist.destroy_process_group()


class DistributedReviewBlackBoxTests(unittest.TestCase):
    def run_workers(self, world, device):
        with tempfile.TemporaryDirectory() as directory:
            processes = mp.spawn(_worker, args=(world, device, directory), nprocs=world, join=False)
            deadline = time.monotonic() + 180
            try:
                while not processes.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("Review acceptance tests exceeded 180 seconds")
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            for rank in range(world):
                self.assertTrue(json.loads(Path(directory, f"rank-{rank}.json").read_text())["passed"])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
    def test_two_rank_gloo_acceptance(self):
        self.run_workers(2, "cpu")

    @unittest.skipUnless(os.environ.get("DEEPOPT_EXTENDED_DDP") == "1" and dist.is_gloo_available(), "Enable extended four-rank coverage")
    def test_four_rank_gloo_acceptance(self):
        self.run_workers(4, "cpu")

    @unittest.skipUnless(torch.cuda.device_count() >= 2 and dist.is_nccl_available(), "Two CUDA GPUs and NCCL are required")
    def test_two_rank_nccl_acceptance(self):
        self.run_workers(2, "cuda")

    @unittest.skipUnless(torch.cuda.device_count() >= 4 and dist.is_nccl_available(), "Four CUDA GPUs and NCCL are required")
    def test_four_rank_nccl_acceptance(self):
        self.run_workers(4, "cuda")


if __name__ == "__main__":
    unittest.main()
