from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import torch
from torch import distributed as dist, nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, TensorDataset

from deepopt import Checkpoint, DDPCore, FiniteCheck, LossResult, MultiOptimizer, Trainer, TrainingCore, TrainingModule, TrainingRuntime
from deepopt.data import ResumableDataLoader, build_loaders
from deepopt.engine.checkpointing.checkpoint import read_checkpoint

if __package__:
    from .launcher import run_torchrun
else:
    from launcher import run_torchrun


def weighted_step(model, batch):
    # Deliberately use a parameter directly; the complete step must enter DDP.
    values = (model.weight * batch).square()
    return {"loss": LossResult.from_mean(values.mean(), weight=values.numel()), "sample_count": values.numel()}


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right, (left, right)


class HookModule(TrainingModule):
    def training_step(self, batch):
        prediction = self(batch)
        return prediction.square().mean()


class SeededIndexData(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return index, torch.initial_seed()


def _distributed_worker(rank, directory, device_type="cpu"):
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}" if device_type == "cuda" else "cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo", init_method=Path(directory, "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        model = nn.Linear(1, 1, bias=False).double().to(device)
        nn.init.ones_(model.weight)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01),
                          step_fn=weighted_step, accumulation_steps=2, device=device)
        assert isinstance(trainer.runtime.core, DDPCore)
        batches = [torch.full((rank * 2 + 1, 1), float(index + rank + 1), dtype=torch.double)
                   for index in range(3)]
        result = trainer.train_epoch(batches)
        expected = torch.ones((), dtype=torch.double, device=device)
        for indices in ((0, 1), (2,)):
            inputs = torch.cat([torch.full((other * 2 + 1,), float(index + other + 1), dtype=torch.double, device=device)
                                for index in indices for other in range(2)])
            expected -= .01 * 2 * expected * inputs.square().mean()
        torch.testing.assert_close(model.weight.squeeze(), expected, rtol=1e-12, atol=1e-12)
        assert result["samples"] == 12

        # Fake memory tracing uses the local core, then all ranks agree on a fit.
        from deepopt.memory import estimate_training_memory
        from deepopt.memory.planning import plan_batch_size
        plan_model = nn.Linear(1, 1).to(device)
        planned = Trainer(plan_model, torch.optim.SGD(plan_model.parameters(), lr=.01), loss_fn=nn.functional.mse_loss,
                          device=device)
        sample = lambda size: (torch.ones(size, 1), torch.zeros(size, 1))
        report = estimate_training_memory(planned, sample(2))
        budget = report.peak_bytes_by_device[str(device)] if rank else 1000000
        plan = plan_batch_size(planned, requested_batch_size=8, sample_batch_fn=sample, memory_budget_bytes=budget)
        assert plan["new_batch_size"] <= 2
        sizes = [None, None]
        dist.all_gather_object(sizes, plan["new_batch_size"])
        assert sizes[0] == sizes[1]

        # TrainingModule hooks also enter DDP's forward, without bypassing it.
        module = HookModule(nn.Linear(1, 1, bias=False)).to(device)
        hooked = Trainer(module, torch.optim.SGD(module.parameters(), lr=.01), device=device)
        hooked.train_epoch([torch.ones(2, 1) * (rank + 1)])
        weights = [torch.empty_like(module.model.weight) for _ in range(2)]
        dist.all_gather(weights, module.model.weight.detach())
        _equal(weights[0], weights[1])

        # Manual execution and optimizer routing share the same DDP core.
        branches = nn.ModuleDict({"a": nn.Linear(1, 1, bias=False), "b": nn.Linear(1, 1, bias=False)}).to(device)
        for parameter in branches.parameters():
            nn.init.ones_(parameter)
        optimizers = MultiOptimizer({"a": torch.optim.SGD(branches["a"].parameters(), lr=.1),
                                     "b": torch.optim.SGD(branches["b"].parameters(), lr=.2)})
        runtime = TrainingRuntime(branches, optimizers, device=device,
                                  ddp_options={"find_unused_parameters": True},
                                  checks=[FiniteCheck(kinds={"weights"})] if rank else False)
        loss = runtime.compute(torch.ones(1, 1), lambda batch: branches["a" if rank == 0 else "b"](batch).sum())
        runtime.backward(loss)
        runtime.step()
        torch.testing.assert_close(branches["a"].weight, torch.full_like(branches["a"].weight, .95))
        torch.testing.assert_close(branches["b"].weight, torch.full_like(branches["b"].weight, .9))

        # Sharded shuffled indices, consumed cursors and rebatching.
        loader = ResumableDataLoader(SeededIndexData(), batch_size=2, num_workers=2, prefetch_factor=3,
                                     multiprocessing_context="spawn", timeout=30,
                                     generator=torch.Generator().manual_seed(7), collate_fn=list)
        iterator = iter(loader)
        first_batch = next(iterator)
        first = [index for index, seed in first_batch]
        seeds = [None, None]
        dist.all_gather_object(seeds, first_batch[0][1])
        assert seeds[0] != seeds[1], "Different ranks must not reuse worker RNG streams"
        saved = loader.state_dict()
        assert saved["sampler"]["position"] == 2
        assert loader.batch_sampler.position > 2
        rest = [index for batch in iterator for index, seed in batch]
        shards = [None, None]
        dist.all_gather_object(shards, first + rest)
        assert sorted(shards[0] + shards[1]) == list(range(12))
        loader.set_batch_size(3)
        loader.load_state_dict(saved)
        assert [index for batch in loader for index, seed in batch] == rest

        # Unequal training exhaustion must fail on both ranks before backward.
        bad = nn.Linear(1, 1).to(device)
        uneven = Trainer(bad, torch.optim.SGD(bad.parameters(), lr=.01), loss_fn=nn.functional.mse_loss, device=device)
        try:
            uneven.train_epoch([(torch.ones(1, 1), torch.zeros(1, 1))] * (rank + 1))
        except ValueError as error:
            assert "equal batch counts" in str(error)
        else:
            raise AssertionError("Unequal training shards were accepted")

        # A nonfinite loss on one rank must stop all ranks before DDP backward.
        bad = nn.Linear(1, 1).to(device)
        failing = Trainer(bad, torch.optim.SGD(bad.parameters(), lr=.01),
                          step_fn=lambda model, batch: model(batch).sum() * (float("nan") if rank else 1), device=device)
        try:
            failing.train_epoch([torch.ones(1, 1)])
        except (FloatingPointError, RuntimeError):
            pass
        else:
            raise AssertionError("Nonfinite distributed loss was accepted")

        def make():
            torch.manual_seed(17)
            model = nn.Sequential(nn.Linear(1, 3), nn.Dropout(.2), nn.Linear(3, 1)).double().to(device)
            trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.001, momentum=.9),
                              loss_fn=nn.functional.mse_loss, accumulation_steps=2, device=device)
            data = TensorDataset(torch.arange(12, dtype=torch.double).reshape(-1, 1), torch.ones(12, 1).double())
            # Only one validation example: rank one has an empty shard.
            train, valid, generators = build_loaders(data, TensorDataset(torch.ones(1, 1).double(),
                                                                        torch.zeros(1, 1).double()),
                                                     batch_size=2, seed=7, resumable=True)
            torch.manual_seed(123 + rank)
            return trainer, train, valid, generators

        reference, train, valid, generators = make()
        full = Checkpoint(Path(directory, "full"), every_steps=1, generators=generators)
        reference.fit(train, valid, epochs=2, callbacks=[full], progress_bar=False)
        expected_model = {key: value.clone() for key, value in reference.model.state_dict().items()}
        expected_rng = torch.get_rng_state().clone()
        expected_cuda_rng = torch.cuda.get_rng_state(device) if device_type == "cuda" else None
        expected_generators = {key: value.get_state() for key, value in generators.items()}
        assert full.history[-1]["validation"]["samples"] == 1
        assert full.history[-1]["train"]["samples"] == 12

        class Interrupted(Exception):
            pass

        for boundary in ("step", "epoch"):
            interrupted, train, valid, generators = make()
            checkpoint = Checkpoint(Path(directory, boundary), every_steps=1, generators=generators)

            def interrupt_step(current, event):
                if current.state.optimizer_steps == 1:
                    raise Interrupted()

            def interrupt_epoch(current, row):
                if current.state.completed_epochs == 1:
                    raise Interrupted()

            try:
                interrupted.fit(train, valid, epochs=2, callbacks=[checkpoint] +
                                ([interrupt_epoch] if boundary == "epoch" else []),
                                step_callbacks=[interrupt_step] if boundary == "step" else [], progress_bar=False)
            except Interrupted:
                pass
            if boundary == "step":
                # Restore the same runtime even after a failure between forward
                # and backward, when DDP's old reducer still has pending work.
                loss = interrupted.runtime.compute(torch.ones(1, 1).double(),
                    lambda batch: interrupted.model(batch).sum() * (float("nan") if rank else 1))
                try:
                    interrupted.runtime.backward(loss)
                except (FloatingPointError, RuntimeError):
                    pass
                else:
                    raise AssertionError("Expected a coordinated failure before backward")
                del loss
                resumed, restored = interrupted, checkpoint
            else:
                resumed, train, valid, generators = make()
                restored = Checkpoint(Path(directory, boundary), every_steps=1, generators=generators)
            restored.restore(resumed, train, valid)
            resumed.fit(train, valid, epochs=2, callbacks=[restored], progress_bar=False)
            _equal(resumed.model.state_dict(), expected_model)
            _equal(resumed.optimizer.state_dict(), reference.optimizer.state_dict())
            _equal(torch.get_rng_state(), expected_rng)
            if expected_cuda_rng is not None:
                _equal(torch.cuda.get_rng_state(device), expected_cuda_rng)
            _equal({key: value.get_state() for key, value in generators.items()}, expected_generators)
            for actual, expected_row in zip(restored.history, full.history):
                _equal(actual["validation"], expected_row["validation"])
                _equal(actual["train"]["loss"], expected_row["train"]["loss"])
            payload = read_checkpoint(Path(directory, boundary, "last.pt"))
            assert payload["distributed"]["world_size"] == 2
            assert len(payload["distributed"]["ranks"]) == 2
            assert not any(key.startswith("module.") for key in payload["model"])
        Path(directory, f"rank-{rank}.json").write_text(json.dumps({"ok": True}))
    finally:
        dist.destroy_process_group()


class DistributedBlackBoxTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
    def test_two_rank_training_and_resume(self):
        self._run_ranks("cpu")

    @unittest.skipUnless(torch.cuda.device_count() >= 2 and dist.is_nccl_available(), "Two CUDA devices and NCCL are required")
    def test_two_gpu_training_and_resume(self):
        self._run_ranks("cuda")

    def _run_ranks(self, device_type):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.spawn(_distributed_worker, args=(directory, device_type), nprocs=2, join=False)
            deadline = time.monotonic() + 120
            try:
                while not context.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("Distributed tests exceeded 120 seconds")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            for rank in range(2):
                self.assertTrue(json.loads(Path(directory, f"rank-{rank}.json").read_text())["ok"])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
    def test_torchrun_configured_train_resume_and_single_process_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "ddp.toml"
            config.write_text('name="ddp"\nseed=7\n[training]\nepochs=1\nbatch_size=2\n'
                              'learning_rate=0.01\ncheckpoint_every_steps=1\n[application.synthetic]\n'
                              'train_samples=12\nvalidation_samples=1\ninput_dim=2\nnum_classes=2\n')
            command = ["--module", "deepopt"]
            environment = {**os.environ, "OMP_NUM_THREADS": "1", "DEEPOPT_LOGGING_LEVEL": "ERROR"}
            for arguments in (["train", "--config", str(config), "--output", str(root / "run"),
                               "--batch-size-mode", "manual", "--no-progress"],
                              ["train", "--resume", str(root / "run/last.pt"), "--epochs", "2",
                               "--batch-size-mode", "manual", "--no-progress"]):
                result = run_torchrun(command + arguments, env=environment, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = read_checkpoint(root / "run/last.pt")
            self.assertEqual(payload["state"]["completed_epochs"], 2)
            self.assertEqual(payload["history"][-1]["validation"]["samples"], 1)
            self.assertEqual(len((root / "run/metrics.jsonl").read_text().splitlines()), 2)
            self.assertEqual(len(list((root / "run").glob("*.pt"))), 3)
            result = subprocess.run([sys.executable, "-m", "deepopt", "evaluate", "--checkpoint",
                                     str(root / "run/last.pt"), "--device", "cpu"],
                                    env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
