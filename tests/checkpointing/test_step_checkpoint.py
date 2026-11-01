from contextlib import redirect_stdout
from dataclasses import replace
from functools import partial
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset
from torchmetrics import Metric

from deepopt.config import config_from_dict
from deepopt.data import build_loaders
from deepopt.data.synthetic import SyntheticDataSettings, build_datasets
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, require_step_loader, restore_checkpoint, save_checkpoint
from deepopt.baseline import build_baseline_experiment
from deepopt.optimizers import MultiOptimizer
from deepopt.run import train


class InterruptedRun(RuntimeError):
    pass


class NoisyDataset(Dataset):
    """Use both global RNGs to detect replaying already-consumed examples."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        inputs, target = self.dataset[index]
        return inputs + 0.01 * torch.randn_like(inputs) + 0.01 * random.random(), target


def noisy_data(settings, seed):
    training, validation = build_datasets(settings, seed)
    return NoisyDataset(training), validation


def noisy_experiment(config, **overrides):
    settings = SyntheticDataSettings.from_dict(config.application.get("synthetic", {}))
    return build_baseline_experiment(
        config, dataset_factory=partial(noisy_data, settings, config.seed), **overrides)


def tensor_batch(samples):
    return torch.tensor([inputs for inputs, _ in samples]), torch.tensor([target for _, target in samples])


class PickledTensorBatch:
    """Use ordinary IPC; restricted environments may lack torch_shm_manager."""

    def __init__(self, samples):
        self.samples = samples

    def __reduce__(self):
        return tensor_batch, (self.samples,)


def prefetched_loaders(training, validation, *, batch_size, seed):
    samples = [(inputs.tolist(), target.item()) for inputs, target in training]
    valid_samples = [(inputs.tolist(), target.item()) for inputs, target in validation]
    return build_loaders(samples, valid_samples, batch_size=batch_size, seed=seed, resumable=True,
                         num_workers=2, prefetch_factor=3, persistent_workers=True,
                         multiprocessing_context="spawn", timeout=20, collate_fn=PickledTensorBatch)


class CustomLossMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, output):
        self.total += output.loss_sum().detach().double()
        self.count += output.sample_count

    def compute(self):
        return {"loss": self.total / self.count, "samples": self.count.clone()}


def multiple_optimizers(model, settings):
    return MultiOptimizer({
        "first": torch.optim.AdamW(model.layers[0].parameters(), lr=settings.learning_rate),
        "last": torch.optim.SGD(model.layers[3].parameters(), lr=settings.learning_rate, momentum=0.9),
    })


class StepCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def config(self, samples=31):
        return config_from_dict({
            "name": "step-resume", "seed": 42,
            "training": {"epochs": 3, "batch_size": 7, "learning_rate": 0.02,
                         "accumulation_steps": 2, "lr_gamma": 0.8, "checkpoint_every_steps": 1},
            "application": {"synthetic": {"train_samples": samples, "validation_samples": 13}},
        })

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(a, b)

    def interrupt(self, config, output, step, factory=build_baseline_experiment):
        def save_then_interrupt(path, trainer, *args, **kwargs):
            save_checkpoint(path, trainer, *args, **kwargs)
            if kwargs.get("kind") == "step" and trainer.state.optimizer_steps == step:
                raise InterruptedRun("simulated process interruption after durable step save")

        with patch("deepopt.run.save_checkpoint", side_effect=save_then_interrupt):
            with self.assertRaises(InterruptedRun):
                train(config, output=output, experiment_factory=factory)

    def compare_completed_runs(self, full, resumed):
        expected, actual = read_checkpoint(full / "last.pt"), read_checkpoint(resumed / "last.pt")
        for key in ("model", "optimizer", "scheduler", "state", "rng", "generators"):
            self.assert_nested_equal(expected[key], actual[key])
        self.assertEqual(len(expected["history"]), len(actual["history"]))
        for left, right in zip(expected["history"], actual["history"]):
            for key in ("epoch", "learning_rates", "optimizer_steps", "validation"):
                self.assert_nested_equal(left[key], right[key])
            for key in ("loss", "samples"):
                self.assertEqual(left["train"][key], right["train"][key])
        self.assert_nested_equal(read_checkpoint(full / "best.pt")["model"],
                                 read_checkpoint(resumed / "best.pt")["model"])

    def test_mid_epoch_and_final_partial_window_resume_are_exact(self):
        # 31/7 -> 5 microbatches -> 3 optimizer updates per epoch.
        # Stop inside epoch one, after its final partial window, and in epoch two.
        factory = noisy_experiment
        with redirect_stdout(io.StringIO()):
            full = self.root / "full"
            train(self.config(), output=full, experiment_factory=factory)
            for step in (1, 3, 4):
                with self.subTest(step=step):
                    resumed = self.root / f"resume-{step}"
                    self.interrupt(self.config(), resumed, step, factory)
                    saved = read_checkpoint(resumed / "last_step.pt")
                    self.assertEqual(saved["kind"], "step")
                    self.assertEqual(saved["state"]["optimizer_steps"], step)
                    self.assertEqual(len(saved["history"]), saved["state"]["completed_epochs"])
                    train(resume=resumed / "last_step.pt", experiment_factory=factory)
                    self.compare_completed_runs(full, resumed)

    def test_final_full_accumulation_window_does_not_repeat_epoch(self):
        config = self.config(samples=28)  # Exactly two full update windows.
        with redirect_stdout(io.StringIO()):
            full, resumed = self.root / "full", self.root / "resumed"
            train(config, output=full)
            self.interrupt(config, resumed, 2)
            train(resume=resumed / "last_step.pt")
            self.compare_completed_runs(full, resumed)

    def test_prefetched_worker_resume_matches_uninterrupted_training(self):
        factory = partial(build_baseline_experiment, loader_factory=prefetched_loaders)
        with redirect_stdout(io.StringIO()):
            full = self.root / "full-workers"
            train(self.config(), output=full, experiment_factory=factory)
            # Mid-epoch, the last partial window, and an epoch checkpoint.
            for step, checkpoint in ((1, "last_step.pt"), (3, "last_step.pt"), (4, "last.pt")):
                with self.subTest(step=step, checkpoint=checkpoint):
                    resumed = self.root / f"workers-resume-{step}"
                    self.interrupt(self.config(), resumed, step, factory)
                    train(resume=resumed / checkpoint, experiment_factory=factory)
                    self.compare_completed_runs(full, resumed)

    def test_custom_metrics_and_multiple_optimizer_states_resume_exactly(self):
        config = self.config()
        config = replace(config, training=replace(config.training, optimizer="custom", monitor="loss", monitor_mode="min"))
        factory = partial(noisy_experiment, optimizer_factory=multiple_optimizers,
                          metric_factory=CustomLossMetric)
        with redirect_stdout(io.StringIO()):
            full, resumed = self.root / "full", self.root / "resumed"
            train(config, output=full, experiment_factory=factory)
            self.interrupt(config, resumed, 4, factory)
            saved = read_checkpoint(resumed / "last_step.pt")
            self.assertGreater(saved["epoch_progress"]["metrics"]["state"]["count"].item(), 0)
            train(resume=resumed / "last_step.pt", experiment_factory=factory)
            self.compare_completed_runs(full, resumed)

    def test_exactly_three_checkpoint_files_and_step_cadence(self):
        config = self.config()
        config = replace(config, training=replace(config.training, checkpoint_every_steps=2))
        with redirect_stdout(io.StringIO()):
            train(config, output=self.root / "run")
        self.assertEqual({p.name for p in (self.root / "run").glob("*.pt")}, {"best.pt", "last.pt", "last_step.pt"})
        last = read_checkpoint(self.root / "run" / "last.pt")
        step = read_checkpoint(self.root / "run" / "last_step.pt")
        self.assertEqual(last["state"]["optimizer_steps"], 9)
        self.assertEqual(step["state"]["optimizer_steps"], 8)
        with self.assertRaisesRegex(ValueError, "older than last.pt"):
            train(resume=self.root / "run" / "last_step.pt", epochs=4)

    def test_step_saving_disabled(self):
        config = self.config()
        config = replace(config, training=replace(config.training, checkpoint_every_steps=0))
        with redirect_stdout(io.StringIO()):
            train(config, output=self.root / "run")
        self.assertFalse((self.root / "run" / "last_step.pt").exists())

    def test_step_checkpoint_requires_update_boundary_and_resumable_loader(self):
        config = self.config()
        trainer, loader, validation, generators = build_baseline_experiment(config)
        with self.assertRaisesRegex(ValueError, "update boundary"):
            save_checkpoint(self.root / "bad.pt", trainer, config, generators=generators, history=[],
                            kind="step", train_loader=loader)

        def save_with_wrong_loader(trainer, event):
            save_checkpoint(self.root / "bad.pt", trainer, config, generators=generators, history=[],
                            kind="step", train_loader=validation)

        with self.assertRaisesRegex(ValueError, "resumable loader"):
            trainer.train_epoch(loader, step_callbacks=[save_with_wrong_loader])
        self.assertFalse((self.root / "bad.pt").exists())

    def test_step_restore_requires_loader_and_matching_metric(self):
        with redirect_stdout(io.StringIO()):
            self.interrupt(self.config(), self.root / "run", 1)
        payload = read_checkpoint(self.root / "run" / "last_step.pt")
        trainer, loader, _, generators = build_baseline_experiment(self.config())
        with self.assertRaisesRegex(ValueError, "resumable loader"):
            restore_checkpoint(payload, trainer, self.config(), generators=generators)
        trainer.metric_factory = CustomLossMetric
        with self.assertRaisesRegex(ValueError, "same metric class"):
            restore_checkpoint(payload, trainer, self.config(), generators=generators, train_loader=loader)

    def test_epoch_restore_discards_a_previously_restored_step_cursor(self):
        with redirect_stdout(io.StringIO()):
            self.interrupt(self.config(), self.root / "run", 4)
        trainer, loader, _, generators = build_baseline_experiment(self.config())
        restore_checkpoint(read_checkpoint(self.root / "run" / "last_step.pt"), trainer,
                           self.config(), generators=generators, train_loader=loader)
        restore_checkpoint(read_checkpoint(self.root / "run" / "last.pt"), trainer,
                           self.config(), generators=generators, train_loader=loader)
        metrics = trainer.train_epoch(loader)
        self.assertEqual(metrics["samples"], 31)
        self.assertEqual(trainer.state.optimizer_steps, 6)

    def test_step_loader_interface_is_independent_of_worker_count(self):
        # Interface checks only: these objects do not simulate worker execution.
        for attributes in ({}, {"num_workers": 0}, {"num_workers": 4}):
            with self.subTest(attributes=attributes):
                loader = SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda state: None,
                                         **attributes)
                require_step_loader(loader)

    def test_step_loader_requires_both_callable_state_methods(self):
        for loader in (None, object(), SimpleNamespace(state_dict=lambda: {}),
                       SimpleNamespace(load_state_dict=lambda state: None),
                       SimpleNamespace(state_dict={}, load_state_dict=lambda state: None),
                       SimpleNamespace(state_dict=lambda: {}, load_state_dict=None)):
            with self.subTest(loader=loader):
                with self.assertRaisesRegex(ValueError, "resumable loader"):
                    require_step_loader(loader)


if __name__ == "__main__":
    unittest.main()
