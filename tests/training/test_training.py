from contextlib import redirect_stdout
from dataclasses import replace
import copy
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepopt.config import config_from_dict
from deepopt.engine import Trainer
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.baseline import build_baseline_experiment
from deepopt.run import evaluate, train
from deepopt.tasks.classification import ClassificationTask
from deepopt.tasks.loss import LossResult
from deepopt.tasks.regression import RegressionTask


def small_config(epochs=4):
    return config_from_dict({
        "name": "test", "seed": 42,
        "training": {"epochs": epochs, "batch_size": 7, "learning_rate": 0.03,
                     "accumulation_steps": 2, "lr_gamma": 0.9, "grad_clip_norm": 1.0,
                     "monitor": "accuracy", "monitor_mode": "max"},
        "application": {"synthetic": {"train_samples": 31, "validation_samples": 13, "input_dim": 5, "num_classes": 3}},
    })


class TrainingTests(unittest.TestCase):
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

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(a, b)

    def test_accumulation_matches_large_batches_with_uneven_tail(self):
        # N=11 produces windows of 6 and 5 samples. N=13 also exercises a
        # final window with fewer microbatches than accumulation_steps.
        for count in (11, 13):
            with self.subTest(samples=count):
                torch.manual_seed(4)
                data = TensorDataset(torch.randn(count, 5), torch.randint(0, 3, (count,)))
                model = nn.Linear(5, 3)
                reference = copy.deepcopy(model)
                trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1),
                                  task=ClassificationTask(), accumulation_steps=2)
                large = Trainer(reference, torch.optim.SGD(reference.parameters(), lr=0.1), task=ClassificationTask())
                metrics = trainer.train_epoch(DataLoader(data, batch_size=3))
                large.train_epoch(DataLoader(data, batch_size=6))
                for left, right in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
                self.assertEqual(trainer.state.optimizer_steps, (count + 5) // 6)
                self.assertEqual(metrics["samples"], count)

    def test_validation_weights_examples_and_restores_mode_without_gradients(self):
        torch.manual_seed(2)
        model = nn.Linear(5, 3)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), task=ClassificationTask())
        x, y = torch.randn(11, 5), torch.randint(0, 3, (11,))
        expected = nn.functional.cross_entropy(model(x), y).item()
        before = copy.deepcopy(model.state_dict())
        for mode in (True, False):
            model.train(mode)
            metrics = trainer.evaluate(DataLoader(TensorDataset(x, y), batch_size=4))
            self.assertAlmostEqual(metrics["loss"], expected, places=6)
            self.assertEqual(metrics["samples"], 11)
            self.assertEqual(model.training, mode)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assert_nested_equal(before, model.state_dict())

    def test_nonfinite_loss_blocks_checkpoint_and_clears_gradients(self):
        config = small_config()
        trainer, _, _, generators = build_baseline_experiment(config)
        before = copy.deepcopy(trainer.model.state_dict())
        bad = DataLoader(TensorDataset(torch.full((2, 5), float("nan")), torch.zeros(2, dtype=torch.long)))
        with self.assertRaises(FloatingPointError):
            trainer.train_epoch(bad)
        self.assert_nested_equal(before, trainer.model.state_dict())
        self.assertTrue(all(p.grad is None for p in trainer.model.parameters()))
        with self.assertRaisesRegex(ValueError, "epoch boundary"):
            save_checkpoint(self.root / "bad.pt", trainer, config, generators=generators, history=[])
        with self.assertRaisesRegex(RuntimeError, "restore"):
            trainer.train_epoch(bad)

    def test_empty_loaders_fail_and_evaluation_restores_mode(self):
        trainer, _, _, _ = build_baseline_experiment(small_config())
        empty = DataLoader(TensorDataset(torch.empty(0, 5), torch.empty(0, dtype=torch.long)))
        with self.assertRaisesRegex(ValueError, "empty"):
            trainer.evaluate(empty)
        self.assertTrue(trainer.model.training)
        with self.assertRaisesRegex(ValueError, "empty"):
            trainer.train_epoch(empty)
        self.assertEqual(trainer.state.completed_epochs, 0)

    def test_loss_contract_accepts_explicit_batch_mean(self):
        trainer, loader, _, _ = build_baseline_experiment(small_config())
        trainer.task = ClassificationTask(
            lambda predictions, targets: LossResult.from_mean(
                nn.functional.cross_entropy(predictions, targets), weight=targets.shape[0]),
        )
        self.assertEqual(trainer.train_epoch(loader)["samples"], 31)

    def test_custom_classification_loss_controls_gradients(self):
        trainer, loader, _, _ = build_baseline_experiment(small_config())
        calls = []

        def custom_loss(logits, targets):
            calls.append(targets.numel())
            return nn.functional.cross_entropy(logits, targets, reduction="none").mean() * 0.0

        trainer.task = ClassificationTask(loss_fn=custom_loss)
        before = copy.deepcopy(trainer.model.state_dict())
        metrics = trainer.train_epoch(loader)
        self.assertEqual(sum(calls), 31)
        self.assertEqual(metrics["loss"], 0.0)
        self.assert_nested_equal(before, trainer.model.state_dict())

    def test_custom_regression_loss(self):
        model = nn.Linear(3, 1)
        x, y = torch.ones(5, 3), torch.zeros(5, 1)

        def absolute_loss(predictions, targets):
            return (predictions - targets).abs().mean()

        output = RegressionTask(loss_fn=absolute_loss).compute(model, (x, y))
        torch.testing.assert_close(output.loss, absolute_loss(model(x), y))
        self.assertEqual(output.objective.normalization_weight, 1)
        with self.assertRaisesRegex(ValueError, "scalar"):
            RegressionTask(loss_fn=nn.MSELoss(reduction="none")).compute(model, (x, y))

    def test_fit_controls_epoch_order_and_callback_stop(self):
        trainer, loader, validation, _ = build_baseline_experiment(small_config())
        epochs_seen = []

        def stop_after_two(trainer, metrics):
            epochs_seen.append(metrics["epoch"])
            self.assertEqual(metrics["epoch"], trainer.state.completed_epochs)
            self.assertIn("validation", metrics)
            self.assertTrue(trainer.at_epoch_boundary)
            if metrics["epoch"] == 2:
                trainer.request_stop()

        trainer.fit(loader, validation, epochs=10, callbacks=[stop_after_two])
        self.assertEqual(epochs_seen, [1, 2])
        self.assertEqual(trainer.state.completed_epochs, 2)
        self.assertEqual(list(self.root.iterdir()), [])  # The core creates no files.

    def test_failed_callback_requires_restore_before_continuing(self):
        trainer, loader, validation, _ = build_baseline_experiment(small_config())

        def fail(trainer, metrics):
            raise OSError("log write failed")

        with self.assertRaisesRegex(OSError, "log write failed"):
            trainer.fit(loader, validation, epochs=3, callbacks=[fail])
        self.assertFalse(trainer.at_epoch_boundary)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            trainer.train_epoch(loader)

    def test_training_reduces_loss(self):
        config = small_config(20)
        config = replace(config, training=replace(config.training, lr_gamma=1.0))
        trainer, loader, _, _ = build_baseline_experiment(config)
        before = trainer.evaluate(loader)["loss"]
        for _ in range(20):
            trainer.train_epoch(loader)
        after = trainer.evaluate(loader)["loss"]
        self.assertLess(after, before * 0.5)

    def test_same_engine_supports_regression(self):
        torch.manual_seed(7)
        x = torch.randn(17, 3)
        y = x.sum(dim=1, keepdim=True)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        model = nn.Linear(3, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1),
                          task=RegressionTask(), accumulation_steps=2)
        before = trainer.evaluate(loader)
        for _ in range(20):
            trainer.train_epoch(loader)
        after = trainer.evaluate(loader)
        self.assertLess(after["loss"], before["loss"] * 0.01)
        self.assertLess(after["mse"], before["mse"])
        self.assertNotIn("accuracy", after)

    def test_factory_injection_and_minimum_loss_checkpoint_selection(self):
        def factory(config):
            def dataset_factory():
                generator = torch.Generator().manual_seed(config.seed)
                x = torch.randn(31, 5, generator=generator)
                return (TensorDataset(x[:20], x[:20].sum(1, keepdim=True)),
                        TensorDataset(x[20:], x[20:].sum(1, keepdim=True)))

            return build_baseline_experiment(
                config, model_factory=lambda: nn.Linear(5, 1),
                dataset_factory=dataset_factory, task_factory=RegressionTask,
                scheduler_factory=lambda optimizer, training: None)
        config = small_config(3)
        config = replace(config, training=replace(config.training, monitor="loss", monitor_mode="min"))
        run_dir = self.root / "custom"
        with redirect_stdout(io.StringIO()):
            train(config, output=run_dir, experiment_factory=factory)
            train(resume=run_dir / "last.pt", epochs=4, experiment_factory=factory)
        last = read_checkpoint(run_dir / "last.pt")
        best = evaluate(run_dir / "best.pt", experiment_factory=factory)
        self.assertEqual(best["validation"]["loss"], min(row["validation"]["loss"] for row in last["history"]))
        self.assertIn("mse", best["validation"])

    def test_validation_restores_mixed_submodule_modes(self):
        trainer, _, loader, _ = build_baseline_experiment(small_config())
        trainer.model.train()
        trainer.model.layers[2].eval()
        before = [module.training for module in trainer.model.modules()]
        trainer.evaluate(loader)
        self.assertEqual([module.training for module in trainer.model.modules()], before)

    def test_checkpoint_rejects_different_component_type(self):
        config = small_config()
        trainer, _, _, generators = build_baseline_experiment(config)
        path = self.root / "state.pt"
        save_checkpoint(path, trainer, config, generators=generators, history=[])
        trainer.task = RegressionTask()
        with self.assertRaisesRegex(ValueError, "component types"):
            restore_checkpoint(read_checkpoint(path), trainer, config, generators=generators)

    def test_resume_matches_uninterrupted_dropout_shuffle_adam_scheduler(self):
        full, resumed = self.root / "full", self.root / "resumed"
        with redirect_stdout(io.StringIO()):
            train(small_config(4), output=full)
            train(small_config(2), output=resumed)
            # Log repair must remove records not present in the saved checkpoint.
            with (resumed / "metrics.jsonl").open("a") as file:
                file.write('{"epoch": 999}\n')
            train(resume=resumed / "last.pt", epochs=4)
        expected, actual = read_checkpoint(full / "last.pt"), read_checkpoint(resumed / "last.pt")
        for key in ("model", "optimizer", "scheduler", "state", "rng", "generators"):
            self.assert_nested_equal(expected[key], actual[key])
        for left, right in zip(expected["history"], actual["history"]):
            for key in ("epoch", "learning_rate", "learning_rates", "optimizer_steps", "validation"):
                self.assertEqual(left[key], right[key])
            for key in ("loss", "accuracy", "samples"):
                self.assertEqual(left["train"][key], right["train"][key])
        self.assertEqual(len((resumed / "metrics.jsonl").read_text().splitlines()), 4)
        best = read_checkpoint(resumed / "best.pt")
        self.assertEqual(evaluate(resumed / "best.pt")["validation"]["accuracy"], actual["state"]["best_score"])
        best_epoch = max(actual["history"], key=lambda row: row["validation"]["accuracy"])["epoch"]
        self.assertEqual(best["state"]["completed_epochs"], best_epoch)

    def test_checkpoint_restores_python_and_torch_rng(self):
        config = small_config()
        trainer, _, _, generators = build_baseline_experiment(config)
        path = self.root / "state.pt"
        save_checkpoint(path, trainer, config, generators=generators, history=[])
        expected = random.random(), torch.rand(4), torch.rand(4, generator=generators["train"])
        random.seed(999)
        torch.manual_seed(999)
        restore_checkpoint(read_checkpoint(path), trainer, config, generators=generators)
        actual = random.random(), torch.rand(4), torch.rand(4, generator=generators["train"])
        self.assert_nested_equal(expected, actual)

    def test_failed_checkpoint_write_keeps_previous_file(self):
        config = small_config()
        trainer, _, _, generators = build_baseline_experiment(config)
        path = self.root / "state.pt"
        save_checkpoint(path, trainer, config, generators=generators, history=[])
        before = path.read_bytes()
        with patch("deepopt.engine.checkpointing.checkpoint.torch.save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                save_checkpoint(path, trainer, config, generators=generators, history=[])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_resume_rejects_changed_configuration_without_overwriting(self):
        run = self.root / "run"
        with redirect_stdout(io.StringIO()):
            train(small_config(2), output=run)
        before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        changed = replace(small_config(3), training=replace(small_config(3).training, learning_rate=0.04))
        with self.assertRaisesRegex(ValueError, "only epochs"):
            train(changed, resume=run / "last.pt")
        with self.assertRaisesRegex(ValueError, "already exists"):
            train(small_config(2), output=run)
        with self.assertRaisesRegex(ValueError, "cannot be less"):
            train(resume=run / "last.pt", epochs=1)
        self.assertEqual(before, {str(p.relative_to(run)): p.read_bytes() for p in run.rglob("*") if p.is_file()})

    def test_resume_completed_budget_repairs_logs_without_training_more(self):
        run = self.root / "completed"
        with redirect_stdout(io.StringIO()):
            train(small_config(1), output=run)
        before = (run / "last.pt").read_bytes()
        history = read_checkpoint(run / "last.pt")["history"]
        with (run / "metrics.jsonl").open("a") as file:
            file.write('{"epoch": 999}\n')

        with patch.object(Trainer, "fit", side_effect=AssertionError("unexpected extra training")):
            restored = train(resume=run / "last.pt", epochs=1)

        self.assertEqual(restored, run.resolve())
        self.assertEqual((run / "last.pt").read_bytes(), before)
        self.assertEqual([json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()], history)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA hardware unavailable")
    def test_bf16_training_and_evaluation(self):
        config = small_config(1)
        config = replace(config, runtime=replace(config.runtime, device="cuda", precision="bf16"))
        trainer, loader, validation, _ = build_baseline_experiment(config)
        before = copy.deepcopy(trainer.model.state_dict())
        trainer.train_epoch(loader)
        self.assertGreater(trainer.state.optimizer_steps, 0)
        self.assertTrue(any(not torch.equal(before[key], value) for key, value in trainer.model.state_dict().items()))
        self.assertGreaterEqual(trainer.evaluate(validation)["loss"], 0)


if __name__ == "__main__":
    unittest.main()
