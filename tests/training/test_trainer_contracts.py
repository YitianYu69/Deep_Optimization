from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from deepopt.config import config_from_dict
from deepopt.data.loaders import build_loaders
from deepopt.engine import Trainer
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.tasks.classification import ClassificationTask
from deepopt.tasks.loss import LossResult
from deepopt.tasks.regression import RegressionTask


class FixedUpdateOptimizer(torch.optim.Optimizer):
    """A valid optimizer whose groups do not contain a learning-rate setting."""

    def __init__(self, parameters):
        super().__init__(parameters, defaults={})
        self.calls = 0

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-0.01)
        self.calls += 1


class InterruptedTraining(RuntimeError):
    pass


class TrainerContractTests(unittest.TestCase):
    def setUp(self):
        self.batches = [(torch.ones(2, 2), torch.ones(2, 1))]

    def trainer(self):
        model = nn.Linear(2, 1, bias=False)
        return Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), task=RegressionTask())

    def assert_model_equal(self, expected, model):
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(expected[name], value), name)

    def test_manual_epoch_does_not_lock_the_first_fit_monitor(self):
        trainer = self.trainer()
        trainer.train_epoch(self.batches)
        self.assertEqual(trainer.state.completed_epochs, 1)
        self.assertIsNone(trainer.state.monitor_mode)
        trainer.fit(self.batches, self.batches, epochs=2, monitor="mse")
        self.assertEqual(trainer.state.completed_epochs, 2)
        self.assertEqual(trainer.state.monitor, "mse")
        self.assertEqual(trainer.state.monitor_mode, "min")
        self.assertEqual(trainer.state.best_epoch, 2)

    def test_disabled_selection_remains_a_locked_fit_choice(self):
        trainer = self.trainer()
        trainer.fit(self.batches, self.batches, epochs=1, monitor=None)
        steps = trainer.state.optimizer_steps
        model_state = deepcopy(trainer.model.state_dict())
        for monitor, mode in (("loss", "min"), (None, "max")):
            with self.subTest(monitor=monitor, mode=mode):
                with self.assertRaisesRegex(ValueError, "Cannot change the monitored metric"):
                    trainer.fit(self.batches, self.batches, epochs=2, monitor=monitor, monitor_mode=mode)
                self.assertEqual(trainer.state.optimizer_steps, steps)
                self.assert_model_equal(model_state, trainer.model)
        trainer.fit(self.batches, self.batches, epochs=2, monitor=None)
        self.assertEqual(trainer.state.completed_epochs, 2)
        self.assertIsNone(trainer.state.best_score)
        self.assertEqual(trainer.state.best_epoch, 0)

    def test_first_epoch_step_resume_keeps_the_original_monitor_before_any_update(self):
        config = config_from_dict({"name": "monitor-resume", "seed": 7,
                                   "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.01}})
        dataset = TensorDataset(torch.ones(6, 2), torch.ones(6, 1))
        for original_monitor, changed_monitor in ((None, "loss"), ("loss", "mse")):
            with self.subTest(monitor=original_monitor), tempfile.TemporaryDirectory() as directory:
                trainer = self.trainer()
                loader, validation, generators = build_loaders(dataset, dataset, batch_size=2, seed=7, resumable=True)
                path = Path(directory) / "last_step.pt"

                def save_then_interrupt(current, event):
                    save_checkpoint(path, current, config, generators=generators, history=[],
                                    kind="step", train_loader=loader)
                    raise InterruptedTraining("stop after the first saved update")

                with self.assertRaises(InterruptedTraining):
                    trainer.fit(loader, validation, epochs=1, monitor=original_monitor,
                                step_callbacks=[save_then_interrupt])
                restored = self.trainer()
                loader, validation, generators = build_loaders(dataset, dataset, batch_size=2, seed=7, resumable=True)
                restore_checkpoint(read_checkpoint(path), restored, config,
                                   generators=generators, train_loader=loader)
                self.assertEqual(restored.state.completed_epochs, 0)
                self.assertEqual(restored.state.optimizer_steps, 1)
                before = deepcopy(restored.model.state_dict())
                cursor = loader.state_dict()["sampler"]["position"]
                with self.assertRaisesRegex(ValueError, "Cannot change the monitored metric"):
                    restored.fit(loader, validation, epochs=1, monitor=changed_monitor)
                self.assertEqual(restored.state.optimizer_steps, 1)
                self.assertEqual(loader.state_dict()["sampler"]["position"], cursor)
                self.assert_model_equal(before, restored.model)
                restored.fit(loader, validation, epochs=1, monitor=original_monitor)
                self.assertEqual(restored.state.completed_epochs, 1)
                self.assertEqual(restored.state.optimizer_steps, 3)

    def test_optimizer_without_learning_rate_updates_and_reports_missing_rates(self):
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = FixedUpdateOptimizer(model.parameters())
        trainer = Trainer(model, optimizer, task=RegressionTask())
        before = model.weight.detach().clone()
        rows = []
        trainer.fit(self.batches, self.batches, epochs=2,
                    callbacks=[lambda _, row: rows.append(row)])
        self.assertEqual(optimizer.calls, 2)
        self.assertFalse(torch.equal(before, model.weight))
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIsNone(row["learning_rate"])
            self.assertEqual(row["learning_rates"], [None])
        self.assertNotIn("lr", optimizer.param_groups[0])

    def test_schedulers_must_identify_the_correct_optimizer_and_callable_step(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        other_optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        for scheduler in (SimpleNamespace(step=lambda: None),
                          SimpleNamespace(optimizer=other_optimizer, step=lambda: None)):
            with self.subTest(scheduler=scheduler), self.assertRaisesRegex(ValueError, "Scheduler must belong"):
                Trainer(model, optimizer, task=RegressionTask(), scheduler=scheduler)
        for scheduler in (SimpleNamespace(optimizer=optimizer),
                          SimpleNamespace(optimizer=optimizer, step=None)):
            with self.subTest(scheduler=scheduler), self.assertRaisesRegex(ValueError, "scheduler.step"):
                Trainer(model, optimizer, task=RegressionTask(), scheduler=scheduler)

    def test_replaced_classification_loss_accepts_a_scalar_or_explicit_weights(self):
        task = ClassificationTask(num_classes=3)
        model = nn.Linear(2, 3)
        batch = torch.ones(4, 2), torch.tensor([0, 1, 2, 0])
        task.compute(model, batch)  # Exercise the built-in path before replacing it.
        task.loss_fn = lambda logits, targets: F.cross_entropy(logits, targets, reduction="none")
        with self.assertRaisesRegex(ValueError, "scalar"):
            task.compute(model, batch)
        task.loss_fn = lambda logits, targets: F.cross_entropy(logits, targets, reduction="none").mean()
        reduced = task.compute(model, batch)
        expected = F.cross_entropy(model(batch[0]), batch[1])
        torch.testing.assert_close(reduced.loss, expected)
        self.assertEqual(reduced.objective.normalization_weight, 1)
        self.assertEqual(reduced.loss_report.weight, 1)
        task.loss_fn = lambda logits, targets: LossResult.from_mean(
            F.cross_entropy(logits, targets), weight=targets.shape[0])
        explicit = task.compute(model, batch)
        torch.testing.assert_close(explicit.loss, expected)
        self.assertEqual(explicit.objective.normalization_weight, 4)


if __name__ == "__main__":
    unittest.main()
