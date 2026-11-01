import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepopt.engine import Trainer
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.engine.strategies import ExplicitObjective
from deepopt.tasks.base import BatchOutput
from deepopt.tasks.classification import ClassificationTask
from deepopt.tasks.loss import LossResult
from deepopt.tasks.regression import RegressionTask


class LossScalingTests(unittest.TestCase):
    def test_task_validation_never_converts_the_loss(self):
        scalar = torch.tensor(2.0, requires_grad=True)
        with patch.object(BatchOutput, "loss_sum", side_effect=AssertionError("unexpected sum conversion")):
            for reduction in ("mean", "sum"):
                custom_loss = (lambda predictions, targets: LossResult.from_mean(scalar, weight=targets.shape[0])
                               if reduction == "mean" else
                               LossResult.from_sum(scalar, report_weight=targets.shape[0]))
                for task, target in (
                    (ClassificationTask(loss_fn=custom_loss), torch.zeros(3, dtype=torch.long)),
                    (RegressionTask(loss_fn=custom_loss), torch.zeros(3, 2)),
                ):
                    with self.subTest(task=type(task).__name__, reduction=reduction):
                        output = task.compute(nn.Identity(), (torch.ones(3, 2), target))
                        self.assertIs(output.loss, scalar)
                        self.assertEqual(output.sample_count, 3)

    def test_validation_checks_shape_and_values_without_scaling(self):
        # Finite scalar means should not overflow just because validation
        # multiplies them by a batch size it does not need.
        output = BatchOutput(torch.tensor(2e38), 4, "mean")
        output.validate()
        self.assertTrue(torch.isfinite(output.loss))
        with self.assertRaisesRegex(FloatingPointError, "accumulated loss"):
            output.loss_sum()
        for invalid in (BatchOutput(torch.tensor(float("nan")), 1),
                        BatchOutput(torch.ones(2), 2, "mean"),
                        BatchOutput(torch.tensor(2, dtype=torch.long), 2)):
            with self.assertRaises((ValueError, FloatingPointError)):
                invalid.validate()

    def test_normal_scalar_losses_pass_through_unchanged(self):
        for reduction in ("mean", "sum"):
            loss = torch.tensor(2.0, requires_grad=True)
            prepared = ExplicitObjective().prepare(BatchOutput(loss, 3, reduction), accumulation_steps=1)
            self.assertIs(prepared.loss, loss)
            self.assertIsNone(prepared.normalization_weight)

    def test_normal_training_matches_plain_pytorch_without_sum_conversion(self):
        for reduction in ("mean", "sum"):
            with self.subTest(reduction=reduction):
                torch.manual_seed(8)
                model = nn.Linear(3, 2).double()
                reference = copy.deepcopy(model)
                data = TensorDataset(torch.randn(7, 3, dtype=torch.double), torch.randint(0, 2, (7,)))
                loader = DataLoader(data, batch_size=3)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
                trainer = Trainer(model, optimizer, task=ClassificationTask(num_classes=2, reduction_type=reduction))
                with patch.object(BatchOutput, "loss_sum", side_effect=AssertionError("unexpected sum conversion")):
                    trainer.train_epoch(loader)
                    trainer.evaluate(loader)
                for inputs, targets in loader:
                    reference_optimizer.zero_grad()
                    nn.functional.cross_entropy(reference(inputs), targets, reduction=reduction).backward()
                    reference_optimizer.step()
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_accumulation_preserves_sum_objective_including_partial_window(self):
        torch.manual_seed(9)
        model = nn.Linear(3, 2).double()
        reference = copy.deepcopy(model)
        data = TensorDataset(torch.randn(13, 3, dtype=torch.double), torch.randint(0, 2, (13,)))
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                          task=ClassificationTask(reduction_type="sum"), accumulation_steps=2)
        trainer.train_epoch(DataLoader(data, batch_size=3))
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        for inputs, targets in DataLoader(data, batch_size=6):
            optimizer.zero_grad()
            nn.functional.cross_entropy(reference(inputs), targets, reduction="sum").backward()
            optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(trainer.state.optimizer_steps, 3)

    def test_unreduced_loss_uses_mean_without_accumulation(self):
        loss = torch.tensor([1.0, 3.0], requires_grad=True)
        prepared = ExplicitObjective().prepare(BatchOutput(loss, 2, "none"), accumulation_steps=1)
        self.assertIsNone(prepared.normalization_weight)
        prepared.loss.backward()
        torch.testing.assert_close(loss.grad, torch.tensor([0.5, 0.5]))

    def test_resume_rejects_previous_implicit_normalization_policy(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), task=RegressionTask())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, trainer, {}, generators={}, history=[])
            payload = read_checkpoint(path)
        payload["trainer_settings"].pop("loss_scaling")
        with self.assertRaisesRegex(ValueError, "trainer settings"):
            restore_checkpoint(payload, trainer, {}, generators={})

    def test_resume_rejects_previous_literal_scalar_policy_before_mutation(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), task=RegressionTask())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, trainer, {}, generators={}, history=[])
            payload = read_checkpoint(path)
        payload["trainer_settings"]["loss_scaling"] = "explicit_objective_v2"
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.)
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, "trainer settings"):
            restore_checkpoint(payload, trainer, {}, generators={})
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        # Rejection occurs before restore invalidates the healthy runtime.
        trainer.train_epoch([(torch.ones(2, 2), torch.zeros(2, 1))])
        self.assertEqual(trainer.state.optimizer_steps, 1)

    def test_mixed_objectives_cannot_silently_change_window_normalization(self):
        class MixedTask(RegressionTask):
            def compute(self, model, batch):
                output = super().compute(model, batch)
                self.reduction_type = "sum"
                return output

        model = nn.Linear(2, 1)
        before = copy.deepcopy(model.state_dict())
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1),
                          task=MixedTask(), accumulation_steps=2)
        batches = [(torch.ones(3, 2), torch.zeros(3, 1))] * 2
        with self.assertRaisesRegex(ValueError, "Cannot mix weighted and unweighted"):
            trainer.train_epoch(batches)
        self.assertEqual(trainer.state.optimizer_steps, 0)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
