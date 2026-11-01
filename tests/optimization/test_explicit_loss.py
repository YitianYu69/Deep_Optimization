import copy
from fractions import Fraction
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt.engine import Trainer, TrainingRuntime
from deepopt.engine.reporting.metrics import MetricAccumulator
from deepopt.module import TrainingModule
from deepopt.tasks.base import BatchOutput
from deepopt.tasks.classification import ClassificationTask
from deepopt.tasks.loss import LossReport, LossResult
from deepopt.tasks.regression import RegressionTask


class LiteralObjectiveModule(TrainingModule):
    def training_step(self, batch):
        inputs, targets = batch
        return BatchOutput(LossResult((self(inputs) - targets).square().sum()), inputs.shape[0])


class WeightedObjectiveModule(TrainingModule):
    def training_step(self, batch):
        inputs, targets, weights = batch
        denominator = weights.sum().item()
        mean = ((self(inputs) - targets).square() * weights).sum() / denominator
        return BatchOutput(LossResult.from_mean(mean, weight=denominator), inputs.shape[0])


class VectorObjectiveModule(TrainingModule):
    def training_step(self, batch):
        inputs, targets = batch
        return BatchOutput((self(inputs) - targets).square(), inputs.shape[0])


class ExplicitLossTests(unittest.TestCase):
    def test_plain_custom_scalar_losses_fit_with_equal_batch_weight_and_partial_windows(self):
        for task_type, loss_function in ((ClassificationTask, nn.functional.cross_entropy),
                                         (RegressionTask, nn.functional.mse_loss)):
            with self.subTest(task=task_type.__name__):
                torch.manual_seed(31)
                model = nn.Linear(2, 3).double()
                reference = copy.deepcopy(model)
                batches = [(torch.randn(count, 2, dtype=torch.double),
                            torch.randint(0, 3, (count,)) if task_type is ClassificationTask else
                            torch.randn(count, 3, dtype=torch.double)) for count in (4, 1, 2)]
                task = task_type(loss_fn=loss_function)
                trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.02),
                                  task=task, accumulation_steps=2)
                rows = []
                trainer.fit(batches, batches, epochs=1,
                            callbacks=[lambda trainer, row: rows.append(row)])

                optimizer = torch.optim.SGD(reference.parameters(), lr=0.02)
                batch_losses = []
                for start in range(0, len(batches), 2):
                    objectives = [loss_function(reference(inputs), targets)
                                  for inputs, targets in batches[start:start + 2]]
                    batch_losses.extend(loss.detach().item() for loss in objectives)
                    optimizer.zero_grad()
                    (sum(objectives) / len(objectives)).backward()
                    optimizer.step()
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                self.assertEqual(trainer.state.optimizer_steps, 2)
                self.assertEqual(trainer.state.best_epoch, 1)
                self.assertEqual(rows[0]["train"]["samples"], 7)
                self.assertAlmostEqual(rows[0]["train"]["loss"], sum(batch_losses) / len(batch_losses), places=12)
                with torch.no_grad():
                    expected_validation = sum(loss_function(reference(x), y).item() for x, y in batches) / len(batches)
                self.assertAlmostEqual(rows[0]["validation"]["loss"], expected_validation, places=12)

    def test_scalar_identity_reaches_backward_once_without_accumulation(self):
        for task_type, targets in ((ClassificationTask, torch.tensor([0, 1])),
                                   (RegressionTask, torch.zeros(2, 3))):
            with self.subTest(task=task_type.__name__):
                model = nn.Linear(2, 3)
                produced = []

                def custom_loss(predictions, targets):
                    scalar = predictions.square().mean()
                    produced.append(scalar)
                    return scalar

                trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                                  task=task_type(loss_fn=custom_loss))
                with patch.object(trainer.runtime, "backward", wraps=trainer.runtime.backward) as backward:
                    with patch.object(BatchOutput, "loss_sum", side_effect=AssertionError("unexpected loss scaling")):
                        result = trainer.train_epoch([(torch.ones(2, 2), targets)])
                backward.assert_called_once()
                self.assertIs(backward.call_args.args[0], produced[0])
                self.assertEqual(result["loss"], produced[0].detach().double().item())

    def test_plain_batch_output_reports_batch_average_independent_of_sample_count(self):
        metric = MetricAccumulator()
        for value, count in ((1., 1), (5., 3)):
            output = BatchOutput(torch.tensor(value), count)
            self.assertEqual(output.objective.normalization_weight, 1)
            self.assertEqual(output.loss_report.weight, 1)
            metric.update(output)
        self.assertEqual(metric.compute()["loss"].item(), 3.)
        self.assertEqual(metric.compute()["samples"].item(), 4)

    def test_literal_scalar_accumulation_ignores_sample_counts_and_has_no_loss_report(self):
        torch.manual_seed(12)
        module = LiteralObjectiveModule(nn.Linear(2, 1).double())
        reference = copy.deepcopy(module)
        batches = [(torch.randn(count, 2, dtype=torch.double), torch.randn(count, 1, dtype=torch.double))
                   for count in (1, 3, 2)]
        trainer = Trainer(module, torch.optim.SGD(module.parameters(), lr=0.03), accumulation_steps=2)
        rows = []
        trainer.fit(batches, batches, epochs=1, monitor=None,
                    callbacks=[lambda trainer, row: rows.append(row)])

        optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
        for start in range(0, len(batches), 2):
            optimizer.zero_grad()
            for inputs, targets in batches[start:start + 2]:
                (reference(inputs) - targets).square().sum().backward()
            optimizer.step()

        for actual, expected in zip(module.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(trainer.state.optimizer_steps, 2)
        self.assertEqual(trainer.state.best_epoch, 0)
        self.assertIsNone(trainer.state.best_score)
        for phase in ("train", "validation"):
            self.assertNotIn("loss", rows[0][phase])
            self.assertEqual(rows[0][phase]["samples"], 6)

    def test_fractional_weights_and_partial_windows_match_handwritten_weighted_mean(self):
        torch.manual_seed(23)
        module = WeightedObjectiveModule(nn.Linear(2, 1).double())
        reference = copy.deepcopy(module)
        weight_lists = ([0.25, 0.5], [0.125], [0.75, 0.25, 0.125], [0.5], [0.125, 0.25])
        batches = [(torch.randn(len(weights), 2, dtype=torch.double),
                    torch.randn(len(weights), 1, dtype=torch.double),
                    torch.tensor(weights, dtype=torch.double).reshape(-1, 1))
                   for weights in weight_lists]
        trainer = Trainer(module, torch.optim.SGD(module.parameters(), lr=0.02), accumulation_steps=2)
        result = trainer.train_epoch(batches)

        optimizer = torch.optim.SGD(reference.parameters(), lr=0.02)
        report_total, report_weight = 0., 0.
        for start in range(0, len(batches), 2):
            window = batches[start:start + 2]
            inputs = torch.cat([batch[0] for batch in window])
            targets = torch.cat([batch[1] for batch in window])
            weights = torch.cat([batch[2] for batch in window])
            numerator = ((reference(inputs) - targets).square() * weights).sum()
            denominator = weights.sum()
            report_total += numerator.detach().item()
            report_weight += denominator.item()
            optimizer.zero_grad()
            (numerator / denominator).backward()
            optimizer.step()

        for actual, expected in zip(module.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(trainer.state.optimizer_steps, 3)
        self.assertEqual(result["samples"], 9)
        self.assertNotEqual(report_weight, result["samples"])
        self.assertAlmostEqual(result["loss"], report_total / report_weight, places=12)

    def test_reporting_uses_its_own_numerator_and_weight(self):
        metric = MetricAccumulator()
        first = LossResult(torch.tensor(100., requires_grad=True), normalization_weight=0.25,
                           report=LossReport(torch.tensor(12., requires_grad=True), 3.))
        second = LossResult(torch.tensor(200., requires_grad=True), normalization_weight=0.75,
                            report=LossReport(torch.tensor(10., requires_grad=True), 0.5))
        metric.update(BatchOutput(first, 100))
        metric.update(BatchOutput(second, 5))
        result = metric.compute()
        self.assertEqual(result["samples"].item(), 105)
        self.assertAlmostEqual(result["loss"].item(), 22 / 3.5)
        self.assertFalse(result["loss"].requires_grad)

    def test_fraction_weights_work_for_helpers_direct_results_and_accumulation(self):
        for use_helper in (True, False):
            with self.subTest(use_helper=use_helper):
                class FractionModule(TrainingModule):
                    def training_step(self, batch):
                        inputs, targets, denominator = batch
                        total = (self(inputs) - targets).square().sum()
                        value = total / float(denominator)
                        result = (LossResult.from_mean(value, weight=denominator) if use_helper else
                                  LossResult(value, normalization_weight=denominator,
                                             report=LossReport(total.detach(), denominator)))
                        return BatchOutput(result, inputs.shape[0])

                torch.manual_seed(21)
                module = FractionModule(nn.Linear(2, 1).double())
                reference = copy.deepcopy(module)
                batches = [(torch.randn(count, 2, dtype=torch.double),
                            torch.randn(count, 1, dtype=torch.double), denominator)
                           for count, denominator in ((2, Fraction(1, 2)), (1, Fraction(3, 4)))]
                trainer = Trainer(module, torch.optim.SGD(module.parameters(), lr=0.01), accumulation_steps=2)
                result = trainer.train_epoch(batches)
                numerator = sum((reference(inputs) - targets).square().sum()
                                for inputs, targets, _ in batches)
                denominator = sum(batch[2] for batch in batches)
                expected_loss = numerator / float(denominator)
                optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
                expected_loss.backward()
                optimizer.step()
                for actual, expected in zip(module.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                self.assertAlmostEqual(result["loss"], expected_loss.detach().item(), places=12)

    def test_manual_runtime_accepts_fraction_gradient_divisor(self):
        torch.manual_seed(22)
        model = nn.Linear(1, 1).double()
        reference = copy.deepcopy(model)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1))
        inputs = torch.ones(2, 1, dtype=torch.double)
        runtime.backward(model(inputs).square().sum())
        runtime.step(gradient_divisor=Fraction(1, 2))

        optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        (reference(inputs).square().sum() / 0.5).backward()
        optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_custom_function_reduces_arbitrary_rank_and_preserves_gradients(self):
        for reduction, expected_gradient in (("mean", 1 / 24), ("sum", 1.)):
            with self.subTest(reduction=reduction):
                values = torch.arange(24, dtype=torch.double).reshape(2, 3, 4).requires_grad_()
                task = RegressionTask(loss_fn=lambda predictions, targets: getattr(predictions, reduction)())
                output = task.compute(nn.Identity(), (values, torch.zeros_like(values)))
                expected = getattr(values, reduction)()
                torch.testing.assert_close(output.loss, expected)
                self.assertEqual(output.objective.normalization_weight, 1)
                self.assertEqual(output.loss_report.weight, 1)
                torch.testing.assert_close(output.loss_report.total, expected.detach().double())
                output.loss.backward()
                torch.testing.assert_close(values.grad, torch.full_like(values, expected_gradient))

    def test_custom_unreduced_loss_requires_reduction_inside_the_function(self):
        inputs = torch.ones(2, 3)
        for task_type, targets in ((ClassificationTask, torch.tensor([0, 1])),
                                   (RegressionTask, torch.zeros(2, 3))):
            with self.subTest(task=task_type.__name__):
                task = task_type(loss_fn=lambda predictions, targets: predictions.square())
                with self.assertRaisesRegex(ValueError, "scalar"):
                    task.compute(nn.Identity(), (inputs, targets))

    def test_custom_loss_result_preserves_scalar_identity(self):
        scalar = torch.tensor(3., requires_grad=True)
        explicit = LossResult(scalar)
        for task_type, targets in ((ClassificationTask, torch.tensor([0, 1])),
                                   (RegressionTask, torch.zeros(2, 3))):
            with self.subTest(task=task_type.__name__):
                task = task_type(loss_fn=lambda predictions, targets: explicit)
                output = task.compute(nn.Identity(), (torch.ones(2, 3), targets))
                self.assertIs(output.loss, scalar)
                self.assertIs(output.objective, explicit)
                self.assertIsNone(output.objective.normalization_weight)
                self.assertIsNone(output.loss_report)

    def test_raw_vector_default_strategy_fails_without_parameter_update(self):
        module = VectorObjectiveModule(nn.Linear(2, 1))
        before = copy.deepcopy(module.state_dict())
        trainer = Trainer(module, torch.optim.SGD(module.parameters(), lr=0.1))
        with self.assertRaisesRegex(ValueError, "scalar"):
            trainer.train_epoch([(torch.ones(2, 2), torch.zeros(2, 1))])
        self.assertEqual(trainer.state.optimizer_steps, 0)
        for name, parameter in module.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad)

    def test_mixed_reporting_is_rejected_without_changing_accumulated_state(self):
        reported = BatchOutput(LossResult.from_mean(torch.tensor(2.), weight=0.5), 3)
        unreported = BatchOutput(LossResult(torch.tensor(4.)), 7)
        for first, second in ((reported, unreported), (unreported, reported)):
            with self.subTest(first_reports=first.loss_report is not None):
                metric = MetricAccumulator()
                metric.update(first)
                before = metric.compute()
                with self.assertRaisesRegex(ValueError, "every batch"):
                    metric.update(second)
                after = metric.compute()
                self.assertEqual(after.keys(), before.keys())
                for name in before:
                    torch.testing.assert_close(after[name], before[name])

    def test_invalid_objective_and_reporting_weights_are_rejected(self):
        invalid = (0, -1, False, float("nan"), float("inf"), "2", torch.tensor(2.), 10 ** 1000)
        for index, weight in enumerate(invalid):
            with self.subTest(case=index):
                with self.assertRaisesRegex(ValueError, "positive finite real"):
                    LossResult(torch.tensor(1.), normalization_weight=weight).validate()
                with self.assertRaisesRegex(ValueError, "positive finite real"):
                    LossReport(torch.tensor(1.), weight=weight).validate()
                with self.assertRaisesRegex(ValueError, "positive finite real"):
                    LossResult.from_mean(torch.tensor(1.), weight=weight)
                with self.assertRaisesRegex(ValueError, "positive finite real"):
                    LossResult.from_sum(torch.tensor(1.), report_weight=weight)

    def test_metric_based_scheduler_is_rejected_before_optimizer_updates(self):
        model = nn.Linear(2, 1)
        before = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaisesRegex(ValueError, "no required arguments"):
                Trainer(model, optimizer, task=RegressionTask(), scheduler=scheduler)
            step.assert_not_called()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
