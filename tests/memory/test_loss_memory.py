"""Backward must preserve gradients without materializing logging tensors."""

import unittest
from unittest.mock import PropertyMock, patch

import torch

from deepopt import TrainingRuntime
from deepopt.engine.strategies import ExplicitObjective, MicrobatchMean
from deepopt.tasks.base import BatchOutput
from deepopt.tasks.loss import LossReport, LossResult


class LossMemoryTests(unittest.TestCase):
    def test_backward_preparation_does_not_materialize_reporting_tensors(self):
        for strategy in (ExplicitObjective(), MicrobatchMean()):
            for accumulation in (1, 2):
                with self.subTest(strategy=type(strategy).__name__, accumulation=accumulation):
                    parameter = torch.tensor(3., requires_grad=True)
                    output = BatchOutput(parameter.square())
                    with patch.object(BatchOutput, "loss_report", new_callable=PropertyMock,
                                      side_effect=AssertionError("backward requested a report")):
                        prepared = strategy.prepare(output, accumulation_steps=accumulation)
                    self.assertIs(prepared.loss, output.loss)
                    prepared.loss.backward()
                    self.assertEqual(parameter.grad.item(), 6.)

    def test_legacy_objectives_are_recomputed_after_in_place_changes(self):
        values = torch.tensor([1., 3.], requires_grad=True)
        output = BatchOutput(values, 2, "none")
        self.assertEqual(output.objective.value.item(), 2.)
        with torch.no_grad():
            values.add_(2.)
        self.assertEqual(output.objective.value.item(), 4.)
        self.assertEqual(output.loss_report.total.item(), 8.)
        prepared = ExplicitObjective().prepare(output, accumulation_steps=2)
        self.assertEqual(prepared.loss.item(), 8.)
        self.assertFalse(output.loss_report.total.requires_grad)
        prepared.loss.backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))

    def test_validation_is_not_cached_across_calls(self):
        for structured in (False, True):
            with self.subTest(structured=structured):
                loss = torch.tensor(2., requires_grad=True)
                output = BatchOutput(LossResult(loss) if structured else loss)
                output.validate()
                with torch.no_grad():
                    loss.fill_(float("nan"))
                with self.assertRaises(FloatingPointError):
                    output.validate()
                prepared = ExplicitObjective().prepare(output, accumulation_steps=1)
                with self.assertRaises(FloatingPointError):
                    TrainingRuntime(torch.nn.Identity()).backward(prepared.loss)

    def test_weighting_overflow_still_fails_before_backward(self):
        output = BatchOutput(LossResult(torch.tensor(2e38, requires_grad=True), 4))
        prepared = ExplicitObjective().prepare(output, accumulation_steps=2)
        with self.assertRaisesRegex(FloatingPointError, "backward loss"):
            TrainingRuntime(torch.nn.Identity()).backward(prepared.loss)
        self.assertIsNone(output.loss.grad)

    def test_legacy_report_weights_keep_validation_and_float_conversion(self):
        for reduction in ("mean", "sum"):
            for count in (None, 0, False, -1):
                output = BatchOutput(torch.tensor(2.), count, reduction)
                for name in ("objective", "loss_report"):
                    with self.subTest(reduction=reduction, count=count, accessor=name):
                        with self.assertRaisesRegex(ValueError, "positive finite real"):
                            getattr(output, name)
            output = BatchOutput(torch.tensor(2.), 2 ** 64, reduction)
            report = output.loss_report
            self.assertEqual(report.weight, float(2 ** 64))
            self.assertEqual(report.total.item(), 2. * 2 ** 64 if reduction == "mean" else 2.)

    def test_existing_objective_overrides_remain_effective(self):
        class CustomOutput(BatchOutput):
            @property
            def objective(self):
                original = super().objective
                return LossResult(original.value * 2, original.normalization_weight,
                                  LossReport(original.report.total * 3, original.report.weight))

        for strategy in (ExplicitObjective(), MicrobatchMean()):
            parameter = torch.tensor(3., requires_grad=True)
            output = CustomOutput(parameter.square())
            prepared = strategy.prepare(output, accumulation_steps=1)
            self.assertEqual(prepared.loss.item(), 18.)
            self.assertEqual(output.loss_sum().item(), 18.)
            self.assertEqual(output.loss_report.total.item(), 27.)
            self.assertFalse(output.loss_report.total.requires_grad)
            prepared.loss.backward()
            self.assertEqual(parameter.grad.item(), 12.)

    def test_overridden_objective_overflow_is_caught_before_runtime_backward(self):
        class OverflowingOutput(BatchOutput):
            @property
            def objective(self):
                return LossResult(self.loss * 2, 1)

        output = OverflowingOutput(torch.tensor(2e38, requires_grad=True))
        prepared = ExplicitObjective().prepare(output, accumulation_steps=2)
        with self.assertRaisesRegex(FloatingPointError, "backward loss"):
            TrainingRuntime(torch.nn.Identity()).backward(prepared.loss)
        self.assertIsNone(output.loss.grad)


if __name__ == "__main__":
    unittest.main()
