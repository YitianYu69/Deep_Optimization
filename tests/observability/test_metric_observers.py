import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchmetrics import Metric

from deepopt import Trainer, TrainingRuntime
from deepopt.engine.reporting.metric_lifecycle import MetricLifecycle
from deepopt.engine.reporting.metrics import MetricAccumulator
from deepopt.observers import CheckObservers, FiniteCheck
from deepopt.tasks.base import BatchOutput
from deepopt.tasks.loss import LossReport, LossResult


class FixedMetric(Metric):
    def __init__(self, results):
        super().__init__()
        self.results = results
        self.add_state("batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, output):
        self.batches += 1

    def compute(self):
        return self.results


class RecordingObserver:
    kinds = frozenset({"report", "metrics"})

    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append((event.kind, event.name, len(event.tensors),
                            torch.is_grad_enabled(),
                            any(tensor.requires_grad or tensor.grad_fn is not None for tensor in event.tensors)))


def make_trainer(checks=True, **kwargs):
    model = nn.Linear(2, 1)
    runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.1), checks=checks)
    return Trainer(runtime, loss_fn=nn.functional.mse_loss, **kwargs)


class MetricObserverTests(unittest.TestCase):
    def test_standalone_default_checks_reject_loss_but_disabled_checks_do_not_scan(self):
        output = BatchOutput(torch.tensor(float("nan")), 2)
        checked = MetricAccumulator()
        with self.assertRaises(FloatingPointError):
            checked.update(output)
        self.assertEqual(checked.batches.item(), 0)

        unchecked = MetricAccumulator(checks=False)
        with patch("torch.isfinite", side_effect=AssertionError("Disabled checks scanned a tensor")):
            unchecked.update(output)
            result = unchecked.compute()
        self.assertTrue(math.isnan(result["loss"].item()))
        self.assertEqual(result["samples"].item(), 2)

    def test_report_checks_can_be_selected_independently(self):
        report_only = MetricAccumulator(checks=[FiniteCheck(kinds={"report"})])
        report_only.update(BatchOutput(LossResult(torch.tensor(float("nan")),
                                                report=LossReport(torch.tensor(3.), 1.)), 2))
        self.assertEqual(report_only.compute()["loss"].item(), 3.)

        loss_only = MetricAccumulator(checks=[FiniteCheck(kinds={"loss"})])
        loss_only.update(BatchOutput(LossResult(torch.tensor(1.),
                                              report=LossReport(torch.tensor(float("nan")), 1.)), 2))
        self.assertTrue(math.isnan(loss_only.compute()["loss"].item()))

        for enabled in (True, False):
            metric = MetricAccumulator(checks=[FiniteCheck(kinds={"report"}, enabled=enabled)])
            # Make the stored numerator finite before testing accumulated overflow.
            report = LossReport(torch.tensor(torch.finfo(torch.float64).max, dtype=torch.float64), 1.)
            metric.update(BatchOutput(LossResult(torch.tensor(1.), report=report)))
            if enabled:
                with self.assertRaises(FloatingPointError):
                    metric.update(BatchOutput(LossResult(torch.tensor(1.), report=report)))
                self.assertEqual(metric.batches.item(), 1)
            else:
                metric.update(BatchOutput(LossResult(torch.tensor(1.), report=report)))
                self.assertTrue(math.isinf(metric.compute()["loss"].item()))

    def test_training_uses_shared_observers_without_duplicate_accumulator_loss_events(self):
        events = []

        def record(event):
            events.append((event.kind, event.name))

        trainer = make_trainer(checks=[record])
        batch = (torch.ones(2, 2), torch.zeros(2, 1))
        trainer.train_epoch([batch])
        trainer.evaluate([batch])
        self.assertEqual(events.count(("loss", "loss")), 2)
        self.assertEqual(events.count(("loss", "backward loss")), 1)
        self.assertEqual(events.count(("report", "accumulated reporting loss or weight")), 2)
        self.assertEqual(events.count(("metrics", "metric results")), 2)

    def test_selected_observers_receive_only_detached_metric_tensors(self):
        observer = RecordingObserver()
        trainer = make_trainer(checks=[observer])
        batch = (torch.ones(2, 2), torch.zeros(2, 1))
        trainer.train_epoch([batch])
        trainer.evaluate([batch])
        self.assertEqual([event[0] for event in observer.events], ["report", "metrics", "report", "metrics"])
        for _, _, count, grad_enabled, graph_retained in observer.events:
            self.assertGreater(count, 0)
            self.assertFalse(grad_enabled)
            self.assertFalse(graph_retained)

    def test_disabled_training_checks_have_no_hidden_metric_or_loss_scans(self):
        trainer = make_trainer(checks=False)
        batch = (torch.ones(2, 2), torch.zeros(2, 1))
        with patch("torch.isfinite", side_effect=AssertionError("Disabled checks scanned a tensor")):
            trainer.train_epoch([batch])
            trainer.evaluate([batch])
        self.assertEqual(trainer.state.optimizer_steps, 1)

    def test_observer_failure_blocks_runtime_before_reporting_state_mutation(self):
        class RejectReport:
            kinds = frozenset({"report"})

            def __call__(self, event):
                raise RuntimeError("report observer failed")

        trainer = make_trainer(checks=[RejectReport()])
        batch = (torch.ones(2, 2), torch.zeros(2, 1))
        with self.assertRaisesRegex(RuntimeError, "report observer failed"):
            trainer.train_epoch([batch])
        self.assertEqual(trainer.state.optimizer_steps, 0)
        self.assertEqual(trainer.epoch_progress.metrics.batches.item(), 0)
        self.assertEqual(trainer.epoch_progress.metrics.loss_sum.item(), 0.)
        with self.assertRaisesRegex(RuntimeError, "previous|failed"):
            trainer.runtime.backward(trainer.model(batch[0]).sum())
        self.assertTrue(all(parameter.grad is None for parameter in trainer.model.parameters()))

    def test_custom_metric_factory_results_use_runtime_policy(self):
        for checks in (True, False):
            with self.subTest(checks=checks):
                trainer = make_trainer(checks=checks, metric_factory=lambda: FixedMetric({"score": torch.tensor(float("nan"))}))
                batch = (torch.ones(2, 2), torch.zeros(2, 1))
                if checks:
                    with self.assertRaisesRegex(ValueError, "finite scalar"):
                        trainer.evaluate([batch])
                    with self.assertRaisesRegex(RuntimeError, "previous|failed"):
                        trainer.runtime.backward(trainer.model(batch[0]).sum())
                else:
                    self.assertTrue(math.isnan(trainer.evaluate([batch])["score"]))
                self.assertTrue(trainer.model.training)

    def test_loss_is_checked_again_after_custom_metric_mutation(self):
        class MutatingMetric(FixedMetric):
            def update(self, output):
                output.loss.fill_(float("nan"))
                self.batches += 1

        trainer = make_trainer(metric_factory=lambda: MutatingMetric({"score": torch.tensor(1.)}))
        before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
        batch = (torch.ones(2, 2), torch.zeros(2, 1))
        with self.assertRaisesRegex(FloatingPointError, "backward loss"):
            trainer.train_epoch([batch])
        self.assertEqual(trainer.state.optimizer_steps, 0)
        for name, parameter in trainer.model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad)
        with self.assertRaisesRegex(RuntimeError, "previous|failed"):
            trainer.runtime.backward(trainer.model(batch[0]).sum())

    def test_disabled_checks_preserve_shape_weight_count_and_device_contracts(self):
        invalid = (
            BatchOutput(torch.ones(2), 2),
            BatchOutput(torch.tensor(1.), 0),
            BatchOutput(LossResult(torch.tensor(1.), report=LossReport(torch.tensor(2.), 0.))),
            BatchOutput(LossResult(torch.tensor(1.), report=LossReport(torch.ones(2), 1.))),
            BatchOutput(torch.tensor(1, dtype=torch.long)),
            BatchOutput(torch.empty((), device="meta")),
            BatchOutput(LossResult(torch.tensor(1.), report=LossReport(torch.empty((), device="meta"), 1.))),
        )
        for output in invalid:
            with self.subTest(output=output):
                metric = MetricAccumulator(checks=False)
                with self.assertRaises(ValueError):
                    metric.update(output)
                self.assertEqual(metric.batches.item(), 0)

        metric = MetricAccumulator(checks=False)
        metric.update(BatchOutput(torch.tensor(1.)))
        with self.assertRaisesRegex(ValueError, "every batch"):
            metric.update(BatchOutput(LossResult(torch.tensor(2.))))
        self.assertEqual(metric.batches.item(), 1)

    def test_disabled_final_checks_preserve_metric_result_contracts(self):
        observe = CheckObservers(False).observe
        invalid = (
            {"score": torch.ones(2)},
            {"score": 1.},
            {2: torch.tensor(1.)},
            {"seconds": torch.tensor(1.)},
            {"samples": torch.tensor(0)},
            {"batches": torch.tensor(2.)},
        )
        for results in invalid:
            with self.subTest(results=results):
                metric = FixedMetric(results)
                metric.update(None)
                with self.assertRaises(ValueError):
                    MetricLifecycle.compute(metric, observe=observe)


if __name__ == "__main__":
    unittest.main()
