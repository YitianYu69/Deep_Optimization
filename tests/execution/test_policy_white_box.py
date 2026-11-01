from collections import defaultdict, namedtuple
from dataclasses import dataclass
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import DiagnosticPolicy, Trainer, TrainingRuntime
from deepopt.engine.execution.batch import move_to_device
from deepopt.engine.reporting.metrics import MetricAccumulator
from deepopt.engine.reporting.metric_state import snapshot_metrics, restore_metrics
from deepopt.tasks.base import BatchOutput
from deepopt.tasks.loss import LossResult


class PolicyWhiteBoxTests(unittest.TestCase):
    def test_max_speed_does_not_scan_materialize_events_or_collect_gradients(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.01),
                          loss_fn=nn.functional.mse_loss, execution_mode="max_speed")
        with patch("torch.isfinite", side_effect=AssertionError("passive scan")), \
                patch("deepopt.observers.TensorEvent", side_effect=AssertionError("event constructed")), \
                patch.object(trainer.runtime._stepper, "_values", side_effect=AssertionError("gradient traversal")):
            trainer.train_epoch([(torch.ones(2, 2), torch.zeros(2, 1))])
        with patch.object(trainer.runtime.core, "observe", side_effect=AssertionError("Invalid event reached backend")):
            with self.assertRaisesRegex(ValueError, "nonempty kind"):
                trainer.runtime.observe("", [])

    def test_balanced_cadence_uses_optimizer_updates_not_microbatches(self):
        events = []
        model = nn.Linear(1, 1)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.01),
                                  diagnostics=DiagnosticPolicy(loss_finite_every=2, gradient_finite_every=2),
                                  checks=[lambda event: events.append((event.kind, runtime.mode_controller.completed_updates + 1))])
        trainer = Trainer(runtime, loss_fn=nn.functional.mse_loss, accumulation_steps=2)
        trainer.train_epoch([(torch.ones(2, 1), torch.zeros(2, 1))] * 5)
        self.assertTrue(any(kind == "loss" for kind, _ in events))
        self.assertTrue(any(kind == "gradients" for kind, _ in events))
        self.assertTrue(all(update == 2 for kind, update in events if kind in ("loss", "gradients", "report")))
        self.assertEqual(runtime.mode_controller.completed_updates, 3)

    def test_metric_update_has_no_scalar_reads_and_restores_reporting_presence(self):
        metric = MetricAccumulator(checks=False)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            metric.update(BatchOutput(torch.tensor(2.), sample_count=2))
            metric.update(BatchOutput(torch.tensor(3.), sample_count=1))
        self.assertFalse({"aten::item", "aten::_local_scalar_dense"}.intersection(event.key for event in profile.key_averages()))
        snapshot = snapshot_metrics(metric)
        # Also exercise old metric snapshots that lack the Python metadata.
        snapshot["state"]["_extra_state"].pop("report_presence")
        restored = MetricAccumulator(checks=False)
        restore_metrics(restored, snapshot, "cpu")
        with self.assertRaisesRegex(ValueError, "every batch"):
            restored.update(BatchOutput(LossResult(torch.tensor(1.))))
        restored.reset()
        restored.update(BatchOutput(LossResult(torch.tensor(1.))))

    def test_nonblocking_propagates_through_all_nested_batch_containers(self):
        @dataclass(frozen=True)
        class Batch:
            x: object

        Pair = namedtuple("Pair", "x label")
        value = Batch(defaultdict(list, {"items": [Pair(torch.ones(1), "x"), (torch.zeros(1),)]}))
        original = torch.Tensor.to
        seen = []

        def transfer(tensor, device, **kwargs):
            seen.append(kwargs.get("non_blocking"))
            return original(tensor, device, **kwargs)

        with patch("torch.Tensor.to", new=transfer):
            moved = move_to_device(value, "cpu", non_blocking=True)
        self.assertEqual(seen, [True, True])
        self.assertIsInstance(moved.x, defaultdict)
        self.assertIsInstance(moved.x["items"][0], Pair)
        self.assertEqual(moved.x["items"][0].label, "x")
        self.assertIsNot(moved, value)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_metric_bookkeeping_has_no_host_scalar_reads(self):
        metric = MetricAccumulator(checks=False).cuda()
        output = BatchOutput(torch.ones((), device="cuda"), sample_count=2)
        metric.update(output)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            metric.update(output)
        self.assertFalse({"aten::item", "aten::_local_scalar_dense"}.intersection(event.key for event in profile.key_averages()))


if __name__ == "__main__":
    unittest.main()
