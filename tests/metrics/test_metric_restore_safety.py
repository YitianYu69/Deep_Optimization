from copy import deepcopy
import unittest

import torch
from torchmetrics import MeanMetric, Metric
from torchmetrics.classification import BinaryAUROC

from deepopt.engine.reporting.metric_state import restore_metrics, snapshot_metrics


class BufferedMetric(Metric):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.tensor(0.))
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.extra = torch.tensor(2., requires_grad=True) * 3

    def get_extra_state(self):
        return {"nested": [self.extra]}

    def set_extra_state(self, state):
        self.extra = state["nested"][0]

    def update(self, value):
        self.total.add_(value)
        self.count += 1

    def compute(self):
        return self.total / self.count


class NestedMetric(Metric):
    def __init__(self):
        super().__init__()
        self.average = MeanMetric()
        self.auc = BinaryAUROC()
        self.add_state("samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        self.average.update(predictions)
        self.auc.update(predictions, targets)
        self.samples += targets.numel()

    def compute(self):
        return {"average": self.average.compute(), "auc": self.auc.compute(),
                "samples": self.samples.clone()}


class MetricRestoreSafetyTests(unittest.TestCase):
    def test_restore_invalidates_cached_compute_result(self):
        metric = MeanMetric()
        metric.update(torch.tensor(2.))
        saved = snapshot_metrics(metric)
        metric.update(torch.tensor(10.))
        self.assertEqual(metric.compute().item(), 6.)

        restore_metrics(metric, saved, "cpu")

        self.assertEqual(metric.compute().item(), 2.)
        self.assertEqual(metric.update_count, 1)

    def test_restored_updates_do_not_mutate_saved_tensor_states(self):
        original = MeanMetric()
        original.update(torch.tensor(2.))
        saved = snapshot_metrics(original)
        restored = MeanMetric()
        restore_metrics(restored, saved, "cpu")

        restored.update(torch.tensor(6.))

        self.assertEqual(restored.compute().item(), 4.)
        self.assertEqual(saved["state"]["mean_value"].item(), 2.)
        self.assertEqual(saved["state"]["weight"].item(), 1.)
        # The same snapshot can be restored again without accumulating updates.
        restore_metrics(restored, saved, "cpu")
        self.assertEqual(restored.compute().item(), 2.)

    def test_snapshot_copies_buffers_and_detaches_nested_extra_state(self):
        original = BufferedMetric()
        original.update(torch.tensor(2.))
        saved = snapshot_metrics(original)
        copied_extra = saved["state"]["_extra_state"]["nested"][0]
        self.assertFalse(copied_extra.requires_grad)
        self.assertIsNone(copied_extra.grad_fn)
        self.assertTrue(hasattr(saved["state"], "_metadata"))

        original.update(torch.tensor(3.))
        with torch.no_grad():
            original.extra.add_(4.)

        self.assertEqual(saved["state"]["total"].item(), 2.)
        self.assertEqual(saved["state"]["count"].item(), 1)
        self.assertEqual(copied_extra.item(), 6.)
        restored = BufferedMetric()
        restore_metrics(restored, saved, "cpu")
        restored.update(torch.tensor(8.))
        restored.extra.add_(1.)
        self.assertEqual(saved["state"]["total"].item(), 2.)
        self.assertEqual(copied_extra.item(), 6.)

    def test_nested_caches_counters_and_list_states_restore_independently(self):
        metric = NestedMetric()
        metric.update(torch.tensor([0.1, 0.9]), torch.tensor([0, 1]))
        expected = metric.compute()
        saved = snapshot_metrics(metric)
        saved_prediction = saved["state"]["auc.preds"][0].clone()
        metric.update(torch.tensor([0.8, 0.7]), torch.tensor([0, 1]))
        self.assertEqual(metric.compute()["samples"].item(), 4)
        metric.auc.preds[0].add_(0.01)
        torch.testing.assert_close(saved["state"]["auc.preds"][0], saved_prediction)

        restore_metrics(metric, saved, "cpu")

        for child in (metric, metric.average, metric.auc):
            self.assertIsNone(child._computed)
            self.assertEqual(child.update_count, 1)
        for name, value in metric.compute().items():
            torch.testing.assert_close(value, expected[name])
        metric.update(torch.tensor([0.4, 0.6]), torch.tensor([0, 1]))
        metric.auc.preds[0].add_(0.02)
        self.assertEqual(len(saved["state"]["auc.preds"]), 1)
        torch.testing.assert_close(saved["state"]["auc.preds"][0], saved_prediction)
        self.assertEqual(saved["state"]["samples"].item(), 2)

    def test_restore_clears_forward_and_synchronization_caches(self):
        metric = MeanMetric(
            dist_sync_fn=lambda value, group=None: [value, value],
            distributed_available_fn=lambda: True,
        )
        metric(torch.tensor(2.))
        saved = snapshot_metrics(metric)
        self.assertIsNotNone(metric._forward_cache)
        metric.update(torch.tensor(10.))
        metric.sync(dist_sync_fn=metric.dist_sync_fn)
        self.assertTrue(metric._is_synced)
        self.assertIsNotNone(metric._cache)

        restore_metrics(metric, saved, "cpu")

        self.assertIsNone(metric._forward_cache)
        self.assertIsNone(metric._cache)
        self.assertFalse(metric._is_synced)
        self.assertEqual(metric.compute().item(), 2.)
        self.assertEqual(metric.mean_value.item(), 2.)

    def test_invalid_counter_is_rejected_before_replacing_existing_state(self):
        metric = MeanMetric()
        metric.update(torch.tensor(2.))
        saved = deepcopy(snapshot_metrics(metric))
        saved["updates"][""] = -1
        metric.update(torch.tensor(6.))
        with self.assertRaisesRegex(ValueError, "update counter"):
            restore_metrics(metric, saved, "cpu")
        self.assertEqual(metric.compute().item(), 4.)


if __name__ == "__main__":
    unittest.main()
