from datetime import timedelta
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torchmetrics import Metric, MeanSquaredError
from torchmetrics.classification import BinaryF1Score, BinaryAUROC

from deepopt.engine.reporting.metrics import MetricAccumulator
from deepopt.engine.reporting.metric_state import snapshot_metrics, restore_metrics
from deepopt.tasks.base import BatchOutput
from deepopt.tasks.regression import RegressionTask
from deepopt.tasks.classification import ClassificationTask
from deepopt.config import config_from_dict
from deepopt.baseline import build_baseline_experiment


class CustomLossMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total", torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, output):
        self.total += output.loss_sum().detach().double()
        self.count += output.sample_count

    def compute(self):
        return {"loss": self.total / self.count, "samples": self.count.clone(),
                "custom_score": 1 / (1 + self.total / self.count)}


def batch(losses):
    return BatchOutput(torch.tensor(losses, dtype=torch.float32), len(losses), reduction="none")


def update_predictions(metrics, output):
    for metric in metrics.values():
        metric.update(*output.metric_data)


def distributed_worker(rank, directory):
    dist.init_process_group("gloo", init_method=f"file://{directory}/store", rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        output = batch([1.0]) if rank == 0 else batch([3.0, 5.0, 7.0])
        synchronized = MetricAccumulator()
        synchronized.update(output)
        global_result = {key: value.item() for key, value in synchronized.compute().items()}
        local_samples_after_compute = synchronized.samples.item()
        local = MetricAccumulator(sync_on_compute=False)
        local.update(output)
        empty_rank = MetricAccumulator()
        if rank == 1:
            empty_rank.update(batch([2.0, 6.0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            empty_result = {key: value.item() for key, value in empty_rank.compute().items()}
        Path(directory, f"rank-{rank}.json").write_text(json.dumps({
            "global": global_result, "local": {k: v.item() for k, v in local.compute().items()},
            "local_samples_after_compute": local_samples_after_compute, "empty_rank": empty_result,
        }))
    finally:
        dist.destroy_process_group()


class MetricTests(unittest.TestCase):
    def test_subclasses_metric_and_weights_mean_sum_and_unreduced_losses(self):
        metric = MetricAccumulator()
        self.assertIsInstance(metric, Metric)
        metric.update(BatchOutput(torch.tensor(1.), 1, "mean"))
        metric.update(BatchOutput(torch.tensor(5.), 3, "mean"))
        self.assertEqual(metric.compute()["loss"].item(), 4)
        metric.reset()
        metric.update(BatchOutput(torch.tensor(16.), 4, reduction="sum"))
        self.assertEqual(metric.compute()["loss"].item(), 4)
        metric.reset()
        metric.update(batch([1., 3., 5., 7.]))
        self.assertEqual(metric.compute()["loss"].item(), 4)

    def test_nested_metrics_compute_whole_phase_not_average_batch_scores(self):
        child = BinaryF1Score(sync_on_compute=True)
        metric = MetricAccumulator({"f1": child}, update_metrics=update_predictions)
        first = BatchOutput(torch.tensor(1.), 2, "mean", metric_data=(torch.tensor([1, 1]), torch.tensor([1, 0])))
        second = BatchOutput(torch.tensor(1.), 1, "mean", metric_data=(torch.tensor([0]), torch.tensor([1])))
        metric.update(first)
        self.assertIsNone(child._computed)  # update does not compute after every batch.
        metric.update(second)
        self.assertEqual(metric.compute()["f1"].item(), 0.5)
        self.assertEqual(metric.compute()["samples"].item(), 3)
        self.assertTrue(child.sync_on_compute)
        metric.reset()
        metric.update(second)
        self.assertEqual(metric.compute()["f1"].item(), 0.)
        self.assertEqual(child.update_count, 1)

    def test_update_invalidates_compute_cache_and_reset_clears_state(self):
        metric = MetricAccumulator()
        metric.update(batch([1.0]))
        first = metric.compute()
        metric.update(batch([3.0]))
        self.assertEqual(metric.compute()["loss"].item(), 2)
        metric.reset()
        self.assertEqual(metric.samples.item(), 0)
        metric.update(batch([9.0]))
        self.assertEqual(metric.compute()["loss"].item(), 9)
        self.assertEqual(first["samples"].item(), 1)

    def test_forward_keeps_parent_and_child_epoch_states(self):
        metric = MetricAccumulator({"f1": BinaryF1Score()}, update_metrics=update_predictions)
        metric.update(BatchOutput(torch.tensor(1.), 2, "mean",
                                  metric_data=(torch.tensor([1, 1]), torch.tensor([1, 0]))))
        current = metric(BatchOutput(torch.tensor(3.), 1, "mean",
                                     metric_data=(torch.tensor([0]), torch.tensor([1]))))
        self.assertEqual(current["loss"].item(), 3.)
        self.assertEqual(current["f1"].item(), 0.)
        self.assertEqual(metric.compute()["samples"].item(), 3)
        self.assertEqual(metric.compute()["f1"].item(), 0.5)
        self.assertEqual(metric.task_metrics[0].update_count, 2)

    def test_states_do_not_retain_autograd_graph(self):
        task = RegressionTask()
        metric = MetricAccumulator(task.create_metrics(), update_metrics=task.update_metrics)
        predictions = torch.tensor([1., 2.], requires_grad=True)
        output = task.compute(nn.Identity(), (predictions, torch.zeros(2)))
        metric.update(output)
        self.assertTrue(all(not value.requires_grad for value in metric.compute().values()))
        output.loss_sum().backward()
        torch.testing.assert_close(predictions.grad, 2 * predictions.detach())

    def test_invalid_loss_does_not_update_metric_state(self):
        metric = MetricAccumulator()
        metric.update(batch([1.0]))
        for output in (batch([float("nan")]), BatchOutput(torch.ones(2), 2, "mean"),
                       BatchOutput(torch.tensor(1.), 0), BatchOutput(torch.tensor(1.), True),
                       BatchOutput(torch.tensor(1.), 2, reduction="invalid")):
            with self.subTest(output=output):
                with self.assertRaises((ValueError, FloatingPointError)):
                    metric.update(output)
                self.assertEqual(metric.samples.item(), 1)
                self.assertEqual(metric.loss_sum.item(), 1)

    def test_metric_factories_validate_types_names_and_aliases(self):
        child = MeanSquaredError()
        for metrics in ({"loss": child}, {"batches": child}, {"": child}, {"bad": object()},
                        {"a": child, "b": child}, []):
            with self.subTest(metrics=metrics), self.assertRaises(ValueError):
                MetricAccumulator(metrics, update_metrics=update_predictions)
        metric = MetricAccumulator({"z": MeanSquaredError(), "a": MeanSquaredError()},
                                   update_metrics=update_predictions)
        self.assertEqual(metric.metric_names, ("a", "z"))

    def test_tasks_create_isolated_metrics_and_do_not_update_during_compute(self):
        for task in (RegressionTask(), ClassificationTask(num_classes=2)):
            with self.subTest(task=type(task).__name__):
                first, second = task.create_metrics(), task.create_metrics()
                self.assertTrue(all(first[name] is not second[name] for name in first))
                self.assertTrue(all(metric.sync_on_compute for metric in first.values()))
        task = RegressionTask()
        metric = task.create_metrics()["mse"]
        task.compute(nn.Identity(), (torch.tensor([2.]), torch.tensor([0.])))
        self.assertEqual(metric.update_count, 0)
        self.assertEqual(ClassificationTask().create_metrics(), {})  # No assumed class count.

    def test_nested_metric_states_and_counters_restore_including_list_states(self):
        def factory():
            return MetricAccumulator({"auc": BinaryAUROC(), "f1": BinaryF1Score()},
                                     update_metrics=update_predictions)
        original = factory()
        original.update(BatchOutput(torch.tensor(1.), 2, "mean",
                                   metric_data=(torch.tensor([0.1, 0.8]), torch.tensor([0, 1]))))
        state = copy.deepcopy(snapshot_metrics(original))
        restored = factory()
        restore_metrics(restored, state, "cpu")
        extra = BatchOutput(torch.tensor(3.), 2, "mean",
                            metric_data=(torch.tensor([0.7, 0.9]), torch.tensor([0, 1])))
        for metric in (original, restored):
            metric.update(extra)
        for key in original.compute():
            torch.testing.assert_close(original.compute()[key], restored.compute()[key], rtol=0, atol=0)
        for child in restored.task_metrics:
            self.assertEqual(child.update_count, 2)
        changed = MetricAccumulator({"auc": BinaryAUROC(), "other": BinaryF1Score()},
                                    update_metrics=update_predictions)
        with self.assertRaisesRegex(ValueError, "same metric names"):
            restore_metrics(changed, state, "cpu")

    def test_sync_on_compute_reduces_states_and_restores_local_state(self):
        gathered = []
        remote = MetricAccumulator(sync_on_compute=False)
        remote.update(batch([3., 5., 7.]))

        def gather(value, group=None):
            # The remote loss numerator and denominator are different float
            # states. Match their names instead of guessing from tensor dtype.
            for name in ("samples", "loss_sum", "loss_weight", "batches", "counted_batches", "loss_batches"):
                if value is getattr(metric, name):
                    gathered.append(name)
                    return [value, getattr(remote, name).clone()]
            raise AssertionError("Unexpected synchronized metric state")

        metric = MetricAccumulator(dist_sync_fn=gather, distributed_available_fn=lambda: True)
        metric.update(batch([1.0]))
        result = metric.compute()
        self.assertEqual(result["loss"].item(), 4)
        self.assertEqual(result["samples"].item(), 4)
        self.assertEqual(metric.samples.item(), 1)
        self.assertEqual(set(gathered), {"samples", "loss_sum", "loss_weight", "batches", "counted_batches", "loss_batches"})
        local = MetricAccumulator(sync_on_compute=False, dist_sync_fn=gather,
                                  distributed_available_fn=lambda: True)
        local.update(batch([1.0]))
        self.assertEqual(local.compute()["loss"].item(), 1)
        self.assertEqual(len(gathered), 6)

    def test_trainer_accepts_independent_metric_subclass(self):
        config = config_from_dict({"name": "custom-metric", "seed": 1,
                                   "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.01},
                                   "application": {"synthetic": {"train_samples": 9, "validation_samples": 5}}})
        instances = []

        def factory():
            metric = CustomLossMetric()
            instances.append(metric)
            return metric

        trainer, train, validation, _ = build_baseline_experiment(config, metric_factory=factory)
        results = []
        trainer.fit(train, validation, epochs=1, monitor="custom_score", monitor_mode="max",
                    callbacks=[lambda trainer, row: results.append(row)])
        self.assertEqual(len(instances), 2)
        self.assertIsNot(instances[0], instances[1])
        self.assertEqual(results[0]["train"]["samples"], 9)
        self.assertEqual(results[0]["validation"]["samples"], 5)
        trainer.metric_factory = lambda: object()
        with self.assertRaisesRegex(ValueError, "torchmetrics.Metric subclass"):
            trainer.evaluate(validation)
        trainer.metric_factory = lambda: instances[0]
        with self.assertRaisesRegex(ValueError, "fresh instances"):
            trainer.evaluate(validation)

    def test_custom_metrics_do_not_bypass_loss_validation(self):
        config = config_from_dict({"name": "invalid-loss", "seed": 1,
                                   "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.01},
                                   "application": {"synthetic": {"train_samples": 5}}})
        trainer, loader, _, _ = build_baseline_experiment(config, metric_factory=CustomLossMetric)
        trainer.task = ClassificationTask(
            loss_fn=lambda logits, targets: logits.sum() * float("nan"))
        with self.assertRaises(FloatingPointError):
            trainer.train_epoch(loader)
        self.assertEqual(trainer.state.optimizer_steps, 0)

    @unittest.skipUnless(os.environ.get("DEEPOPT_TEST_DISTRIBUTED") == "1" and dist.is_gloo_available(),
                         "Set DEEPOPT_TEST_DISTRIBUTED=1 to run the two-process Gloo test")
    def test_sync_on_compute_with_unequal_and_empty_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=(directory,), nprocs=2, join=True)
            for rank in range(2):
                result = json.loads(Path(directory, f"rank-{rank}.json").read_text())
                self.assertEqual(result["global"], {"samples": 4, "loss": 4})
                self.assertEqual(result["local"]["loss"], 1 if rank == 0 else 5)
                self.assertEqual(result["local_samples_after_compute"], 1 if rank == 0 else 3)
                self.assertEqual(result["empty_rank"], {"samples": 2, "loss": 4})


if __name__ == "__main__":
    unittest.main()
