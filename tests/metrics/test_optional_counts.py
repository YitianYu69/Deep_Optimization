import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Metric

from deepopt.data.resumable import ResumableDataLoader
from deepopt.engine import Trainer
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.engine.reporting.metric_lifecycle import MetricLifecycle
from deepopt.engine.reporting.metric_state import restore_metrics, snapshot_metrics
from deepopt.engine.reporting.metrics import MetricAccumulator
from deepopt.module import TrainingModule
from deepopt.tasks.base import BatchOutput
from deepopt.tasks.loss import LossResult


class UnknownCountModule(TrainingModule):
    def training_step(self, batch):
        inputs, targets = batch
        return BatchOutput((self(inputs) - targets).square().mean())


class CountedModule(UnknownCountModule):
    def training_step(self, batch):
        output = super().training_step(batch)
        return BatchOutput(output.loss, len(batch[1]))


class MixedCountModule(TrainingModule):
    def training_step(self, batch):
        return BatchOutput((self(batch["x"]) - batch["y"]).square().mean(), batch["count"])


class LossOnlyMetric(Metric):
    """A valid custom aggregate need not claim to count examples."""
    def __init__(self):
        super().__init__()
        self.add_state("total", torch.tensor(0., dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("batch_count", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, output):
        self.total += output.loss.detach().double()
        self.batch_count += 1

    def compute(self):
        return {"loss": self.total / self.batch_count.clamp_min(1)}


class OptionalCountTests(unittest.TestCase):
    def test_batch_output_accepts_unknown_count_without_inferring_loss_weight(self):
        output = BatchOutput(torch.tensor(3.))
        output.validate()
        self.assertIsNone(output.sample_count)
        self.assertEqual(output.objective.normalization_weight, 1)
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "sample_count"):
                BatchOutput(torch.tensor(3.), count).validate()
        for reduction in ("mean", "sum", "none"):
            with self.subTest(reduction=reduction), self.assertRaisesRegex(ValueError, "explicit sample_count"):
                BatchOutput(torch.tensor(3.), reduction=reduction).validate()

    def test_known_counts_keep_existing_output_schema(self):
        metric = MetricAccumulator()
        metric.update(BatchOutput(torch.tensor(1.), 3))
        metric.update(BatchOutput(torch.tensor(5.), 7))
        result = metric.compute()
        self.assertEqual(set(result), {"loss", "samples"})
        self.assertEqual(result["samples"].item(), 10)
        self.assertEqual(result["loss"].item(), 3)

    def test_unknown_or_mixed_counts_report_batches_without_partial_sample_totals(self):
        for counts in ((None, None, None), (2, None, 5), (None, 3, 5)):
            with self.subTest(counts=counts):
                metric = MetricAccumulator()
                for loss, count in zip((1., 3., 5.), counts):
                    metric.update(BatchOutput(torch.tensor(loss), count))
                result = metric.compute()
                self.assertEqual(set(result), {"loss", "batches"})
                self.assertEqual(result["batches"].item(), 3)
                self.assertEqual(result["loss"].item(), 3)
        metric = MetricAccumulator()
        metric.update(BatchOutput(LossResult(torch.tensor(1.))))
        self.assertEqual(set(metric.compute()), {"batches"})

    def test_synchronization_omits_samples_when_a_remote_rank_has_unknown_counts(self):
        remote = MetricAccumulator(sync_on_compute=False)
        remote.update(BatchOutput(torch.tensor(3.)))

        def gather(value, group=None):
            for name in metric._defaults:
                if value is getattr(metric, name):
                    return [value, getattr(remote, name).clone()]
            raise AssertionError("Unexpected metric state")

        metric = MetricAccumulator(dist_sync_fn=gather, distributed_available_fn=lambda: True)
        metric.update(BatchOutput(torch.tensor(1.), 4))
        result = metric.compute()
        self.assertEqual(set(result), {"loss", "batches"})
        self.assertEqual(result["loss"].item(), 2)
        self.assertEqual(result["batches"].item(), 2)
        self.assertEqual(metric.samples.item(), 4)
        self.assertEqual(metric.counted_batches.item(), 1)

    def test_custom_aggregate_can_report_only_loss(self):
        module = UnknownCountModule(nn.Linear(1, 1))
        trainer = Trainer(module, torch.optim.SGD(module.parameters(), lr=0.1), metric_factory=LossOnlyMetric)
        batches = [(torch.ones(2, 1), torch.zeros(2, 1))]
        rows = []
        trainer.fit(batches, batches, epochs=1, callbacks=[lambda trainer, row: rows.append(row)])
        self.assertEqual(set(rows[0]["train"]), {"loss", "seconds"})
        self.assertEqual(set(rows[0]["validation"]), {"loss"})

    def test_unknown_count_rejects_empty_phases_even_with_a_constant_custom_metric(self):
        module = UnknownCountModule(nn.Linear(1, 1))
        trainer = Trainer(module, torch.optim.SGD(module.parameters(), lr=0.1), metric_factory=LossOnlyMetric)
        with self.assertRaisesRegex(ValueError, "empty"):
            trainer.evaluate([])
        self.assertTrue(module.training)
        with self.assertRaisesRegex(ValueError, "empty"):
            trainer.train_epoch([])
        self.assertEqual(trainer.state.optimizer_steps, 0)

    def test_optional_metric_counts_still_require_positive_integer_tensors(self):
        class ReportMetric(Metric):
            def __init__(self, report):
                super().__init__()
                self.report = report

            def update(self):
                pass

            def compute(self):
                return self.report

        for name in ("samples", "batches"):
            for count in (torch.tensor(0), torch.tensor(-1), torch.tensor(1.), torch.tensor(True)):
                with self.subTest(name=name, count=count):
                    metric = ReportMetric({name: count})
                    metric.update()
                    with self.assertRaisesRegex(ValueError, "positive integer tensor"):
                        MetricLifecycle.compute(metric)

    def test_progress_stays_unknown_after_a_missing_count_and_omits_throughput(self):
        module = MixedCountModule(nn.Linear(1, 1))
        trainer = Trainer(module, torch.optim.SGD(module.parameters(), lr=0.1))
        batches = [{"x": torch.ones(2, 1), "y": torch.zeros(2, 1), "count": count}
                   for count in (2, None, 2)]
        counts = []
        result = trainer.train_epoch(batches, step_callbacks=[lambda trainer, event: counts.append(event["samples_processed"])])
        self.assertEqual(counts, [2, None, None])
        self.assertEqual(result["batches"], 3)
        self.assertNotIn("samples", result)
        self.assertNotIn("samples_per_second", result)

    def test_old_counted_metric_state_is_migrated_without_mutating_snapshot(self):
        metric = MetricAccumulator()
        metric.update(BatchOutput(torch.tensor(1.), 2))
        saved = snapshot_metrics(metric)
        saved["state"].pop("counted_batches")
        before = copy.deepcopy(saved)
        restored = MetricAccumulator()
        restore_metrics(restored, saved, "cpu")
        self.assertEqual(restored.counted_batches.item(), 1)
        self.assertNotEqual(restored.counted_batches.data_ptr(), restored.batches.data_ptr())
        restored.update(BatchOutput(torch.tensor(3.), 5))
        self.assertEqual(restored.counted_batches.item(), 2)
        self.assertEqual(restored.compute()["samples"].item(), 7)
        self.assertEqual(restored.compute()["loss"].item(), 2)
        self.assertEqual(saved.keys(), before.keys())
        self.assertEqual(saved["types"], before["types"])
        self.assertEqual(saved["updates"], before["updates"])
        self.assertEqual(saved["state"].keys(), before["state"].keys())
        for name, value in before["state"].items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(saved["state"][name], value, rtol=0, atol=0)
            else:
                self.assertEqual(saved["state"][name], value)

    def test_migration_rejects_other_missing_or_extra_states_and_subclasses(self):
        metric = MetricAccumulator()
        metric.update(BatchOutput(torch.tensor(1.), 2))
        old_state = snapshot_metrics(metric)
        old_state["state"].pop("counted_batches")
        for missing in ("samples", "batches", "loss_weight", "_extra_state"):
            with self.subTest(missing=missing):
                changed = copy.deepcopy(old_state)
                changed["state"].pop(missing)
                with self.assertRaisesRegex(ValueError, "same registered metric states"):
                    restore_metrics(MetricAccumulator(), changed, "cpu")
        changed = copy.deepcopy(old_state)
        changed["state"]["unexpected"] = torch.tensor(1)
        with self.assertRaisesRegex(ValueError, "same registered metric states"):
            restore_metrics(MetricAccumulator(), changed, "cpu")

        class CustomAccumulator(MetricAccumulator):
            pass

        custom = CustomAccumulator()
        custom.update(BatchOutput(torch.tensor(1.), 2))
        custom_state = snapshot_metrics(custom)
        custom_state["state"].pop("counted_batches")
        with self.assertRaisesRegex(ValueError, "same registered metric states"):
            restore_metrics(CustomAccumulator(), custom_state, "cpu")

    def test_migration_rejects_malformed_batch_counter_and_leaves_new_state_unchanged(self):
        metric = MetricAccumulator()
        metric.update(BatchOutput(torch.tensor(1.)))
        saved = snapshot_metrics(metric)
        restored = MetricAccumulator()
        restore_metrics(restored, saved, "cpu")
        self.assertEqual(restored.counted_batches.item(), 0)
        self.assertEqual(set(restored.compute()), {"loss", "batches"})
        for value in (torch.tensor(1.), torch.tensor([1]), torch.tensor(-1), 1):
            with self.subTest(counter=value):
                changed = copy.deepcopy(saved)
                changed["state"].pop("counted_batches")
                changed["state"]["batches"] = value
                with self.assertRaisesRegex(ValueError, "Invalid saved batch count"):
                    restore_metrics(MetricAccumulator(), changed, "cpu")

    @staticmethod
    def checkpoint_components(module_type=UnknownCountModule):
        torch.manual_seed(17)
        module = module_type(nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.2), nn.Linear(3, 1)))
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
        trainer = Trainer(module, optimizer, accumulation_steps=2)
        data = TensorDataset(torch.arange(22, dtype=torch.float32).reshape(11, 2) / 22, torch.zeros(11, 1))
        generator = torch.Generator().manual_seed(8)
        return (trainer, ResumableDataLoader(data, batch_size=4, generator=generator),
                DataLoader(data, batch_size=4), {"train": generator})

    def test_old_counted_step_checkpoint_resumes_like_current_schema(self):
        full, train, validation, _ = self.checkpoint_components(CountedModule)
        expected = []
        full.fit(train, validation, epochs=2, callbacks=[lambda trainer, row: expected.append(row)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_step.pt"
            interrupted, train, validation, generators = self.checkpoint_components(CountedModule)

            def stop(trainer, event):
                save_checkpoint(path, trainer, {}, generators=generators, history=[],
                                kind="step", train_loader=train)
                raise InterruptedError()

            with self.assertRaises(InterruptedError):
                interrupted.fit(train, validation, epochs=2, step_callbacks=[stop])
            payload = read_checkpoint(path)
            payload["epoch_progress"]["metrics"]["state"].pop("counted_batches")
            old_samples = payload["epoch_progress"]["metrics"]["state"]["samples"].clone()
            resumed, train, validation, generators = self.checkpoint_components(CountedModule)
            restore_checkpoint(payload, resumed, {}, generators=generators, train_loader=train)
            actual = []
            resumed.fit(train, validation, epochs=2, callbacks=[lambda trainer, row: actual.append(row)])
            self.assertEqual(full.state, resumed.state)
            for left, right in zip(full.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            for left, right in zip(expected, actual):
                self.assertEqual(left["train"]["loss"], right["train"]["loss"])
                self.assertEqual(left["train"]["samples"], right["train"]["samples"])
                self.assertEqual(left["validation"], right["validation"])
            self.assertNotIn("counted_batches", payload["epoch_progress"]["metrics"]["state"])
            torch.testing.assert_close(payload["epoch_progress"]["metrics"]["state"]["samples"], old_samples)

    def test_unknown_counts_resume_exactly_inside_and_at_end_of_epoch(self):
        setup = self.checkpoint_components

        full, train, validation, _ = setup()
        expected = []
        full.fit(train, validation, epochs=2, callbacks=[lambda trainer, row: expected.append(row)])
        for stop_step in (1, 2):
            with self.subTest(step=stop_step), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "last_step.pt"
                interrupted, train, validation, generators = setup()

                def stop(trainer, event):
                    self.assertIsNone(event["samples_processed"])
                    if event["optimizer_step"] == stop_step:
                        save_checkpoint(path, trainer, {}, generators=generators, history=[],
                                        kind="step", train_loader=train)
                        raise InterruptedError()

                with self.assertRaises(InterruptedError):
                    interrupted.fit(train, validation, epochs=2, step_callbacks=[stop])
                payload = read_checkpoint(path)
                self.assertIsNone(payload["epoch_progress"]["samples"])
                resumed, train, validation, generators = setup()
                restore_checkpoint(payload, resumed, {}, generators=generators, train_loader=train)
                actual = []
                resumed.fit(train, validation, epochs=2, callbacks=[lambda trainer, row: actual.append(row)])
                self.assertEqual(full.state, resumed.state)
                for left, right in zip(full.model.parameters(), resumed.model.parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                for left, right in zip(expected, actual):
                    self.assertEqual(left["train"]["loss"], right["train"]["loss"])
                    self.assertEqual(left["train"]["batches"], right["train"]["batches"])
                    self.assertEqual(left["validation"], right["validation"])
                    self.assertNotIn("samples_per_second", right["train"])


if __name__ == "__main__":
    unittest.main()
