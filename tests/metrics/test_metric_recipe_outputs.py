from collections import UserDict
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Metric

from deepopt import Trainer
from deepopt.data.resumable import ResumableDataLoader
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.engine.reporting.metric_lifecycle import MetricLifecycle
from deepopt.observers import CheckObservers


class CompoundError(Metric):
    gradient_flags = []

    def __init__(self):
        super().__init__()
        self.add_state("absolute_sum", default=torch.tensor(0., dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("squared_sum", default=torch.tensor(0., dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions, targets):
        type(self).gradient_flags.append((predictions.requires_grad, targets.requires_grad, torch.is_grad_enabled()))
        errors = (predictions - targets).double()
        self.absolute_sum += errors.abs().sum()
        self.squared_sum += errors.square().sum()
        self.count += errors.numel()

    def compute(self):
        return {"mae": self.absolute_sum / self.count,
                "moments": UserDict({"mse": self.squared_sum / self.count})}


class FixedResults(Metric):
    def __init__(self, results):
        super().__init__()
        self.results = results
        self.add_state("count", default=torch.tensor(0))

    def update(self, *args):
        self.count += 1

    def compute(self):
        return self.results


class MetricRecipeOutputTests(unittest.TestCase):
    def setUp(self):
        CompoundError.gradient_flags.clear()
        self.addCleanup(CompoundError.gradient_flags.clear)

    def computed(self, results, *, checks=True):
        metric = FixedResults(results)
        metric.update()
        return MetricLifecycle.compute(metric, observe=CheckObservers(checks).observe)

    def test_compound_metric_fits_and_can_be_monitored_without_result_wrappers(self):
        torch.manual_seed(21)
        model = nn.Linear(2, 1).double()
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1), loss_fn=F.mse_loss,
                          metrics={"quality": CompoundError()})
        batches = [(torch.randn(count, 2, dtype=torch.double),
                    torch.randn(count, 1, dtype=torch.double, requires_grad=True)) for count in (3, 1)]
        rows = []
        trainer.fit(batches, batches, epochs=2, monitor="quality/mae",
                    callbacks=[lambda trainer, row: rows.append(row)])
        with torch.no_grad():
            errors = torch.cat([model(inputs) - targets for inputs, targets in batches])
            expected_mae, expected_mse = errors.abs().mean().item(), errors.square().mean().item()
        self.assertAlmostEqual(rows[-1]["validation"]["quality/mae"], expected_mae)
        self.assertAlmostEqual(rows[-1]["validation"]["quality/moments/mse"], expected_mse)
        self.assertEqual(trainer.state.best_score, min(row["validation"]["quality/mae"] for row in rows))
        self.assertTrue(CompoundError.gradient_flags)
        self.assertTrue(all(flags == (False, False, False) for flags in CompoundError.gradient_flags))

    def test_flat_names_stay_unchanged_and_nested_scalar_values_are_not_reduced(self):
        results = self.computed(UserDict({"loss": torch.tensor(4.), "samples": torch.tensor(3),
                                         "already/named": torch.tensor(2.),
                                         "quality": {"mae": torch.tensor(7.), "extra": {"mse": torch.tensor(11.)}}}))
        self.assertEqual(results, {"loss": 4., "samples": 3, "already/named": 2.,
                                   "quality/mae": 7., "quality/extra/mse": 11.})

    def test_name_collisions_are_rejected_in_either_iteration_order(self):
        for results in ({"a/b": torch.tensor(1.), "a": {"b": torch.tensor(2.)}},
                        {"a": {"b": torch.tensor(1.)}, "a/b": torch.tensor(2.)}):
            with self.subTest(results=results), self.assertRaisesRegex(ValueError, "collide.*a/b"):
                self.computed(results)

    def test_root_reserved_fields_keep_their_contracts_but_namespaced_leaves_are_user_metrics(self):
        for name in ("loss", "samples", "batches"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Reserved root metric"):
                self.computed({name: {"nested": torch.tensor(1.)}})
        for name in ("seconds", "samples_per_second"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "reserved timing"):
                self.computed({name: {"nested": torch.tensor(1.)}})
        results = self.computed({"quality": {"loss": torch.tensor(2.), "samples": torch.tensor(-.5),
                                             "seconds": torch.tensor(3.)}})
        self.assertEqual(results, {"quality/loss": 2., "quality/samples": -.5, "quality/seconds": 3.})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.computed({"samples": torch.tensor(0), "quality": {"score": torch.tensor(1.)}})

    def test_vector_outputs_request_an_explicit_reduction_or_named_scalars(self):
        for results in ({"quality": torch.tensor([1., 2.])},
                        {"quality": {"per_class": torch.tensor([1., 2.])}}):
            with self.subTest(results=results), self.assertRaisesRegex(ValueError, "reduce it explicitly.*named scalar"):
                self.computed(results)
        self.assertEqual(self.computed({"quality": {"class_0": torch.tensor(1.), "class_1": torch.tensor(2.)}}),
                         {"quality/class_0": 1., "quality/class_1": 2.})

    def test_invalid_nested_names_values_and_cycles_fail_clearly(self):
        for results in ({"quality": {1: torch.tensor(1.)}}, {"quality": {"": torch.tensor(1.)}},
                        {"quality": {"mae": 1.}}):
            with self.subTest(results=results), self.assertRaises(ValueError):
                self.computed(results)
        cycle = {}
        cycle["child"] = cycle
        # Invoke the boundary with an unwrapped compute method: TorchMetrics'
        # own output traversal also expects acyclic data.
        class Cyclic:
            def compute(self):
                return cycle
        with self.assertRaisesRegex(ValueError, "cycles"):
            MetricLifecycle.compute(Cyclic())

    def test_nested_metrics_respect_runtime_default_and_disabled_finite_checks(self):
        for checks in (True, False):
            with self.subTest(checks=checks):
                model = nn.Linear(2, 1)
                trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1), loss_fn=F.mse_loss,
                                  checks=checks, metrics={"quality": FixedResults({"mae": torch.tensor(float("nan"))})})
                batch = (torch.ones(2, 2), torch.zeros(2, 1))
                if checks:
                    with self.assertRaisesRegex(ValueError, "finite scalar"):
                        trainer.evaluate([batch])
                else:
                    with patch("torch.isfinite", side_effect=AssertionError("Disabled metric scans")):
                        results = trainer.evaluate([batch])
                    self.assertTrue(math.isnan(results["quality/mae"]))

    def test_final_metric_observer_receives_detached_flattened_leaf_tensors(self):
        parameter = torch.tensor(2., requires_grad=True)
        value = parameter.square()
        observed = []

        def record(event):
            observed.append((event.kind, event.name, tuple(tensor.item() for tensor in event.tensors)))
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(all(not tensor.requires_grad and tensor.grad_fn is None for tensor in event.tensors))

        self.assertEqual(self.computed({"quality": {"score": value, "other": value + 1}}, checks=[record]),
                         {"quality/score": 4., "quality/other": 5.})
        self.assertEqual(observed, [("metrics", "metric results", (4., 5.))])
        value.backward()
        self.assertEqual(parameter.grad.item(), 4.)

    def resumable_setup(self):
        torch.manual_seed(52)
        data_generator = torch.Generator().manual_seed(53)
        dataset = TensorDataset(torch.randn(9, 2, dtype=torch.double, generator=data_generator),
                                torch.randn(9, 1, dtype=torch.double, generator=data_generator))
        model = nn.Linear(2, 1).double()
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1, momentum=.5),
                          loss_fn=F.mse_loss, metrics={"quality": CompoundError()}, accumulation_steps=2)
        generator = torch.Generator().manual_seed(54)
        loader = ResumableDataLoader(dataset, batch_size=2, generator=generator)
        return trainer, loader, DataLoader(dataset, batch_size=3), {"train": generator}

    def test_step_resume_keeps_metric_state_schema_and_compound_results(self):
        full, loader, validation, _ = self.resumable_setup()
        expected_rows = []
        full.fit(loader, validation, epochs=2, monitor="quality/mae",
                 callbacks=[lambda trainer, row: expected_rows.append(row)])

        with tempfile.TemporaryDirectory() as directory:
            interrupted, loader, validation, generators = self.resumable_setup()
            checkpoint = Path(directory) / "last_step.pt"

            def save_and_interrupt(trainer, event):
                if event["optimizer_step"] == 1:
                    save_checkpoint(checkpoint, trainer, {}, generators=generators, history=[],
                                    kind="step", train_loader=loader)
                    raise InterruptedError("saved")

            with self.assertRaises(InterruptedError):
                interrupted.fit(loader, validation, epochs=2, monitor="quality/mae", step_callbacks=[save_and_interrupt])
            payload = read_checkpoint(checkpoint)
            saved_metrics = payload["epoch_progress"]["metrics"]
            self.assertEqual(saved_metrics["types"]["task_metrics.0"], f"{CompoundError.__module__}.{CompoundError.__qualname__}")
            self.assertIn("task_metrics.0.absolute_sum", saved_metrics["state"])
            self.assertNotIn("quality/mae", saved_metrics["state"])
            saved_total = saved_metrics["state"]["task_metrics.0.absolute_sum"].clone()

            resumed, loader, validation, generators = self.resumable_setup()
            actual_rows = restore_checkpoint(payload, resumed, {}, generators=generators, train_loader=loader)
            resumed.fit(loader, validation, epochs=2, monitor="quality/mae",
                        callbacks=[lambda trainer, row: actual_rows.append(row)])
            for name, value in full.model.state_dict().items():
                torch.testing.assert_close(resumed.model.state_dict()[name], value, rtol=0, atol=0)
            self.assertEqual(resumed.state, full.state)
            for actual, expected in zip(actual_rows, expected_rows):
                self.assertEqual(actual["validation"], expected["validation"])
                for name in ("loss", "samples", "quality/mae", "quality/moments/mse"):
                    self.assertEqual(actual["train"][name], expected["train"][name])
            torch.testing.assert_close(saved_metrics["state"]["task_metrics.0.absolute_sum"], saved_total, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
