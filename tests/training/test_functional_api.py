import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanMetric, MeanSquaredError

from deepopt.engine import Trainer
from deepopt.data.resumable import ResumableDataLoader
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.engine.reporting.metrics import MetricAccumulator
from deepopt.module import TrainingModule
from deepopt.tasks.base import BatchOutput
from deepopt.tasks.loss import LossResult
from deepopt.tasks.regression import RegressionTask


class RecordingMSE(MeanSquaredError):
    updates = []

    def update(self, preds, target):
        type(self).updates.append((self, preds.requires_grad, target.requires_grad, torch.is_grad_enabled()))
        return super().update(preds, target)


class RecordingMean(MeanMetric):
    updates = []

    def update(self, value, weight=1.):
        type(self).updates.append((self, value.requires_grad, torch.is_grad_enabled()))
        return super().update(value, weight)


def squared_error_step(model, batch):
    inputs, targets = batch
    return F.mse_loss(model(inputs), targets)


def absolute_error_step(model, batch):
    inputs, targets = batch
    return F.l1_loss(model(inputs), targets)


class FunctionalAPITests(unittest.TestCase):
    def setUp(self):
        RecordingMSE.updates.clear()
        RecordingMean.updates.clear()
        self.addCleanup(RecordingMSE.updates.clear)
        self.addCleanup(RecordingMean.updates.clear)

    def assert_parameters_equal(self, actual, expected):
        for left, right in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def resumable_setup(self, step_fn=squared_error_step):
        torch.manual_seed(51)
        data_generator = torch.Generator().manual_seed(52)
        dataset = TensorDataset(torch.randn(13, 2, generator=data_generator, dtype=torch.double),
                                torch.randn(13, 1, generator=data_generator, dtype=torch.double))
        model = nn.Sequential(nn.Linear(2, 5), nn.Dropout(0.3), nn.Tanh(), nn.Linear(5, 1)).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        trainer = Trainer(model, optimizer, step_fn=step_fn, accumulation_steps=2, scheduler=scheduler)
        generator = torch.Generator().manual_seed(53)
        loader = ResumableDataLoader(dataset, batch_size=3, generator=generator)
        return trainer, loader, DataLoader(dataset, batch_size=4), {"train": generator}

    def test_public_imports_expose_the_existing_training_implementations(self):
        from deepopt import MultiOptimizer, Trainer as PublicTrainer, TrainingModule as PublicModule, TrainingRuntime
        from deepopt.engine import TrainingRuntime as EngineRuntime
        from deepopt.optimizers import MultiOptimizer as OptimizerImplementation

        self.assertIs(PublicTrainer, Trainer)
        self.assertIs(PublicModule, TrainingModule)
        self.assertIs(TrainingRuntime, EngineRuntime)
        self.assertIs(MultiOptimizer, OptimizerImplementation)

    def test_scalar_function_step_resumes_exactly_through_uneven_and_partial_windows(self):
        full, train_loader, validation, generators = self.resumable_setup()
        expected_rows = []
        full.fit(train_loader, validation, epochs=3,
                 callbacks=[lambda trainer, row: expected_rows.append(row)])
        expected_rng = torch.get_rng_state().clone()
        expected_generator = generators["train"].get_state().clone()

        for stop_step in (1, 3, 4):
            with self.subTest(stop_step=stop_step), tempfile.TemporaryDirectory() as directory:
                interrupted, loader, validation, generators = self.resumable_setup()
                checkpoint = Path(directory) / "last_step.pt"
                rows = []

                def save_then_interrupt(trainer, event):
                    if event["optimizer_step"] == stop_step:
                        save_checkpoint(checkpoint, trainer, {}, generators=generators, history=rows,
                                        kind="step", train_loader=loader)
                        raise InterruptedError("resume this completed optimizer boundary")

                with self.assertRaises(InterruptedError):
                    interrupted.fit(loader, validation, epochs=3,
                                    callbacks=[lambda trainer, row: rows.append(row)],
                                    step_callbacks=[save_then_interrupt])
                resumed, loader, validation, generators = self.resumable_setup()
                rows = restore_checkpoint(read_checkpoint(checkpoint), resumed, {},
                                          generators=generators, train_loader=loader)
                resumed.fit(loader, validation, epochs=3,
                            callbacks=[lambda trainer, row: rows.append(row)])
                self.assert_nested_equal(resumed.model.state_dict(), full.model.state_dict())
                self.assert_nested_equal(resumed.optimizer.state_dict(), full.optimizer.state_dict())
                self.assert_nested_equal(resumed.scheduler.state_dict(), full.scheduler.state_dict())
                self.assertEqual(resumed.state, full.state)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                torch.testing.assert_close(generators["train"].get_state(), expected_generator, rtol=0, atol=0)
                self.assertEqual(len(rows), 3)
                for actual, expected in zip(rows, expected_rows):
                    for key in ("epoch", "learning_rates", "optimizer_steps", "validation"):
                        self.assert_nested_equal(actual[key], expected[key])
                    self.assertEqual(actual["train"]["loss"], expected["train"]["loss"])
                    self.assertEqual(actual["train"]["batches"], 5)
                    self.assertNotIn("samples", actual["train"])

    def test_changed_named_step_is_rejected_before_restoring_any_state(self):
        original, loader, validation, generators = self.resumable_setup()
        history = []
        original.fit(loader, validation, epochs=1, callbacks=[lambda trainer, row: history.append(row)])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            save_checkpoint(checkpoint, original, {}, generators=generators, history=history)
            saved = read_checkpoint(checkpoint)

        changed, loader, _, generators = self.resumable_setup(absolute_error_step)
        changed.train_epoch(loader)
        model_state = copy.deepcopy(changed.model.state_dict())
        optimizer_state = copy.deepcopy(changed.optimizer.state_dict())
        scheduler_state = copy.deepcopy(changed.scheduler.state_dict())
        state = copy.deepcopy(changed.state)
        generator_state = generators["train"].get_state().clone()
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, "trainer settings"):
            restore_checkpoint(saved, changed, {}, generators=generators, train_loader=loader)
        self.assert_nested_equal(changed.model.state_dict(), model_state)
        self.assert_nested_equal(changed.optimizer.state_dict(), optimizer_state)
        self.assert_nested_equal(changed.scheduler.state_dict(), scheduler_state)
        self.assertEqual(changed.state, state)
        torch.testing.assert_close(generators["train"].get_state(), generator_state, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        changed.train_epoch(loader)
        self.assertEqual(changed.state.optimizer_steps, state.optimizer_steps + 3)

    def test_plain_loss_function_fits_nonlinear_model_without_a_task(self):
        torch.manual_seed(40)
        model = nn.Sequential(nn.Linear(2, 5), nn.SiLU(), nn.Linear(5, 3)).double()
        reference = copy.deepcopy(model)
        batches = [(torch.randn(count, 2, dtype=torch.double), torch.randn(count, 3, dtype=torch.double))
                   for count in (4, 1, 2)]
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.02),
                          loss_fn=F.smooth_l1_loss, accumulation_steps=2)
        rows = []
        trainer.fit(batches, batches, epochs=1, callbacks=[lambda trainer, row: rows.append(row)])

        optimizer = torch.optim.SGD(reference.parameters(), lr=0.02)
        batch_losses = []
        for start in range(0, len(batches), 2):
            losses = [F.smooth_l1_loss(reference(inputs), targets)
                      for inputs, targets in batches[start:start + 2]]
            batch_losses.extend(loss.item() for loss in losses)
            optimizer.zero_grad()
            (sum(losses) / len(losses)).backward()
            optimizer.step()
        self.assert_parameters_equal(model, reference)
        self.assertEqual(trainer.state.optimizer_steps, 2)
        self.assertAlmostEqual(rows[0]["train"]["loss"], sum(batch_losses) / 3, places=12)
        self.assertEqual(rows[0]["train"]["samples"], 7)
        self.assertIn("samples_per_second", rows[0]["train"])

    def test_contrastive_step_accepts_arbitrary_batches_and_separate_validation(self):
        torch.manual_seed(41)
        model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 3)).double()
        reference = copy.deepcopy(model)
        batches = [{"first_view": torch.randn(count, 2, dtype=torch.double),
                    "second_view": torch.randn(count, 2, dtype=torch.double), "temperature": 0.3}
                   for count in (3, 2, 4)]
        phases = []

        def objective(current, batch):
            left = F.normalize(current(batch["first_view"]), dim=-1)
            right = F.normalize(current(batch["second_view"]), dim=-1)
            scores = left @ right.T / batch["temperature"]
            return -scores.log_softmax(dim=-1).diagonal().mean()

        def train_step(current, batch):
            phases.append(("train", current.training, torch.is_grad_enabled()))
            return objective(current, batch)

        def validate_step(current, batch):
            phases.append(("validation", current.training, torch.is_grad_enabled()))
            return (current(batch["first_view"]) - current(batch["second_view"])).abs().mean()

        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                          step_fn=train_step, validation_step_fn=validate_step, accumulation_steps=2)
        rows = []
        trainer.fit(batches, batches, epochs=1, callbacks=[lambda trainer, row: rows.append(row)])
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        for start in range(0, len(batches), 2):
            losses = [objective(reference, batch) for batch in batches[start:start + 2]]
            optimizer.zero_grad()
            (sum(losses) / len(losses)).backward()
            optimizer.step()
        self.assert_parameters_equal(model, reference)
        self.assertEqual(phases, [("train", True, True)] * 3 + [("validation", False, False)] * 3)
        self.assertTrue(model.training)
        for phase in ("train", "validation"):
            self.assertNotIn("samples", rows[0][phase])
            self.assertNotIn("samples_per_second", rows[0][phase])
        with torch.no_grad():
            expected = sum((reference(batch["first_view"]) - reference(batch["second_view"])).abs().mean().item()
                           for batch in batches) / len(batches)
        self.assertAlmostEqual(rows[0]["validation"]["loss"], expected, places=12)

    def test_metric_templates_are_cloned_per_phase_and_hook_inputs_are_detached(self):
        model = nn.Linear(2, 1)
        template = RecordingMSE()
        template.update(torch.tensor([9.]), torch.tensor([0.]))
        RecordingMSE.updates.clear()
        batches = [{"inputs": torch.ones(2, 2), "targets": torch.zeros(2, 1, requires_grad=True)}]
        hooks = []

        def step(current, batch):
            predictions = current(batch["inputs"])
            return {"loss": (predictions - batch["targets"]).square().mean(), "predictions": predictions,
                    "sample_count": batch["targets"].shape[0], "extra": "available to the metric hook"}

        def metric_inputs(output, batch):
            hooks.append(output["extra"])
            return output["predictions"], batch["targets"]

        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), step_fn=step,
                          metrics={"mse": template}, metric_inputs=metric_inputs)
        rows = []
        trainer.fit(batches, batches, epochs=1, callbacks=[lambda trainer, row: rows.append(row)])
        self.assertEqual(len(hooks), 2)
        self.assertEqual(len(RecordingMSE.updates), 2)
        train_metric, validation_metric = (record[0] for record in RecordingMSE.updates)
        self.assertIsNot(train_metric, validation_metric)
        self.assertIsNot(train_metric, template)
        self.assertIsNot(validation_metric, template)
        self.assertEqual(template.update_count, 1)
        self.assertEqual(template.compute().item(), 81.)
        for _, predictions_require_grad, targets_require_grad, grad_enabled in RecordingMSE.updates:
            self.assertFalse(predictions_require_grad)
            self.assertFalse(targets_require_grad)
            self.assertFalse(grad_enabled)
        with torch.no_grad():
            expected = model(batches[0]["inputs"]).square().mean().item()
        self.assertAlmostEqual(rows[0]["validation"]["mse"], expected, places=6)

    def test_loss_function_supplies_prediction_target_metrics_automatically(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                          loss_fn=F.mse_loss, metrics={"mse": MeanSquaredError()})
        inputs, targets = torch.ones(3, 2), torch.zeros(3, 1)
        result = trainer.evaluate([(inputs, targets)])
        self.assertAlmostEqual(result["mse"], F.mse_loss(model(inputs), targets).item(), places=6)
        self.assertEqual(result["samples"], 3)

    def test_per_metric_input_functions_preserve_default_inputs_for_other_metrics(self):
        torch.manual_seed(43)
        model = nn.Linear(2, 1)
        mse_template, mean_template = RecordingMSE(), RecordingMean()
        mse_template.update(torch.tensor([3.]), torch.tensor([0.]))
        mean_template.update(torch.tensor([10.]))
        RecordingMSE.updates.clear()
        RecordingMean.updates.clear()
        batches = [(torch.randn(count, 2), torch.randn(count, 1, requires_grad=True))
                   for count in (3, 1)]
        trainer = Trainer(
            model, torch.optim.SGD(model.parameters(), lr=0.01), loss_fn=F.mse_loss,
            metrics={"mse": mse_template, "prediction_mean": mean_template},
            metric_inputs={"prediction_mean": lambda output, batch: (output["predictions"],)},
        )
        rows = []
        trainer.fit(batches, batches, epochs=1, callbacks=[lambda trainer, row: rows.append(row)])
        self.assertEqual(len(RecordingMSE.updates), 4)
        self.assertEqual(len(RecordingMean.updates), 4)
        for records, template in ((RecordingMSE.updates, mse_template), (RecordingMean.updates, mean_template)):
            self.assertEqual(len({id(record[0]) for record in records}), 2)
            for metric, *gradient_flags in records:
                self.assertIsNot(metric, template)
                self.assertFalse(any(gradient_flags))
                self.assertEqual(metric.update_count, 2)
        self.assertEqual(mse_template.update_count, 1)
        self.assertEqual(mean_template.update_count, 1)
        self.assertEqual(mse_template.compute().item(), 9.)
        self.assertEqual(mean_template.compute().item(), 10.)
        with torch.no_grad():
            predictions = torch.cat([model(inputs) for inputs, _ in batches])
            targets = torch.cat([targets for _, targets in batches])
            expected_mse = F.mse_loss(predictions, targets).item()
            expected_mean = predictions.mean().item()
        self.assertAlmostEqual(rows[0]["validation"]["mse"], expected_mse, places=6)
        self.assertAlmostEqual(rows[0]["validation"]["prediction_mean"], expected_mean, places=6)

    def test_per_metric_input_mapping_rejects_unknown_names_and_noncallables(self):
        cases = (
            ({"mean": MeanMetric()}, {"missing": lambda output, batch: (output["predictions"],)}),
            ({"mean": MeanMetric()}, {"mean": 3}),
            (None, {"mean": lambda output, batch: (output["predictions"],)}),
        )
        for index, (metrics, metric_inputs) in enumerate(cases):
            with self.subTest(case=index):
                model = nn.Linear(2, 1)
                with self.assertRaises((ValueError, TypeError)):
                    Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                            loss_fn=F.mse_loss, metrics=metrics, metric_inputs=metric_inputs)

    def test_step_dictionary_metric_data_supports_keyword_arguments(self):
        model = nn.Linear(2, 1)

        def step(current, batch):
            predictions = current(batch["x"])
            return {"loss": F.mse_loss(predictions, batch["y"]), "sample_count": len(batch["y"]),
                    "metric_data": {"preds": predictions, "target": batch["y"]}}

        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                          step_fn=step, metrics={"mse": MeanSquaredError()})
        batch = {"x": torch.ones(2, 2), "y": torch.zeros(2, 1)}
        result = trainer.evaluate([batch])
        self.assertEqual(result["samples"], 2)
        self.assertAlmostEqual(result["mse"], result["loss"], places=6)

    def test_step_can_return_batch_output_with_explicit_count(self):
        model = nn.Linear(2, 1)

        def step(current, batch):
            return BatchOutput(current(batch).square().mean(), sample_count=batch.shape[0])

        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), step_fn=step)
        result = trainer.train_epoch([torch.ones(3, 2)])
        self.assertEqual(result["samples"], 3)
        self.assertIn("samples_per_second", result)

    def test_step_can_return_weighted_loss_result_without_sample_count(self):
        torch.manual_seed(42)
        model = nn.Linear(2, 1).double()
        reference = copy.deepcopy(model)
        batches = [{"x": torch.randn(n, 2, dtype=torch.double),
                    "weights": torch.tensor(weights, dtype=torch.double).reshape(-1, 1)}
                   for n, weights in ((2, [0.25, 0.5]), (1, [0.125]))]

        def step(current, batch):
            weights = batch["weights"]
            total = (current(batch["x"]).square() * weights).sum()
            denominator = weights.sum().item()
            return LossResult.from_mean(total / denominator, weight=denominator)

        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                          step_fn=step, accumulation_steps=2)
        result = trainer.train_epoch(batches)
        total = sum((reference(batch["x"]).square() * batch["weights"]).sum() for batch in batches)
        denominator = sum(batch["weights"].sum().item() for batch in batches)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        (total / denominator).backward()
        optimizer.step()
        self.assert_parameters_equal(model, reference)
        self.assertAlmostEqual(result["loss"], total.detach().item() / denominator, places=12)
        self.assertNotIn("samples", result)
        self.assertNotIn("samples_per_second", result)

    def test_scalar_or_mapping_targets_do_not_imply_a_sample_count(self):
        for target in (torch.tensor(0.), {"value": torch.zeros(3, 1)}):
            with self.subTest(target_type=type(target).__name__):
                model = nn.Linear(2, 1)

                def loss(predictions, targets):
                    expected = targets["value"] if isinstance(targets, dict) else targets
                    return (predictions - expected).square().mean()

                trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), loss_fn=loss)
                result = trainer.train_epoch([(torch.ones(3, 2), target)])
                self.assertIn("loss", result)
                self.assertNotIn("samples", result)
                self.assertNotIn("samples_per_second", result)

    def test_conflicting_owners_and_metric_routes_are_rejected(self):
        for options in (
            {"loss_fn": F.mse_loss, "step_fn": lambda model, batch: model(batch).mean()},
            {"task": RegressionTask(), "loss_fn": F.mse_loss},
            {"task": RegressionTask(), "step_fn": lambda model, batch: model(batch).mean()},
            {"validation_step_fn": lambda model, batch: model(batch).mean()},
            {"loss_fn": F.mse_loss, "metrics": {"mse": MeanSquaredError()}, "metric_factory": MetricAccumulator},
        ):
            with self.subTest(options=tuple(options)):
                model = nn.Linear(2, 1)
                with self.assertRaises((ValueError, TypeError)):
                    Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), **options)

    def test_invalid_functional_arguments_are_rejected(self):
        for options in ({"loss_fn": 1}, {"step_fn": 1},
                        {"loss_fn": F.mse_loss, "validation_step_fn": 1},
                        {"step_fn": lambda model, batch: model(batch).mean(), "metric_inputs": 1},
                        {"loss_fn": F.mse_loss, "metrics": {"score": object()}}):
            with self.subTest(options=tuple(options)):
                model = nn.Linear(2, 1)
                with self.assertRaises((ValueError, TypeError)):
                    Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), **options)

    def test_step_metrics_require_declared_metric_inputs(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                          step_fn=lambda current, batch: current(batch).square().mean(),
                          metrics={"mse": MeanSquaredError()})
        with self.assertRaisesRegex(ValueError, "metric"):
            trainer.train_epoch([torch.ones(2, 2)])
        self.assertEqual(trainer.state.optimizer_steps, 0)

    def test_invalid_step_results_fail_before_any_update(self):
        for invalid in (None, {"other": torch.tensor(1.)}, (torch.tensor(1.),),
                        {"loss": torch.ones(2)}, {"loss": torch.tensor(1.), "sample_count": 0},
                        {"loss": torch.tensor(1.), "sample_count": True}):
            with self.subTest(result_type=type(invalid).__name__):
                model = nn.Linear(2, 1)
                before = copy.deepcopy(model.state_dict())
                trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                                  step_fn=lambda current, batch: invalid)
                with self.assertRaises((ValueError, TypeError)):
                    trainer.train_epoch([torch.ones(2, 2)])
                self.assertEqual(trainer.state.optimizer_steps, 0)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
                    self.assertIsNone(parameter.grad)

    def test_existing_training_module_path_remains_usable(self):
        module = TrainingModule(nn.Linear(2, 1), task=RegressionTask())
        trainer = Trainer(module, torch.optim.SGD(module.parameters(), lr=0.01))
        batches = [(torch.ones(2, 2), torch.zeros(2, 1))]
        trainer.fit(batches, batches, epochs=1)
        self.assertEqual(trainer.state.optimizer_steps, 1)
        self.assertEqual(trainer.state.completed_epochs, 1)


if __name__ == "__main__":
    unittest.main()
