import copy
from collections import OrderedDict, UserDict, defaultdict, namedtuple
from collections.abc import Mapping
from configparser import ConfigParser
from dataclasses import dataclass, field
from types import MappingProxyType
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import MeanSquaredError

from deepopt import Trainer, TrainingModule
from deepopt.engine.execution.batch import move_to_device
from deepopt.engine.training.steps import StepAdapter
from deepopt.tasks.base import BatchOutput
from deepopt.tasks.loss import LossResult


class RecipeStepTests(unittest.TestCase):
    def test_mapping_constructor_cannot_reinterpret_entries_as_metadata(self):
        metadata = ConfigParser()
        metadata["data"] = {"batch": "2", "split": "train"}
        original = {name: dict(section) for name, section in metadata.items()}
        moved = move_to_device({"inputs": torch.ones(2, 1), "metadata": metadata}, "meta")
        self.assertEqual(moved["inputs"].device.type, "meta")
        self.assertEqual(moved["metadata"], original)
        self.assertIsInstance(moved["metadata"], dict)
        self.assertEqual({name: dict(section) for name, section in metadata.items()}, original)

    def test_scalar_and_dictionary_module_hooks_match_pytorch_updates(self):
        for dictionary in (False, True):
            with self.subTest(dictionary=dictionary):
                class Recipe(TrainingModule):
                    def training_step(self, batch):
                        inputs, targets = batch
                        predictions = self(inputs)
                        loss = F.mse_loss(predictions, targets)
                        return ({"loss": loss, "predictions": predictions, "targets": targets}
                                if dictionary else loss)

                    def validation_step(self, batch):
                        inputs, targets = batch
                        predictions = self(inputs)
                        loss = F.l1_loss(predictions, targets)
                        return ({"loss": loss, "predictions": predictions, "targets": targets}
                                if dictionary else loss)

                torch.manual_seed(17)
                module = Recipe(nn.Linear(2, 1).double())
                reference = copy.deepcopy(module.model)
                batches = [(torch.randn(count, 2, dtype=torch.double), torch.randn(count, 1, dtype=torch.double))
                           for count in (3, 1)]
                metrics = {"mse": MeanSquaredError()} if dictionary else None
                trainer = Trainer(module, torch.optim.SGD(module.parameters(), lr=.1),
                                  accumulation_steps=2, metrics=metrics)
                rows = []
                trainer.fit(batches, batches, epochs=1, callbacks=[lambda trainer, row: rows.append(row)])

                optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
                objective = sum(F.mse_loss(reference(inputs), targets) for inputs, targets in batches) / 2
                objective.backward()
                optimizer.step()
                for actual, expected in zip(module.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected_validation = sum(F.l1_loss(reference(inputs), targets).item() for inputs, targets in batches) / 2
                self.assertAlmostEqual(rows[0]["validation"]["loss"], expected_validation)
                self.assertNotIn("samples", rows[0]["train"])
                self.assertEqual(rows[0]["train"]["batches"], 2)
                if dictionary:
                    expected_mse = F.mse_loss(torch.cat([reference(inputs) for inputs, _ in batches]),
                                             torch.cat([targets for _, targets in batches])).item()
                    self.assertAlmostEqual(rows[0]["validation"]["mse"], expected_mse, places=6)

    def test_task_return_forms_keep_task_metric_update_ownership(self):
        for form in ("scalar", "mapping", "result", "batch"):
            with self.subTest(form=form):
                class RecipeTask:
                    calls = 0

                    def compute(self, model, batch):
                        loss = model(batch).square().mean()
                        if form == "mapping":
                            return {"loss": loss}
                        if form == "result":
                            return LossResult(loss)
                        if form == "batch":
                            return BatchOutput(loss)
                        return loss

                    def create_metrics(self):
                        return {"own_metric": MeanSquaredError()}

                    def update_metrics(self, metrics, output):
                        self.calls += 1
                        self.received = output
                        metrics["own_metric"].update(output.loss.detach(), torch.zeros_like(output.loss))

                model = nn.Linear(2, 1)
                task = RecipeTask()
                trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1), task=task)
                results = trainer.train_epoch([torch.ones(2, 2)])
                self.assertEqual(task.calls, 1)
                self.assertIsInstance(task.received, BatchOutput)
                self.assertAlmostEqual(results["own_metric"], task.received.loss.detach().square().item(), places=6)
                self.assertEqual(trainer.state.optimizer_steps, 1)

    def test_mapping_prediction_targets_route_metrics_without_inferring_counts(self):
        model = nn.Linear(2, 1)
        inputs, targets = torch.randn(3, 2), torch.randn(3, 1)

        def step(model, batch):
            predictions = model(batch[0])
            return {"loss": F.mse_loss(predictions, batch[1]), "predictions": predictions, "targets": batch[1]}

        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1), step_fn=step,
                          metrics={"mse": MeanSquaredError()})
        results = trainer.evaluate([(inputs, targets)])
        self.assertAlmostEqual(results["mse"], F.mse_loss(model(inputs), targets).item(), places=6)
        self.assertNotIn("samples", results)
        self.assertEqual(results["batches"], 1)

    def test_explicit_metric_data_including_none_takes_precedence(self):
        for explicit_none in (False, True):
            with self.subTest(explicit_none=explicit_none):
                model = nn.Linear(2, 1)

                def step(model, batch):
                    predictions = model(batch)
                    zeros = torch.zeros_like(predictions)
                    return {"loss": predictions.square().mean(), "predictions": predictions, "targets": zeros,
                            "metric_data": None if explicit_none else (zeros, zeros)}

                trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1), step_fn=step,
                                  metrics={"mse": MeanSquaredError()})
                if explicit_none:
                    with self.assertRaisesRegex(ValueError, "Metrics need inputs"):
                        trainer.train_epoch([torch.ones(2, 2)])
                    self.assertEqual(trainer.state.optimizer_steps, 0)
                else:
                    self.assertEqual(trainer.evaluate([torch.ones(2, 2)])["mse"], 0.)

    def test_metric_input_callback_receives_original_module_output(self):
        class Recipe(TrainingModule):
            def training_step(self, batch):
                predictions = self(batch)
                self.raw = {"loss": predictions.square().mean(), "predictions": predictions,
                            "targets": torch.zeros_like(predictions), "extra": "available"}
                return self.raw

        module = Recipe(nn.Linear(2, 1))
        received = []

        def route(raw, batch):
            received.append(raw)
            self.assertEqual(raw["extra"], "available")
            return raw["predictions"], raw["targets"]

        adapter = StepAdapter(module, metrics={"mse": MeanSquaredError()}, metric_inputs=route)
        output = adapter.training_step(torch.ones(2, 2))
        self.assertIs(received[0], module.raw)
        self.assertIs(output.loss, module.raw["loss"])
        self.assertTrue(module.raw["predictions"].requires_grad)
        self.assertFalse(output.metric_data[0].requires_grad)
        self.assertNotIn("metric_data", module.raw)

    def test_existing_result_and_batch_metadata_are_preserved(self):
        class Recipe(TrainingModule):
            def training_step(self, batch):
                return self.output

        module = Recipe(nn.Linear(1, 1))
        loss = module(torch.ones(2, 1)).square().mean()
        result = LossResult.from_mean(loss, weight=.5)
        adapter = StepAdapter(module)
        module.output = result
        self.assertIs(adapter.training_step(None).objective, result)
        module.output = BatchOutput(result, sample_count=2, metric_data={"custom": "data"})
        self.assertIs(adapter.training_step(None), module.output)

    def test_mapping_transfer_preserves_common_container_types_and_original_data(self):
        Pair = namedtuple("Pair", "tensor label")
        builders = (dict, UserDict, OrderedDict, MappingProxyType,
                    lambda data: defaultdict(list, data))
        for build in builders:
            with self.subTest(container=build):
                source = build({"inputs": torch.ones(2), "nested": [Pair(torch.zeros(1), "item")]})
                moved = move_to_device(source, torch.device("meta"))
                self.assertIsInstance(moved, type(source))
                self.assertIsNot(moved, source)
                self.assertEqual(moved["inputs"].device.type, "meta")
                self.assertIsInstance(moved["nested"], list)
                self.assertIsInstance(moved["nested"][0], Pair)
                self.assertEqual(moved["nested"][0].tensor.device.type, "meta")
                self.assertEqual(moved["nested"][0].label, "item")
                self.assertEqual(source["inputs"].device.type, "cpu")
                self.assertEqual(source["nested"][0].tensor.device.type, "cpu")
                if isinstance(source, defaultdict):
                    self.assertIs(moved.default_factory, source.default_factory)

    def test_mapping_with_required_metadata_falls_back_without_mutating_source(self):
        class TaggedMapping(Mapping):
            def __init__(self, items, tag):
                self.items_by_key, self.tag = dict(items), tag

            def __getitem__(self, key):
                return self.items_by_key[key]

            def __iter__(self):
                return iter(self.items_by_key)

            def __len__(self):
                return len(self.items_by_key)

        source = TaggedMapping({"inputs": torch.ones(2)}, "source")
        moved = move_to_device(source, "meta")
        self.assertIsInstance(moved, dict)
        self.assertEqual(moved["inputs"].device.type, "meta")
        self.assertEqual(source["inputs"].device.type, "cpu")
        self.assertEqual(source.tag, "source")

    def test_dataclass_transfer_preserves_frozen_slots_and_does_not_rerun_initialization(self):
        initializations = []

        @dataclass(frozen=True, slots=True)
        class Batch:
            inputs: torch.Tensor
            metadata: dict
            cached: torch.Tensor = field(init=False)
            optional_cache: torch.Tensor | None = field(init=False, default=None)

            def __post_init__(self):
                initializations.append(1)
                object.__setattr__(self, "cached", self.inputs.sum())

        source = Batch(torch.ones(2), {"targets": torch.zeros(2), "label": "unchanged"})
        moved = move_to_device(source, "meta")
        self.assertIsInstance(moved, Batch)
        self.assertIsNot(moved, source)
        self.assertEqual(initializations, [1])
        self.assertEqual(moved.inputs.device.type, "meta")
        self.assertEqual(moved.cached.device.type, "meta")
        self.assertEqual(moved.metadata["targets"].device.type, "meta")
        self.assertEqual(moved.metadata["label"], "unchanged")
        self.assertIsNone(moved.optional_cache)
        self.assertEqual(source.inputs.device.type, "cpu")
        self.assertEqual(source.cached.device.type, "cpu")
        self.assertEqual(source.metadata["targets"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
