"""Dictionary recipes exercise the same public training and recovery paths."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import torch
from torch import nn
from torch.utils.data import TensorDataset

from deepopt.baseline import baseline_sample_batch, build_baseline_experiment
from deepopt.config import ConfigError, ExperimentConfig, config_from_dict
from deepopt.experiment import build_experiment
from deepopt.run import evaluate, train


def configuration():
    return {
        "name": "dictionary-recipe", "seed": 19,
        "training": {"epochs": 2, "batch_size": 3, "learning_rate": 0.03, "optimizer": "sgd"},
        "application": {"synthetic": {"train_samples": 8, "validation_samples": 4,
                                       "input_dim": 3, "num_classes": 2}},
        "runtime": {"fp8": {"enabled": True}, "compile": {"enabled": False}},
    }


def recipe(config):
    inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24
    dataset = TensorDataset(inputs, inputs.sum(dim=1, keepdim=True))
    return build_experiment(config, model_factory=lambda: nn.Linear(3, 1),
                            dataset_factory=lambda: (dataset, dataset), loss_fn=nn.functional.mse_loss)


class DictionaryRunBlackBoxTests(unittest.TestCase):
    def test_experiment_factory_mapping_matches_typed_training(self):
        raw = configuration()
        before = deepcopy(raw)
        mapped, batches, validation, _ = recipe(raw)
        typed, typed_batches, typed_validation, _ = recipe(config_from_dict(raw))
        mapped.train_epoch(batches)
        typed.train_epoch(typed_batches)
        self.assertTrue(mapped.runtime.fp8.enabled)
        self.assertEqual(mapped.state.optimizer_steps, 3)
        for name, tensor in mapped.model.state_dict().items():
            torch.testing.assert_close(tensor, typed.model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(mapped.evaluate(validation)["loss"], typed.evaluate(typed_validation)["loss"])
        self.assertEqual(raw, before)

    def test_dictionary_run_overrides_resume_and_evaluation_match_uninterrupted(self):
        raw = configuration()
        before = deepcopy(raw)
        seen = []

        def factory(config):
            seen.append(config)
            self.assertIsInstance(config, ExperimentConfig)
            return recipe(config)

        options = {"experiment_factory": factory, "batch_size_mode": "manual", "progress_bar": False}
        with tempfile.TemporaryDirectory() as directory:
            resumed_path = train(raw, output=Path(directory, "resumed"), epochs=1, **options)
            train(raw, resume=resumed_path / "last.pt", epochs=2, **options)
            reference_path = train(config_from_dict(raw), output=Path(directory, "reference"), **options)
            resumed = torch.load(resumed_path / "last.pt", weights_only=True)
            reference = torch.load(reference_path / "last.pt", weights_only=True)
            self.assertEqual(resumed["config"], reference["config"])
            self.assertEqual(resumed["state"], reference["state"])
            for name, tensor in resumed["model"].items():
                torch.testing.assert_close(tensor, reference["model"][name], rtol=0, atol=0)
            result = evaluate(resumed_path / "last.pt", experiment_factory=factory)
            self.assertEqual(result["epoch"], 2)
            self.assertTrue(torch.isfinite(torch.tensor(result["validation"]["loss"])))
        self.assertEqual([config.training.epochs for config in seen], [1, 2, 2, 2])
        self.assertEqual(raw, before)

    def test_baseline_factory_and_sample_batch_accept_dictionary(self):
        raw = configuration()
        before = deepcopy(raw)
        trainer, batches, validation, _ = build_baseline_experiment(raw)
        self.assertEqual(trainer.train_epoch(batches)["samples"], 8)
        self.assertEqual(trainer.evaluate(validation)["samples"], 4)
        inputs, targets = baseline_sample_batch(raw, 5)
        self.assertEqual(inputs.shape, (5, 3))
        self.assertEqual(targets.shape, (5,))
        self.assertEqual(raw, before)

    def test_selected_toml_file_controls_factories_and_managed_run(self):
        template = '''name = "{name}"
seed = 19
[training]
epochs = 1
batch_size = {batch_size}
learning_rate = 0.03
optimizer = "sgd"
[application.synthetic]
train_samples = 8
validation_samples = 4
input_dim = 3
num_classes = 2
'''
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory, "first.toml")
            second = Path(directory, "second.toml")
            first.write_text(template.format(name="first-choice", batch_size=2))
            second.write_text(template.format(name="second-choice", batch_size=4))
            for source, expected in ((first, 2), (str(second), 4)):
                with self.subTest(source=source):
                    trainer, loader, _, _ = recipe(source)
                    self.assertEqual(loader.batch_size, expected)
                    trainer.train_epoch(loader)
                    self.assertEqual(trainer.state.optimizer_steps, 8 // expected)
            run_dir = train(second, output=Path(directory, "selected-run"),
                            batch_size_mode="manual", progress_bar=False)
            saved = torch.load(run_dir / "last.pt", weights_only=True)
            self.assertEqual(saved["config"]["name"], "second-choice")
            self.assertEqual(saved["config"]["training"]["batch_size"], 4)
            self.assertEqual(saved["state"]["optimizer_steps"], 2)

    def test_invalid_dictionary_is_rejected_before_factories_or_output_creation(self):
        raw = configuration()
        raw["runtime"]["unknown_option"] = True
        model_factory, dataset_factory = Mock(), Mock()
        with self.assertRaises(ConfigError):
            build_experiment(raw, model_factory=model_factory, dataset_factory=dataset_factory)
        model_factory.assert_not_called()
        dataset_factory.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "invalid")
            with self.assertRaises(ConfigError):
                train(raw, output=output, progress_bar=False)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
