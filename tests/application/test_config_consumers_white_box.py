"""Factory normalization keeps typed callback contracts and validation boundaries."""

from collections import UserDict
from dataclasses import asdict
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from deepopt.config import CompileConfig, ConfigError, TrainingConfig
from deepopt.data.synthetic import SyntheticDataSettings, build_datasets
from deepopt.engine.execution.compilation import compile_model
from deepopt.optimizer_factory import build_optimizer
from deepopt.scheduling.lr import build_scheduler


class ConfigConsumerWhiteBoxTests(unittest.TestCase):
    def test_custom_optimizer_callback_gets_typed_settings_and_original_parameters(self):
        typed = TrainingConfig(epochs=2, batch_size=3, learning_rate=.02, optimizer="custom")
        for settings in (typed, asdict(typed), UserDict(asdict(typed))):
            with self.subTest(settings_type=type(settings).__name__):
                model = nn.Linear(4, 2)
                calls = []

                def apply(parameters, config):
                    calls.append(config)
                    self.assertIsInstance(config, TrainingConfig)
                    self.assertEqual(config, typed)
                    parameters = tuple(parameters)
                    self.assertTrue(all(actual is original for actual, original in zip(parameters, model.parameters())))
                    return torch.optim.SGD(parameters, lr=config.learning_rate)

                optimizer = build_optimizer(model, settings, apply_func=apply)
                self.assertEqual(len(calls), 1)
                self.assertEqual(optimizer.param_groups[0]["lr"], typed.learning_rate)
                if settings is typed:
                    self.assertIs(calls[0], typed)

    def test_invalid_training_dict_is_rejected_before_optimizer_callback(self):
        valid = {"epochs": 2, "batch_size": 3, "learning_rate": .02}
        invalid = ({"learning_rate": .02}, {**valid, "batch_size": 0},
                   {**valid, "unknown": True}, {**valid, "learning_rate": float("nan")})
        model = nn.Linear(4, 2)
        for config in invalid:
            with self.subTest(config=config):
                callback = Mock()
                with self.assertRaises(ConfigError):
                    build_optimizer(model, config, apply_func=callback)
                callback.assert_not_called()

    def test_invalid_training_dict_does_not_construct_or_step_scheduler(self):
        optimizer = torch.optim.SGD(nn.Linear(4, 2).parameters(), lr=.02)
        with patch("deepopt.scheduling.lr.lr.ExponentialLR") as scheduler:
            with self.assertRaises(ConfigError):
                build_scheduler(optimizer, {"lr_gamma": .5})
        scheduler.assert_not_called()
        self.assertEqual(optimizer.param_groups[0]["lr"], .02)

    def test_mapping_scheduler_settings_use_training_defaults(self):
        optimizer = torch.optim.SGD(nn.Linear(4, 2).parameters(), lr=.02)
        raw = UserDict({"epochs": 2, "batch_size": 3, "learning_rate": .02, "lr_gamma": .5})
        scheduler = build_scheduler(optimizer, raw)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], .01)
        self.assertNotIn("optimizer", raw)

    def test_disabled_compile_dict_does_not_invoke_torch_compile(self):
        for settings in ({}, {"enabled": False}, UserDict({"enabled": False}), CompileConfig()):
            with self.subTest(settings=settings):
                model = nn.Linear(4, 2)
                with patch("torch.compile") as compiler:
                    self.assertIs(compile_model(model, settings), model)
                compiler.assert_not_called()
                self.assertIsNone(model._compiled_call_impl)

    def test_compile_dict_preserves_dispatch_options_and_parameter_identity(self):
        model = nn.Linear(4, 2)
        parameters = tuple(model.parameters())
        raw = UserDict({"enabled": True, "backend": "eager", "dynamic": False,
                        "fullgraph": True, "options": {"custom_option": 2}})
        with patch("torch.compile", return_value=Mock()) as compiler:
            self.assertIs(compile_model(model, raw), model)
        compiler.assert_called_once_with(model._call_impl, backend="eager", dynamic=False,
                                         fullgraph=True, options={"custom_option": 2})
        self.assertTrue(all(actual is original for actual, original in zip(model.parameters(), parameters)))
        self.assertEqual(raw["options"], {"custom_option": 2})

    def test_invalid_compile_dict_is_rejected_before_model_mutation(self):
        for settings in ({"enabled": 1}, {"enabled": True, "unknown": True},
                         {"enabled": True, "backend": ""}, {"options": {"bad": []}}):
            with self.subTest(settings=settings):
                model = nn.Linear(4, 2)
                with patch("torch.compile") as compiler, self.assertRaises(ConfigError):
                    compile_model(model, settings)
                compiler.assert_not_called()
                self.assertIsNone(model._compiled_call_impl)

    def test_synthetic_typed_settings_are_not_reparsed_and_dict_uses_existing_validation(self):
        typed = SyntheticDataSettings(train_samples=3, validation_samples=2, input_dim=2, num_classes=2)
        with patch.object(SyntheticDataSettings, "from_dict", wraps=SyntheticDataSettings.from_dict) as parse:
            build_datasets(typed, seed=7)
            parse.assert_not_called()
            raw = asdict(typed)
            build_datasets(raw, seed=7)
            parse.assert_called_once_with(raw)
        for raw in ({"train_samples": 0}, {"input_dim": True}, {"unknown": 1}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                build_datasets(raw, seed=7)


if __name__ == "__main__":
    unittest.main()
