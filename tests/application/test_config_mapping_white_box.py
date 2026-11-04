"""Configuration boundaries keep typed identity and own validated mapping data."""

from collections import UserDict
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

from deepopt.config import (
    ConfigError, ExperimentConfig, TrainingConfig, compile_config_from_dict,
    config_from_dict, fp8_config_from_dict, load_config, normalize_config,
    normalize_training_config, training_config_from_dict,
)


def training_table():
    return {"epochs": 2, "batch_size": 4, "learning_rate": .01}


def experiment_table():
    return {"name": "mapping", "seed": 7, "training": training_table()}


class ConfigMappingWhiteBoxTests(unittest.TestCase):
    def test_typed_normalizers_preserve_existing_object_identity(self):
        training = TrainingConfig(2, 4, .01)
        config = ExperimentConfig("mapping", 7, training)
        self.assertIs(normalize_training_config(training), training)
        self.assertIs(normalize_config(config), config)
        self.assertIs(load_config(config), config)
        self.assertIs(config.training, training)

    def test_training_table_parses_without_inventing_experiment_fields(self):
        actual = training_config_from_dict(training_table())
        self.assertEqual(actual, TrainingConfig(epochs=2, batch_size=4, learning_rate=.01))
        self.assertEqual(normalize_training_config(UserDict(training_table())), actual)
        self.assertEqual(config_from_dict(experiment_table()).training, actual)

    def test_nested_mapping_tables_use_the_same_schema(self):
        raw = MappingProxyType({
            "name": "mapping", "seed": 7,
            "training": UserDict({**training_table(), "monitor": False,
                                  "lr_schedule": MappingProxyType({"type": "piecewise", "milestones": [2, 4]})}),
            "runtime": MappingProxyType({
                "mode_schedule": UserDict({"2": "analyze"}),
                "compile": UserDict({"options": MappingProxyType({"some_flag": True})}),
                "fp8": MappingProxyType({"block_shapes": UserDict({"weight": [32, 64]})}),
            }),
            "application": UserDict({"data": MappingProxyType({"splits": ["train", "valid"]})}),
            "schedules": MappingProxyType({"aux": MappingProxyType({
                "type": "sequential", "durations": [2, 3],
                "schedules": [MappingProxyType({"type": "constant", "value": .1}),
                              UserDict({"type": "constant", "value": .2})],
            })}),
        })
        config = normalize_config(raw)
        self.assertIsNone(config.training.monitor)
        self.assertEqual(config.training.lr_schedule["milestones"], [2, 4])
        self.assertEqual(config.runtime.fp8.block_shapes["weight"], (32, 64))
        self.assertEqual(config.runtime.mode_schedule, {"2": "analyze"})
        self.assertEqual(config.runtime.compile.options, {"some_flag": True})
        self.assertEqual(config.application, {"data": {"splits": ["train", "valid"]}})
        self.assertEqual(config.schedules["aux"]["durations"], [2, 3])
        self.assertEqual(config_from_dict(asdict(config)), config)

    def test_mapping_normalization_copies_mutable_data_without_changing_input(self):
        raw = experiment_table()
        raw.update({
            "application": {"data": {"splits": ["train", "valid"]}},
            "runtime": {"mode_schedule": {"2": "analyze"},
                        "fp8": {"block_shapes": {"weight": [32, 64]}}},
            "schedules": {"aux": {"type": "piecewise", "milestones": [2], "values": [.1, .2]}},
        })
        original = deepcopy(raw)
        config = load_config(raw)
        self.assertEqual(raw, original)
        raw["application"]["data"]["splits"].append("test")
        raw["runtime"]["mode_schedule"]["2"] = "balanced"
        raw["runtime"]["fp8"]["block_shapes"]["weight"][0] = 64
        raw["schedules"]["aux"]["values"][0] = .9
        self.assertEqual(config.application["data"]["splits"], ["train", "valid"])
        self.assertEqual(config.runtime.mode_schedule, {"2": "analyze"})
        self.assertEqual(config.runtime.fp8.block_shapes["weight"], (32, 64))
        self.assertEqual(config.schedules["aux"]["values"], [.1, .2])

    def test_training_lr_lists_are_owned_by_the_parsed_config(self):
        raw = {**training_table(), "lr_schedule": {"type": "piecewise", "milestones": [2, 5]}}
        config = normalize_training_config(raw)
        raw["lr_schedule"]["milestones"].append(8)
        self.assertEqual(config.lr_schedule["milestones"], [2, 5])

    def test_default_mutable_tables_are_independent_between_normalizations(self):
        first = normalize_config(experiment_table())
        second = normalize_config(experiment_table())
        first.runtime.compile.options["flag"] = True
        first.runtime.fp8.block_shapes["grad"] = (32, 32)
        first.application["data"] = []
        self.assertEqual(second.runtime.compile.options, {})
        self.assertIsNone(second.runtime.fp8.block_shapes["grad"])
        self.assertEqual(second.application, {})

    def test_training_parser_preserves_existing_validation_rules(self):
        for values in ({"epochs": 0}, {"batch_size": True}, {"learning_rate": float("inf")},
                       {"optimizer": "unknown"}, {"monitor": "samples"},
                       {"scheduler_interval": "batch"}, {"unknown": 1},
                       {"lr_gamma": .5, "lr_schedule": {"type": "constant"}}):
            with self.subTest(values=values), self.assertRaises(ConfigError):
                normalize_training_config({**training_table(), **values})
        for missing in training_table():
            raw = training_table()
            raw.pop(missing)
            with self.subTest(missing=missing), self.assertRaisesRegex(ConfigError, "missing fields"):
                training_config_from_dict(raw)

    def test_wrong_boundary_types_and_nonstring_schema_keys_raise_config_errors(self):
        for normalizer in (normalize_config, normalize_training_config):
            for value in (None, True, [], "baseline"):
                with self.subTest(normalizer=normalizer.__name__, value=value), self.assertRaises(ConfigError):
                    normalizer(value)
        with self.assertRaisesRegex(ConfigError, "keys must be strings"):
            normalize_training_config({**training_table(), 1: "not a field name"})

    def test_compile_and_fp8_parsers_accept_mapping_views(self):
        compiler = compile_config_from_dict(MappingProxyType({
            "enabled": True, "options": UserDict({"flag": True}),
        }))
        fp8 = fp8_config_from_dict(UserDict({
            "enabled": True, "block_shapes": MappingProxyType({"activation": [32, 64]}),
        }))
        self.assertEqual(compiler.options, {"flag": True})
        self.assertEqual(fp8.block_shapes["activation"], (32, 64))

    def test_filesystem_loading_retains_the_existing_path_keyword(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "config.toml")
            path.write_text('name="mapping"\nseed=7\n[training]\nepochs=2\nbatch_size=4\nlearning_rate=0.01\n')
            self.assertEqual(load_config(path=path), normalize_config(experiment_table()))

    def test_only_explicit_builtin_prefix_uses_the_package_loader(self):
        expected = normalize_config(experiment_table())
        with patch("deepopt.configs.load_preset", return_value=expected) as preset:
            self.assertIs(load_config("builtin:baseline"), expected)
            preset.assert_called_once_with("baseline")
            preset.reset_mock()
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(FileNotFoundError):
                    load_config(Path(directory, "baseline.toml"))
            preset.assert_not_called()

    def test_all_mapping_entry_points_remain_torch_free(self):
        result = subprocess.run([sys.executable, "-c", """
import sys
from collections import UserDict
from deepopt.config import load_config, normalize_config, normalize_training_config
training = UserDict(epochs=2, batch_size=4, learning_rate=.01)
config = UserDict(name='mapping', seed=7, training=training)
assert load_config(config) == normalize_config(config)
assert normalize_training_config(training).batch_size == 4
assert 'torch' not in sys.modules
assert 'triton' not in sys.modules
"""], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
