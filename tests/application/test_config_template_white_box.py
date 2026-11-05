"""Template completeness follows the public schema and keeps optional features inactive."""

from dataclasses import fields
from importlib.resources import files
import tomllib
import unittest

from deepopt import load_config_template
from deepopt.config import (
    CompileConfig, ExperimentConfig, FP8Config, RuntimeConfig, TrainingConfig,
    config_from_dict, reset_config,
)


class ConfigTemplateWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_dictionary_covers_every_dataclass_field_and_documented_option_table(self):
        template = load_config_template()
        for table, schema in (
            (template, ExperimentConfig), (template["training"], TrainingConfig),
            (template["runtime"], RuntimeConfig),
            (template["runtime"]["compile"], CompileConfig),
            (template["runtime"]["fp8"], FP8Config),
        ):
            with self.subTest(schema=schema.__name__):
                self.assertEqual(set(table), {field.name for field in fields(schema)})
        self.assertEqual(set(template["runtime"]["ddp"]), {
            "find_unused_parameters", "broadcast_buffers", "gradient_as_bucket_view",
            "bucket_cap_mb", "static_graph", "validation_sync_buffers",
        })
        self.assertEqual(set(template["runtime"]["cuda_graph"]), {
            "device", "warmup", "allow_unused_input", "autograd", "capture_error_mode",
        })
        self.assertEqual(set(template["loader"]), {
            "num_workers", "pin_memory", "persistent_workers", "prefetch_factor", "drop_last",
        })
        self.assertEqual(set(template["logging"]), {"level", "file_level"})
        self.assertEqual(set(template["runtime"]["fp8"]["block_shapes"]), {"activation", "weight", "grad"})
        config_from_dict(template)

    def test_nested_mutable_defaults_are_independent_between_calls(self):
        first, second = load_config_template(), load_config_template()
        self.assertIsNot(first["runtime"]["compile"]["options"], second["runtime"]["compile"]["options"])
        self.assertIsNot(first["runtime"]["fp8"]["block_shapes"], second["runtime"]["fp8"]["block_shapes"])
        first["runtime"]["compile"]["options"]["example.option"] = True
        first["runtime"]["fp8"]["block_shapes"]["activation"] = [32, 32]
        first["loader"]["num_workers"] = 2
        first["application"]["user_metadata"] = {"versions": ["changed"]}
        self.assertNotIn("example.option", second["runtime"]["compile"]["options"])
        self.assertIsNone(second["runtime"]["fp8"]["block_shapes"]["activation"])
        self.assertEqual(second["loader"]["num_workers"], 0)
        self.assertNotIn("user_metadata", second["application"])
        self.assertEqual(load_config_template(), second)

    def test_optional_features_remain_inactive_and_have_commented_toml_examples(self):
        template = load_config_template()
        self.assertIsNone(template["training"]["lr_schedule"])
        self.assertIsNone(template["training"]["grad_clip_norm"])
        self.assertIsNone(template["runtime"]["diagnostics"])
        self.assertIsNone(template["runtime"]["mode_schedule"])
        self.assertIsNone(template["runtime"]["compile"]["mode"])
        self.assertIsNone(template["runtime"]["compile"]["dynamic"])
        self.assertIsNone(template["runtime"]["fp8"]["out_dtype"])
        self.assertFalse(template["runtime"]["compile"]["enabled"])
        self.assertFalse(template["runtime"]["fp8"]["enabled"])
        resource = files("deepopt.configs").joinpath("templates", "complete.toml")
        text = resource.read_text(encoding="utf-8")
        raw = tomllib.loads(text)
        self.assertNotIn("lr_schedule", raw["training"])
        self.assertNotIn("diagnostics", raw["runtime"])
        comments = "\n".join(line for line in text.splitlines() if line.lstrip().startswith("#"))
        for field in ("training.lr_schedule", "runtime.diagnostics", "runtime.mode_schedule",
                      "runtime.compile.options", "grad_clip_norm", "out_dtype", "prefetch_factor"):
            with self.subTest(field=field):
                self.assertIn(field, comments)


if __name__ == "__main__":
    unittest.main()
