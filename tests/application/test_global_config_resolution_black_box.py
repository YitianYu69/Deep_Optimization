"""Configuration precedence, snapshot stability, and final merged validation."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from deepopt import get_config, load_config, reset_config
from deepopt._config_resolution import (
    ResolvedConfig, _UNSET, config_scope, resolve_config, resolve_section, resolve_training_config,
)
from deepopt.config import ConfigError, TrainingConfig, config_from_dict


def settings(name="global"):
    return {"name": name, "seed": 7,
            "training": {"epochs": 3, "batch_size": 4, "learning_rate": .01},
            "runtime": {"fp8": {"enabled": True}},
            "application": {"labels": ["a", "b"], "nested": {"left": 1, "right": 2}}}


class GlobalConfigResolutionTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_no_source_preserves_empty_legacy_configuration(self):
        resolved = resolve_config()
        self.assertIsInstance(resolved, ResolvedConfig)
        self.assertEqual(resolved, {})
        with self.assertRaisesRegex(ConfigError, "complete experiment configuration"):
            resolve_config(require_complete=True)

    def test_partial_inputs_do_not_expose_scaffold_required_values(self):
        resolved = resolve_config({"training": {"batch_size": 8}, "runtime": {"checks": False}})
        self.assertNotIn("name", resolved)
        self.assertNotIn("seed", resolved)
        self.assertNotIn("epochs", resolved["training"])
        self.assertNotIn("learning_rate", resolved["training"])
        self.assertEqual(resolved["training"]["batch_size"], 8)
        self.assertEqual(resolved["training"]["optimizer"], "adamw")
        self.assertFalse(resolved["runtime"]["checks"])
        with self.assertRaisesRegex(ConfigError, "training.epochs"):
            resolve_config(resolved, require_complete=True)

    def test_partial_mapping_merges_recursively_and_lists_replace(self):
        load_config(settings())
        resolved = resolve_config({"training": {"batch_size": 8}, "application": {
            "labels": ["only"], "nested": {"left": 9}}})
        self.assertEqual(resolved["training"]["learning_rate"], .01)
        self.assertEqual(resolved["training"]["batch_size"], 8)
        self.assertEqual(resolved["application"], {"labels": ["only"], "nested": {"left": 9, "right": 2}})
        self.assertEqual(get_config().training.batch_size, 4)

    def test_explicit_false_none_and_omission_have_distinct_meanings(self):
        raw = settings()
        raw["training"].update({"grad_clip_norm": 1., "monitor": "loss"})
        raw["runtime"].update({"checks": True, "compile": {"enabled": True}})
        load_config(raw)
        resolved = resolve_config(overrides={
            "training": {"grad_clip_norm": None, "monitor": None, "batch_size": _UNSET},
            "runtime": {"checks": False, "compile": {"enabled": False}, "fp8": {"enabled": False}},
        })
        self.assertIsNone(resolved["training"]["grad_clip_norm"])
        self.assertIsNone(resolved["training"]["monitor"])
        self.assertEqual(resolved["training"]["batch_size"], 4)
        self.assertFalse(resolved["runtime"]["checks"])
        self.assertFalse(resolved["runtime"]["compile"]["enabled"])
        self.assertFalse(resolved["runtime"]["fp8"]["enabled"])

    def test_resolved_snapshot_survives_global_reload_and_reset(self):
        load_config(settings())
        snapshot = resolve_config()
        load_config({**settings("next"), "runtime": {"fp8": {"enabled": False}}})
        self.assertEqual(resolve_config(snapshot), snapshot)
        reset_config()
        self.assertEqual(resolve_config(snapshot), snapshot)
        second = resolve_config(snapshot)
        second["application"]["labels"].append("local")
        self.assertEqual(snapshot["application"]["labels"], ["a", "b"])

    def test_explicit_base_does_not_consult_process_global(self):
        load_config(settings())
        independent = resolve_config({"runtime": {"checks": False}}, base=None)
        self.assertNotIn("fp8", independent["runtime"])
        snapshot = resolve_config({"training": {"batch_size": 6}})
        load_config({**settings("new"), "training": {"epochs": 8, "batch_size": 99, "learning_rate": .2}})
        merged = resolve_config({"training": {"epochs": 5}}, base=snapshot)
        self.assertEqual(merged["training"]["epochs"], 5)
        self.assertEqual(merged["training"]["batch_size"], 6)
        self.assertEqual(merged["name"], "global")

    def test_partial_runtime_setting_does_not_invent_unrequested_execution_policies(self):
        partial = resolve_config({"runtime": {"device": "cpu"}})
        for policy in ("fp8", "compile", "ddp", "cuda_graph"):
            self.assertNotIn(policy, partial["runtime"])
        for section in ("loader", "logging", "schedules", "application", "experiment_identity"):
            self.assertNotIn(section, partial)
        self.assertEqual(resolve_config(partial), partial)
        training_only = resolve_config({"training": {"learning_rate": .1}})
        self.assertNotIn("runtime", training_only)
        complete = resolve_config(config_from_dict(settings()))
        self.assertTrue(complete["runtime"]["fp8"]["enabled"])
        self.assertFalse(complete["runtime"]["compile"]["enabled"])

    def test_complete_typed_file_and_preset_sources_bypass_global(self):
        load_config(settings())
        typed = config_from_dict({**settings("typed"), "runtime": {"fp8": {"enabled": False}}})
        self.assertEqual(resolve_config(typed)["name"], "typed")
        self.assertFalse(resolve_config(typed)["runtime"]["fp8"]["enabled"])
        self.assertFalse(resolve_config("builtin:baseline")["runtime"]["fp8"]["enabled"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "own.toml")
            path.write_text('name="file"\nseed=1\n[training]\nepochs=1\nbatch_size=2\nlearning_rate=0.1\n')
            self.assertEqual(resolve_config(path)["name"], "file")
            self.assertFalse(resolve_config(path)["runtime"]["fp8"]["enabled"])
        self.assertEqual(get_config().name, "global")

    def test_factory_scopes_are_nested_and_do_not_change_real_global(self):
        load_config(settings())
        outer = resolve_config(config_from_dict(settings("outer")))
        inner = resolve_config(config_from_dict(settings("inner")))
        with config_scope(outer):
            self.assertEqual(resolve_config()["name"], "outer")
            self.assertEqual(get_config().name, "global")
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with config_scope(inner):
                    self.assertEqual(resolve_config()["name"], "inner")
                    raise RuntimeError("stop")
            self.assertEqual(resolve_config()["name"], "outer")
            self.assertEqual(resolve_config({"training": {"batch_size": 12}})["name"], "outer")
        self.assertEqual(resolve_config()["name"], "global")
        with config_scope(ResolvedConfig()):
            self.assertEqual(resolve_config(), {})

    def test_final_conflicts_require_explicit_resolution(self):
        load_config({**settings(), "runtime": {"execution_mode": "analyze"}})
        with self.assertRaisesRegex(ConfigError, "Choose runtime.execution_mode"):
            resolve_config({"runtime": {"diagnostics": {"mode": "balanced"}}})
        resolved = resolve_config({"runtime": {"diagnostics": {"mode": "balanced"}}},
                                  overrides={"runtime": {"execution_mode": None}})
        self.assertIsNone(resolved["runtime"]["execution_mode"])
        load_config({**settings(), "training": {**settings()["training"], "lr_gamma": .5}})
        with self.assertRaisesRegex(ConfigError, "Choose training.lr_schedule"):
            resolve_config({"training": {"lr_schedule": {"type": "constant"}}})

    def test_resolve_training_section_preserves_typed_identity_and_required_fields(self):
        typed = TrainingConfig(1, 2, .1)
        self.assertIs(resolve_training_config(typed), typed)
        with self.assertRaisesRegex(ConfigError, "missing fields"):
            resolve_training_config({"learning_rate": .1})
        load_config(settings())
        merged = resolve_training_config({"batch_size": 8})
        self.assertEqual((merged.epochs, merged.batch_size, merged.learning_rate), (3, 8, .01))

    def test_owner_defaults_do_not_leak_into_snapshot_or_legacy_section_contract(self):
        snapshot = resolve_config({"training": {"learning_rate": .03}})
        local = resolve_training_config(snapshot, defaults={"epochs": 1, "batch_size": 1})
        self.assertEqual(local.learning_rate, .03)
        self.assertNotIn("epochs", snapshot["training"])
        self.assertNotIn("batch_size", snapshot["training"])
        with self.assertRaisesRegex(ConfigError, "missing fields"):
            resolve_training_config({"learning_rate": .03}, defaults={"epochs": 1, "batch_size": 1})
        with self.assertRaisesRegex(ConfigError, "learning_rate"):
            resolve_training_config(None, defaults={"epochs": 1, "batch_size": 1})

    def test_framework_sections_validate_without_native_objects_or_cuda(self):
        raw = {**settings(), "runtime": {"ddp": {"broadcast_buffers": False, "bucket_cap_mb": 12},
                                          "cuda_graph": {"device": "cuda:1", "warmup": 2, "autograd": False}},
               "loader": {"num_workers": 2, "prefetch_factor": 3, "persistent_workers": True},
               "logging": {"level": "warning", "file_level": "DEBUG"}}
        resolved = resolve_config(config_from_dict(raw))
        self.assertEqual(resolved["runtime"]["ddp"]["bucket_cap_mb"], 12.)
        self.assertEqual(resolved["runtime"]["cuda_graph"]["device"], "cuda:1")
        self.assertEqual(resolved["logging"]["level"], "WARNING")
        self.assertEqual(json.loads(json.dumps(resolved)), resolved)
        with self.assertRaisesRegex(ConfigError, "requires num_workers"):
            resolve_config(resolved, overrides={"loader": {"num_workers": 0}})

    def test_section_extraction_returns_owned_data_and_honors_absent_default(self):
        self.assertEqual(resolve_section(None, "runtime.ddp", default={}), {})
        load_config(settings())
        fp8 = resolve_section(None, "runtime.fp8")
        fp8["block_shapes"]["weight"] = [32, 32]
        self.assertIsNone(get_config().runtime.fp8.block_shapes["weight"])
        self.assertTrue(fp8["enabled"])

    def test_empty_new_sections_preserve_legacy_configuration_defaults(self):
        config = config_from_dict(settings())
        self.assertEqual(config.runtime.ddp, {})
        self.assertEqual(config.runtime.cuda_graph, {})
        self.assertEqual(config.loader, {})
        self.assertEqual(config.logging, {})
        self.assertEqual(config_from_dict(asdict(config)), config)

    def test_new_section_schemas_reject_unsupported_or_inconsistent_values(self):
        invalid = (
            {"runtime": {"ddp": {"process_group": "native object"}}},
            {"runtime": {"ddp": {"static_graph": 1}}},
            {"runtime": {"ddp": {"bucket_cap_mb": 0}}},
            {"runtime": {"cuda_graph": {"device": "cpu"}}},
            {"runtime": {"cuda_graph": {"warmup": 0}}},
            {"runtime": {"cuda_graph": {"capture_error_mode": "unknown"}}},
            {"loader": {"num_workers": True}},
            {"loader": {"persistent_workers": True}},
            {"loader": {"prefetch_factor": 2}},
            {"logging": {"level": "unknown"}},
            {"logging": {"file_level": False}},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ConfigError):
                resolve_config(override)


if __name__ == "__main__":
    unittest.main()
