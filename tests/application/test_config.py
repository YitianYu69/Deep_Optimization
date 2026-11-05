from dataclasses import FrozenInstanceError, asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from deepopt.config import ConfigError, config_from_dict, load_config, reset_config


VALID = '''
name = "test-run"
seed = 42

[training]
epochs = 5
batch_size = 32
learning_rate = 0.001
'''



class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_config()
        self.addCleanup(reset_config)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "config.toml"
        self.path.write_text(VALID, encoding="utf-8")

    def test_loads_named_settings(self):
        config = load_config(self.path)
        self.assertEqual(config.name, "test-run")
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.training.batch_size, 32)
        self.assertEqual(config.training.learning_rate, 0.001)
        self.assertEqual(config.training.monitor, "loss")
        self.assertEqual(config.training.monitor_mode, "min")

    def test_application_options_roundtrip_without_a_dataset_or_task_schema(self):
        raw = asdict(load_config(self.path))
        raw["application"] = {
            "dataset": {"source": "records.jsonl", "fields": ["prompt", "completion"],
                        "transforms": [{"name": "tokenize", "length": 128}]},
            "model": {"architecture": "caller-defined", "widths": [64, 32]},
            "task": {"objective": "sequence-generation", "temperature": 0.7},
            "optional": None,
        }
        config = config_from_dict(raw)
        self.assertEqual(asdict(config)["application"], raw["application"])
        self.assertEqual(config_from_dict(json.loads(json.dumps(asdict(config)))), config)
        self.assertNotIn("data", asdict(config))
        self.assertFalse(hasattr(config, "data"))

    def test_legacy_data_table_has_a_migration_error(self):
        raw = asdict(load_config(self.path))
        raw["data"] = {"train_samples": 256, "input_dim": 16}
        with self.assertRaisesRegex(ConfigError, "application"):
            config_from_dict(raw)

    def test_application_requires_a_table(self):
        raw = asdict(load_config(self.path))
        with self.assertRaisesRegex(ConfigError, "application.*table"):
            config_from_dict({**raw, "application": "records.jsonl"})

    def test_application_options_do_not_alias_the_input_mapping(self):
        raw = asdict(load_config(self.path))
        raw["application"] = {"dataset": {"splits": ["train", "validation"]}}
        config = config_from_dict(raw)
        raw["application"]["dataset"]["splits"].append("test")
        raw["application"]["dataset"]["source"] = "changed.jsonl"
        self.assertEqual(config.application, {"dataset": {"splits": ["train", "validation"]}})

    def test_application_rejects_values_that_cannot_be_saved_safely(self):
        raw = asdict(load_config(self.path))
        for value in (float("nan"), float("inf"), {1: "non-string key"}, {"a", "b"},
                      Path("records.jsonl"), object()):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ConfigError, "application.dataset"):
                    config_from_dict({**raw, "application": {"dataset": [value]}})

    def test_settings_cannot_be_accidentally_reassigned(self):
        config = load_config(self.path)
        with self.assertRaises(FrozenInstanceError):
            config.training.epochs = 99

    def test_rejects_invalid_values(self):
        replacements = [
            ('name = "test-run"', 'name = " "'),
            ("seed = 42", "seed = -1"),
            ("seed = 42", "seed = true"),
            ("epochs = 5", "epochs = 0"),
            ("batch_size = 32", "batch_size = 1.5"),
            ("learning_rate = 0.001", "learning_rate = -0.1"),
            ("learning_rate = 0.001", "learning_rate = nan"),
            ("learning_rate = 0.001", "learning_rate = inf"),
            ("learning_rate = 0.001", "learning_rate = true"),
        ]

        for old, new in replacements:
            with self.subTest(value=new):
                self.path.write_text(VALID.replace(old, new), encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_config(self.path)

    def test_rejects_unknown_field(self):
        self.path.write_text(VALID + "epoch = 10\n", encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "unknown fields: epoch"):
            load_config(self.path)

    def test_rejects_missing_field(self):
        self.path.write_text(VALID.replace("seed = 42\n", ""), encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "missing fields: seed"):
            load_config(self.path)

    def test_rejects_wrong_table_type(self):
        self.path.write_text('name = "test"\nseed = 0\ntraining = "wrong"\n')
        with self.assertRaisesRegex(ConfigError, "training must be a TOML table"):
            load_config(self.path)

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "deepopt", *args],
            capture_output=True, text=True,
        )

    def test_cli_emits_validated_json(self):
        result = self.run_cli("--config", str(self.path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["training"]["epochs"], 5)

    def test_help_does_not_require_config(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--config", result.stdout)

    def test_missing_file_has_readable_error(self):
        result = self.run_cli("--config", str(self.path.parent / "missing.toml"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_malformed_toml_has_readable_error(self):
        self.path.write_text("[training", encoding="utf-8")
        result = self.run_cli("--config", str(self.path))
        self.assertEqual(result.returncode, 2)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_shipped_baseline_is_valid(self):
        root = Path(__file__).resolve().parents[2]
        config = load_config(root / "configs" / "baseline.toml")
        self.assertEqual(config.name, "baseline")

    def test_rejects_invalid_extended_settings(self):
        config = asdict(load_config(self.path))
        for section, key, value in [
            ("training", "accumulation_steps", 0), ("training", "optimizer", "typo"),
            ("training", "checkpoint_every_steps", -1), ("training", "checkpoint_every_steps", True),
            ("training", "grad_clip_norm", float("inf")), ("training", "monitor_mode", "typo"),
            ("runtime", "device", "cuda:-1"), ("runtime", "precision", "fp16"),
            ("runtime", "deterministic", "true"),
        ]:
            with self.subTest(section=section, key=key):
                changed = {**config, section: {**config[section], key: value}}
                with self.assertRaises(ConfigError):
                    config_from_dict(changed)

    def test_cli_honors_explicit_argv(self):
        from contextlib import redirect_stdout
        import io
        from deepopt.cli import main
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["inspect-config", "--config", str(self.path)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["name"], "test-run")


if __name__ == "__main__":
    unittest.main()

