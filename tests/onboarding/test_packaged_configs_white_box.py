"""Preset resource ownership, validation, and synchronization with checkout examples."""

from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
import unittest
from unittest.mock import patch

from deepopt.config import ConfigError, ExperimentConfig, config_from_dict
from deepopt.configs import list_presets, load_preset


ROOT = Path(__file__).resolve().parents[2]


class PackagedConfigWhiteBoxTests(unittest.TestCase):
    def test_checkout_examples_and_package_resources_are_byte_identical(self):
        names = list_presets()
        self.assertEqual(names, ("baseline", "ddp", "fp8", "scheduling"))
        self.assertEqual(set(names), {path.stem for path in (ROOT / "configs").glob("*.toml")})
        for name in names:
            with self.subTest(preset=name):
                resource = files("deepopt.configs").joinpath(name + ".toml")
                self.assertEqual(resource.read_bytes(), (ROOT / "configs" / (name + ".toml")).read_bytes())

    def test_all_presets_are_validated_and_mapping_results_are_normalized_copies(self):
        for name in list_presets():
            with self.subTest(preset=name):
                typed = load_preset(name)
                plain = load_preset(name + ".toml", as_dict=True)
                self.assertIsInstance(typed, ExperimentConfig)
                self.assertIs(type(plain), dict)
                self.assertEqual(plain, asdict(typed))
                self.assertEqual(config_from_dict(plain), typed)
        self.assertEqual(load_preset(), load_preset("baseline"))
        changed = load_preset("fp8", as_dict=True)
        changed["runtime"]["fp8"]["block_shapes"]["activation"] = [32, 32]
        changed["application"]["synthetic"]["train_samples"] = 999
        fresh = load_preset("fp8", as_dict=True)
        self.assertIsNone(fresh["runtime"]["fp8"]["block_shapes"]["activation"])
        self.assertEqual(fresh["application"]["synthetic"]["train_samples"], 32)

    def test_invalid_names_are_rejected_before_resolving_resources(self):
        for name in (None, 1, "", ".", "..", "../baseline", "/baseline",
                     "baseline/../fp8", "..\\baseline", "builtin:fp8", "baseline.toml.toml"):
            with self.subTest(name=name), patch("deepopt.configs.files") as resolve:
                with self.assertRaises(ConfigError):
                    load_preset(name)
                resolve.assert_not_called()

    def test_unknown_simple_name_reports_available_presets(self):
        with self.assertRaisesRegex(ConfigError, "Unknown built-in preset.*available presets: baseline, ddp, fp8, scheduling"):
            load_preset("missing")


if __name__ == "__main__":
    unittest.main()
