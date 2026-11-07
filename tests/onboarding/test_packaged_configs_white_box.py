"""Preset resource ownership, validation, and synchronization with checkout examples."""

from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from deepopt.config import ConfigError, ExperimentConfig, config_from_dict
from deepopt.configs import list_presets, load_preset


ROOT = Path(__file__).resolve().parents[2]


class PackagedConfigWhiteBoxTests(unittest.TestCase):
    def run_sync(self, root, *arguments):
        return subprocess.run(
            [sys.executable, str(root / "scripts/sync_configs.py"), *arguments],
            cwd=root.parent, capture_output=True, text=True, timeout=10,
        )

    def make_sync_checkout(self, directory):
        root = Path(directory) / "checkout"
        (root / "scripts").mkdir(parents=True)
        shutil.copyfile(ROOT / "scripts/sync_configs.py", root / "scripts/sync_configs.py")
        source = root / "src/deepopt/configs"
        source.mkdir(parents=True)
        return root, source

    def test_sync_check_detects_drift_without_changing_checkout_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source = self.make_sync_checkout(directory)
            (source / "baseline.toml").write_bytes(b"name = 'updated'\r\n")
            (source / "templates").mkdir()
            (source / "templates/complete.toml").write_bytes(b"# complete\n")
            destination = root / "configs"
            destination.mkdir()
            stale = destination / "baseline.toml"
            stale.write_bytes(b"name = 'old'\n")
            before = stale.stat().st_mtime_ns

            result = self.run_sync(root, "--check")

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("out of sync: configs/baseline.toml", result.stderr)
            self.assertIn("out of sync: configs/templates/complete.toml", result.stderr)
            self.assertEqual(stale.read_bytes(), b"name = 'old'\n")
            self.assertEqual(stale.stat().st_mtime_ns, before)
            self.assertFalse((destination / "templates").exists())

            synced = self.run_sync(root)
            self.assertEqual(synced.returncode, 0, synced.stdout + synced.stderr)
            for relative in ("baseline.toml", "templates/complete.toml"):
                self.assertEqual((source / relative).read_bytes(), (destination / relative).read_bytes())
            self.assertEqual(self.run_sync(root, "--check").returncode, 0)
            before = stale.stat().st_mtime_ns
            self.assertEqual(self.run_sync(root).returncode, 0)
            self.assertEqual(stale.stat().st_mtime_ns, before)

    def test_sync_reports_extra_checkout_configs_without_deleting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source = self.make_sync_checkout(directory)
            (source / "baseline.toml").write_bytes(b"# baseline\n")
            (root / "configs").mkdir()
            extra = root / "configs/personal.toml"
            extra.write_bytes(b"# personal experiment\n")
            for arguments in (("--check",), ()):
                with self.subTest(arguments=arguments):
                    result = self.run_sync(root, *arguments)
                    self.assertEqual(result.returncode, 1)
                    self.assertIn("no packaged source: configs/personal.toml", result.stderr)
                    self.assertEqual(extra.read_bytes(), b"# personal experiment\n")

    def test_complete_template_is_packaged_separately_from_named_presets(self):
        resource = files("deepopt.configs").joinpath("templates", "complete.toml")
        self.assertEqual(resource.read_bytes(), (ROOT / "configs/templates/complete.toml").read_bytes())
        self.assertNotIn("complete", list_presets())

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
