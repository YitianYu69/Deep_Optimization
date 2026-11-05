"""Atomic process-local configuration installation and independent snapshots."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Barrier
import unittest

from deepopt import get_config, load_config, reset_config
from deepopt.config import ConfigError, _read_config, config_from_dict
from deepopt.configs import load_preset


def settings(name="first"):
    return {"name": name, "seed": 7,
            "training": {"epochs": 2, "batch_size": 4, "learning_rate": .01},
            "application": {"owner": name, "labels": ["train", "valid"]}}


class GlobalConfigLifecycleTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_unset_and_repeated_reset_are_safe(self):
        self.assertIsNone(get_config())
        reset_config()
        reset_config()
        self.assertIsNone(get_config())

    def test_loaded_return_and_get_results_do_not_alias_global_or_input(self):
        raw = settings()
        loaded = load_config(raw)
        raw["application"]["labels"].append("input mutation")
        loaded.application["labels"].append("return mutation")
        first = get_config()
        self.assertEqual(first.application["labels"], ["train", "valid"])
        first.application["labels"].append("snapshot mutation")
        self.assertEqual(get_config().application["labels"], ["train", "valid"])
        self.assertIsNot(get_config(), get_config())

    def test_new_load_replaces_previous_defaults_instead_of_merging_them(self):
        load_config({**settings("first"), "runtime": {"fp8": {"enabled": True}}})
        load_config({"name": "second", "seed": 9,
                     "training": {"epochs": 1, "batch_size": 2, "learning_rate": .1}})
        current = get_config()
        self.assertEqual(current.name, "second")
        self.assertFalse(current.runtime.fp8.enabled)
        self.assertEqual(current.application, {})
        reset_config()
        self.assertIsNone(get_config())

    def test_failed_file_mapping_and_typed_load_keep_previous_config(self):
        previous = load_config(settings())
        malformed = {**settings("invalid"), "runtime": {"device": "cpu", "precision": "bf16"}}
        with self.assertRaises(ConfigError):
            load_config(malformed)
        self.assertEqual(get_config(), previous)
        with self.assertRaises(ConfigError):
            load_config(replace(previous, seed=-1))
        self.assertEqual(get_config(), previous)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "invalid.toml")
            path.write_text("[training")
            with self.assertRaises(ValueError):
                load_config(path)
        self.assertEqual(get_config(), previous)

    def test_pure_parsers_and_preset_loaders_do_not_install_defaults(self):
        typed = config_from_dict(settings())
        self.assertIs(_read_config(typed), typed)
        self.assertEqual(_read_config(settings()), typed)
        self.assertTrue(load_preset("fp8").runtime.fp8.enabled)
        self.assertIsNone(get_config())
        load_config(settings("active"))
        _read_config("builtin:baseline")
        self.assertEqual(get_config().name, "active")

    def test_typed_install_is_validated_and_copied(self):
        typed = config_from_dict(settings())
        loaded = load_config(typed)
        self.assertEqual(loaded, typed)
        self.assertIsNot(loaded, typed)
        typed.application["labels"].append("new")
        self.assertEqual(get_config().application["labels"], ["train", "valid"])

    def test_concurrent_readers_only_observe_complete_configurations(self):
        load_config(settings("first"))
        start = Barrier(3)

        def write(name):
            start.wait()
            for _ in range(25):
                load_config(settings(name))

        def read():
            start.wait()
            for _ in range(100):
                snapshot = get_config()
                self.assertEqual(snapshot.name, snapshot.application["owner"])
                self.assertEqual(snapshot.training.batch_size, 4)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(write, "first"), executor.submit(write, "second"),
                       executor.submit(read)]
            for future in futures:
                future.result(timeout=30)

    def test_lifecycle_is_torch_free_and_child_process_has_its_own_defaults(self):
        load_config(settings("parent"))
        script = """
import json, sys
from deepopt import get_config, load_config, reset_config
assert get_config() is None
loaded = load_config({'name':'child','seed':1,
                     'training':{'epochs':1,'batch_size':2,'learning_rate':.1}})
assert get_config().name == 'child'
assert 'torch' not in sys.modules and 'triton' not in sys.modules
reset_config()
print(json.dumps({'name': loaded.name, 'reset': get_config() is None}))
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"name": "child", "reset": True})
        self.assertEqual(get_config().name, "parent")


if __name__ == "__main__":
    unittest.main()
