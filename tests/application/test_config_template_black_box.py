"""Public configuration templates are editable, nonactivating, and safe to write."""

from dataclasses import asdict
from importlib.resources import files
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from deepopt import load_config_template, write_config_template
from deepopt.config import ExperimentConfig, config_from_dict, get_config, load_config, reset_config


ROOT = Path(__file__).resolve().parents[2]


class ConfigTemplateBlackBoxTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_template_is_valid_and_only_explicit_loading_activates_edited_values(self):
        template = load_config_template()
        self.assertIs(type(template), dict)
        self.assertIsInstance(config_from_dict(template), ExperimentConfig)
        self.assertIsNone(get_config())
        template["name"] = "edited-user-recipe"
        template["training"]["batch_size"] = 7
        template["training"]["learning_rate"] = 0.031
        parsed = config_from_dict(template)
        self.assertEqual(parsed.training.batch_size, 7)
        self.assertIsNone(get_config())
        loaded = load_config(template)
        self.assertEqual(loaded, parsed)
        self.assertEqual(get_config().name, "edited-user-recipe")
        self.assertEqual(get_config().training.learning_rate, 0.031)

    def test_written_template_matches_resource_and_normalizes_like_the_dictionary(self):
        installed = load_config_template()
        installed["name"] = "already-installed-sentinel"
        previous = load_config(installed)
        template = load_config_template()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "my-training.toml"
            written = write_config_template(path)
            self.assertIsInstance(written, Path)
            self.assertEqual(written, path)
            self.assertEqual(path.read_bytes(),
                             files("deepopt.configs").joinpath("templates", "complete.toml").read_bytes())
            self.assertEqual(get_config(), previous)
            from_file = load_config(path)
        from_dict = config_from_dict(template)
        left, right = asdict(from_file), asdict(from_dict)
        # TOML has no null value. Dataclass defaults normalize its optional
        # scalar fields; these two dictionary fields retain explicit None.
        for config in (left, right):
            graph = config["runtime"]["cuda_graph"]
            if graph.get("device") is None:
                graph.pop("device", None)
            if config["loader"].get("prefetch_factor") is None:
                config["loader"].pop("prefetch_factor", None)
        self.assertEqual(left, right)

    def test_writer_never_overwrites_an_existing_user_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recipe.toml"
            write_config_template(str(path))
            original = b'# user-edited configuration\nname = "keep-this"\n'
            path.write_bytes(original)
            with self.assertRaises(FileExistsError):
                write_config_template(path)
            self.assertEqual(path.read_bytes(), original)
        self.assertIsNone(get_config())

    def test_template_apis_are_torch_free_and_do_not_replace_active_defaults(self):
        script = """
import json
from pathlib import Path
import sys
from deepopt import load_config_template, write_config_template
from deepopt.config import get_config, load_config

assert "torch" not in sys.modules
assert get_config() is None
first = load_config_template()
assert get_config() is None
first["name"] = "already-installed-sentinel"
existing = load_config(first)
second = load_config_template()
assert second["name"] != "already-installed-sentinel"
assert get_config() == existing
written = write_config_template(Path(sys.argv[1]))
assert written.is_file()
assert get_config() == existing
assert "torch" not in sys.modules
print(json.dumps({"active": get_config().name, "torch_loaded": "torch" in sys.modules}))
"""
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new" / "template.toml"
            result = subprocess.run([sys.executable, "-c", script, str(target)],
                                    cwd=directory, env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout), {
            "active": "already-installed-sentinel", "torch_loaded": False,
        })


if __name__ == "__main__":
    unittest.main()
