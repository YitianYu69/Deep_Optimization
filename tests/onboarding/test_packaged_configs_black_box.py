"""Build real distributions and load installed presets away from the checkout."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PRESETS = ("baseline", "ddp", "fp8", "scheduling")


class PackagedConfigBlackBoxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        staging = cls.directory / "source"
        staging.mkdir()
        # Build metadata and egg-info belong to the temporary copy, never the
        # developer's checkout. Build-system dependencies come from this Python.
        for name in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE"):
            shutil.copy2(ROOT / name, staging / name)
        for name in ("src", "configs", "tests", "docs", "examples", "benchmarks"):
            shutil.copytree(ROOT / name, staging / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".DS_Store"))
        distributions = cls.directory / "distributions"
        distributions.mkdir()
        script = """
import json
import sys
from setuptools.build_meta import build_sdist, build_wheel

# Setuptools hooks replace sys.argv while running their setup commands.
output_directory = sys.argv[1]
wheel = build_wheel(output_directory)
sdist = build_sdist(output_directory)
print(json.dumps({"wheel": wheel, "sdist": sdist}))
"""
        result = subprocess.run([sys.executable, "-I", "-c", script, str(distributions)], cwd=staging,
                                capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)
        built = json.loads(result.stdout.splitlines()[-1])
        cls.wheel = distributions / built["wheel"]
        cls.sdist = distributions / built["sdist"]
        for artifact in (cls.wheel, cls.sdist):
            if not artifact.is_file():
                contents = sorted(path.name for path in distributions.iterdir())
                raise AssertionError(
                    f"Build backend reported a missing artifact: {artifact.name}; "
                    f"output directory contains {contents}\n{result.stdout}\n{result.stderr}"
                )

    def test_wheel_and_sdist_contain_schema_and_all_preset_resources(self):
        with zipfile.ZipFile(self.wheel) as wheel:
            self.assertIn("deepopt/config.py", wheel.namelist())
            self.assertIn("deepopt/configs/__init__.py", wheel.namelist())
            for name in PRESETS:
                with self.subTest(archive="wheel", preset=name):
                    member = f"deepopt/configs/{name}.toml"
                    self.assertIn(member, wheel.namelist())
                    self.assertEqual(wheel.read(member), (ROOT / "configs" / (name + ".toml")).read_bytes())
        with tarfile.open(self.sdist) as sdist:
            names = sdist.getnames()
            self.assertTrue(any(name.endswith("/src/deepopt/config.py") for name in names))
            for preset in PRESETS:
                with self.subTest(archive="sdist", preset=preset):
                    members = [name for name in names if name.endswith(f"/src/deepopt/configs/{preset}.toml")]
                    self.assertEqual(len(members), 1)
                    with sdist.extractfile(members[0]) as stream:
                        self.assertEqual(stream.read(), (ROOT / "configs" / (preset + ".toml")).read_bytes())

    def test_installed_wheel_loads_presets_and_builtin_cli_without_checkout_or_torch(self):
        target = self.directory / "installed"
        outside = self.directory / "installed-working-directory"
        outside.mkdir()
        installed = subprocess.run([
            sys.executable, "-I", "-m", "pip", "install", "--no-index", "--no-deps",
            "--disable-pip-version-check", "--target", str(target), str(self.wheel),
        ], cwd=outside, capture_output=True, text=True, timeout=90)
        self.assertEqual(installed.returncode, 0, installed.stdout + installed.stderr)
        script = """
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sys

target, checkout = (Path(value).resolve() for value in sys.argv[1:3])
# -I ignores PYTHONPATH and cwd; remove any editable .pth entry as well.
sys.path = [str(target), *[entry for entry in sys.path
                        if entry and not Path(entry).resolve().is_relative_to(checkout)]]
import deepopt
import deepopt.configs
from deepopt.config import ExperimentConfig, load_config
from deepopt.configs import list_presets, load_preset

assert Path(deepopt.__file__).resolve().is_relative_to(target)
assert Path(deepopt.configs.__file__).resolve().is_relative_to(target)
names = list_presets()
assert names == ("baseline", "ddp", "fp8", "scheduling")
for name in names:
    assert isinstance(load_preset(name), ExperimentConfig)
    assert isinstance(load_preset(name, as_dict=True), dict)
assert load_config("builtin:fp8").runtime.fp8.enabled
assert load_preset("fp8.toml").runtime.fp8.enabled
assert "torch" not in sys.modules

captured = io.StringIO()
sys.argv = ["deepopt", "inspect-config", "--config", "builtin:baseline"]
with redirect_stdout(captured):
    try:
        runpy.run_module("deepopt", run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
inspected = json.loads(captured.getvalue())
assert inspected["name"] == "baseline"
selected_files = []
for filename, name, batch_size in (("baseline.toml", "user-baseline", 3),
                                   ("chosen-settings.toml", "user-custom", 5)):
    path = Path(filename)
    path.write_text(
        f'name = "{name}"\\nseed = 21\\n[training]\\nepochs = 1\\n'
        f'batch_size = {batch_size}\\nlearning_rate = 0.01\\n', encoding="utf-8")
    absolute = deepopt.load_config(path.resolve())
    relative = deepopt.load_config(filename)
    assert absolute == relative
    assert relative.name == name
    assert relative.training.batch_size == batch_size
    captured_file = io.StringIO()
    sys.argv = ["deepopt", "inspect-config", "--config", filename]
    with redirect_stdout(captured_file):
        try:
            runpy.run_module("deepopt", run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 0
    selected = json.loads(captured_file.getvalue())
    assert selected["name"] == name
    assert selected["training"]["batch_size"] == batch_size
    selected_files.append({"file": filename, "name": selected["name"],
                           "batch_size": selected["training"]["batch_size"]})
# A local baseline.toml never changes explicit built-in selection.
assert deepopt.load_config("builtin:baseline").name == "baseline"
assert load_preset("baseline").name == "baseline"
assert "torch" not in sys.modules
print(json.dumps({"presets": names, "cli_name": inspected["name"],
                  "file_configs": selected_files, "torch_loaded": "torch" in sys.modules}))
"""
        result = subprocess.run([sys.executable, "-I", "-c", script, str(target), str(ROOT)], cwd=outside,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout), {
            "presets": list(PRESETS), "cli_name": "baseline", "torch_loaded": False,
            "file_configs": [
                {"file": "baseline.toml", "name": "user-baseline", "batch_size": 3},
                {"file": "chosen-settings.toml", "name": "user-custom", "batch_size": 5},
            ],
        })

    def test_wheel_zip_import_reads_presets_through_traversable_resources(self):
        outside = self.directory / "zip-working-directory"
        outside.mkdir()
        script = """
import json
from pathlib import Path
import sys
import zipimport

wheel, checkout = (Path(value).resolve() for value in sys.argv[1:3])
sys.path = [str(wheel), *[entry for entry in sys.path
                       if entry and not Path(entry).resolve().is_relative_to(checkout)]]
import deepopt
from deepopt.configs import list_presets, load_preset

assert isinstance(deepopt.__loader__, zipimport.zipimporter)
assert Path(deepopt.__loader__.archive).resolve() == wheel
loaded = {name: load_preset(name, as_dict=True)["name"] for name in list_presets()}
assert loaded["baseline"] == "baseline"
assert loaded["fp8"] == "fp8-reference"
assert "torch" not in sys.modules
print(json.dumps(loaded))
"""
        result = subprocess.run([sys.executable, "-I", "-c", script, str(self.wheel), str(ROOT)], cwd=outside,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(set(json.loads(result.stdout)), set(PRESETS))


if __name__ == "__main__":
    unittest.main()
