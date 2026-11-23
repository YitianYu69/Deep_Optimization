"""Public CLI plans, fail-closed receipts and fresh-process package imports."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.compiler import benchmark_vision_compression as runner


ROOT = Path(__file__).resolve().parents[2]


class VisionRunnerTests(unittest.TestCase):
    def args(self, *extra):
        return runner.parser().parse_args(["--output", "/unused/vision", *extra])

    def test_original_28_cells_have_exact_models_profiles_and_scope(self):
        plan = runner.make_plan(self.args())
        self.assertEqual(plan["unique_cells"], 28)
        self.assertEqual(len({row["id"] for row in plan["cells"]}), 28)
        self.assertEqual({name: row["expected_parameter_count"] for name, row in plan["models"].items()},
            {"resnet18": 11689512, "efficientnet-b0": 5288548, "vit-base": 86567656})
        self.assertFalse(plan["cuda_executed"])
        self.assertEqual(plan["runtime_request"]["input_shape"], [128, 3, 224, 224])
        self.assertTrue(all(row["compression"]["group_size"] == 1024 for row in plan["cells"]))
        self.assertTrue(all("compiler_backend" not in row for row in plan["models"].values()))

    def test_changed_geometry_counts_and_source_target_mixing_refuse(self):
        for args in (("--image-size", "32"), ("--group-size", "512"), ("--warmup", "1"),
                     ("--batches", "1"), ("--updates", "1"), ("--source-runtime", "--qualify")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                runner.make_plan(self.args(*args))

    def test_missing_data_dependencies_and_identity_fail_before_model_allocation(self):
        with patch.object(runner.importlib.util, "find_spec", return_value=None):
            result = runner.preflight(self.args())
        self.assertFalse(result["passed"])
        self.assertFalse(result["model_allocated"])
        self.assertEqual(len(result["blockers"]), 3)
        with patch.object(runner, "_preflight", side_effect=RuntimeError("Install the current wheel")):
            result = runner.preflight(self.args())
        self.assertFalse(result["passed"])
        self.assertIn("Install the current wheel", result["blockers"][0])

    def test_clean_export_dry_run_uses_only_stdlib_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "benchmarks", root / "benchmarks", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            output = root / "no-output"
            env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY")}
            result = subprocess.run([sys.executable, "-B", "-S", "-m", "benchmarks.compiler.benchmark_vision_compression",
                "--dry-run", "--output", str(output)], cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["unique_cells"], 28)
            self.assertFalse(output.exists())
            blocked = subprocess.run([sys.executable, "-B", "-S", "-m", "benchmarks.compiler.benchmark_vision_compression",
                "--preflight", "--qualify", "--device", "cpu", "--source-runtime", "--output", str(output)],
                cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(blocked.returncode, 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertFalse(manifest["preflight"]["passed"])
            self.assertEqual(manifest["cells"], [])

    def test_worker_failure_continues_independent_cells_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            command = ["--models", "resnet18", "--bf16-policies", "bf16_storage", "--profiles", "off", "full-two-bit",
                "--executions", "eager", "--device", "cpu", "--qualify", "--source-runtime", "--output", str(root)]
            with patch.object(runner, "preflight", return_value=dict(passed=True)), \
                    patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)):
                self.assertEqual(runner.main(command), 1)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(len(manifest["cells"]), 2)
            self.assertFalse(manifest["qualified"])
            self.assertFalse(manifest["cuda_verified"])
            self.assertTrue(all(not row["passed"] for row in manifest["cells"]))

    def test_cuda_preflight_failure_blocks_all_model_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            with patch.object(runner, "preflight", return_value=dict(passed=True, runtime={})), \
                    patch.object(runner, "launch_stage", return_value=dict(passed=False, error="H200 unavailable")) as launch:
                self.assertEqual(runner.main(["--preflight", "--qualify", "--output", str(root)]), 1)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(launch.call_args.args[2], "preflight")
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertFalse(manifest["hardware_preflight"]["passed"])
            self.assertEqual(manifest["cells"], [])

    @unittest.skipUnless(importlib.util.find_spec("torchvision") is not None, "torchvision unavailable; actual full-model CLI pending")
    def test_actual_cpu_eager_cell_uses_the_public_fresh_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT)))}
            env.pop("DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY", None)
            result = subprocess.run([sys.executable, "-m", "benchmarks.compiler.benchmark_vision_compression",
                "--models", "resnet18", "--bf16-policies", "bf16_storage", "--executions", "eager",
                "--profiles", "full-two-bit", "--device", "cpu", "--qualify", "--source-runtime", "--workers", "0",
                "--output", str(output)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(result.returncode, 0, (result.stderr, manifest))
            self.assertTrue(manifest["qualified"])
            self.assertFalse(manifest["cuda_verified"])
            cell, = manifest["cells"]
            self.assertEqual(cell["stages"][0]["qualification"]["execution"], "eager")
            self.assertEqual(len(cell["stages"][0]["qualification"]["updates"]), 3)
            self.assertEqual(list(output.rglob("*.pt")), [])


if __name__ == "__main__":
    unittest.main()
