"""Installed-library gradient benchmark portability and numerical report contracts."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import deepopt


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/fp8/benchmark_backward_precision.py"


class BackwardPrecisionBenchmarkTests(unittest.TestCase):
    def run_copied(self, *args):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / SCRIPT.name
            shutil.copyfile(SCRIPT, script)
            environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
            result = subprocess.run([sys.executable, str(script), *args], cwd=directory, env=environment,
                                    capture_output=True, text=True, timeout=60)
            output = Path(directory) / "gradient.json"
            return result, json.loads(output.read_text()) if output.exists() else None

    def test_help_is_portable(self):
        result, _ = self.run_copied("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--grad-block", result.stdout)
        self.assertNotIn("sys.path.insert", SCRIPT.read_text())

    def test_real_cpu_linear_and_spatial_pipelines_report_edges_and_gradient_errors(self):
        for pipeline in ("linear", "conv2d"):
            result, report = self.run_copied("--device", "cpu", "--pipeline", pipeline, "--batch-size", "2",
                                             "--width", "8", "--sequence-length", "4", "--spatial", "3",
                                             "--warmup", "1", "--iterations", "2", "--repeats", "1", "--output", "gradient.json")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(report["deepopt"]["path"]), Path(deepopt.__file__).resolve())
            native, auto = report["results"]
            self.assertEqual(native["mode"], "native")
            self.assertEqual(auto["precision_plan"]["analyses"], 1)
            self.assertTrue(any(edge["precision"] == "fp8" for edge in auto["precision_plan"]["backward_edges"]))
            self.assertEqual(auto["errors_vs_native_gradient_transport"]["output"]["max_abs"], 0)
            self.assertIn("dx", auto["errors_vs_native_gradient_transport"])

    def test_invalid_options_fail_before_cuda_imports(self):
        for args, message in ((["--width", "3"], "divisible by two"), (["--iterations", "0"], "must be positive"),
                              (["--grad-block", "1x32", "--scaling-recipe", "delayed"], "tensorwise")):
            with self.subTest(args=args):
                result, _ = self.run_copied(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
