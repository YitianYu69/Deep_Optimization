"""Copied installed-library ReLU benchmark and its CPU numerical reports."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import deepopt


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/activation/benchmark_relu.py"


class ReLUBenchmarkTests(unittest.TestCase):
    def run_copied(self, *args):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / SCRIPT.name
            shutil.copyfile(SCRIPT, script)
            environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
            result = subprocess.run([sys.executable, str(script), *args], cwd=directory, env=environment,
                                    capture_output=True, text=True, timeout=60)
            output = Path(directory) / "relu.json"
            return result, json.loads(output.read_text()) if output.exists() else None

    def test_help_does_not_require_cuda_or_checkout_imports(self):
        result, _ = self.run_copied("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-scaling", result.stdout)
        self.assertNotIn("sys.path.insert", SCRIPT.read_text())

    def test_installed_cpu_all_precision_pairs_masks_and_diagnostics(self):
        result, report = self.run_copied("--device", "cpu", "--shape", "3x33", "--iterations", "2",
                                         "--warmup", "1", "--repeats", "2", "--output", "relu.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(report["deepopt"]["path"]), Path(deepopt.__file__).resolve())
        self.assertEqual(len(report["results"]), 8)
        for row in report["results"]:
            self.assertTrue(row["gradient_exact"])
            self.assertEqual(row["saved_mask_bytes"], 16 if row["compression"] else 99)
            self.assertEqual(len(row["timings"]["wall_ms"]), 2)
            self.assertGreater(row["timings"]["wall_ms_median"], 0)

    def test_invalid_options_fail_before_importing_gpu_code(self):
        for args, message in ((["--iterations", "0"], "must be positive"),
                              (["--fp8-block", "4x4x4"], "two dimensions"),
                              (["--layout", "channels", "--shape", "33"], "at least two dimensions")):
            with self.subTest(args=args):
                result, _ = self.run_copied(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
