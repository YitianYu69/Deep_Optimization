"""Portable installed-library boolean benchmark and real CPU report coverage."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import deepopt


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/activation_compression/benchmark_boolean_mask.py"


class BooleanMaskBenchmarkTests(unittest.TestCase):
    def run_copied(self, *args):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / SCRIPT.name
            shutil.copyfile(SCRIPT, script)
            environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
            result = subprocess.run([sys.executable, str(script), *args], cwd=directory, env=environment,
                                    capture_output=True, text=True, timeout=60)
            path = Path(directory) / "result.json"
            return result, json.loads(path.read_text()) if path.exists() else None

    def test_help_works_outside_checkout_and_has_no_source_path_injection(self):
        result, _ = self.run_copied("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--layouts", result.stdout)
        self.assertNotIn("sys.path.insert", SCRIPT.read_text())

    def test_real_cpu_measurements_include_exactness_payload_and_all_operations(self):
        result, report = self.run_copied("--device", "cpu", "--shape", "3x33", "--iterations", "2",
                                         "--warmup", "1", "--repeats", "2", "--output", "result.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(report["deepopt"]["path"]), Path(deepopt.__file__).resolve())
        self.assertEqual(len(report["results"]), 4)
        for row in report["results"]:
            self.assertTrue(row["exact_roundtrip"])
            self.assertEqual((row["logical_bool_bytes"], row["packed_bytes"]), (99, 16))
            self.assertEqual(set(row["timings"]), {"pack", "unpack", "roundtrip", "clone_bool"})
            for measurement in row["timings"].values():
                self.assertEqual(len(measurement["wall_ms"]), 2)
                self.assertGreater(measurement["wall_ms_median"], 0)
                self.assertIsNone(measurement["cuda_stream_ms_median"])

    def test_invalid_counts_shapes_and_layout_fail_before_measurement(self):
        for args, message in ((["--iterations", "0"], "must be positive"),
                              (["--warmup", "-1"], "warmup may be zero"),
                              (["--shape", "0x32"], "positive dimensions"),
                              (["--shape", "33"], "at least two dimensions")):
            with self.subTest(args=args):
                result, _ = self.run_copied(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
