"""Copied installed-library pooling benchmark smoke and contract tests."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/pooling/benchmark_pooling.py"


class PoolingBenchmarkTests(unittest.TestCase):
    def run_script(self, *options, script=SCRIPT, cwd=None):
        return subprocess.run([sys.executable, str(script), *options], cwd=cwd,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, capture_output=True, text=True, timeout=60)

    def test_copied_benchmark_all_precision_compression_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / SCRIPT.name
            shutil.copyfile(SCRIPT, script)
            result = self.run_script("--device", "cpu", "--shape", "2x3x6x6", "--return-indices",
                                     "--warmup", "0", "--iterations", "1", "--repeats", "1", script=script, cwd=directory)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertNotIn(str(SCRIPT.parents[2] / "src"), report["deepopt"]["path"])
        rows = [r for r in report["results"] if r["implementation"] == "deepopt"]
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertGreater(row["timing"]["eager"]["wall_ms_median"], 0)
            self.assertGreaterEqual(row["eager_phase_profile"]["forward_cpu_ms"], 0)
            self.assertGreaterEqual(row["eager_phase_profile"]["backward_and_flush_cpu_ms"], 0)
            if row["compression"]:
                self.assertEqual(row["execution"]["compression_strategy"], "pack_routing")
                self.assertEqual(row["errors"]["output_max_abs_vs_compression_off"], 0)
                self.assertTrue(row["errors"]["indices_equal_vs_compression_off"])

    def test_zero_state_power_fractional_and_unpool_reports(self):
        for op in ("avg_pool2d", "adaptive_avg_pool2d", "lp_pool2d", "fractional_max_pool2d", "max_unpool2d"):
            result = self.run_script("--device", "cpu", "--shape", "2x3x6x6", "--op", op,
                "--inputs", "bf16", "--outputs", "bf16", "--warmup", "0", "--iterations", "1", "--repeats", "1")
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(result.stdout)["results"][-1]
            self.assertTrue(row["compression"])
            self.assertEqual(row["execution"]["saved_raw_input_bytes"], 0)
            if "avg_pool" in op:
                self.assertEqual(row["execution"]["saved_state_bytes"], 0)
            self.assertEqual(row["errors"]["output_max_abs_vs_compression_off"], 0)

    def test_pipeline_delayed_and_graph_constraints(self):
        result = self.run_script("--device", "cpu", "--shape", "2x3x6x6", "--op", "max_pool2d",
            "--pipeline", "conv", "--scaling-recipe", "delayed", "--inputs", "bf16",
            "--warmup", "0", "--iterations", "1", "--repeats", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [r for r in json.loads(result.stdout)["results"] if r["implementation"] == "deepopt"]
        self.assertTrue(all(not r["precision_plan"]["fallbacks"] for r in rows))
        self.assertTrue(any(e["precision"] == "fp8" for e in rows[-1]["precision_plan"]["backward_edges"]))
        for options, message in ((("--device", "cpu", "--execution", "graph"), "requires a CUDA"),
                                 (("--execution", "graph", "--scaling-recipe", "delayed"), "requires eager")):
            result = self.run_script(*options)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
