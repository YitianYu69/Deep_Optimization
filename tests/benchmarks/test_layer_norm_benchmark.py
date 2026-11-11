"""LayerNorm benchmark portability, suffix validation and real CPU reports."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/normalization/benchmark_normalization.py"


class LayerNormBenchmarkTests(unittest.TestCase):
    def run_script(self, script, *options, cwd=None):
        return subprocess.run([sys.executable, str(script), *options], cwd=cwd,
                              env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                              capture_output=True, text=True, timeout=45)

    def test_copied_installed_library_benchmark_covers_precision_pairs_and_xhat(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / SCRIPT.name
            shutil.copyfile(SCRIPT, script)
            result = self.run_script(script, "--op", "layer_norm", "--device", "cpu", "--pipeline", "none",
                                     "--shape", "2x3x7", "--normalized-shape", "3x7", "--warmup", "0",
                                     "--iterations", "1", "--repeats", "1", cwd=directory)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertNotIn(str(SCRIPT.parents[2] / "src"), report["deepopt"]["path"])
        self.assertEqual(report["normalized_shape"], [3, 7])
        rows = [r for r in report["results"] if r["implementation"] == "deepopt"]
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertEqual(row["execution"]["compression_target"], "x_hat")
            self.assertGreater(row["timing"]["eager"]["wall_ms_median"], 0)
            if row["compression"]:
                self.assertEqual(row["errors"]["output_max_abs_vs_compression_off"], 0)

    def test_activation_linear_pipeline_and_delayed_ownership_reports(self):
        for recipe in ("current", "delayed"):
            result = self.run_script(SCRIPT, "--op", "layer_norm", "--device", "cpu", "--pipeline", "linear",
                                     "--activation", "gelu_tanh", "--shape", "3x8", "--scaling-recipe", recipe,
                                     "--warmup", "0", "--iterations", "1", "--repeats", "1", "--inputs", "bf16")
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [r for r in json.loads(result.stdout)["results"] if r["implementation"] == "deepopt"]
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertFalse(row["precision"]["fallbacks"])
                self.assertTrue(any(e["precision"] == "fp8" for e in row["precision"]["backward_edges"]))

    def test_recenter_off_on_measurements_include_state_and_mean_diagnostics(self):
        result = self.run_script(SCRIPT, "--op", "layer_norm", "--device", "cpu", "--pipeline", "none",
                                 "--shape", "3x33", "--group-size", "17", "--bits", "2",
                                 "--norm-xhat-recenter", "both", "--inputs", "bf16", "--outputs", "bf16",
                                 "--warmup", "0", "--iterations", "1", "--repeats", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [r for r in json.loads(result.stdout)["results"] if r["implementation"] == "deepopt"]
        self.assertEqual([(r["compression"], r["norm_xhat_recenter"]) for r in rows],
                         [(False, False), (True, False), (True, True)])
        off, on = rows[1:]
        self.assertEqual(on["execution"]["saved_state_bytes"] - off["execution"]["saved_state_bytes"], 12)
        self.assertEqual(on["execution"]["recenter_state_bytes"], 12)
        self.assertEqual(on["errors"]["output_max_abs_vs_compression_off"], 0)
        self.assertLess(on["reference_codec_diagnostics"]["mean_max_abs_error"], 1e-6)
        self.assertLess(on["reference_codec_diagnostics"]["mean_max_abs_error"],
                        off["reference_codec_diagnostics"]["mean_max_abs_error"])

    def test_suffix_graph_and_delayed_constraints_are_validated_before_cuda(self):
        for flags, message in ((["--shape", "2x3x7", "--normalized-shape", "21"], "exact input suffix"),
                               (["--device", "cpu", "--execution", "graph"], "requires a CUDA"),
                               (["--execution", "graph", "--scaling-recipe", "delayed"], "delayed scale updates"),
                               (["--scaling-recipe", "delayed", "--fp8-block", "1x32"], "tensorwise")):
            result = self.run_script(SCRIPT, "--op", "layer_norm", "--pipeline", "none", *flags)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
