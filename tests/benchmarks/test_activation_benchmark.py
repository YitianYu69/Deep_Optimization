"""Portable installed-library measurements and pre-device validation."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/activation/benchmark_activations.py"
SPEC = importlib.util.spec_from_file_location("activation_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class ActivationBenchmarkTests(unittest.TestCase):
    def test_defaults_cover_every_variant_precision_pair_and_compression_policy(self):
        args = benchmark.parser().parse_args([])
        self.assertEqual(args.activations, ["sigmoid", "silu", "gelu", "gelu_tanh", "tanh", "relu6"])
        self.assertEqual(args.inputs, ["bf16", "fp8"])
        self.assertEqual(args.outputs, ["bf16", "fp8"])
        self.assertEqual(args.compression, ["off", "on"])
        self.assertEqual((args.bits, args.group_size), (2, 512))

    def test_copied_script_runs_against_installed_library_with_real_reports(self):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / SCRIPT.name
            shutil.copyfile(SCRIPT, script)
            result = subprocess.run([sys.executable, str(script), "--device", "cpu", "--shape", "3x33",
                                     "--warmup", "0", "--iterations", "1", "--repeats", "1"],
                                    cwd=directory, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertNotIn(str(SCRIPT.parents[2] / "src"), report["deepopt"]["path"])
            rows = [r for r in report["results"] if r["implementation"] == "deepopt"]
            self.assertEqual(len(rows), 6 * 2 * 2 * 2)
            self.assertTrue(all(r["timing"]["eager"]["wall_ms_median"] > 0 for r in rows))
            for r in rows:
                self.assertIn("saved_state_bytes", r["execution"])
                self.assertIn("dx_max_abs_vs_preprocessed_fp32", r["errors"])

    def test_linear_and_conv_pipeline_measurements_report_planned_fp8_edges(self):
        for pipeline, shape in (("linear", "3x8"), ("conv", "2x8x3x3")):
            result = subprocess.run([sys.executable, str(SCRIPT), "--device", "cpu", "--pipeline", pipeline,
                                     "--shape", shape, "--activations", "silu", "--warmup", "0",
                                     "--iterations", "1", "--repeats", "1"], capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            for row in json.loads(result.stdout)["results"]:
                if row["implementation"] == "native_bf16_pipeline":
                    continue
                self.assertFalse(row["precision"]["fallbacks"])
                self.assertTrue(any(e["precision"] == "fp8" for e in row["precision"]["backward_edges"]))

    def test_invalid_graph_and_shape_options_fail_without_cuda_initialization(self):
        for options, text in ((["--device", "cpu", "--execution", "graph"], "requires a CUDA"),
                              (["--pipeline", "conv", "--shape", "3x8"], "3, 4 or 5"),
                              (["--iterations", "0"], "must be positive")):
            result = subprocess.run([sys.executable, str(SCRIPT), *options], capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(text, result.stderr)


if __name__ == "__main__":
    unittest.main()
