"""Copied installed-library pooling benchmark smoke and contract tests."""

from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/pooling/benchmark_pooling.py"
SPEC = importlib.util.spec_from_file_location("pooling_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class PoolingBenchmarkTests(unittest.TestCase):
    def run_script(self, *options, script=SCRIPT, cwd=None):
        return subprocess.run([sys.executable, str(script), *options], cwd=cwd,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, capture_output=True, text=True, timeout=60)

    def test_parser_backend_defaults_and_help_without_site_packages(self):
        from deepopt.opt.nn.pooling.spec import SPECS
        args = benchmark.parser().parse_args([])
        self.assertEqual(args.deepopt_backend, "triton")
        self.assertEqual(args.deepopt_kernel_policy, {})
        self.assertEqual(tuple(benchmark.OPERATIONS), tuple(SPECS))
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--deepopt-backend {triton,tilelang,mixed}", result.stdout)
        self.assertIn("--deepopt-kernel-policy", result.stdout)

    def test_invalid_backend_policy_fails_before_cuda_setup(self):
        import torch
        cases = (
            ["--deepopt-kernel-policy", "not-json"],
            ["--deepopt-kernel-policy", "null"],
            ["--deepopt-kernel-policy", "[]"],
            ["--deepopt-backend", "tilelang", "--deepopt-kernel-policy", '{"backend":"triton"}'],
            ["--deepopt-backend", "mixed", "--deepopt-kernel-policy", '{"roles":{"unknown":"tilelang"}}'],
        )
        for options in cases:
            with self.subTest(options=options), patch.object(sys, "argv", [str(SCRIPT), *options]), \
                    redirect_stderr(io.StringIO()), patch.object(torch.cuda, "set_device", side_effect=AssertionError("CUDA setup")) as cuda, \
                    patch.object(benchmark, "run", side_effect=AssertionError("measurement")) as run, self.assertRaises(SystemExit):
                benchmark.main()
            cuda.assert_not_called()
            run.assert_not_called()

    def test_cpu_modes_propagate_to_preparation_autocast_and_report(self):
        import deepopt
        import deepopt.opt.policy as policy_module
        importlib.import_module("deepopt.opt.nn.functional")
        from deepopt.kernel_policy import current_kernel_policy

        for backend in ("triton", "tilelang", "mixed"):
            with self.subTest(backend=backend):
                options = ["--device", "cpu", "--shape", "1x2x4x4", "--op", "max_pool2d",
                    "--inputs", "bf16", "--outputs", "bf16", "--warmup", "0", "--iterations", "1", "--repeats", "1",
                    "--deepopt-backend", backend]
                if backend == "mixed":
                    options += ["--deepopt-kernel-policy", '{"preferred":"tilelang","families":{"pooling":"triton"}}']
                original = policy_module.prepare_fp8_model
                observed = []

                def prepare(*args, **kwargs):
                    observed.append(current_kernel_policy().backend)
                    return original(*args, **kwargs)

                output = io.StringIO()
                with patch.object(sys, "argv", [str(SCRIPT), *options]), redirect_stdout(output), \
                        patch.dict(sys.modules, {"triton": None, "tilelang": None}), patch.object(benchmark.gc, "collect"), \
                        patch.object(policy_module, "prepare_fp8_model", side_effect=prepare) as prepared, \
                        patch.object(deepopt, "autocast", wraps=deepopt.autocast) as autocast:
                    benchmark.main()
                report = json.loads(output.getvalue())
                self.assertEqual(report["kernel_backend"]["policy"]["backend"], backend)
                self.assertEqual(report["kernel_backend"]["execution"], "cpu_reference")
                self.assertEqual(report["kernel_backend"]["packages"], {})
                self.assertEqual(observed, [backend, backend])
                self.assertTrue(all(call.args[1].backend == "reference" for call in prepared.call_args_list))
                self.assertTrue(all(call.kwargs["config"]["runtime"]["kernels"]["backend"] == backend
                    for call in prepared.call_args_list))
                self.assertTrue(autocast.call_args_list)
                self.assertTrue(all(call.kwargs["config"]["runtime"]["kernels"]["backend"] == backend
                    for call in autocast.call_args_list))
                self.assertEqual(len(report["results"]), 3)
                self.assertTrue(report["results"][-1]["output_finite"])
                self.assertEqual(current_kernel_policy().backend, "triton")

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
