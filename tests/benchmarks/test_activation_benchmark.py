"""Portable installed-library measurements and pre-device validation."""

import importlib.util
from contextlib import redirect_stdout, redirect_stderr
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
        self.assertEqual((args.bits, args.group_size), (2, 1024))
        self.assertEqual(args.deepopt_backend, "triton")
        self.assertEqual(args.deepopt_kernel_policy, {})

    def test_backend_help_is_available_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--help"],
            capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--deepopt-backend {triton,tilelang,mixed}", result.stdout)
        self.assertIn("--deepopt-kernel-policy", result.stdout)

    def test_invalid_backend_policies_fail_before_cuda_setup(self):
        import torch
        cases = (
            ["--deepopt-kernel-policy", "not-json"],
            ["--deepopt-kernel-policy", "null"],
            ["--deepopt-kernel-policy", "[]"],
            ["--deepopt-backend", "tilelang", "--deepopt-kernel-policy", '{"backend":"triton"}'],
            ["--deepopt-backend", "mixed", "--deepopt-kernel-policy", '{"families":{"unknown":"tilelang"}}'],
        )
        for options in cases:
            with self.subTest(options=options), patch.object(sys, "argv", [str(SCRIPT), *options]), \
                    redirect_stderr(io.StringIO()), patch.object(torch.cuda, "set_device", side_effect=AssertionError("CUDA setup")) as cuda, \
                    patch.object(benchmark, "run", side_effect=AssertionError("measurement")) as run, self.assertRaises(SystemExit):
                benchmark.main()
            cuda.assert_not_called()
            run.assert_not_called()

    def test_cpu_modes_propagate_policy_to_plans_autocast_and_measurement(self):
        import deepopt
        import deepopt.opt.policy as policy_module
        importlib.import_module("deepopt.opt.nn.functional")
        from deepopt.kernel_policy import current_kernel_policy

        for backend in ("triton", "tilelang", "mixed"):
            with self.subTest(backend=backend):
                options = ["--device", "cpu", "--shape", "3x8", "--activations", "silu",
                    "--inputs", "bf16", "--outputs", "bf16", "--compression", "off",
                    "--warmup", "0", "--iterations", "1", "--repeats", "1", "--deepopt-backend", backend]
                if backend == "mixed":
                    options += ["--deepopt-kernel-policy", '{"preferred":"tilelang","families":{"activation":"triton"}}']
                observed = []

                def measure(step, *args):
                    observed.append(current_kernel_policy().backend)
                    step()
                    return {"eager": {"wall_ms_median": 1.0}}

                output = io.StringIO()
                with patch.object(sys, "argv", [str(SCRIPT), *options]), redirect_stdout(output), \
                        patch.dict(sys.modules, {"triton": None, "tilelang": None}), \
                        patch.object(benchmark, "measurements", side_effect=measure), \
                        patch.object(policy_module, "activation_plan", wraps=policy_module.activation_plan) as plans, \
                        patch.object(deepopt, "autocast", wraps=deepopt.autocast) as autocast:
                    benchmark.main()
                report = json.loads(output.getvalue())
                self.assertEqual(report["kernel_backend"]["policy"]["backend"], backend)
                self.assertEqual(report["kernel_backend"]["execution"], "cpu_reference")
                self.assertEqual(report["kernel_backend"]["packages"], {})
                self.assertTrue(observed)
                self.assertTrue(all(value == backend for value in observed))
                self.assertEqual(plans.call_args.kwargs["fp8"]["backend"], "reference")
                self.assertEqual(plans.call_args.kwargs["config"]["runtime"]["kernels"]["backend"], backend)
                self.assertTrue(autocast.call_args_list)
                self.assertTrue(all(call.kwargs["config"]["runtime"]["kernels"]["backend"] == backend
                    for call in autocast.call_args_list))
                self.assertEqual(current_kernel_policy().backend, "triton")

    def test_cpu_pipeline_preparation_receives_selected_backend_policy(self):
        import deepopt
        import deepopt.opt.policy as policy_module
        importlib.import_module("deepopt.opt.nn.functional")
        options = ["--device", "cpu", "--pipeline", "linear", "--shape", "3x8", "--activations", "silu",
            "--compression", "off", "--warmup", "0", "--iterations", "1", "--repeats", "1", "--deepopt-backend", "tilelang"]

        def measure(step, *args):
            step()
            return {"eager": {"wall_ms_median": 1.0}}

        with patch.object(sys, "argv", [str(SCRIPT), *options]), redirect_stdout(io.StringIO()), \
                patch.dict(sys.modules, {"triton": None, "tilelang": None}), \
                patch.object(benchmark, "measurements", side_effect=measure), \
                patch.object(policy_module, "prepare_fp8_model", wraps=policy_module.prepare_fp8_model) as prepared, \
                patch.object(deepopt, "autocast", wraps=deepopt.autocast) as autocast:
            benchmark.main()
        self.assertEqual(prepared.call_args.args[1].backend, "reference")
        self.assertEqual(prepared.call_args.kwargs["config"]["runtime"]["kernels"]["backend"], "tilelang")
        self.assertTrue(all(call.kwargs["config"]["runtime"]["kernels"]["backend"] == "tilelang"
            for call in autocast.call_args_list))

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
