"""The comparison benchmark must exercise compiler capture with the same policy."""

import json
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/compiler/benchmark_compiler.py"
SPEC = importlib.util.spec_from_file_location("compiler_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class CompilerBenchmarkTests(unittest.TestCase):
    def test_phase_total_excludes_the_already_measured_total_step(self):
        row = {"forward_loss_ms": 2., "backward_ms": 3., "optimizer_ms": 1., "step_ms": 7.}
        self.assertEqual(benchmark.phase_total(row), 6.)

    def test_invalid_kernel_policies_fail_before_cuda_availability_or_setup(self):
        cases = (
            ("tilelang", "null"), ("tilelang", "[]"), ("tilelang", "{"),
            ("tilelang", '{"backend":"triton"}'),
            ("mixed", '{"families":{"unknown":"tilelang"}}'),
            ("mixed", '{"roles":{"linear.dw":"unknown"}}'),
        )
        for backend, policy in cases:
            with self.subTest(backend=backend, policy=policy):
                error = io.StringIO()
                with patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA query")), \
                        redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                    benchmark.main(["--device", "cuda:1", "--deepopt-backend", backend,
                                    "--deepopt-kernel-policy", policy])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("error:", error.getvalue())
                self.assertNotIn("CUDA query", error.getvalue())

    def test_mixed_policy_reaches_preparation_autocast_and_compiler_and_output_file(self):
        from deepopt.kernel_policy import current_kernel_policy

        stdout, observed = io.StringIO(), []
        prepare = benchmark.prepare_fp8_model

        def prepare_owned(*args, **kwargs):
            observed.append(current_kernel_policy())
            return prepare(*args, **kwargs)

        policy = {"preferred": "tilelang", "roles": {"compiler.backward": "triton"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results" / "compiler.json"
            with patch.object(benchmark, "prepare_fp8_model", side_effect=prepare_owned) as preparation, \
                    patch.object(benchmark, "autocast", wraps=benchmark.autocast) as amp, \
                    patch.object(benchmark, "compile_model", wraps=benchmark.compile_model) as compiler, \
                    redirect_stdout(stdout):
                result = benchmark.main([
                    "--device", "cpu", "--steps", "1", "--warmup", "0", "--batch-size", "2",
                    "--fp8", "--compression", "--deepopt-backend", "mixed",
                    "--deepopt-kernel-policy", json.dumps(policy), "--output", str(path),
                ])
            report = json.loads(stdout.getvalue())
            self.assertEqual(report, json.loads(path.read_text()))
        self.assertEqual(result, 0)
        self.assertTrue(observed)
        self.assertTrue(all(owner.backend == "mixed" for owner in observed))
        self.assertEqual(current_kernel_policy().backend, "triton")
        self.assertEqual(report["settings"]["backend"], "deepopt")
        self.assertEqual(report["kernel_backend"]["policy"]["backend"], "mixed")
        self.assertEqual(report["kernel_backend"]["execution"], "cpu_reference")
        self.assertEqual(report["kernel_backend"]["packages"], {})
        for call in preparation.call_args_list:
            self.assertEqual(call.args[1].backend, "reference")
            self.assertEqual(call.kwargs["config"]["runtime"]["kernels"]["backend"], "mixed")
        for call in amp.call_args_list:
            self.assertEqual(call.kwargs["config"]["runtime"]["kernels"]["roles"], policy["roles"])
        self.assertEqual(compiler.call_args.kwargs["config"]["runtime"]["kernels"]["backend"], "mixed")
        self.assertTrue(all(graph["kernel_backend"] == "mixed"
                            for graph in report["cases"][1]["compile_report"]["graphs"]))

    def test_cuda_device_index_wraps_measurement_in_selected_device_context(self):
        from argparse import Namespace
        from contextlib import contextmanager
        from deepopt.kernel_policy import current_kernel_policy

        entered = []

        @contextmanager
        def device_scope(device):
            entered.append((str(device), current_kernel_policy().backend))
            yield

        args = Namespace(device="cuda:1", deepopt_backend="tilelang", deepopt_kernel_policy={})
        with patch.object(torch.cuda, "device", side_effect=device_scope), \
                patch.object(benchmark, "_benchmark", return_value=("result", "state")) as execute:
            self.assertEqual(benchmark.benchmark(args, "eager"), ("result", "state"))
        self.assertEqual(entered, [("cuda:1", "tilelang")])
        self.assertEqual(execute.call_args.args[2].policy.backend, "tilelang")

    def test_vit_block_profiles_native_statistics_and_lowered_projection_training(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [
                sys.executable, str(root / "benchmarks/compiler/benchmark_compiler.py"),
                "--model", "vit-block", "--width", "32", "--heads", "4", "--tokens", "7",
                "--batch-size", "2", "--steps", "1", "--warmup", "1",
                "--fp8", "--compression", "--precision", "bf16", "--profile",
            ],
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = json.loads(result.stdout)
        self.assertTrue(result["numerical_parity"])
        compiled = result["cases"][1]
        self.assertEqual(compiled["compile_report"]["lowered_deepopt_operations"], 5) # Four projections plus SDPA.
        self.assertGreater(compiled["compile_report"]["native_autodiff_operations"], 0)
        self.assertGreaterEqual(compiled["compile_report"]["compiled_deepopt_backward_regions"],
                                compiled["compile_report"]["lowered_deepopt_operations"])
        for case in result["cases"]:
            operations = case["profile"]["normalization_operations"]
            self.assertEqual(operations["aten::native_layer_norm"], 2)
            self.assertEqual(operations["aten::native_layer_norm_backward"], 2)

    def test_cpu_fp8_compression_smoke_reports_capture_and_numerical_parity(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [
                sys.executable,
                str(root / "benchmarks/compiler/benchmark_compiler.py"),
                "--steps",
                "1",
                "--warmup",
                "1",
                "--fp8",
                "--compression",
            ],
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["numerical_parity"])
        self.assertEqual(
            [case["backend"] for case in report["cases"]], ["eager", "deepopt"]
        )
        self.assertGreater(
            report["cases"][1]["compile_report"]["deepopt_operations"], 0
        )
        self.assertGreater(report["cases"][1]["saved_storage_bytes"], 0)
        self.assertIsNone(report["cases"][1]["peak_cuda_allocated_bytes"])


if __name__ == "__main__":
    unittest.main()
