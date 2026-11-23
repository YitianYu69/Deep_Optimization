"""Normalization benchmark portability and pre-CUDA option validation."""

from contextlib import nullcontext, redirect_stderr
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/normalization/benchmark_normalization.py"
SPEC = importlib.util.spec_from_file_location("normalization_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class NormalizationBenchmarkTests(unittest.TestCase):
    def test_help_is_portable_without_repository_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / SCRIPT.name
            shutil.copyfile(SCRIPT, path)
            environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
            result = subprocess.run([sys.executable, str(path), "--help"], cwd=directory, env=environment,
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--inputs", result.stdout)
        self.assertIn("--pipeline", result.stdout)
        self.assertNotIn("sys.path.insert", SCRIPT.read_text())

    def test_defaults_cover_both_input_precisions_and_xhat_compression(self):
        args = benchmark.parser().parse_args([])
        self.assertEqual(args.inputs, ["bf16", "fp8"])
        self.assertEqual((args.group_size, args.bits), (1024, 2))
        self.assertEqual(args.pipeline, "conv")
        self.assertEqual(args.deepopt_backend, "triton")
        self.assertEqual(args.deepopt_kernel_policy, {})

    def test_kernel_policy_validation_happens_before_gpu_lease_or_setup(self):
        for backend, policy in (("tilelang", "null"), ("tilelang", "[]"), ("tilelang", "{"),
                                ("tilelang", '{"backend":"triton"}'),
                                ("mixed", '{"families":{"unknown":"triton"}}')):
            with self.subTest(backend=backend, policy=policy):
                error = io.StringIO()
                argv = [str(SCRIPT), "--deepopt-backend", backend, "--deepopt-kernel-policy", policy]
                with patch.object(sys, "argv", argv), patch.object(benchmark, "_lease_cli", side_effect=AssertionError("GPU lease")), \
                        redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                    benchmark.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("error:", error.getvalue())

    def test_triton_schedule_controls_reject_tilelang_routes_before_cuda(self):
        for flags in (("--deepopt-backend", "tilelang"),
                      ("--deepopt-backend", "mixed", "--deepopt-kernel-policy",
                       '{"roles":{"normalization.backward":"tilelang"}}')):
            with self.subTest(flags=flags):
                error = io.StringIO()
                argv = [str(SCRIPT), "--kernel-variant", "candidate", *flags]
                with patch.object(sys, "argv", argv), patch.object(benchmark, "_lease_cli", side_effect=AssertionError("GPU lease")), \
                        redirect_stderr(error), self.assertRaises(SystemExit):
                    benchmark.main()
                self.assertIn("Triton-only", error.getvalue())

    def test_policy_scope_covers_variant_selection_and_normalization_runner(self):
        from deepopt.kernel_policy import current_kernel_policy

        observed = []

        def variant(args):
            observed.append(("variant", current_kernel_policy().backend))
            return nullcontext()

        def execute(args, kernels):
            observed.append(("run", current_kernel_policy().backend))
            self.assertEqual(kernels.config["runtime"]["kernels"]["backend"], "mixed")
            return 17

        argv = [str(SCRIPT), "--deepopt-backend", "mixed", "--deepopt-kernel-policy",
                '{"preferred":"tilelang","families":{"normalization":"triton"}}']
        with patch.object(sys, "argv", argv), patch.object(benchmark, "_lease_cli", return_value=None), \
                patch.object(benchmark, "variant_scope", side_effect=variant), \
                patch.object(benchmark, "norm_main", side_effect=execute):
            self.assertEqual(benchmark.main(), 17)
        self.assertEqual(observed, [("variant", "mixed"), ("run", "mixed")])
        self.assertEqual(current_kernel_policy().backend, "triton")

    def test_invalid_options_fail_before_gpu_initialization(self):
        for args, message in ((["--shape", "2x4x5"], "four dimensions"),
                              (["--op", "group_norm", "--groups", "7"], "divide channels"),
                              (["--op", "group_norm", "--shape", "2x4", "--groups", "2", "--pipeline", "conv"], "3, 4 or 5"),
                              (["--iterations", "0"], "counts must be positive")):
            with self.subTest(args=args):
                error = io.StringIO()
                with patch.object(sys, "argv", [str(SCRIPT), *args]), redirect_stderr(error), self.assertRaises(SystemExit):
                    benchmark.main()
                self.assertIn(message, error.getvalue())


if __name__ == "__main__":
    unittest.main()
