"""Portable CLI and numerical-report contracts without a CUDA device."""

import argparse
from contextlib import redirect_stderr
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

import torch


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/convolution/benchmark_convolution.py"
SPEC = importlib.util.spec_from_file_location("convolution_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class ConvolutionBenchmarkTests(unittest.TestCase):
    def test_copied_program_help_needs_no_repository_or_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "benchmark_convolution.py"
            shutil.copyfile(SCRIPT, target)
            environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
            result = subprocess.run([sys.executable, str(target), "--help"], cwd=directory,
                                    env=environment, text=True, capture_output=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--group-size", result.stdout)
        self.assertIn("--scaling-recipe", result.stdout)
        self.assertIn("--layout", result.stdout)
        self.assertIn("--fp8-weight-block", result.stdout)
        self.assertIn("--fp8-grad-block", result.stdout)
        self.assertIn("installed DeepOpt", result.stdout)
        self.assertNotIn("sys.path.insert", SCRIPT.read_text())

    def test_default_is_matched_four_modes(self):
        args = benchmark.parser().parse_args([])
        self.assertEqual(args.paths, ["bf16", "fp8"])
        self.assertEqual(args.shape, (8, 64, 56, 56))
        self.assertEqual(args.group_size, 512)
        self.assertEqual(args.bits, 2)
        self.assertEqual(args.scaling_recipe, "current")
        self.assertIsNone(args.fp8_block)
        self.assertIsNone(args.fp8_weight_block)
        self.assertIsNone(args.fp8_grad_block)

    def test_independent_fp8_operand_blocks(self):
        args = benchmark.parser().parse_args([
            "--fp8-block", "1x32", "--fp8-weight-block", "32x64", "--fp8-grad-block", "16x32",
        ])
        self.assertEqual(args.fp8_block, (1, 32))
        self.assertEqual(args.fp8_weight_block, (32, 64))
        self.assertEqual(args.fp8_grad_block, (16, 32))

    def test_transposed_graph_and_profile_options(self):
        args = benchmark.parser().parse_args([
            "--operation", "conv_transpose", "--shape", "8x512x7x7", "--kernel", "4", "--stride", "2",
            "--padding", "1", "--output-padding", "0", "--cuda-graph", "--optimizer-step", "--profile-dir", "profiles",
        ])
        self.assertEqual(args.operation, "conv_transpose")
        self.assertEqual(args.output_padding, (0,))
        self.assertTrue(args.cuda_graph)
        self.assertTrue(args.optimizer_step)
        self.assertEqual(args.profile_dir, Path("profiles"))
        with patch.object(sys, "argv", ["benchmark", "--operation", "conv_transpose", "--padding", "same"]), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.main()
        with patch.object(sys, "argv", ["benchmark", "--cuda-graph", "--scaling-recipe", "delayed"]), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.main()

    def test_spatial_flags_support_all_ranks_and_real_grouped_cases(self):
        for shape in ("2x16x31", "2x16x15x17", "1x16x9x11x13"):
            args = benchmark.parser().parse_args([
                "--shape", shape, "--kernel", "3", "--stride", "2", "--padding", "1",
                "--groups", "16", "--bits", "4", "--group-size", "17",
            ])
            rank = len(args.shape) - 2
            self.assertEqual(benchmark.dimensional(args.kernel, rank, "--kernel"), (3,) * rank)
            self.assertEqual(args.groups, 16)
        self.assertEqual(benchmark.padding_value("same"), "same")
        self.assertEqual(benchmark.padding_value("0x1x2"), (0, 1, 2))
        self.assertEqual(benchmark.dimensional((1, 2, 3), 3, "--dilation"), (1, 2, 3))

    def test_shape_and_bit_validation(self):
        for invalid in ("", "1x0x3", "-2x3", "2xx3", "hello"):
            with self.subTest(shape=invalid), self.assertRaises(argparse.ArgumentTypeError):
                benchmark.positive_shape(invalid)
        with self.assertRaises(argparse.ArgumentTypeError):
            benchmark.padding_value("-1")
        with self.assertRaisesRegex(ValueError, "must have 1 or 3 dimensions"):
            benchmark.dimensional((1, 2), 3, "--stride")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.parser().parse_args(["--bits", "3"])

    def test_incompatible_options_fail_before_cuda_execution(self):
        cases = (
            (["--shape", "1x2"], "spatial dimensions"),
            (["--shape", "1x3x9", "--layout", "channels_last"], "Conv2d and Conv3d"),
            (["--shape", "1x3x9", "--groups", "2"], "divisible"),
            (["--fp8-block", "4x3"], "powers of two"),
            (["--fp8-block", "1x1"], "cannot be 1x1"),
            (["--fp8-weight-block", "32"], "--fp8-weight-block"),
            (["--fp8-grad-block", "4x3"], "--fp8-grad-block"),
            (["--fp8-block", "4x8", "--scaling-recipe", "delayed"], "tensorwise"),
            (["--fp8-weight-block", "32x32", "--scaling-recipe", "delayed"], "tensorwise"),
            (["--fp8-grad-block", "32x32", "--scaling-recipe", "delayed"], "tensorwise"),
            (["--iterations", "0"], "counts must be positive"),
            (["--kernel", "2x3x4"], "must have 1 or 2 dimensions"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                error = io.StringIO()
                with patch.object(sys, "argv", ["benchmark", *arguments]), redirect_stderr(error), self.assertRaises(SystemExit):
                    benchmark.main()
                self.assertIn(message, error.getvalue())

    def test_difference_report_preserves_max_error_and_relative_norm(self):
        expected = {"dw": torch.tensor([3., 4.]), "dx": torch.zeros(2)}
        actual = {"dw": torch.tensor([6., 8.]), "dx": torch.zeros(2)}
        report = benchmark.differences(actual, expected)
        self.assertEqual(report["dw"], {"relative_l2": 1., "max_abs": 4.})
        self.assertEqual(report["dx"], {"relative_l2": 0., "max_abs": 0.})


if __name__ == "__main__":
    unittest.main()
