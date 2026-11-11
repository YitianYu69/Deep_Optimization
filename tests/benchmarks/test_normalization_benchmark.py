"""Normalization benchmark portability and pre-CUDA option validation."""

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
        self.assertEqual((args.group_size, args.bits), (512, 2))
        self.assertEqual(args.pipeline, "conv")

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
