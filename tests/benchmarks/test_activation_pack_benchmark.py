"""Portable pack benchmark validates arguments before loading GPU dependencies."""

from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/activation_compression/benchmark_activation_pack.py"


class ActivationPackBenchmarkTests(unittest.TestCase):
    def test_help_and_invalid_arguments_work_without_site_packages(self):
        for args, code in ((["--help"], 0), (["--output", "unused", "--iterations", "0"], 2),
                           (["--output", "unused", "--shapes", "invalid"], 2),
                           (["--output", "unused", "--group-sizes", "4096"], 2)):
            result = subprocess.run([sys.executable, "-S", str(SCRIPT), *args], capture_output=True, text=True)
            self.assertEqual(result.returncode, code, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
