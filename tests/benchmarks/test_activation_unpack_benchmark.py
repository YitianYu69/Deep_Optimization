"""Reconstruction benchmark controls are portable and validate before CUDA imports."""

import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/activation_compression/benchmark_activation_unpack.py"
SPEC = importlib.util.spec_from_file_location("activation_unpack_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class ActivationUnpackBenchmarkTests(unittest.TestCase):
    def test_defaults_reproduce_dominant_trace_shapes_and_include_all_controls(self):
        args = benchmark.parser().parse_args([])
        self.assertEqual(args.shapes[:2], [(100864, 3072), (100864, 768)])
        self.assertEqual(args.implementations, ["grouped", "copy", "tiled", "auto"])
        self.assertEqual((args.source_dtype, args.dtypes, args.group_sizes, args.bits), ("e4m3fn", ["e4m3fn"], [1024], [2]))

    def test_help_and_invalid_settings_do_not_import_torch_or_triton(self):
        original = __import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "triton", "deepopt"):
                raise AssertionError("Invalid arguments imported " + name)
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", new=guarded), patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
            for options, code in ((["--help"], 0), (["--device", "cpu"], 2), (["--iterations", "0"], 2),
                                  (["--block", "3x32"], 2), (["--shapes", "-1x32"], 2)):
                with self.subTest(options=options), self.assertRaises(SystemExit) as raised:
                    benchmark.main(options)
                self.assertEqual(raised.exception.code, code)


if __name__ == "__main__":
    unittest.main()
