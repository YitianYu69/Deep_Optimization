"""Opt-in independent benchmark oracles, default isolation and calibration."""

from contextlib import redirect_stdout
import builtins
import importlib.util
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from deepopt import _tilelang_benchmarking as qualification


ROOT = Path(__file__).resolve().parents[2]


def benchmark(relative):
    spec = importlib.util.spec_from_file_location("qualification_" + Path(relative).stem,
                                                  ROOT / "benchmarks" / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TileLangQualificationTests(unittest.TestCase):
    def test_helper_import_is_standard_library_only(self):
        completed = subprocess.run([sys.executable, "-S", "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); import deepopt._tilelang_benchmarking; "
            "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules; assert 'tilelang' not in sys.modules",
            str(ROOT / "src")], capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_validation_fails_closed_for_missing_nonfinite_and_corrupt_roles(self):
        check = qualification.tensor_check(torch.ones(8), torch.ones(8))
        result = qualification.validation_result({"output": check}, ["output", "dx"], "fixture")
        self.assertFalse(qualification.validation_valid(result))
        self.assertEqual(result["checks"]["dx"]["status"], "failed")
        for actual in (None, torch.full((8,), float("nan")), torch.ones(8) + .1):
            with self.subTest(actual=actual):
                self.assertEqual(qualification.tensor_check(actual, torch.ones(8))["status"], "failed")
        self.assertTrue(qualification.validation_valid(qualification.validation_result({"output": check}, ["output"], "fixture")))
        valid = qualification.validation_result({"output": check}, ["output"], "fixture")
        for changed in ({"required_checks": None}, {"required_checks": [["output"]]}, {"oracle": ""},
                        {"required_checks": "output"}, {"required_checks": []}):
            self.assertFalse(qualification.validation_valid({**valid, **changed}))

    def test_linear_checks_identical_quantized_operands_and_each_gradient(self):
        worker = benchmark("fp8/benchmark_fp8_eager.py")
        args = worker.parser().parse_args(["--device", "cpu", "--bias"])
        args.deepopt_backend = "reference"
        case = worker.LinearCase(args, "deepopt", (16, 32, 16), "forward-backward", device="cpu")
        result = qualification.qualify_linear(case)
        self.assertTrue(qualification.validation_valid(result, ("output", "dx", "dweight", "dbias")), result)
        self.assertIn("identical", result["oracle"])
        from deepopt.opt.nn.autograd import linear
        original = linear.gemm_nt

        def broken(*args, **kwargs):
            value = original(*args, **kwargs)
            return value + .5 if kwargs.get("diagnostic_role") == "dw" else value

        with patch.object(linear, "gemm_nt", broken):
            failed = qualification.qualify_linear(case)
        self.assertEqual(failed["checks"]["dweight"]["status"], "failed")
        self.assertEqual(failed["checks"]["dbias"]["status"], "passed")

    def _run(self, relative, flags):
        module = benchmark(relative)
        output = StringIO()
        with patch.object(sys, "argv", [str(ROOT / "benchmarks" / relative), *flags]), redirect_stdout(output):
            module.main()
        return json.loads(output.getvalue())

    def test_layernorm_full_precision_pairs_and_actual_compressed_state_oracle(self):
        report = self._run("normalization/benchmark_normalization.py", ["--op", "layer_norm", "--pipeline", "none",
            "--device", "cpu", "--shape", "2x3x7", "--normalized-shape", "3x7", "--iterations", "1",
            "--repeats", "1", "--warmup", "0", "--time-budget", "0", "--qualify-tilelang", "--compression", "on"])
        rows = [row for row in report["results"] if row["implementation"] == "deepopt"]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["compression"] for row in rows))
        for row in rows:
            self.assertTrue(qualification.validation_valid(row["validation"], ("output", "dx", "dweight", "dbias")), row["validation"])
            self.assertIn("saved_gain", row["validation"]["checks"])
            self.assertGreater(row["validation"]["compression_quality"]["xhat"]["relative_l2"], 0)
            self.assertEqual(row["timing"]["eager"]["sampling"]["calibration_calls"], 4)

    def test_pooling_native_oracle_exact_compression_and_compression_only_selection(self):
        report = self._run("pooling/benchmark_pooling.py", ["--device", "cpu", "--shape", "2x3x6x6",
            "--iterations", "1", "--repeats", "1", "--warmup", "0", "--time-budget", "0",
            "--qualify-tilelang", "--compression", "on", "--return-indices"])
        rows = [row for row in report["results"] if row["implementation"] == "deepopt"]
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertTrue(row["compression"])
            self.assertTrue(qualification.validation_valid(row["validation"], ("output", "dx")), row["validation"])
            self.assertEqual(row["validation"]["checks"]["indices"]["value"], 0)
        bad = qualification.qualify_pooling(torch.ones(1, 1, 1, 1), SimpleNamespace(dtype=torch.float32, grad=torch.ones(1, 1, 2, 2)),
            torch.ones(1, 1, 2, 2), torch.ones(1, 1, 1, 1), "max_pool2d", {"kernel_size": 2}, compressed=True,
            compression_errors={"output_max_abs_vs_compression_off": None, "dx_max_abs_vs_compression_off": 0.})
        self.assertEqual(bad["status"], "failed")
        # The legacy compression-off comparison has already overwritten the
        # live gradient. The oracle must check the actual sample's saved dx.
        native = torch.ones(1, 1, 2, 2, requires_grad=True)
        torch.nn.functional.max_pool2d(native, 2).backward(torch.ones(1, 1, 1, 1))
        source = SimpleNamespace(dtype=torch.float32, grad=native.grad)
        corrupt = qualification.qualify_pooling(torch.ones(1, 1, 1, 1), source, native.detach(),
            torch.ones(1, 1, 1, 1), "max_pool2d", {"kernel_size": 2}, actual_dx=native.grad + .5,
            compressed=True, compression_errors={"output_max_abs_vs_compression_off": 0., "dx_max_abs_vs_compression_off": 0.})
        self.assertEqual(corrupt["checks"]["dx"]["status"], "failed")
        self.assertEqual(corrupt["checks"]["compression_dx_max_abs_vs_compression_off"]["status"], "passed")

    def test_qualified_calibration_uses_three_pilots_and_keeps_cold_exhaustive_accounting(self):
        args = SimpleNamespace(warmup=3, iterations=10, repeats=5, time_budget=1.)
        ticks = iter([0., 10., 10., 10.0002, 11., 11.156, 12., 12.0002])
        calls = []
        plan = qualification.qualified_timing_plan(lambda: calls.append(1), args, clock=lambda: next(ticks))
        self.assertEqual(len(calls), 4)
        self.assertEqual((plan.warmup, plan.iterations, plan.repeats), (3, 10, 5))
        self.assertAlmostEqual(plan.pilot_seconds, .0002)
        self.assertEqual(plan.cold_first_call_seconds, 10.)
        args.time_budget = 0.
        ticks = iter([0., 10., 11., 12., 13., 14., 15., 16.])
        plan = qualification.qualified_timing_plan(lambda: None, args, clock=lambda: next(ticks))
        self.assertEqual(plan.cold_first_call_seconds, 10.)
        self.assertEqual(plan.sampling(20.)["calibration_calls"], 4)

    def test_default_reports_and_calibration_do_not_import_qualification_helpers(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "deepopt._tilelang_benchmarking":
                raise AssertionError("default path imported the opt-in helper")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", guarded):
            report = self._run("normalization/benchmark_normalization.py", ["--op", "layer_norm", "--pipeline", "none",
                "--device", "cpu", "--shape", "2x7", "--inputs", "bf16", "--outputs", "bf16",
                "--warmup", "0", "--iterations", "1", "--repeats", "1", "--time-budget", "0"])
        self.assertNotIn("qualify_tilelang", report["configuration"])
        self.assertNotIn("compression", report["configuration"])
        self.assertTrue(all("validation" not in row for row in report["results"]))


if __name__ == "__main__":
    unittest.main()
