"""Stdlib planning and CUDA-unavailable receipts; not fake timing evidence."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from benchmarks.compiler import benchmark_offload_transport as calibration


def _args(output, *extra):
    return calibration.parser().parse_args(["--groups", "1024", "--host-bytes", "1048576",
        "--gpu-bytes", "67108864", "--compute-elements", "1024", "--output", str(output), *extra])


class OffloadTransportCalibrationTests(unittest.TestCase):
    def test_help_and_import_are_torch_free(self):
        command = """
import sys
from benchmarks.compiler import benchmark_offload_transport as module
assert 'torch' not in sys.modules
assert 'deepopt' not in sys.modules
assert '--host-bytes' in module.parser().format_help()
assert 'native' in module.parser().format_help()
"""
        environment = {**os.environ, "PYTHONPATH": str(calibration.ROOT)}
        process = subprocess.run([sys.executable, "-c", command], env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_explicit_fixture_host_gpu_budgets_and_six_case_contract(self):
        args = _args("unused.json")
        plan = calibration.calibration_plan(args)
        self.assertEqual(plan["triplet_bytes"], 1024 * 264)
        self.assertEqual(plan["host_required_bytes"], 1024 * 264 * 3)
        self.assertEqual(plan["GPU_tensor_required_bytes"], 1024 * 264 * 2 + 1024 * 8)
        self.assertEqual(len(plan["cases"]), 6)
        self.assertEqual({case["direction"] for case in plan["cases"]}, {"h2d", "d2h", "bidirectional"})
        self.assertEqual({case["independent_compute"] for case in plan["cases"]}, {False, True})
        self.assertEqual(plan["group_size"], 1024)
        for extra, message in ((["--host-bytes", "1"], "Host budget"),
                               (["--gpu-bytes", "1"], "GPU tensor budget"),
                               (["--device", "cpu"], "cuda:N"),
                               (["--groups", str(2**31)], "signed-int32")):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                calibration.calibration_plan(_args("unused.json", *extra))
        large = _args("unused.json", "--groups", "10000000", "--host-bytes", str(8 * 2**30), "--gpu-bytes", str(8 * 2**30))
        self.assertGreater(calibration.calibration_plan(large)["triplet_bytes"], 2**31)

    def test_unavailable_cuda_fails_before_any_allocation_and_keeps_immutable_failure_receipt(self):
        touched = []

        class NoCuda:
            @staticmethod
            def is_available():
                touched.append("availability")
                return False

            def __getattr__(self, name):
                raise AssertionError(f"Unexpected CUDA operation before availability: {name}")

        @contextmanager
        def runtime():
            yield SimpleNamespace(torch=SimpleNamespace(cuda=NoCuda()), identity={"version": "explicit-no-CUDA-test"})

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            report = calibration.run(_args(output, "--preflight-only"), runtime_context=runtime)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["stage"], "installed_CUDA_preflight")
            self.assertEqual(report["error"]["type"], "RuntimeError")
            self.assertIn("CUDA unavailable", report["error"]["message"])
            self.assertIn("cuda_preflight", report["error"]["traceback"])
            self.assertFalse(report["CUDA_fixture_execution_observed"])
            self.assertFalse(report["native_AutoDiff_qualified"])
            self.assertEqual(json.loads(output.read_text()), json.loads(json.dumps(report)))
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                calibration.run(_args(output), runtime_context=runtime)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(touched, ["availability"])
            self.assertEqual(tuple(output.with_name("result-artifacts").iterdir()), ())

    def test_invalid_budget_writes_failure_without_loading_torch_or_installation(self):
        @contextmanager
        def forbidden():
            raise AssertionError("Runtime loaded before bounded admission")
            yield

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "bad.json"
            report = calibration.run(_args(output, "--gpu-bytes", "1"), runtime_context=forbidden)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["stage"], "planning")
            self.assertEqual(report["error"]["type"], "ValueError")
            self.assertNotIn("installed", report)

    def test_all_owned_stream_drains_run_and_primary_exception_survives(self):
        calls, evidence = [], {}
        primary = ValueError("original compute failure")

        class Stream:
            def __init__(self, name, fails=False):
                self.name, self.fails = name, fails

            def synchronize(self):
                calls.append(self.name)
                if self.fails:
                    raise RuntimeError("secondary drain fault")

        def operation(owned):
            owned.update(D2H=Stream("D2H"), H2D=Stream("H2D", True), compute=Stream("compute"))
            evidence["completed_cases"] = 2
            raise primary

        with self.assertRaises(ValueError) as caught:
            calibration._with_owned_stream_cleanup(operation, evidence)
        self.assertIs(caught.exception, primary)
        self.assertEqual(calls, ["D2H", "H2D", "compute"])
        self.assertEqual(evidence["completed_cases"], 2)
        self.assertFalse(evidence["owned_stream_cleanup"]["succeeded"])
        self.assertEqual(evidence["owned_stream_cleanup"]["errors"][0]["stream"], "H2D")
        self.assertTrue(any("secondary drain fault" in note for note in primary.__notes__))

    def test_cleanup_fault_without_primary_still_fails_and_drains_other_streams(self):
        calls, evidence = [], {}

        class Stream:
            def __init__(self, name):
                self.name = name

            def synchronize(self):
                calls.append(self.name)
                if self.name == "D2H":
                    raise OSError("copy stream unavailable")

        def operation(owned):
            owned.update(D2H=Stream("D2H"), H2D=Stream("H2D"), compute=Stream("compute"))
            return "otherwise successful"

        with self.assertRaisesRegex(RuntimeError, "cleanup failed") as caught:
            calibration._with_owned_stream_cleanup(operation, evidence)
        self.assertIsInstance(caught.exception.__cause__, OSError)
        self.assertEqual(calls, ["D2H", "H2D", "compute"])
        self.assertFalse(evidence["owned_stream_cleanup"]["succeeded"])

    def test_cost_profile_admission_excludes_zero_overlap_and_labels_positive_span_unverified(self):
        class Measurements:
            def __init__(self):
                self.calls = []

            def observe(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        measurements = Measurements()
        copy = dict(direction="H2D", bytes=264, GPU_copy_ms=1., compute_stream_span_intersection_ms=0.)
        status = calibration._observe_cost(measurements, copy, direction="h2d", contended=True)
        self.assertIn("excluded_zero", status)
        self.assertFalse(measurements.calls)
        copy["compute_stream_span_intersection_ms"] = .5
        status = calibration._observe_cost(measurements, copy, direction="h2d", contended=True)
        self.assertIn("unverified", status)
        self.assertEqual(measurements.calls[-1][1]["load"], "compute")
        before = len(measurements.calls)
        status = calibration._observe_cost(measurements, copy, direction="bidirectional", contended=False)
        self.assertIn("excluded_unmodelled", status)
        self.assertEqual(len(measurements.calls), before)


if __name__ == "__main__":
    unittest.main()
