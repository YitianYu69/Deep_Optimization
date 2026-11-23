"""Opt-in qualification must not alter the legacy Triton benchmark path."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/benchmark_kernel_backends.py"
SPEC = importlib.util.spec_from_file_location("tilelang_qualified_suite", SCRIPT)
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


def checked_row():
    check = {"status": "passed", "finite": True, "metric": "max_abs", "value": 0., "limit": 1e-5}
    return {"implementation": "deepopt", "input": "bf16", "output": "bf16", "compression": False,
            "timing": {"eager": {"wall_ms_median": 1.}},
            "validation": {"schema_version": 1, "status": "passed", "oracle": "independent native",
                           "required_checks": ["output", "dx"],
                           "checks": {"output": dict(check), "dx": dict(check)}}}


class TileLangQualifiedSuiteTests(unittest.TestCase):
    def test_default_triton_commands_and_settings_keep_legacy_contract(self):
        args = suite.parser().parse_args(["--workloads", "pooling", "--backends", "triton"])
        case, = suite.matrix(args)
        self.assertNotIn("--qualify-tilelang", case["command"])
        self.assertNotIn("qualification", case)
        self.assertNotIn("qualify_tilelang", suite.settings_snapshot(args))
        self.assertNotIn("compression", suite.settings_snapshot(args))
        # The old detector keeps its behavior; only the new mode closes this gap.
        self.assertFalse(suite.invalid_report({"results": [{"output_finite": False}]}))

    def test_focused_flags_reach_only_the_supported_workers(self):
        args = suite.parser().parse_args([
            "--qualify-tilelang", "--workloads", "linear", "layer_norm", "pooling",
            "--backends", "triton", "tilelang", "--inputs", "bf16", "--outputs", "bf16",
            "--compression", "on", "--linear-shapes", "32x64x96", "--linear-modes", "forward",
            "--max-worker-runs", "6"])
        cases = suite.matrix(args)
        self.assertEqual(len(cases), 6)
        for case in cases:
            command = case["command"]
            self.assertIn("--qualify-tilelang", command)
            self.assertEqual(case["qualification"]["schema_version"], 1)
            if case["workload"] == "linear":
                self.assertNotIn("--compression", command)
                self.assertEqual(command[command.index("--shapes") + 1], "32x64x96")
                self.assertIn("--modes", command)
            else:
                self.assertEqual(command[command.index("--compression") + 1], "on")
                self.assertEqual(command[command.index("--inputs") + 1], "bf16")

    def test_work_cap_rejects_before_gpu_discovery_or_output_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "never-created"
            with patch.object(suite, "run_provenance") as discover, self.assertRaises(SystemExit):
                suite.main(["--qualify-tilelang", "--workloads", "pooling", "--backends", "triton", "tilelang",
                            "--trials", "3", "--max-worker-runs", "4", "--output", str(output)])
            discover.assert_not_called()
            self.assertFalse(output.exists())

    def test_strict_checks_reject_missing_failed_and_nonfinite_required_evidence(self):
        valid = {"results": [checked_row()]}
        self.assertFalse(suite.invalid_qualified_report(valid, "pooling"))
        mutations = [
            lambda r: r.pop("validation"),
            lambda r: r.update(output_finite=False),
            lambda r: r["validation"]["checks"].pop("dx"),
            lambda r: r["validation"]["checks"]["dx"].update(value=None),
            lambda r: r["validation"]["checks"]["dx"].update(value=float("nan")),
            lambda r: r["validation"]["checks"]["dx"].update(value=2., limit=1.),
            lambda r: r["validation"]["checks"]["dx"].update(finite=False),
            lambda r: r["validation"].update(status="failed"),
        ]
        for mutation in mutations:
            candidate = deepcopy(valid)
            mutation(candidate["results"][0])
            with self.subTest(mutation=mutation):
                self.assertTrue(suite.invalid_qualified_report(candidate, "pooling"))

    def test_optional_null_indices_and_native_controls_are_not_missing_checks(self):
        report = {"results": [{"implementation": "native_bf16", "timing": {}}, checked_row()]}
        report["results"][1]["indices_equal_vs_compression_off"] = None
        self.assertFalse(suite.invalid_qualified_report(report, "pooling"))
        self.assertTrue(suite.invalid_qualified_report({"results": report["results"][:1]}, "pooling"))

    def test_linear_nested_cases_require_each_requested_gradient(self):
        row = checked_row()
        row.pop("implementation")
        row["mode"] = "forward-backward"
        row["shape_m_k_n"] = [32, 64, 96]
        report = {"settings": {"bias": True}, "results": [{"implementation": "deepopt", "cases": [row]}]}
        self.assertTrue(suite.invalid_qualified_report(report, "linear"))
        for role in ("dweight", "dbias"):
            row["validation"]["checks"][role] = dict(row["validation"]["checks"]["dx"])
        self.assertFalse(suite.invalid_qualified_report(report, "linear"))

    def test_malformed_reports_and_missing_linear_mode_fail_closed(self):
        for report in (None, [], {"settings": None}, {"settings": []}):
            with self.subTest(report=report):
                self.assertTrue(suite.invalid_qualified_report(report, "linear"))
        row = checked_row()
        row["shape_m_k_n"] = [32, 64, 96]
        self.assertTrue(suite.invalid_qualified_report({"results": [row]}, "linear"))

    def test_returned_inventory_cannot_drop_requested_dtype_or_mode(self):
        row = checked_row()
        report = {"configuration": {"shape": [2, 3, 6, 6]}, "results": [row]}
        case = {"command": ["python", "worker", "--shape", "2x3x6x6", "--inputs", "bf16",
                            "--outputs", "bf16", "--compression", "off"]}
        self.assertFalse(suite.invalid_qualified_report(report, "pooling", case))
        case["command"] = ["python", "worker", "--shape", "2x3x6x6", "--inputs", "bf16",
                           "--outputs", "bf16", "fp8", "--compression", "off"]
        self.assertTrue(suite.invalid_qualified_report(report, "pooling", case))

    def test_retry_budget_is_persistent_and_checked_before_any_worker_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            args = suite.parser().parse_args(["--qualify-tilelang", "--workloads", "pooling", "--backends", "tilelang",
                "--max-worker-runs", "1", "--device", "cpu", "--output", directory])
            cases = suite.matrix(args)
            provenance = {"resume_eligible": True}
            failed = lambda case, timeout: {**case, "status": "error", "measurements": {}, "process_wall_seconds": .1}
            with patch.object(suite, "execute", side_effect=failed) as execute:
                self.assertEqual(suite.run_matrix(args, cases, provenance), 1)
                self.assertEqual(execute.call_count, 1)
            args.resume = True
            with patch.object(suite, "execute") as execute, self.assertRaisesRegex(ValueError, "batch exhausted"):
                suite.run_matrix(args, suite.matrix(args), provenance)
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
