"""Lossless diagnostic artifacts and bounded parent receipt views."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from examples.vae.diagnostic_reporting import (
    MAX_FAILED_NAMES, MAX_TEXT_CHARS, externalize_compiler_reports, summary_for_parent,
)


class VAEDiagnosticReportingTests(unittest.TestCase):
    def case_result(self):
        comparison = {"name": "gradient/input", "passed": False,
            "failure": "relative_l2_exceeds_storage_domain_limit", "relative_l2": .0031,
            "relative_l2_limit": 9.5367431640625e-7, "max_abs": 1.49e-7,
            "expected": {"present": True, "dtype": "torch.float32", "shape": [1, 3, 224, 224]},
            "actual": {"present": True, "dtype": "torch.float32", "shape": [1, 3, 224, 224]},
            "context": {"phase": "gradient_comparison", "route": "native", "update": 1}}
        rows = [{"name": f"output/{index}", "passed": True} for index in range(6)]
        rows += [{"name": f"recipe/{index}", "passed": True} for index in range(3)]
        rows += [comparison]
        rows += [{"name": f"gradient/encoder.{index}.weight", "passed": index >= 183,
                  "max_abs": 0.0001 if index < 183 else 0.0} for index in range(332)]
        oracle = {"comparisons": rows, "first_failure": comparison, "failure_count": 184,
                  "completed_comparison_count": 342, "unconsumed_count": 1000,
                  "reference_memory_plan": {"fields": [{"name": f"optimizer/{index}", "bytes": 4096}
                                                        for index in range(1703)]}}
        return {"schema_version": 1, "case": {"name": "streamed_eager_native", "kind": "trajectory",
                    "reference": {"execution": "eager", "profile": "streamed"},
                    "candidate": {"execution": "native", "profile": "streamed"}},
                "settings_sha256": "a" * 64, "installed": {"version": "candidate", "source_sha256": "b" * 64},
                "status": "failed", "qualification_eligible": False, "timing_eligible": False,
                "error": "NumericalComparisonError: gradient/input failed\nsecondary context",
                "traceback": "Original IPC traceback\n  original comparison frame\ncaused by numerical failure\n",
                "failure_context": {"phase": "gradient_comparison", "route": "native", "update": 1},
                "oracle_report": oracle, "qualification_evidence": {"oracle_report": deepcopy(oracle)},
                "compiler_reports": [
                    {"slot": "reference", "compiler_report_scope": "diagnostic_route",
                     "compiler_report_status": "not_captured", "compiler_report": None,
                     "future_metadata": {"preserved": [1, 2, 3]}},
                    {"slot": "candidate", "compiler_report_scope": "diagnostic_route",
                     "compiler_report_status": "available", "eager_dtype_edges": None,
                     "compiler_report": {"captured_graphs": 1, "runtime_fallbacks": 0,
                         "native_autodiff_regions": 2, "fusion_candidates": [{"why": "accepted"}],
                         "graphs": [{"generated_code": "kernel source", "forward_ir": ["x", "y"],
                                     "backward_ir": ["dx", "dy"]}]}}]}

    def write_case(self, path, result):
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    def test_full_entries_are_lossless_compact_artifacts_with_verified_identity(self):
        result = self.case_result()
        original = deepcopy(result)
        with tempfile.TemporaryDirectory() as directory:
            returned = externalize_compiler_reports(result, directory)
            self.assertIs(returned, result)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)
            for before, after in zip(original["compiler_reports"], result["compiler_reports"]):
                reference = after["artifact"]
                path = Path(directory) / reference["path"]
                payload = path.read_bytes()
                self.assertEqual(json.loads(payload), before)
                self.assertEqual(payload, json.dumps(before, sort_keys=True, separators=(",", ":"),
                                                    ensure_ascii=False, allow_nan=False).encode())
                self.assertEqual(reference["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual(reference["bytes"], len(payload))
                self.assertNotIn("compiler_report", after)
                self.assertTrue(after["compiler_report_externalized"])
                self.assertEqual(after["slot"], before["slot"])
            result_path = Path(directory) / "result.json"
            self.write_case(result_path, result)
            canonical = json.loads(result_path.read_bytes())
        self.assertEqual(canonical["traceback"], original["traceback"])
        self.assertEqual(canonical["qualification_evidence"], original["qualification_evidence"])
        self.assertEqual(canonical["oracle_report"], original["oracle_report"])
        self.assertEqual(result["compiler_reports"][1]["counts"]["graphs"], 1)
        self.assertEqual(result["compiler_reports"][1]["counts"]["runtime_fallbacks"], 0)

    def test_parent_summary_stays_small_with_full_ir_and_all_comparisons_saved(self):
        result = self.case_result()
        huge_ir = "generated native IR line\n" * 50000
        result["compiler_reports"][1]["compiler_report"]["graphs"][0]["forward_ir"] = huge_ir
        result["compiler_reports"][1]["new_ir_view"] = huge_ir
        result["future_detailed_evidence"] = {"all_ir": huge_ir}
        result["localization"] = {"scope": "matched_selected_boundaries_only", "enabled": True,
            "complete": False, "global_first_divergence_verified": False,
            "first_tolerance_failure": None,
            "first_bitwise_difference": {"key": "backward/encoder.0/cotangent/1", "phase": "backward",
                "dtype": "torch.bfloat16", "relative_l2": .0074, "passed": True, "future_ir": huge_ir}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / "receipt-artifacts" / "streamed_eager_native"
            case.mkdir(parents=True)
            externalize_compiler_reports(result, case)
            result_path = case / "result.json"
            self.write_case(result_path, result)
            summary = summary_for_parent(result, result_path, root)
            encoded = json.dumps(summary, allow_nan=False).encode()
            self.assertLess(len(encoded), 24 * 1024)
            self.assertNotIn(huge_ir[:200], encoded.decode())
            self.assertNotIn("comparisons", summary)
            self.assertNotIn("qualification_evidence", summary)
            self.assertNotIn("oracle_report", summary)
            self.assertNotIn("future_detailed_evidence", summary)
            self.assertNotIn("reference_memory_plan", encoded.decode())
            self.assertNotIn("traceback", summary)
            self.assertNotIn("future_ir", summary["localization"]["first_bitwise_difference"])
            self.assertEqual(summary["comparison"]["name"], "gradient/input")
            self.assertEqual(summary["comparison"]["expected"]["dtype"], "torch.float32")
            self.assertEqual(summary["comparison"]["context"]["update"], 1)
            counts = summary["comparison_summary"]
            self.assertEqual(counts["completed"], 342)
            self.assertEqual(counts["failure_count"], 184)
            self.assertEqual(counts["gradient_count"], 333)
            self.assertEqual(counts["failed_gradient_count"], 184)
            self.assertEqual(counts["failed_name_count"], 184)
            self.assertEqual(len(counts["failed_names"]), MAX_FAILED_NAMES)
            self.assertEqual(counts["failed_names_truncated"], 184 - MAX_FAILED_NAMES)
            reference = summary["result_file"]
            self.assertEqual(root / reference["path"], result_path)
            self.assertEqual(reference["sha256"], hashlib.sha256(result_path.read_bytes()).hexdigest())
            self.assertEqual(reference["bytes"], result_path.stat().st_size)
            artifact = summary["compiler_reports"][1]["artifact"]
            artifact_path = root / artifact["path"]
            self.assertEqual(artifact["sha256"], hashlib.sha256(artifact_path.read_bytes()).hexdigest())
            self.assertEqual(artifact["bytes"], artifact_path.stat().st_size)
            self.assertEqual(json.loads(artifact_path.read_bytes())["new_ir_view"], huge_ir)
            saved = json.loads(result_path.read_bytes())
        self.assertEqual(len(saved["oracle_report"]["comparisons"]), 342)
        self.assertEqual(len(saved["oracle_report"]["reference_memory_plan"]["fields"]), 1703)
        self.assertEqual(saved["future_detailed_evidence"]["all_ir"], huge_ir)
        self.assertEqual(summary["error"], "NumericalComparisonError: gradient/input failed")
        self.assertFalse(summary["qualification_eligible"])
        self.assertFalse(summary["timing_eligible"])
        self.assertFalse(summary["cuda_qualified"])

    def test_serialization_failure_preserves_original_entries_and_publishes_no_partial_files(self):
        result = self.case_result()
        entries = result["compiler_reports"]
        marker = object()
        entries[1]["future_metadata"] = marker
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TypeError):
                externalize_compiler_reports(result, directory)
            self.assertIs(result["compiler_reports"], entries)
            self.assertIs(entries[1]["future_metadata"], marker)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_artifact_is_immutable_and_identical_calls_are_idempotent(self):
        result = self.case_result()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conflicting = root / "compiler-report-00-reference.json"
            conflicting.write_bytes(b"historical evidence")
            entries = result["compiler_reports"]
            with self.assertRaises(FileExistsError):
                externalize_compiler_reports(result, root)
            self.assertIs(result["compiler_reports"], entries)
            self.assertEqual(conflicting.read_bytes(), b"historical evidence")
        with tempfile.TemporaryDirectory() as directory:
            externalize_compiler_reports(result, directory)
            first = deepcopy(result)
            externalize_compiler_reports(result, directory)
            self.assertEqual(result, first)
            artifact = Path(directory) / result["compiler_reports"][0]["artifact"]["path"]
            artifact.write_bytes(b"changed evidence")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                externalize_compiler_reports(result, directory)

    def test_slots_cannot_escape_the_artifact_directory(self):
        result = {"compiler_reports": [{"slot": "../../outside", "compiler_report": {"graphs": []}}]}
        with tempfile.TemporaryDirectory() as directory:
            externalize_compiler_reports(result, directory)
            reference = result["compiler_reports"][0]["artifact"]
            self.assertEqual(Path(reference["path"]).name, reference["path"])
            self.assertTrue((Path(directory) / reference["path"]).is_file())

    def test_missing_reports_and_unknown_fields_leave_canonical_evidence_untouched(self):
        result = {"case": {"name": "early_failure", "kind": "trajectory", "unexpected": "huge" * 10000},
                  "status": "failed", "error": "X" * (MAX_TEXT_CHARS * 3),
                  "traceback": "original numerical cause", "new_field": {"payload": list(range(10000))}}
        original = deepcopy(result)
        with tempfile.TemporaryDirectory() as directory:
            self.assertIs(externalize_compiler_reports(result, directory), result)
            path = Path(directory) / "result.json"
            self.write_case(path, result)
            summary = summary_for_parent(result, path, directory)
        self.assertEqual(result, original)
        self.assertLessEqual(len(summary["error"]), MAX_TEXT_CHARS)
        self.assertNotIn("unexpected", summary["case"])
        self.assertNotIn("new_field", summary)
        self.assertEqual(summary["comparison_summary"]["completed"], 0)
        self.assertEqual(summary["compiler_reports"], [])

    def test_successful_trajectory_counts_updates_without_promoting_qualification(self):
        result = {"case": {"name": "repeat", "kind": "trajectory"}, "status": "passed",
                  "qualification_eligible": True, "timing_eligible": True,
                  "trajectory": {"updates": [{"oracle": {"comparisons": [
                      {"name": "output/reconstruction", "passed": True},
                      {"name": "gradient/input", "passed": True}]}}]},
                  "operators": {"packed": {"passed": True, "future_ir": "not for parent"}},
                  "repeatability": {"completed": True, "future_ir": "not for parent"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            self.write_case(path, result)
            summary = summary_for_parent(result, path, directory)
        self.assertEqual(summary["comparison_summary"]["completed"], 2)
        self.assertEqual(summary["comparison_summary"]["gradient_count"], 1)
        self.assertNotIn("comparison", summary)
        self.assertEqual(summary["operators"], {"packed": {"passed": True}})
        self.assertEqual(summary["repeatability"], {"completed": True})
        self.assertFalse(summary["qualification_eligible"])

    def test_native_compiler_cause_is_visible_without_loading_full_traceback(self):
        result = {"status": "failed", "error": "BackendCompilerFailed: backend='backend' raised:\n"
            "NotImplementedError: Native AutoDiff coverage incomplete: flat_apply_capture\n\n"
            "Set TORCHDYNAMO_VERBOSE=1 for the full Dynamo stack trace"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            self.write_case(path, result)
            summary = summary_for_parent(result, path, directory)
        self.assertIn("NotImplementedError: Native AutoDiff coverage incomplete", summary["error"])
        self.assertNotIn("TORCHDYNAMO_VERBOSE", summary["error"])
        self.assertNotIn("\n", summary["error"])

    def test_independent_criteria_localization_and_repeat_controls_remain_actionable(self):
        result = self.case_result()
        first = result["oracle_report"]["first_failure"]
        first.update(elementwise_passed=True, relative_l2_passed=False,
                     failure_reasons=["relative_l2_exceeds_storage_domain_limit"])
        result["localization"] = {"complete": False, "coverage_complete": True, "context_succeeded": False,
                                  "first_tolerance_failure": first}
        result["repeatability"] = {"completed": True, "repeat_stable": False,
            "deterministic_blocked": True, "cuda_execution_observed": True, "controls": [
                {"name": "recorded_nondeterministic", "repeat_stable": False, "completed": True,
                 "deterministic_blocked": False, "fixed_inputs_unchanged": True,
                 "probes": [{"detailed_evidence": "retained in canonical case"}]},
                {"name": "strict_deterministic", "repeat_stable": True, "completed": True,
                 "deterministic_blocked": False, "fixed_inputs_unchanged": True}]}
        result["diagnostic_reporting_error"] = "OSError: artifact disk unavailable"
        result["determinism_blocker"] = {"error_type": "RuntimeError", "message": "No deterministic kernel"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            self.write_case(path, result)
            summary = summary_for_parent(result, path, directory)
        self.assertTrue(summary["comparison"]["elementwise_passed"])
        self.assertFalse(summary["comparison"]["relative_l2_passed"])
        self.assertEqual(summary["comparison"]["failure_reasons"], first["failure_reasons"])
        self.assertTrue(summary["localization"]["coverage_complete"])
        self.assertFalse(summary["localization"]["context_succeeded"])
        self.assertFalse(summary["repeatability"]["repeat_stable"])
        self.assertTrue(summary["repeatability"]["deterministic_blocked"])
        self.assertTrue(summary["repeatability"]["cuda_execution_observed"])
        controls = summary["repeatability"]["controls"]
        self.assertFalse(controls[0]["repeat_stable"])
        self.assertTrue(controls[1]["repeat_stable"])
        self.assertEqual(controls[1]["name"], "strict_deterministic")
        self.assertNotIn("probes", controls[0])
        self.assertEqual(summary["diagnostic_reporting_error"], result["diagnostic_reporting_error"])
        self.assertEqual(summary["error"], "NumericalComparisonError: gradient/input failed")
        self.assertEqual(summary["determinism_blocker"], result["determinism_blocker"])

    def test_module_has_no_torch_or_optional_dependency_requirement(self):
        completed = subprocess.run([sys.executable, "-S", "-c",
            "import sys; from examples.vae import diagnostic_reporting; assert 'torch' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
