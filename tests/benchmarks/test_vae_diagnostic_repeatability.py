"""Fixed-input operator repeatability reports never imply CUDA qualification."""

import json
import unittest
from unittest.mock import patch

import torch

from examples.vae import diagnostic_repeatability as diagnostics


class RepeatabilityDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = diagnostics.run_repeatability_diagnostics("cpu", repeats=3, original_shapes=False)

    def test_real_cpu_replays_are_complete_stable_and_metadata_only(self):
        report = self.report
        self.assertTrue(report["completed"], json.dumps(report, allow_nan=False))
        self.assertTrue(report["passed"])
        self.assertTrue(report["repeat_stable"])
        self.assertEqual(report["status"], "unqualified")
        self.assertEqual(report["execution_scope"], "CPU_semantic_only")
        self.assertFalse(report["qualification_eligible"])
        self.assertFalse(report["cuda_qualified"])
        self.assertFalse(report["cuda_execution_observed"])
        self.assertFalse(report["math_correctness_assessed"])
        self.assertEqual(report["tensor_checkpoints_written"], 0)
        self.assertEqual(report["geometry"]["source_shape"], [1, 8, 5, 7])
        self.assertEqual(report["geometry"]["output_shape"], [1, 8, 3, 4])
        self.assertEqual(report["geometry"]["pads"], [1, 2, 1, 2])
        self.assertEqual([row["name"] for row in report["controls"]],
                         ["recorded_nondeterministic", "strict_deterministic"])
        for control in report["controls"]:
            self.assertEqual(len(control["probes"]), 4)
            self.assertTrue(control["fixed_inputs_unchanged"])
            self.assertTrue(control["comparisons"]["raw_vs_metadata_first_outputs"]["bitwise_equal"])
            for probe in control["probes"]:
                self.assertEqual(probe["output"]["dtype"], "torch.bfloat16")
                self.assertTrue(probe["same_arguments_each_repeat"])
                self.assertEqual(len(probe["repetitions"]), 3)
                self.assertEqual(len({row["logical_sha256"] for row in probe["repetitions"]}), 1)
                self.assertFalse(probe["cuda_execution_observed"])
        json.dumps(report, allow_nan=False)

    def test_identical_cotangents_are_reused_across_policies_and_padding_abis(self):
        nondeterministic, strict = self.report["controls"]
        for left, right in zip(nondeterministic["probes"], strict["probes"]):
            self.assertEqual(left["inputs"], right["inputs"])
        left, right = nondeterministic["probes"][:2]
        self.assertEqual(left["inputs"]["cotangent"], right["inputs"]["cotangent"])
        self.assertEqual(self.report["fixed_inputs"]["weights"]["shape"], [8, 1, 4, 4])

    def test_flags_and_caller_rng_are_restored(self):
        previous = diagnostics._flags()
        before = torch.get_rng_state().clone()
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            torch.utils.deterministic.fill_uninitialized_memory = False
            expected = diagnostics._flags()
            report = diagnostics.run_repeatability_diagnostics("cpu", seed=73, repeats=2, original_shapes=False)
            self.assertTrue(report["passed"])
            self.assertEqual(diagnostics._flags(), expected)
            self.assertEqual(report["initial_flags"], expected)
            self.assertTrue(report["caller_flags_restored"])
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
        finally:
            torch.use_deterministic_algorithms(previous["deterministic_algorithms"],
                                             warn_only=previous["deterministic_warn_only"])
            torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
            torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
            torch.utils.deterministic.fill_uninitialized_memory = previous["fill_uninitialized_memory"]

    def test_deterministic_rejection_is_data_and_never_falls_back(self):
        from deepopt.opt.nn.padding import padding_adjoint

        def reject_strict(*args):
            if torch.are_deterministic_algorithms_enabled():
                raise RuntimeError("mock operator does not have a deterministic implementation")
            return padding_adjoint(*args)

        with patch("deepopt.opt.nn.padding.padding_adjoint", side_effect=reject_strict) as call:
            report = diagnostics.run_repeatability_diagnostics("cpu", repeats=2, original_shapes=False)
        self.assertTrue(report["completed"])
        self.assertTrue(report["passed"])
        self.assertTrue(report["deterministic_blocked"])
        self.assertIsNone(report["repeat_stable"])
        strict = report["controls"][1]["probes"][1]
        self.assertEqual(strict["status"], "deterministic_blocked")
        self.assertEqual(strict["error"]["type"], "builtins.RuntimeError")
        self.assertFalse(strict["repetitions"])
        self.assertEqual(call.call_count, 3)
        self.assertTrue(report["caller_flags_restored"])

    def test_unexpected_failure_retains_original_error_and_other_probes_continue(self):
        with patch("deepopt.opt.nn.padding.padding_adjoint", side_effect=ValueError("fixture failure")):
            report = diagnostics.run_repeatability_diagnostics("cpu", repeats=2, original_shapes=False)
        self.assertFalse(report["completed"])
        self.assertFalse(report["passed"])
        for control in report["controls"]:
            failed = control["probes"][1]
            self.assertEqual(failed["error"]["type"], "builtins.ValueError")
            self.assertEqual(failed["error"]["message"], "fixture failure")
            self.assertIn("ValueError", failed["error"]["traceback"])
            self.assertTrue(all(row["completed"] for i, row in enumerate(control["probes"]) if i != 1))
        self.assertTrue(report["caller_flags_restored"])

    def test_unexpected_runtime_error_mentioning_determinism_is_not_a_policy_rejection(self):
        before = diagnostics._flags()
        with patch("deepopt.opt.nn.padding.padding_adjoint",
                   side_effect=RuntimeError("unexpected deterministic state corruption")):
            report = diagnostics.run_repeatability_diagnostics("cpu", repeats=2, original_shapes=False)
        self.assertFalse(report["completed"])
        self.assertFalse(report["passed"])
        self.assertFalse(report["deterministic_blocked"])
        for control in report["controls"]:
            self.assertEqual(control["probes"][1]["status"], "execution_failure")
            self.assertEqual(control["probes"][1]["error"]["type"], "builtins.RuntimeError")
        self.assertTrue(report["caller_flags_restored"])
        self.assertEqual(diagnostics._flags(), before)

    def test_variation_is_measured_by_region_without_failing_execution(self):
        from deepopt.opt.nn.padding import padding_adjoint

        calls = 0
        def vary(*args):
            nonlocal calls
            calls += 1
            output = padding_adjoint(*args).clone()
            if calls % 2 == 0:
                output[0, 0, 1, 1] += 1  # Both reflected axes: corner.
                output[0, 0, 1, 0] += 1  # One reflected axis: edge.
                output[0, 0, 0, 0] += 1  # Neither: interior.
            return output

        with patch("deepopt.opt.nn.padding.padding_adjoint", side_effect=vary):
            report = diagnostics.run_repeatability_diagnostics("cpu", repeats=3, original_shapes=False)
        self.assertTrue(report["completed"])
        self.assertTrue(report["passed"])
        self.assertFalse(report["repeat_stable"])
        varied = report["controls"][0]["probes"][1]["repetitions"][1]
        self.assertEqual(varied["mismatched_elements"], 3)
        self.assertEqual(varied["mismatched_regions"], {"corner": 1, "edge": 1, "interior": 1})
        self.assertGreater(varied["max_abs"], 0)
        self.assertGreater(varied["relative_l2"], 0)
        self.assertFalse(report["qualification_eligible"])

    def test_invalid_requests_and_missing_cuda_remain_unqualified(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            unavailable = diagnostics.run_repeatability_diagnostics()
        self.assertFalse(unavailable["completed"])
        self.assertFalse(unavailable["cuda_execution_observed"])
        self.assertFalse(unavailable["controls"])
        self.assertIn("unavailable", unavailable["error"]["message"])
        for options in ({"device": "meta"}, {"seed": True}, {"repeats": 1}, {"original_shapes": 1}):
            with self.subTest(options=options):
                report = diagnostics.run_repeatability_diagnostics(**options)
                self.assertFalse(report["passed"])
                self.assertFalse(report["controls"])
                self.assertEqual(report["status"], "unqualified")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA repeatability evidence requires a GPU")
class RepeatabilityCudaDiagnosticTests(unittest.TestCase):
    def test_original_encoder12_geometry_repeats_and_reports_policy_rejections(self):
        report = diagnostics.run_repeatability_diagnostics("cuda:0", repeats=3)
        self.assertTrue(report["completed"], json.dumps(report, allow_nan=False))
        self.assertTrue(report["passed"])
        self.assertTrue(report["cuda_execution_observed"])
        self.assertEqual(report["geometry"]["source_shape"], [1, 512, 56, 56])
        self.assertEqual(report["geometry"]["padded_shape"], [1, 512, 59, 59])
        self.assertEqual(report["geometry"]["output_shape"], [1, 512, 28, 28])
        self.assertTrue(report["caller_flags_restored"])
        self.assertFalse(report["qualification_eligible"])
        for control in report["controls"]:
            for probe in control["probes"]:
                if probe["deterministic_blocked"]:
                    self.assertEqual(probe["error"]["type"], "builtins.RuntimeError")
                    self.assertEqual(probe["status"], "deterministic_blocked")
                else:
                    self.assertEqual(probe["output"]["dtype"], "torch.bfloat16")
                    self.assertTrue(probe["cuda_execution_observed"])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
