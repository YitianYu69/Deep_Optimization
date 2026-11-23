"""Packed replay diagnostics keep CPU semantics separate from CUDA evidence."""

import json
import unittest
from unittest.mock import patch

import torch

from examples.vae import diagnostic_operators as diagnostics


class PackedOperatorDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = diagnostics.run_packed_diagnostics(device="cpu")

    def test_real_cpu_eager_and_native_ir_replays_are_metadata_only(self):
        report = self.report
        self.assertTrue(report["passed"], json.dumps(report, allow_nan=False))
        self.assertEqual(report["status"], "unqualified")
        self.assertEqual(report["execution_scope"], "CPU_semantic_only")
        self.assertFalse(report["qualified"])
        self.assertFalse(report["cuda_execution_verified"])
        self.assertFalse(report["timing_eligible"])
        self.assertEqual(report["tensor_checkpoints_written"], 0)
        self.assertEqual(len(report["cases"]), 6)
        for case in report["cases"]:
            self.assertEqual(case["native"]["owner"], "DeepOpt.AOTAutoDiff")
            self.assertTrue(case["native"]["same_saved_components_bound"])
            self.assertEqual(len(case["native"]["gradient_targets"]), 2)
            self.assertIn("mul", case["native"]["backward_before_fusion"])
            self.assertFalse(case["native"]["selected_patterns"])
            self.assertTrue(all(row["reason"] == "device" and not row["selected"]
                                for row in case["native"]["compiler_decisions"]))
            self.assertFalse(case["native_packed_cuda_execution_verified"])
            for route, state in zip(case["eager_execution"]["multiply_backward"], case["saved_coefficients"]):
                if state == "packed":
                    self.assertEqual((route["route"], route["rejection_reason"]), ("materialized", "device"))
        json.dumps(report, allow_nan=False)

    def test_both_packed_operands_scalar_dtype_and_predeclared_budgets(self):
        cases = {case["name"]: case for case in self.report["cases"]}
        for name in ("se_both_packed_channels_last", "swiglu_both_packed_tail", "output_tile_tail_both_packed"):
            self.assertEqual(cases[name]["saved_coefficients"], ["packed", "packed"])
        scalar = cases["residual_fp32_scalar"]
        self.assertEqual(scalar["saved_coefficients"], ["exact", "packed"])
        for values in scalar["comparisons"].values():
            self.assertEqual(values[1]["actual_dtype"], "torch.float32")
            self.assertEqual(values[1]["expected_dtype"], "torch.float32")
            self.assertEqual(values[1]["budget"], {"rtol": .015, "atol": .015, "relative_l2_limit": .015})
            self.assertEqual(values[1]["full_model_storage_budget_context"]["relative_l2_limit"],
                             8 * torch.finfo(torch.float32).eps)
        direct = cases["swiglu_both_packed_tail"]["comparisons"]["shared_vs_oracle"]
        self.assertTrue(all(value["bitwise_equal"] for value in direct))
        self.assertTrue(all(value["budget"]["rtol"] == 0 for value in direct))
        self.assertEqual(direct[0]["full_model_storage_budget_context"], {
            "rtol": 2 * torch.finfo(torch.bfloat16).eps, "atol": 2 * torch.finfo(torch.float32).eps,
            "relative_l2_limit": torch.finfo(torch.bfloat16).eps})
        self.assertEqual(cases["scratch_cap_tail"]["effective_scratch_cap_bytes"], 128)
        cancellation = cases["reduction_grid_stride_cancellation"]
        self.assertEqual(cancellation["cotangent_recipe"], "alternating_noninteger_BF16")
        self.assertGreater(cancellation["comparisons"]["native_vs_oracle"][1]["nonzero_reference"], 0)

    def test_only_one_eager_invocation_produces_each_saved_pack(self):
        from deepopt.opt.nn.binary import quantize_pack_activation

        with patch("deepopt.opt.nn.binary.quantize_pack_activation", wraps=quantize_pack_activation) as pack:
            report = diagnostics.run_packed_diagnostics(device="cpu")
        self.assertTrue(report["passed"])
        expected = sum(case["saved_coefficients"].count("packed") for case in report["cases"])
        self.assertEqual(pack.call_count, expected)

    def test_caller_rng_is_preserved(self):
        before = torch.get_rng_state().clone()
        report = diagnostics.run_packed_diagnostics(device="cpu", seed=73)
        self.assertTrue(report["passed"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_unavailable_cuda_and_invalid_requests_are_unqualified(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            report = diagnostics.run_packed_diagnostics(device="cuda:0")
        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "unqualified")
        self.assertFalse(report["cases"])
        self.assertIn("unavailable", report["error"]["message"])
        for options in ({"device": "meta"}, {"group_size": 512}, {"scratch_bytes": 0}, {"seed": True}):
            with self.subTest(options=options):
                report = diagnostics.run_packed_diagnostics(**options)
                self.assertEqual(report["status"], "unqualified")
                self.assertFalse(report["passed"])
                self.assertFalse(report["cases"])

    def test_case_exceptions_preserve_original_error_and_continue(self):
        execute = diagnostics._run_case
        def fail_first(case, row, **options):
            if case["name"] == "se_both_packed_channels_last":
                raise ArithmeticError("diagnostic probe failure")
            return execute(case, row, **options)

        with patch.object(diagnostics, "_run_case", side_effect=fail_first):
            report = diagnostics.run_packed_diagnostics(device="cpu")
        self.assertFalse(report["passed"])
        first = report["cases"][0]
        self.assertEqual(first["status"], "execution_failure")
        self.assertEqual(first["error"]["type"], "builtins.ArithmeticError")
        self.assertEqual(first["error"]["message"], "diagnostic probe failure")
        self.assertIn("ArithmeticError", first["error"]["traceback"])
        self.assertTrue(all(case["passed"] for case in report["cases"][1:]))

    def test_numerical_case_failures_do_not_stop_later_cases(self):
        compare = diagnostics._compare
        calls = 0
        def fail_first(*args, **kwargs):
            nonlocal calls
            calls += 1
            result = compare(*args, **kwargs)
            if calls == 1:
                result["passed"] = False
            return result

        with patch.object(diagnostics, "_compare", side_effect=fail_first):
            report = diagnostics.run_packed_diagnostics(device="cpu")
        self.assertFalse(report["passed"])
        self.assertEqual(report["cases"][0]["status"], "numerical_failure")
        self.assertTrue(all(case["passed"] for case in report["cases"][1:]))

    def test_full_target_is_explicit_and_never_allocated_on_cpu(self):
        report = diagnostics.run_packed_diagnostics(device="cpu", full_shape=True)
        self.assertEqual(len(report["cases"]), 7)
        target = report["cases"][-1]
        self.assertEqual(target["shape"], [128, 768, 224, 224])
        self.assertEqual(target["status"], "unqualified")
        self.assertFalse(target["cuda_execution_verified"])
        self.assertFalse(report["passed"])
        self.assertNotIn("operands", target)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA packed replay diagnostics require a GPU")
class PackedOperatorCudaDiagnosticTests(unittest.TestCase):
    def test_real_cuda_eager_shared_and_native_packed_routes(self):
        report = diagnostics.run_packed_diagnostics(device="cuda:0")
        self.assertTrue(report["passed"], json.dumps(report, allow_nan=False))
        self.assertTrue(report["cuda_execution_verified"])
        for case in report["cases"]:
            self.assertTrue(case["native_packed_cuda_execution_verified"])
            for route, state in zip(case["eager_execution"]["multiply_backward"], case["saved_coefficients"]):
                if state == "packed":
                    self.assertEqual(route["route"], "packed_streaming")
                    self.assertLessEqual(route["scratch_bytes"], case["effective_scratch_cap_bytes"])
            self.assertTrue(case["native"]["selected_patterns"])
            self.assertTrue(all(row["selected"] for row in case["native"]["compiler_decisions"]))


if __name__ == "__main__":
    unittest.main()
