"""Eager dtype diagnostics never retain checkpoint tensors or qualify timing."""

import gc
import json
import unittest
import weakref

import torch
from torch import nn

from deepopt import autocast
from deepopt.opt.policy import prepare_fp8_model
from deepopt.opt.precision.diagnostics import DTypeEdgeRecorder


class DTypeDiagnosticsTests(unittest.TestCase):
    def test_snapshot_dtype_and_policy_survive_scope_exit_without_tensor_retention(self):
        model, _ = prepare_fp8_model(nn.Sequential(nn.Linear(17, 67), nn.GELU()),
                                    config={"runtime": {"bf16_policy": "bf16_storage"}})
        value = torch.randn(3, 17)
        value_ref = weakref.ref(value)
        with torch.no_grad(), autocast("cpu", dtype=torch.bfloat16, config={}):
            with DTypeEdgeRecorder(model) as recorder:
                result = model(value)
        result_ref = weakref.ref(result)
        report = recorder.report()
        self.assertTrue(report["complete"])
        self.assertFalse(report["timing_eligible"])
        root_output, = (edge for edge in report["edges"]
                        if edge["module"] == "<root>" and edge["role"] == "output")
        self.assertEqual(root_output["dtype"], "torch.bfloat16")
        self.assertEqual(root_output["bf16_policy"], {"name": "bf16_storage", "version": 1})
        json.dumps(report, allow_nan=False)
        del value, result
        gc.collect()
        self.assertIsNone(value_ref())
        self.assertIsNone(result_ref())
        self.assertFalse(model._forward_hooks)
        self.assertFalse(model._forward_pre_hooks)

    def test_bounds_are_explicit_and_report_is_owned(self):
        model = nn.Sequential(nn.Identity(), nn.Identity())
        with DTypeEdgeRecorder(model, max_edges=2) as recorder:
            model(torch.randn(2, 1031))
        report = recorder.report()
        self.assertFalse(report["complete"])
        self.assertEqual(report["edge_count"], 2)
        self.assertGreater(report["omitted"], 0)
        report["edges"].clear()
        self.assertEqual(recorder.report()["edge_count"], 2)

    def test_complete_inventory_can_fail_nearest_storage_contract(self):
        from deepopt._bf16_policy import bf16_policy_scope
        from examples.vae.qualification import assert_dtype_edge_contracts

        class BrokenNearest(nn.Upsample):
            def forward(self, value):
                return value.float()

        model = nn.Sequential(BrokenNearest(scale_factor=2, mode="nearest"))
        with bf16_policy_scope("bf16_storage", "cpu"), DTypeEdgeRecorder(model) as recorder:
            model(torch.ones(1, 2, 3, 5, dtype=torch.bfloat16))
        report = recorder.report()
        self.assertTrue(report["complete"])
        validation = report["storage_policy_validation"]
        self.assertFalse(validation["valid"])
        self.assertEqual(validation["checked_edges"], 1)
        self.assertEqual(validation["contracts"][0]["actual_dtype"], "torch.float32")
        with self.assertRaisesRegex(RuntimeError, "activation dtype contract"):
            assert_dtype_edge_contracts(report)

    def test_legitimate_fp32_output_is_separate_from_nearest_edge_contract(self):
        from deepopt._bf16_policy import bf16_policy_scope
        from examples.vae.qualification import assert_dtype_edge_contracts

        class FP32Statistic(nn.Module):
            def forward(self, value):
                return value.float().mean()

        model = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), FP32Statistic())
        with bf16_policy_scope("bf16_storage", "cpu"), DTypeEdgeRecorder(model) as recorder:
            result = model(torch.ones(1, 2, 3, 5, dtype=torch.bfloat16))
        report = recorder.report()
        assert_dtype_edge_contracts(report)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(report["storage_policy_validation"]["checked_edges"], 1)
        self.assertFalse(report["storage_policy_validation"]["full_activation_graph_verified"])

    def test_exception_cleans_hooks_and_does_not_mask_original_error(self):
        class Broken(nn.Module):
            def forward(self, value):
                raise RuntimeError("original failure")

        model = Broken()
        with self.assertRaisesRegex(RuntimeError, "original failure"):
            with DTypeEdgeRecorder(model) as recorder:
                model(torch.randn(2, 1031))
        self.assertFalse(recorder.report()["complete"])
        self.assertFalse(model._forward_hooks)
        self.assertFalse(model._forward_pre_hooks)

    def test_compiled_model_is_rejected_before_hooks(self):
        model = nn.Identity()
        model._compiled_call_impl = lambda value: value
        with self.assertRaisesRegex(ValueError, "separate eager diagnostic"):
            with DTypeEdgeRecorder(model):
                pass
        self.assertFalse(model._forward_hooks)


if __name__ == "__main__":
    unittest.main()
