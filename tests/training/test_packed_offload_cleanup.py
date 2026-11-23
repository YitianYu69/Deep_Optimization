"""A transport cleanup failure cannot mask a training failure or retain grads."""

import unittest
from unittest.mock import patch

import torch

from deepopt._packed_offload import PackedActivationOffloadPolicy
from deepopt.engine.execution.runtime import TrainingRuntime


class PackedOffloadRuntimeFailureTests(unittest.TestCase):
    def test_inherited_owned_policy_is_preserved_in_runtime_configuration(self):
        model = torch.nn.Linear(2, 1)
        owned = PackedActivationOffloadPolicy(host_pinned_bytes=1024, device_staging_bytes=1024, d2h_inflight_bytes=1024)
        model._deepopt_packed_activation_offload_policy = owned
        runtime = TrainingRuntime(model, device="cpu", checks=False, config={"runtime": {"precision": "fp32"}})
        self.assertEqual(runtime.config["runtime"]["packed_activation_offload"], owned.as_config())
        self.assertIsNone(runtime._packed_offload_policy)

    def test_inherited_enabled_policy_cannot_bypass_CPU_rejection(self):
        model = torch.nn.Linear(2, 1)
        model._deepopt_packed_activation_offload_policy = PackedActivationOffloadPolicy(enabled=True,
            host_pinned_bytes=1024, device_staging_bytes=1024, d2h_inflight_bytes=1024)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            TrainingRuntime(model, device="cpu", checks=False, config={"runtime": {"precision": "fp32",
                "compile": {"enabled": True, "backend": "deepopt", "dynamic": False,
                            "options": {"use_native_autodiff": True}}}})

    def make_runtime(self):
        model = torch.nn.Linear(2, 1)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.01), device="cpu", checks=False)
        # Explicit fault injection into cleanup only. No CPU offload execution.
        runtime._packed_offload_policy = PackedActivationOffloadPolicy(enabled=True, host_pinned_bytes=1024,
            device_staging_bytes=1024, d2h_inflight_bytes=1024)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        runtime._forward_pending = runtime._pending_gradients = True
        return runtime

    def test_primary_failure_survives_secondary_transport_drain_failure(self):
        runtime = self.make_runtime()
        primary = ValueError("kernel-primary")
        with patch("deepopt.compiler.runtime.manager.invalidate_packed_offload_runtime", side_effect=RuntimeError("drain-secondary")):
            runtime._mark_failed(primary_error=primary)
        self.assertTrue(runtime._failed)
        self.assertFalse(runtime._forward_pending)
        self.assertFalse(runtime._pending_gradients)
        self.assertTrue(all(parameter.grad is None for parameter in runtime.model.parameters()))
        self.assertIn("drain-secondary", "\n".join(primary.__notes__))

    def test_explicit_invalidation_reports_drain_error_after_clearing_state(self):
        runtime = self.make_runtime()
        with patch("deepopt.compiler.runtime.manager.invalidate_packed_offload_runtime", side_effect=RuntimeError("drain-secondary")):
            with self.assertRaisesRegex(RuntimeError, "drain-secondary"):
                runtime.invalidate()
        self.assertTrue(runtime._failed)
        self.assertTrue(all(parameter.grad is None for parameter in runtime.model.parameters()))
        self.assertFalse(runtime._forward_pending or runtime._pending_gradients)


if __name__ == "__main__":
    unittest.main()
