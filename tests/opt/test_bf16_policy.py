"""Versioned storage-policy ownership, independent of target GPU qualification."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import unittest
from unittest.mock import patch
import subprocess
import sys

import torch
from torch import nn
from torch.nn import functional as F

from deepopt._bf16_policy import (BF16Policy, active_bf16_policy, bf16_policy_scope,
                                 current_bf16_policy, resolve_bf16_policy)
from deepopt._config_resolution import ResolvedConfig, resolve_config
from deepopt.config import ConfigError, RuntimeConfig, config_from_dict
from deepopt.opt._autocast import autocast
from deepopt.opt._recipe import bf16_recipe
from deepopt.opt.policy import linear_plan, prepare_fp8_model
from deepopt.opt.policy import normalization_plan


class BF16PolicyTests(unittest.TestCase):
    def test_descriptor_is_frozen_and_strict(self):
        policy = BF16Policy("bf16_storage")
        self.assertTrue(policy.storage_enabled)

        self.assertEqual(resolve_bf16_policy(policy.as_dict()), policy)
        for value in (True, "bf16", {"name": "torch_amp"}, {"name": "torch_amp", "version": True},
                      {"name": "torch_amp", "version": 2}, {**policy.as_dict(), "extra": False}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_bf16_policy(value)
        with self.assertRaises(FrozenInstanceError):
            policy.name = "torch_amp"
        descriptor = policy.as_dict()
        descriptor["name"] = "torch_amp"
        self.assertTrue(policy.storage_enabled)

    def test_config_policy_import_does_not_initialize_torch(self):
        result = subprocess.run([sys.executable, "-c", """
import sys
from deepopt.config import RuntimeConfig
from deepopt._config_resolution import resolve_config
from deepopt._bf16_policy import BF16Policy
assert RuntimeConfig().bf16_policy == 'torch_amp'
assert resolve_config({'runtime': {'bf16_policy': 'bf16_storage'}}, base=None)['runtime']['bf16_policy'] == 'bf16_storage'
assert BF16Policy().version == 1
assert 'torch' not in sys.modules
"""], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_scopes_are_device_local_and_restore_on_exception(self):
        self.assertIsNone(active_bf16_policy("cpu"))
        with bf16_policy_scope("bf16_storage", "cpu"):
            self.assertTrue(current_bf16_policy("cpu").storage_enabled)
            self.assertFalse(current_bf16_policy("cuda").storage_enabled)
            with self.assertRaisesRegex(RuntimeError, "test"):
                with bf16_policy_scope("torch_amp", "cpu"):
                    raise RuntimeError("test")
            self.assertTrue(current_bf16_policy("cpu").storage_enabled)
        self.assertFalse(current_bf16_policy("cpu").storage_enabled)

    def test_config_defaults_and_complete_cross_field_validation(self):
        base = {"name": "policy", "seed": 1,
                "training": {"epochs": 1, "batch_size": 1, "learning_rate": 0.1}}
        self.assertEqual(RuntimeConfig().bf16_policy, "torch_amp")
        self.assertEqual(config_from_dict(base).runtime.bf16_policy, "torch_amp")
        runtime = {"device": "cuda", "precision": "bf16", "bf16_policy": "bf16_storage"}
        self.assertEqual(config_from_dict({**base, "runtime": runtime}).runtime.bf16_policy, "bf16_storage")
        for change in ({"precision": "fp32"}, {"fp8": {"enabled": True}}):
            with self.subTest(change=change), self.assertRaises(ConfigError):
                config_from_dict({**base, "runtime": {**runtime, **change}})
        restored = config_from_dict(asdict(config_from_dict({**base, "runtime": runtime})))
        self.assertEqual(restored.runtime.bf16_policy, "bf16_storage")

    def test_partial_snapshot_does_not_invent_precision(self):
        snapshot = resolve_config({"runtime": {"bf16_policy": "bf16_storage"}}, base=None)
        self.assertEqual(snapshot["runtime"]["bf16_policy"], "bf16_storage")
        self.assertNotIn("precision", snapshot["runtime"])
        self.assertNotIn("device", snapshot["runtime"])
        self.assertNotIn("bf16_policy", resolve_config({"runtime": {}}, base=None)["runtime"])

    def test_autocast_nested_and_disabled_scope(self):
        with autocast("cpu", dtype=torch.bfloat16, bf16_policy="bf16_storage", config={}):
            self.assertTrue(current_bf16_policy("cpu").storage_enabled)
            with autocast("cpu"):
                self.assertTrue(current_bf16_policy("cpu").storage_enabled)
            with autocast("cpu", enabled=False):
                self.assertFalse(current_bf16_policy("cpu").storage_enabled)
                self.assertEqual(torch.ones(1, dtype=torch.bfloat16).dtype, torch.bfloat16)
                with autocast("cpu", enabled=True, dtype=torch.bfloat16):
                    self.assertTrue(current_bf16_policy("cpu").storage_enabled)
                self.assertFalse(current_bf16_policy("cpu").storage_enabled)
            self.assertTrue(current_bf16_policy("cpu").storage_enabled)
        self.assertFalse(current_bf16_policy("cpu").storage_enabled)

    def test_autocast_rejects_non_bf16_and_fp8(self):
        with self.assertRaisesRegex(ValueError, "BF16 autocast"):
            with autocast("cpu", dtype=torch.float16, bf16_policy="bf16_storage", config={}):
                pass
        with self.assertRaises(ValueError):
            autocast("cpu", dtype=torch.bfloat16, fp8=True, bf16_policy="bf16_storage", config={})
        with autocast("cpu", dtype=torch.bfloat16, fp8={"enabled": True, "backend": "reference"}, config={}):
            with self.assertRaisesRegex(ValueError, "FP8 disabled"):
                with autocast("cpu", dtype=torch.bfloat16, fp8=None, bf16_policy="bf16_storage", config={}):
                    self.fail("Scope must reject inherited FP8 before effects")
        self.assertFalse(current_bf16_policy("cpu").storage_enabled)

    def test_owned_plan_and_model_preserve_parameter_identity(self):
        config = ResolvedConfig({"runtime": {"bf16_policy": "bf16_storage"}})
        plan = linear_plan(config=config, activation_compression={"enabled": False, "group_size": 1024})
        self.assertEqual(plan["bf16_policy"], BF16Policy("bf16_storage").as_dict())
        self.assertFalse(plan["activation_compression"]["enabled"])
        model = nn.Sequential(nn.Linear(17, 67), nn.GELU())
        parameters = tuple(model.parameters())
        prepared, report = prepare_fp8_model(model, config=config)
        self.assertIsNotNone(report)
        self.assertEqual(tuple(map(id, prepared.parameters())), tuple(map(id, parameters)))
        self.assertEqual(prepared._deepopt_bf16_policy, plan["bf16_policy"])
        with autocast("cpu", dtype=torch.bfloat16, config={}):
            output = prepared(torch.randn(3, 17))
        self.assertEqual(output.dtype, torch.bfloat16)
        output.float().sum().backward()
        self.assertTrue(all(parameter.grad.dtype == torch.float32 for parameter in parameters))
        copied = deepcopy(prepared)
        with autocast("cpu", dtype=torch.bfloat16, config={}):
            self.assertEqual(copied(torch.randn(3, 17)).dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "differs"):
            prepare_fp8_model(prepared, config={"runtime": {"bf16_policy": "torch_amp"}})

    def test_prepared_scope_disable_keeps_explicit_input_dtype(self):
        prepared, _ = prepare_fp8_model(nn.GELU(), config={"runtime": {"bf16_policy": "bf16_storage"}})
        value = torch.randn(3, 1031)
        with autocast("cpu", dtype=torch.bfloat16, config={}, bf16_policy="bf16_storage"):
            with autocast("cpu", enabled=False):
                result = prepared(value)
                self.assertEqual(result.dtype, torch.float32)
        self.assertFalse(current_bf16_policy("cpu").storage_enabled)

    def test_default_and_storage_recipes_are_distinct(self):
        self.assertEqual(bf16_recipe(), bf16_recipe("torch_amp"))
        self.assertNotEqual(bf16_recipe(), bf16_recipe("bf16_storage"))

    def test_storage_admission_does_not_enable_compression(self):
        with autocast("cpu", dtype=torch.bfloat16, bf16_policy="bf16_storage", config={},
                      activation_compression={"enabled": False, "group_size": 1024}), \
                patch("deepopt.opt.nn.functional.do_linear", return_value=torch.zeros(3, 67)) as call:
            F.linear(torch.randn(3, 17), torch.randn(67, 17))
        plan = call.call_args.kwargs["plan"]
        self.assertFalse(plan["activation_compression"]["enabled"])
        self.assertEqual(plan["bf16_policy"], BF16Policy("bf16_storage").as_dict())

    def test_owned_storage_rejects_ambient_fp8_before_forward(self):
        class Probe(nn.Module):
            def forward(self, value):
                raise AssertionError("forward must not execute")

        model, _ = prepare_fp8_model(Probe(), config={"runtime": {"bf16_policy": "bf16_storage"}})
        with autocast("cpu", dtype=torch.bfloat16, fp8={"enabled": True, "backend": "reference"}, config={}):
            with self.assertRaisesRegex(ValueError, "FP8 disabled"):
                model(torch.ones(3, 1031, dtype=torch.bfloat16))

    def test_functional_owned_normalization_policy_executes_without_root_wrapper(self):
        from deepopt.opt.nn.functional import do_group_norm

        plan = normalization_plan(config={"runtime": {"bf16_policy": "bf16_storage"}},
                                  activation_compression={"enabled": False, "group_size": 1024})
        value = torch.randn(2, 4, 257, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = do_group_norm(value, 2, plan=plan)
        self.assertEqual(output.dtype, torch.bfloat16)
        output.float().square().sum().backward()
        self.assertEqual(value.grad.dtype, torch.float32)

    def test_prepared_policy_binds_integer_model_inputs_without_changing_them(self):
        class Probe(nn.Module):
            def forward(self, indices):
                return current_bf16_policy(indices.device.type), indices

        model, _ = prepare_fp8_model(Probe(), config={"runtime": {"bf16_policy": "bf16_storage"}})
        indices = torch.arange(7, dtype=torch.int64)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            policy, result = model(indices)
        self.assertTrue(policy.storage_enabled)
        self.assertIs(result, indices)
        self.assertEqual(result.dtype, torch.int64)

    def test_model_exp_preserves_exact_input_adjoint_and_fp32_loss_island(self):
        torch.manual_seed(5402)
        value = torch.randn(4096, dtype=torch.bfloat16, requires_grad=True)
        gradient = torch.randn_like(value)
        with autocast("cpu", dtype=torch.bfloat16, config={}, bf16_policy="bf16_storage",
                      activation_compression={"enabled": False, "group_size": 1024}):
            output = torch.exp(value)
            loss_operand = value.detach().float().requires_grad_()
            loss_output = loss_operand.exp()
        with torch.autocast("cpu", enabled=False):
            output.backward(gradient)
            loss_output.sum().backward()
        coefficient = value.detach().float().exp()
        torch.testing.assert_close(output, coefficient.bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(value.grad, (gradient.float() * coefficient).bfloat16(), rtol=0, atol=0)
        self.assertEqual(loss_output.dtype, torch.float32)
        torch.testing.assert_close(loss_operand.grad, loss_output.detach(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
