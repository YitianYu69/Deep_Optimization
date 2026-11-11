"""Scoped policy lifetime, interception boundaries, and lazy public exports."""

import subprocess
import sys
import unittest
from contextvars import copy_context
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import FP8Config, autocast
from deepopt.opt._autocast import _get_fp8_plan, _FP8LinearMode, fp8_context
from deepopt.opt.nn import functional
from deepopt.opt.nn.autograd import linear as linear_autograd_func
from deepopt.opt.nn.linear import DOLinear
from deepopt.opt.policy import fp8_plan


class AutocastWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(813)
        self.x = torch.randn(5, 7)
        self.weight = torch.randn(4, 7)

    def test_exception_restores_fp8_and_native_amp_cache_state(self):
        previous_enabled = torch.is_autocast_enabled("cpu")
        previous_cache = torch.is_autocast_cache_enabled()
        self.assertIsNone(_get_fp8_plan("cpu"))
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with autocast("cpu", dtype=torch.bfloat16, cache_enabled=not previous_cache, fp8=True):
                self.assertTrue(torch.is_autocast_enabled("cpu"))
                self.assertIsInstance(_get_fp8_plan("cpu"), dict)
                raise RuntimeError("intentional")
        self.assertIsNone(_get_fp8_plan("cpu"))
        self.assertEqual(torch.is_autocast_enabled("cpu"), previous_enabled)
        self.assertEqual(torch.is_autocast_cache_enabled(), previous_cache)

    def test_failed_enter_does_not_leak_native_amp_or_outer_fp8_policy(self):
        with autocast("cpu", enabled=False, fp8=True):
            outer = _get_fp8_plan("cpu")
            with self.assertRaises(ValueError):
                with autocast("cpu", dtype=torch.bfloat16, fp8={"backend": "invalid"}):
                    self.fail("Invalid configuration entered")
            self.assertIs(_get_fp8_plan("cpu"), outer)
            self.assertFalse(torch.is_autocast_enabled("cpu"))
        self.assertIsNone(_get_fp8_plan("cpu"))

    def test_identical_nested_policy_does_not_install_another_mode(self):
        with patch.object(_FP8LinearMode, "__enter__", autospec=True,
                          side_effect=_FP8LinearMode.__enter__) as enter:
            with autocast("cpu", enabled=False, fp8=True):
                with fp8_context("cpu", FP8Config(enabled=True)):
                    self.assertIsInstance(_get_fp8_plan("cpu"), dict)
                with autocast("cpu", enabled=False, fp8=None):
                    self.assertIsInstance(_get_fp8_plan("cpu"), dict)
        self.assertEqual(enter.call_count, 1)

    def test_fp8_only_scope_preserves_native_amp_and_device_specific_policy(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with fp8_context("cpu", True):
                self.assertTrue(torch.is_autocast_enabled("cpu"))
                self.assertEqual(F.linear(self.x, self.weight).dtype, torch.bfloat16)
        with fp8_context("cuda", True):
            self.assertIsNone(_get_fp8_plan("cpu"))
            self.assertIsInstance(_get_fp8_plan("cuda"), dict)
            with patch.object(functional, "do_linear", wraps=functional.do_linear) as converted:
                F.linear(self.x, self.weight)
            converted.assert_not_called()

    def test_context_does_not_monkeypatch_torch_or_change_parameter_registration(self):
        native_linear, native_forward = F.linear, nn.Linear.forward
        model = nn.Linear(7, 4)
        parameters = tuple(model.parameters())
        keys = tuple(model.state_dict())
        with autocast("cpu", enabled=False, fp8=True):
            model(self.x)
            self.assertIs(F.linear, native_linear)
            self.assertIs(nn.Linear.forward, native_forward)
        self.assertEqual(tuple(model.state_dict()), keys)
        self.assertTrue(all(actual is saved for actual, saved in zip(model.parameters(), parameters)))

    def test_existing_do_module_is_quantized_only_once_in_nested_enabled_scopes(self):
        model = DOLinear.from_linear(nn.Linear(7, 4), plan=fp8_plan(FP8Config(enabled=True)))
        with patch.object(linear_autograd_func, "quantize_fp8",
                          wraps=linear_autograd_func.quantize_fp8) as quantizer:
            with autocast("cpu", enabled=False, fp8=True):
                with autocast("cpu", enabled=False, fp8=True):
                    model(self.x)
        self.assertEqual(quantizer.call_count, 2)  # activation and weight, one forward

    def test_disabled_scope_bypasses_fp8_autograd_for_existing_do_module(self):
        model = DOLinear.from_linear(nn.Linear(7, 4), plan=fp8_plan(FP8Config(enabled=True)))
        with patch.object(linear_autograd_func, "quantize_fp8",
                          side_effect=AssertionError("FP8 quantizer was called")):
            with autocast("cpu", enabled=False, fp8=False):
                model(self.x).sum().backward()
        self.assertIsNotNone(model.weight.grad)

    def test_caller_config_mutation_cannot_change_active_plan_or_saved_backward(self):
        config = {"grad_dtype": "e5m2", "block_shapes": {
            "activation": (2, 3), "weight": (2, 3), "grad": (2, 3),
        }}
        x = self.x.clone().requires_grad_()
        weight = self.weight.clone().requires_grad_()
        with autocast("cpu", enabled=False, fp8=config):
            result = F.linear(x, weight)
            config["grad_dtype"] = "e4m3fn"
            config["block_shapes"]["grad"] = None
            self.assertEqual(_get_fp8_plan("cpu")["block_shapes"]["grad"], (2, 3))
        with patch.object(linear_autograd_func, "quantize_fp8",
                          wraps=linear_autograd_func.quantize_fp8) as quantizer:
            result.sum().backward()
        self.assertEqual(quantizer.call_args.kwargs["fp8_dtype"], torch.float8_e5m2)
        self.assertEqual(quantizer.call_args.kwargs["block_shape"], (2, 3))

    def test_reentrant_context_instance_restores_outer_scope(self):
        scope = autocast("cpu", enabled=False, fp8=True)
        with scope:
            outer = _get_fp8_plan("cpu")
            with scope:
                self.assertIs(_get_fp8_plan("cpu"), outer)
            self.assertIs(_get_fp8_plan("cpu"), outer)
        self.assertIsNone(_get_fp8_plan("cpu"))
        with scope:
            self.assertIsInstance(_get_fp8_plan("cpu"), dict)

    def test_repeated_scopes_do_not_accumulate_context_variables(self):
        with autocast("cpu", enabled=False, fp8=True):
            pass
        initial = len(copy_context())
        for _ in range(20):
            with autocast("cpu", enabled=False, fp8=True):
                pass
        self.assertEqual(len(copy_context()), initial)

    def test_malformed_linear_arguments_keep_native_errors(self):
        with autocast("cpu", enabled=False, fp8=True):
            with self.assertRaises(TypeError):
                F.linear(self.x, self.weight, input=self.x)
            with self.assertRaises(TypeError):
                F.linear(self.x, self.weight, unsupported=True)

    def test_config_exports_remain_torch_free_and_context_exports_do_not_collide(self):
        programs = (
            "import sys; from deepopt import FP8Config; from deepopt.opt import FP8Config as Other; "
            "assert FP8Config is Other; assert 'torch' not in sys.modules",
            "from deepopt.opt.nn.functional import do_linear; import deepopt, deepopt.opt; "
            "from deepopt.opt._autocast import autocast; "
            "assert deepopt.autocast is autocast; assert deepopt.opt.autocast is autocast",
        )
        for code in programs:
            with self.subTest(code=code):
                result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                        text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
