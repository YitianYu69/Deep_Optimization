"""Accumulation precision is explicit, validated, and propagated to every GEMM."""

import unittest
from unittest.mock import patch

import torch

from deepopt.config import FP8Config, ConfigError, fp8_config_from_dict
from deepopt.opt.backends import gemm_nt
from deepopt.opt.do_nn.AOTAutograd_Func import linear_autograd_func
from deepopt.opt.do_nn.functional import do_linear, validate_fp8_plan
from deepopt.opt.policy import fp8_plan
from deepopt.opt.quantization.quantizer import quantize_fp8


class FP8AccumulationPolicyTests(unittest.TestCase):
    def test_default_is_unchanged_and_valid_limits_round_trip(self):
        self.assertEqual(FP8Config().max_num_imprecise_acc, 32)
        for limit in (32, 64, 128, 256, 1024):
            config = fp8_config_from_dict({"enabled": True, "max_num_imprecise_acc": limit})
            self.assertEqual(fp8_plan(config)["fp8_training"]["max_num_imprecise_acc"], limit)

    def test_invalid_limits_fail_at_config_plan_and_dispatch_boundaries(self):
        operand = quantize_fp8(torch.ones(2, 3))
        for limit in (True, None, "128", 32., 0, -32, 31, 33, 2147483648):
            with self.subTest(limit=limit):
                with self.assertRaises(ConfigError):
                    fp8_config_from_dict({"max_num_imprecise_acc": limit})
                plan = fp8_plan(True)
                plan["fp8_training"]["max_num_imprecise_acc"] = limit
                with self.assertRaisesRegex(ValueError, "max_num_imprecise_acc"):
                    validate_fp8_plan(plan)
                with self.assertRaisesRegex(ValueError, "max_num_imprecise_acc"):
                    gemm_nt(operand, operand, backend="reference", max_num_imprecise_acc=limit)

    def test_forward_and_both_gradients_receive_the_selected_policy(self):
        x = torch.randn(7, 5, requires_grad=True)
        weight = torch.randn(3, 5, requires_grad=True)
        plan = fp8_plan(FP8Config(enabled=True, max_num_imprecise_acc=128))
        with patch.object(linear_autograd_func, "gemm_nt", wraps=gemm_nt) as calls:
            do_linear(x, weight, plan=plan).square().sum().backward()
        self.assertEqual(len(calls.call_args_list), 3)
        self.assertTrue(all(call.kwargs["max_num_imprecise_acc"] == 128 for call in calls.call_args_list))
