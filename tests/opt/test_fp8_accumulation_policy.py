"""Native accumulation is a versioned numerical policy, with no forced limit."""

import ast
from dataclasses import asdict
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from deepopt.config import FP8Config, ConfigError, fp8_config_from_dict
from deepopt.opt.backends import gemm_nt
from deepopt.opt.do_nn.AOTAutograd_Func import linear_autograd_func
from deepopt.opt.do_nn.functional import do_linear, validate_fp8_plan
from deepopt.opt.policy import fp8_plan


class FP8AccumulationPolicyTests(unittest.TestCase):
    def test_removed_setting_has_an_explicit_migration_error(self):
        self.assertNotIn("max_num_imprecise_acc", asdict(FP8Config()))
        for limit in (32, 128, None):
            with self.assertRaisesRegex(ConfigError, "removed.*native Triton"):
                fp8_config_from_dict({"max_num_imprecise_acc": limit})
            plan = fp8_plan(True)
            plan["fp8_training"]["max_num_imprecise_acc"] = limit
            with self.assertRaisesRegex(ValueError, "removed"):
                validate_fp8_plan(plan)

    def test_forward_and_both_gradients_leave_accumulation_to_the_backend(self):
        x = torch.randn(7, 5, requires_grad=True)
        weight = torch.randn(3, 5, requires_grad=True)
        with patch.object(linear_autograd_func, "gemm_nt", wraps=gemm_nt) as calls:
            do_linear(x, weight, plan=fp8_plan(True)).square().sum().backward()
        self.assertEqual(len(calls.call_args_list), 3)
        self.assertTrue(all("max_num_imprecise_acc" not in call.kwargs for call in calls.call_args_list))
        self.assertEqual(x.grad.dtype, x.dtype)
        self.assertEqual(weight.grad.dtype, weight.dtype)

    def test_native_dot_calls_omit_limit_and_keep_chained_fp32_accumulator(self):
        path = Path(__file__).resolve().parents[2] / "src/deepopt/opt/backends/triton_fp8.py"
        tree = ast.parse(path.read_text())
        dots = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "dot"]
        self.assertEqual(len(dots), 3)
        self.assertTrue(all("max_num_imprecise_acc" not in {kw.arg for kw in node.keywords} for node in dots))
        self.assertEqual(ast.unparse(dots[0].args[2]), "acc")
        self.assertEqual(ast.unparse(dots[0].keywords[0].value), "tl.float32")
