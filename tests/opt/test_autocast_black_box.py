"""Public AMP/FP8 scopes on ordinary PyTorch models and existing DO modules."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import FP8Config, autocast
from deepopt.opt.nn.functional import do_linear
from deepopt.opt.nn.linear import DOLinear
from deepopt.opt.policy import fp8_plan
from tests.opt.fp8_support import round_fp8


class AutocastBlackBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(811)
        self.x = torch.randn(5, 7)
        self.weight = torch.randn(4, 7)
        self.bias = torch.randn(4)

    def test_native_linear_forward_and_backward_use_fp8_without_transformation(self):
        settings = FP8Config(enabled=True, grad_dtype="e5m2",
                             block_shapes={"activation": (2, 3), "weight": (3, 4), "grad": (4, 2)})
        module = nn.Linear(7, 4)
        x = self.x.clone().requires_grad_()
        original_weight = module.weight
        original_type = type(module)
        with autocast("cpu", enabled=False, fp8=settings):
            output = module(x)
        grad = torch.randn_like(output)
        output.backward(grad)  # The backward policy survives scope exit.

        qx = round_fp8(x, torch.float8_e4m3fn, (2, 3))
        qw = round_fp8(module.weight, torch.float8_e4m3fn, (3, 4))
        qg = round_fp8(grad, torch.float8_e5m2, (4, 2))
        torch.testing.assert_close(output, qx @ qw.T + module.bias)
        torch.testing.assert_close(x.grad, qg @ qw)
        torch.testing.assert_close(module.weight.grad, qg.T @ qx)
        torch.testing.assert_close(module.bias.grad, grad.sum(0))
        self.assertIs(type(module), original_type)
        self.assertIs(module.weight, original_weight)

    def test_functional_positional_and_keyword_calls(self):
        plan = fp8_plan(FP8Config(enabled=True))
        expected = do_linear(self.x, self.weight, self.bias, plan=plan)
        with autocast("cpu", enabled=False, fp8=True):
            outputs = (
                F.linear(self.x, self.weight, self.bias),
                F.linear(input=self.x, weight=self.weight, bias=self.bias),
                F.linear(self.x, weight=self.weight, bias=self.bias),
            )
        for actual in outputs:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(expected, F.linear(self.x, self.weight, self.bias)))

    def test_fp8_dictionary_enables_without_explicit_enabled_field(self):
        config = {"backend": "reference", "activation_dtype": "e5m2", "weight_dtype": "e5m2"}
        expected = do_linear(self.x, self.weight, plan=fp8_plan(FP8Config(enabled=True, **config)))
        with autocast("cpu", enabled=False, fp8=config):
            actual = F.linear(self.x, self.weight)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_nested_disable_inherit_and_reenable_restore_the_outer_policy(self):
        dense = F.linear(self.x, self.weight, self.bias)
        quantized = do_linear(self.x, self.weight, self.bias, plan=fp8_plan(FP8Config(enabled=True)))
        with autocast("cpu", enabled=False, fp8=True):
            with autocast("cpu", enabled=False):
                torch.testing.assert_close(F.linear(self.x, self.weight, self.bias), quantized)
            with autocast("cpu", enabled=False, fp8=False):
                torch.testing.assert_close(F.linear(self.x, self.weight, self.bias), dense, rtol=0, atol=0)
                with autocast("cpu", enabled=False, fp8=True):
                    torch.testing.assert_close(F.linear(self.x, self.weight, self.bias), quantized)
                torch.testing.assert_close(F.linear(self.x, self.weight, self.bias), dense, rtol=0, atol=0)
            torch.testing.assert_close(F.linear(self.x, self.weight, self.bias), quantized)
        torch.testing.assert_close(F.linear(self.x, self.weight, self.bias), dense, rtol=0, atol=0)

    def test_existing_do_module_obeys_override_and_dense_disable(self):
        module = DOLinear.from_linear(nn.Linear(7, 4), plan=fp8_plan(FP8Config(enabled=True)))
        original_plan = deepcopy(module.plan)
        override = FP8Config(enabled=True, activation_dtype="e5m2", weight_dtype="e5m2")
        expected = do_linear(self.x, module.weight, module.bias, plan=fp8_plan(override))
        with autocast("cpu", enabled=False, fp8=override):
            torch.testing.assert_close(module(self.x), expected)
            with autocast("cpu", enabled=False, fp8=False):
                torch.testing.assert_close(module(self.x), F.linear(self.x, module.weight, module.bias),
                                           rtol=0, atol=0)
        self.assertEqual(module.plan, original_plan)

    def test_native_bfloat16_amp_and_fp8_have_independent_switches(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            dense_bf16 = F.linear(self.x, self.weight, self.bias)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False):
            actual_dense = F.linear(self.x, self.weight, self.bias)
        torch.testing.assert_close(actual_dense, dense_bf16, rtol=0, atol=0)

        plan = fp8_plan(FP8Config(enabled=True))
        expected_fp32 = do_linear(self.x, self.weight, self.bias, plan=plan)
        with autocast("cpu", dtype=torch.bfloat16, fp8=True):
            actual_fp8_bf16 = F.linear(self.x, self.weight, self.bias)
        torch.testing.assert_close(actual_fp8_bf16, expected_fp32.to(torch.bfloat16), rtol=0, atol=0)
        with autocast("cpu", enabled=False, fp8=True):
            actual_fp8_fp32 = F.linear(self.x, self.weight, self.bias)
        torch.testing.assert_close(actual_fp8_fp32, expected_fp32, rtol=0, atol=0)

    def test_fp8_out_dtype_can_override_native_amp_output(self):
        with autocast("cpu", dtype=torch.bfloat16, fp8={"out_dtype": "fp32"}):
            actual = F.linear(self.x, self.weight, self.bias)
        self.assertEqual(actual.dtype, torch.float32)
        expected = do_linear(self.x, self.weight, self.bias,
                             plan=fp8_plan(FP8Config(enabled=True, out_dtype="fp32")))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_decorator_is_reusable_and_backward_remains_available(self):
        @autocast("cpu", enabled=False, fp8=True)
        def projection(x, weight):
            return F.linear(x, weight)

        expected = do_linear(self.x, self.weight, plan=fp8_plan(FP8Config(enabled=True)))
        self.assertEqual(projection.__name__, "projection")
        for _ in range(2):
            x = self.x.clone().requires_grad_()
            result = projection(x, self.weight)
            torch.testing.assert_close(result, expected)
            result.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_fp64_and_empty_linear_operands_keep_native_semantics(self):
        double_x, double_weight = self.x.double(), self.weight.double()
        expected = F.linear(double_x, double_weight)
        with autocast("cpu", enabled=False, fp8=True):
            torch.testing.assert_close(F.linear(double_x, double_weight), expected, rtol=0, atol=0)
            empty = F.linear(torch.empty(0, 7), self.weight)
        self.assertEqual(empty.shape, (0, 4))

    def test_sparse_weights_keep_the_native_linear_path(self):
        weight = self.weight.masked_fill(self.weight.abs() < .5, 0).to_sparse()
        expected = F.linear(self.x, weight)
        with autocast("cpu", enabled=False, fp8=True):
            actual = F.linear(self.x, weight)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unrelated_matmul_and_attention_are_unchanged(self):
        query = torch.randn(2, 2, 3, 4)
        matmul = self.x @ self.weight.T
        attention = F.scaled_dot_product_attention(query, query, query)
        with autocast("cpu", enabled=False, fp8=True):
            torch.testing.assert_close(self.x @ self.weight.T, matmul, rtol=0, atol=0)
            torch.testing.assert_close(F.scaled_dot_product_attention(query, query, query),
                                       attention, rtol=0, atol=0)

    def test_concurrent_threads_keep_distinct_fp8_policies(self):
        barrier = Barrier(2)
        dense = F.linear(self.x, self.weight)
        expected = do_linear(self.x, self.weight, plan=fp8_plan(FP8Config(enabled=True)))

        def run(enabled):
            with autocast("cpu", enabled=False, fp8=enabled):
                barrier.wait(timeout=10)
                result = F.linear(self.x, self.weight)
                barrier.wait(timeout=10)
                return result

        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = pool.submit(run, True), pool.submit(run, False)
            torch.testing.assert_close(first.result(timeout=15), expected, rtol=0, atol=0)
            torch.testing.assert_close(second.result(timeout=15), dense, rtol=0, atol=0)
        torch.testing.assert_close(F.linear(self.x, self.weight), dense, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
