"""One functional linear entry point selects compute and saved-input policy."""

from copy import deepcopy
from contextlib import ExitStack
from itertools import product
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from deepopt import autocast, reset_config
from deepopt.opt.backends import gemm_nt
from deepopt.opt.nn.functional import do_linear
from deepopt.opt.policy import linear_plan
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.opt.quantization.tensor import QuantizedTensor


def native_oracle(x, weight, bias, grad, compressed, group_size):
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
        output = F.linear(x, weight, bias)
    dx, dw, db = torch.autograd.grad(output, (x, weight, bias), grad)
    if compressed:
        restored = dequantize_unpack_activation(quantize_pack_activation(x.detach().bfloat16().reshape(-1, weight.shape[1]),
                                                    group_size), torch.bfloat16)
        dw = (grad.reshape(-1, weight.shape[0]).T @ restored).float()
    return output.detach(), dx, dw, db


def fp8_oracle(x, weight, bias, grad, compressed, group_size):
    qx = quantize_fp8(x.detach().reshape(-1, weight.shape[1]))
    qw = quantize_fp8(weight.detach())
    qg = quantize_fp8(grad.reshape(-1, weight.shape[0]))
    output = gemm_nt(qx, qw, backend="reference", bias=bias.detach(), out_dtype=torch.bfloat16)
    dx = gemm_nt(qg, qw.transpose(contiguous=False), backend="reference", out_dtype=x.dtype).reshape(x.shape)
    if compressed:
        packed = quantize_pack_activation(qx.data.float(), group_size)
        restored = dequantize_unpack_activation(packed, torch.float8_e4m3fn, transpose=True)
        saved = QuantizedTensor(restored, qx.scale_inv.T)
    else:
        saved = qx.transpose(contiguous=False)
    dw = gemm_nt(qg.transpose(contiguous=False), saved, backend="reference",
                 out_dtype=torch.bfloat16).float()
    return output.reshape(*x.shape[:-1], weight.shape[0]), dx, dw, grad.float().reshape(-1, weight.shape[0]).sum(0)


class UnifiedLinearTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(392)
        reset_config()
        self.addCleanup(reset_config)

    def tensors(self):
        return (torch.randn(2, 3, 7, requires_grad=True),
                torch.randn(5, 7, requires_grad=True),
                torch.randn(5, requires_grad=True),
                torch.randn(2, 3, 5, dtype=torch.bfloat16))

    def assert_projection(self, call, fp8, compressed, group_size=17):
        x, weight, bias, grad = self.tensors()
        oracle = fp8_oracle if fp8 else native_oracle
        expected = oracle(x, weight, bias, grad, compressed, group_size)
        saved = []
        target = "deepopt.opt.nn.autograd.linear.gemm_nt"
        with ExitStack() as stack:
            gemms = stack.enter_context(patch(target, wraps=gemm_nt))
            packs = [stack.enter_context(patch(target, wraps=quantize_pack_activation)) for target in
                     ("deepopt.opt.nn.autograd.native_linear.quantize_pack_activation",
                      "deepopt.opt.quantization.activation_fp8.quantize_pack_activation")]
            unpacks = [stack.enter_context(patch(target, wraps=dequantize_unpack_activation)) for target in
                       ("deepopt.opt.nn.autograd.native_linear.dequantize_unpack_activation",
                        "deepopt.opt.quantization.activation.dequantize_unpack_activation")]
            with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value,
                                                        lambda value: value):
                output = call(x, weight, bias)
            # Backward deliberately outlives both AMP and DeepOpt policy scopes.
            output.backward(grad)
        for actual, reference in zip((output, x.grad, weight.grad, bias.grad), expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(gemms.call_count, 3 if fp8 else 0)
        self.assertEqual(sum(call.call_count for call in packs), int(compressed))
        self.assertEqual(sum(call.call_count for call in unpacks), int(compressed))
        self.assertEqual(any(value.dtype == torch.int32 for value in saved), compressed)
        self.assertEqual(any(value.dtype == torch.float8_e4m3fn for value in saved), fp8)
        if compressed:
            self.assertFalse(any(value.data_ptr() == x.data_ptr() for value in saved))

    def test_explicit_plan_selects_all_compute_and_compression_combinations(self):
        for fp8, compressed in product((False, True), repeat=2):
            with self.subTest(fp8=fp8, compressed=compressed):
                plan = linear_plan(fp8, activation_compression={"enabled": compressed, "group_size": 17},
                                   config={})

                def call(x, weight, bias):
                    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
                        return do_linear(x, weight, bias, plan=plan)

                self.assert_projection(call, fp8, compressed)

    def test_nn_functional_linear_supports_the_pytorch_input_keyword(self):
        import deepopt.opt.nn as do_nn

        x, weight, bias, _ = self.tensors()
        for fp8 in (False, True):
            plan = linear_plan(fp8, activation_compression=False, config={})
            actual = do_nn.functional.linear(input=x, weight=weight, bias=bias, plan=plan)
            expected = do_linear(x, weight, bias, plan=plan)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_configuration_independently_selects_compute_and_compression(self):
        for fp8, compressed in product((False, True), repeat=2):
            with self.subTest(fp8=fp8, compressed=compressed):
                config = {"runtime": {"fp8": {"enabled": fp8},
                                      "activation_compression": {"enabled": compressed, "group_size": 17}}}
                plan = linear_plan(config=config)

                def call(x, weight, bias):
                    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
                        return do_linear(x, weight, bias, plan=plan)

                self.assert_projection(call, fp8, compressed)

    def test_scoped_torch_linear_uses_the_same_policy_combinations(self):
        for fp8, compressed in product((False, True), repeat=2):
            with self.subTest(fp8=fp8, compressed=compressed):
                def call(x, weight, bias):
                    with autocast("cpu", dtype=torch.bfloat16, fp8=fp8, cache_enabled=False,
                                  activation_compression={"enabled": compressed, "group_size": 17}, config={}):
                        return F.linear(x, weight, bias)

                self.assert_projection(call, fp8, compressed)

    def test_omitted_plan_uses_native_linear_and_inherits_active_scope(self):
        x, weight, bias, _ = self.tensors()
        torch.testing.assert_close(do_linear(x, weight, bias), F.linear(x, weight, bias), rtol=0, atol=0)
        for fp8, compressed in product((False, True), repeat=2):
            with self.subTest(fp8=fp8, compressed=compressed):
                def call(x, weight, bias):
                    with autocast("cpu", dtype=torch.bfloat16, fp8=fp8, cache_enabled=False,
                                  activation_compression={"enabled": compressed, "group_size": 17}, config={}):
                        return do_linear(x, weight, bias)

                self.assert_projection(call, fp8, compressed)

    def test_explicit_native_plan_owns_policy_inside_an_fp8_scope(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                plan = linear_plan(False, activation_compression={"enabled": compressed, "group_size": 17},
                                   config={})

                def call(x, weight, bias):
                    with autocast("cpu", dtype=torch.bfloat16, fp8=True,
                                  activation_compression=not compressed, cache_enabled=False, config={}):
                        return do_linear(x, weight, bias, plan=plan)

                self.assert_projection(call, False, compressed)

    def test_explicit_fp8_plan_still_obeys_scoped_fp8_and_codec_disable(self):
        plan = linear_plan(True, activation_compression={"enabled": True, "group_size": 17}, config={})
        for fp8, compressed in ((False, True), (True, False)):
            with self.subTest(fp8=fp8, compressed=compressed):
                def call(x, weight, bias):
                    with autocast("cpu", dtype=torch.bfloat16, fp8=fp8, cache_enabled=False,
                                  activation_compression={"enabled": compressed, "group_size": 17}, config={}):
                        return do_linear(x, weight, bias, plan=plan)

                self.assert_projection(call, fp8, compressed)

    def test_plan_builder_owns_configuration_and_can_be_reused(self):
        config = {"runtime": {"fp8": {"enabled": False},
                              "activation_compression": {"enabled": True, "group_size": 17}}}
        plan = linear_plan(config=config)
        original = deepcopy(plan)
        config["runtime"]["activation_compression"]["group_size"] = 1
        self.assertEqual(plan, original)
        self.assertEqual(plan, {"fp8": False, "activation_compression": {
            "enabled": True, "group_size": 17, "bits": 2, "attention_bits": None, "norm_xhat_recenter": False}})
        for _ in range(2):
            def call(x, weight, bias):
                with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
                    return do_linear(x, weight, bias, plan=plan)

            self.assert_projection(call, False, True)
        self.assertEqual(plan, original)

    def test_reused_native_plan_and_scope_avoid_policy_resolution_in_the_hot_path(self):
        x, weight, _, _ = self.tensors()
        plan = linear_plan(False, activation_compression={"enabled": True, "group_size": 17}, config={})
        scope = autocast("cpu", dtype=torch.bfloat16, fp8=False, cache_enabled=False, config={},
                         activation_compression={"enabled": True, "group_size": 17})
        with scope:
            expected = F.linear(x, weight).detach()

        # Policy construction/validation is allowed before execution. Reusing
        # the plan and autocast scope must dispatch without repeating that work.
        targets = ("deepopt.opt.nn.functional.validate_linear_plan",
                   "deepopt.opt.nn.functional.validate_fp8_plan",
                   "deepopt.opt.nn.functional.compressed_linear",
                   "deepopt.opt.policy.resolve_config",
                   "deepopt.opt._autocast.resolve_config")
        with ExitStack() as stack:
            for target in targets:
                stack.enter_context(patch(target, side_effect=AssertionError(f"hot path reached {target}")))
            for _ in range(2):
                with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
                    torch.testing.assert_close(do_linear(x, weight, plan=plan), expected, rtol=0, atol=0)
                with scope:
                    torch.testing.assert_close(F.linear(x, weight), expected, rtol=0, atol=0)

    def test_native_empty_input_keeps_native_shape_and_gradients(self):
        x = torch.empty(2, 0, 7, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(5, 7, dtype=torch.bfloat16, requires_grad=True)
        plan = linear_plan(False, activation_compression=True, config={})
        output = do_linear(x, weight, plan=plan)
        self.assertEqual(tuple(output.shape), (2, 0, 5))
        output.sum().backward()
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight))
        self.assertEqual(tuple(x.grad.shape), tuple(x.shape))

    def test_native_codec_is_not_saved_without_weight_gradient_or_autograd(self):
        plan = linear_plan(False, activation_compression=True, config={})
        x = torch.randn(3, 7, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(5, 7, dtype=torch.bfloat16)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value,
                                                    lambda value: value):
            do_linear(x, weight, plan=plan).sum().backward()
        self.assertFalse(any(value.dtype == torch.int32 for value in saved))
        with torch.no_grad():
            self.assertFalse(do_linear(x, weight, plan=plan).requires_grad)

    def test_malformed_native_policy_is_rejected(self):
        x, weight, _, _ = self.tensors()
        for plan in ({"fp8": True}, {"fp8": 0}, {"fp8": False, "typo": True},
                     {"fp8": False, "activation_compression": {"group_size": 0}},
                     {"fp8": False, "activation_compression": {"bits": 3}}):
            with self.subTest(plan=plan), self.assertRaises((TypeError, ValueError)):
                do_linear(x, weight, plan=plan)


if __name__ == "__main__":
    unittest.main()
