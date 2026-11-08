"""Fused quantization dispatch preserves validation and metadata-only execution."""

import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from deepopt import DiagnosticPolicy, TrainingRuntime, autocast
from deepopt.opt.do_nn.functional import do_linear
from deepopt.opt.do_nn.AOTAutograd_Func import linear_autograd_func
from deepopt.opt.policy import fp8_plan
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.opt.quantization.quantizer import _quantize_tensorwise_pair
from deepopt.opt.quantization.quantizer import _quantize_tensorwise_transpose
from deepopt.opt.quantization.tensor import QuantizedTensor
from deepopt.opt.scaling_strategy.amax import AmaxScaling


class FP8FusionPolicyTests(unittest.TestCase):
    def test_transpose_only_quantization_keeps_scalar_recipe_and_skips_unused_allocation(self):
        source = torch.randn(37, 65).T
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            actual = _quantize_tensorwise_transpose(source, fp8_dtype=dtype)
            expected = quantize_fp8(source, fp8_dtype=dtype).transpose()
            torch.testing.assert_close(actual.data.float(), expected.data.float(), rtol=0, atol=0)
            torch.testing.assert_close(actual.scale_inv, expected.scale_inv, rtol=0, atol=0)
        with FakeTensorMode(), patch.dict(sys.modules, {"deepopt.opt.backends.triton_quantization": None}), \
             patch.object(torch, "empty_strided", wraps=torch.empty_strided) as allocate:
            source = torch.empty(129, 65, device="cuda")
            transposed = _quantize_tensorwise_transpose(source, fp8_dtype=torch.float8_e4m3fn, check_finite=False)
        requested = [call for call in allocate.call_args_list
                     if torch.device(call.kwargs["device"]).type == "cuda"]
        self.assertEqual(len(requested), 1)
        self.assertEqual(transposed.data.shape, (65, 129))
        self.assertEqual(transposed.data.stride(), (144, 1))

    def test_tensorwise_pair_has_one_scale_and_exactly_transposed_fp8_values(self):
        for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):
            x = torch.randn(37, 65)
            normal, transposed = _quantize_tensorwise_pair(x, fp8_dtype=fp8)
            expected = quantize_fp8(x, fp8_dtype=fp8)
            torch.testing.assert_close(normal.data.float(), expected.data.float(), rtol=0, atol=0)
            torch.testing.assert_close(transposed.data.float(), normal.data.T.float(), rtol=0, atol=0)
            torch.testing.assert_close(transposed.scale_inv, normal.scale_inv, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "tensorwise"):
            _quantize_tensorwise_pair(x, block_shape=(1, 32))

    def test_paired_weight_cache_is_distinct_and_invalidates_after_updates(self):
        from deepopt.opt.quantization.cache import WeightCache

        cache = WeightCache()
        weight = nn.Parameter(torch.randn(37, 65))
        ordinary = cache.quantize(weight, fp8_dtype=torch.float8_e4m3fn, block_shape=None, quantizer=quantize_fp8)
        pair = cache.quantize(weight, fp8_dtype=torch.float8_e4m3fn, block_shape=None,
                              quantizer=_quantize_tensorwise_pair, paired=True)
        self.assertIsInstance(ordinary, QuantizedTensor)
        self.assertIs(pair, cache.quantize(weight, fp8_dtype=torch.float8_e4m3fn, block_shape=None,
                                           quantizer=_quantize_tensorwise_pair, paired=True))
        with torch.no_grad():
            weight.add_(1)
        updated = cache.quantize(weight, fp8_dtype=torch.float8_e4m3fn, block_shape=None,
                                 quantizer=_quantize_tensorwise_pair, paired=True)
        self.assertIsNot(updated, pair)
        self.assertEqual(len(cache._entries), 1)

    def test_fake_cuda_pair_uses_tma_ready_strides_and_mode_checks(self):
        with FakeTensorMode(), patch.dict(sys.modules, {"deepopt.opt.backends.triton_quantization": None}):
            x = torch.empty(129, 65, device="cuda", dtype=torch.bfloat16)
            normal, transposed = _quantize_tensorwise_pair(x, check_finite=False)
            self.assertEqual(normal.data.stride(), (80, 1))
            self.assertEqual(transposed.data.stride(), (144, 1))
            self.assertEqual(normal.data.shape, (129, 65))
            self.assertEqual(transposed.data.shape, (65, 129))
            self.assertIs(normal.scale_inv, transposed.scale_inv)
    def test_fake_cuda_dequantization_avoids_fp32_copies_and_expanded_scales(self):
        with FakeTensorMode(), patch.dict(sys.modules, {"deepopt.opt.backends.triton_quantization": None}):
            data = torch.empty(65, 37, device="cuda", dtype=torch.float8_e4m3fn).T
            inverse = torch.empty_strided((19, 22), (44, 2), device="cuda")
            value = QuantizedTensor(data, inverse, (2, 3))
            with patch.object(torch.Tensor, "float", side_effect=AssertionError("FP32 copy")), \
                 patch.object(torch.Tensor, "repeat_interleave", side_effect=AssertionError("expanded scales")), \
                 patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("dense copy")):
                actual = value.dequantize()
            self.assertEqual(actual.shape, data.shape)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(actual.device, data.device)

    def test_runtime_modes_reach_standalone_cuda_scaling(self):
        runtime = TrainingRuntime(nn.Identity(), execution_mode="max_speed")
        checked = []
        original = AmaxScaling._validate_input

        def validate(strategy, x, dtype, finite):
            checked.append(finite)
            return original(strategy, x, dtype, finite)

        with FakeTensorMode(), patch.object(AmaxScaling, "_validate_input", new=validate):
            x = torch.empty(257, 259, device="cuda", dtype=torch.bfloat16).T
            for mode in ("max_speed", "analyze", "max_speed"):
                runtime.set_execution_mode(mode)
                with runtime.autocast():
                    AmaxScaling((256, 256)).compute(x)
                    AmaxScaling().compute(x, check_finite=True)
                    AmaxScaling().compute(x, check_finite=False)
            self.assertEqual(checked, [False, True, False, True, True, False, False, True, False])

    def test_runtime_check_cadences_and_overrides_reach_fused_cuda_dispatch(self):
        runtime = TrainingRuntime(nn.Identity(), diagnostics=DiagnosticPolicy(
            loss_finite_every=2, gradient_finite_every=3))
        checked = []
        original = AmaxScaling._validate_input

        def validate(strategy, x, dtype, finite):
            checked.append(finite)
            return original(strategy, x, dtype, finite)

        with FakeTensorMode(), patch.object(AmaxScaling, "_validate_input", new=validate):
            x = torch.empty(7, 65, device="cuda")
            for update in range(1, 7):
                runtime.mode_controller.completed_updates = update - 1
                with runtime.autocast():
                    quantize_fp8(x, block_shape=(1, 32))
                with runtime.diagnostic_scope("backward", phase="backward"):
                    quantize_fp8(x, block_shape=(1, 32))
            self.assertEqual(checked, [flag for update in range(1, 7)
                                       for flag in (update % 2 == 0, update % 3 == 0)])
            checked.clear()
            for mode in ("max_speed", "analyze", "max_speed"):
                runtime.set_execution_mode(mode)
                with runtime.autocast():
                    quantize_fp8(x)
                    quantize_fp8(x, check_finite=True)
                    with autocast("cpu", enabled=False, fp8=None, check_finite=False):
                        quantize_fp8(x)
            self.assertEqual(checked, [False, True, False, True, True, False, False, True, False])

    def test_fake_cuda_quantization_produces_only_output_metadata_without_triton(self):
        with patch.dict(sys.modules, {"deepopt.opt.backends.triton_quantization": None}), FakeTensorMode():
            for shape in ((37, 65), (129, 97), (257, 259)):
                for blocks in (None, (1, 32), (32, 32), (128, 128), (256, 256), (1024, 1024)):
                    x = torch.empty(shape, device="cuda", dtype=torch.bfloat16).T
                    with patch.object(AmaxScaling, "compute", side_effect=AssertionError("unfused probe")):
                        result = quantize_fp8(x, block_shape=blocks, check_finite=False)
                    expected = ((1, 1) if blocks is None else
                                tuple((size + block - 1) // block for size, block in zip(x.shape, blocks)))
                    self.assertEqual(result.data.shape, x.shape)
                    self.assertEqual(result.scale_inv.shape, expected)
                    self.assertEqual(result.data.device, x.device)
                    self.assertEqual(result.data.dtype, torch.float8_e4m3fn)
                    self.assertEqual(result.scale_inv.dtype, torch.float32)
                    self.assertTrue(result.data.is_contiguous())

    def test_fused_dispatch_keeps_cuda_structural_validation(self):
        with FakeTensorMode():
            x = torch.empty(2, 7, device="cuda")
            for kwargs, message in (({"block_shape": (1, 1)}, "must not"),
                                    ({"block_shape": (3, 5)}, "powers of two"),
                                    ({"check_finite": 1}, "boolean"),
                                    ({"fp8_dtype": torch.float16}, "Unsupported FP8")):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex((ValueError, TypeError), message):
                    quantize_fp8(x, **kwargs)
            with self.assertRaisesRegex(TypeError, "Input must"):
                quantize_fp8(x.to(torch.float64), check_finite=False)

    def test_training_quantizes_original_transpose_views_without_dense_copies(self):
        plan = fp8_plan({"enabled": True, "block_shapes": {
            "activation": (1, 32), "weight": (32, 32), "grad": (1, 32)}})
        model = nn.Linear(65, 10)
        x = torch.randn(37, 65, requires_grad=True)
        grad = torch.randn(37, 10)
        with patch.object(linear_autograd_func, "quantize_fp8", wraps=quantize_fp8) as quantize:
            do_linear(x, model.weight, model.bias, plan=plan).backward(grad)
        transposes = [call.args[0] for call in quantize.call_args_list
                      if tuple(call.args[0].shape) in ((65, 37), (10, 37))]
        self.assertEqual(len(transposes), 2)
        self.assertTrue(all(not tensor.is_contiguous() for tensor in transposes))
        self.assertEqual(transposes[0].data_ptr(), x.data_ptr())
        self.assertEqual(transposes[1].data_ptr(), grad.data_ptr())
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(model.weight.grad)
