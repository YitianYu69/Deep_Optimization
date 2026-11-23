"""CPU allocation contracts for shape-only convolution and padding adjoints.

Alternate arithmetic uses PyTorch's dtype-specific numerical tolerances.
Exact comparisons are reserved for the same native fallback or padding kernel.
"""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from deepopt.opt.backends.native.convolution import native_input_bias_gradients
from deepopt.opt.backends.native.convolution.ops import native_backward
from deepopt.opt.nn.convolution import ConvGeometry
from deepopt.opt.nn.convolution_transpose.geometry import ConvTransposeGeometry


class SpatialBufferAllocationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(911)

    def test_input_only_all_ranks_groups_stride_dilation_without_dummy(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            for rank in (1, 2, 3):
                for groups in (1, 2, 4):
                    with self.subTest(dtype=dtype, rank=rank, groups=groups):
                        x = torch.randn(2, 4, *([8] * rank), dtype=dtype, requires_grad=True)
                        weight = torch.randn(4, 4 // groups, *([3] * rank), dtype=dtype)
                        geometry = ConvGeometry.from_tensors(x, weight, stride=2, padding=2, dilation=2, groups=groups)
                        output = getattr(F, f"conv{rank}d")(x, weight, stride=2, padding=2, dilation=2, groups=groups)
                        gradient = torch.randn_like(output)
                        expected = torch.autograd.grad(output, x, gradient)[0]
                        info = {}
                        with patch("deepopt.opt.backends.native.convolution.ops.native_backward", side_effect=AssertionError("dense fallback")), \
                             patch("torch.empty", side_effect=AssertionError("shape placeholder")):
                            actual, dw, db = native_input_bias_gradients(gradient, weight, geometry,
                                (True, False, False), execution_info=info)
                        self.assertEqual(info["input_bias_gradient_route"], "shape_only_cpu")
                        self.assertIsNone(dw)
                        self.assertIsNone(db)
                        torch.testing.assert_close(actual, expected)

    def test_bias_only_reduces_without_input_or_weight_placeholders(self):
        for rank in (1, 2, 3):
            with self.subTest(rank=rank):
                x, weight = torch.randn(2, 4, *([5] * rank)), torch.randn(6, 2, *([3] * rank))
                bias = torch.randn(6, requires_grad=True)
                geometry = ConvGeometry.from_tensors(x, weight, padding=1, groups=2)
                output = getattr(F, f"conv{rank}d")(x, weight, bias, padding=1, groups=2)
                gradient = torch.randn_like(output)
                expected = torch.autograd.grad(output, bias, gradient)[0]
                with patch("torch.empty", side_effect=AssertionError("shape placeholder")):
                    dx, dw, actual = native_input_bias_gradients(gradient, None, geometry, (False, False, True))
                self.assertIsNone(dx)
                self.assertIsNone(dw)
                torch.testing.assert_close(actual, expected)
                oracle = gradient.double().sum((0, *range(2, gradient.ndim))).to(actual.dtype)
                torch.testing.assert_close(actual, oracle)
                torch.testing.assert_close(expected, oracle)

    def test_transposed_input_only_handles_output_padding_greater_than_stride(self):
        for rank in (1, 2, 3):
            for dtype in ((torch.float32, torch.float64, torch.bfloat16) if rank == 1 else (torch.bfloat16,)):
                with self.subTest(rank=rank, dtype=dtype):
                    x = torch.randn(2, 4, *([4] * rank), dtype=dtype, requires_grad=True)
                    weight = torch.randn(4, 2, *([3] * rank), dtype=dtype)
                    geometry = ConvTransposeGeometry.from_tensors(x, weight, dilation=3, output_padding=2, groups=2)
                    output = getattr(F, f"conv_transpose{rank}d")(x, weight, dilation=3, output_padding=2, groups=2)
                    gradient = torch.randn_like(output)
                    expected = torch.autograd.grad(output, x, gradient)[0]
                    with patch("torch.empty", side_effect=AssertionError("shape placeholder")):
                        actual, _, _ = native_input_bias_gradients(gradient, weight, geometry,
                            (True, False, False), transposed=True)
                    torch.testing.assert_close(actual, expected)
                    if dtype == torch.float32:
                        # Check both algorithms against a high-precision
                        # adjoint of the same FP32-rounded operands.
                        oracle_x = x.detach().double().requires_grad_()
                        oracle_output = getattr(F, f"conv_transpose{rank}d")(
                            oracle_x, weight.double(), dilation=3, output_padding=2, groups=2)
                        oracle = torch.autograd.grad(oracle_output, oracle_x, gradient.double())[0].to(dtype)
                        torch.testing.assert_close(actual, oracle)
                        torch.testing.assert_close(expected, oracle)

    def test_deterministic_and_depthwise_bias_policies_keep_visible_native_fallback(self):
        x, weight = torch.randn(2, 4, 7, 8), torch.randn(4, 1, 3, 3)
        geometry = ConvGeometry.from_tensors(x, weight, padding=1, groups=4)
        gradient = torch.randn(geometry.output_shape)
        info = {}
        with patch("deepopt.opt.backends.native.convolution.ops.native_backward", wraps=native_backward) as backward:
            actual = native_input_bias_gradients(gradient, weight, geometry, (True, False, True), execution_info=info)
        self.assertEqual(backward.call_count, 1)
        self.assertEqual(info["input_bias_gradient_route"], "native_backward_layout_fallback")
        expected = native_backward(gradient, x, weight, geometry, (True, False, True), bias_sizes=(4,))
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        with patch("torch.are_deterministic_algorithms_enabled", return_value=True), \
             patch("deepopt.opt.backends.native.convolution.ops.native_backward", wraps=native_backward) as backward:
            native_input_bias_gradients(gradient, weight, geometry, (True, False, False), execution_info=info)
        self.assertEqual(backward.call_count, 1)
        self.assertEqual(info["input_bias_gradient_route"], "native_backward_layout_fallback")

    def test_layout_mismatch_and_strided_upstream_preserve_native_selection(self):
        for input_format in (torch.contiguous_format, torch.channels_last):
            for strided in (False, True):
                with self.subTest(input_format=input_format, strided=strided):
                    x = torch.randn(2, 4, 7, 8).contiguous(memory_format=input_format)
                    weight = torch.randn(6, 2, 3, 3)
                    geometry = ConvGeometry.from_tensors(x, weight, padding=1, groups=2)
                    gradient = (torch.randn(2, 6, 7, 16)[..., ::2] if strided else
                        torch.randn(geometry.output_shape).contiguous(memory_format=(
                            torch.channels_last if input_format == torch.contiguous_format else torch.contiguous_format)))
                    expected = native_backward(gradient, x, weight, geometry, (True, False, True), bias_sizes=(6,))
                    info = {}
                    actual = native_input_bias_gradients(gradient, weight, geometry, (True, False, True),
                        input_memory_format=input_format, execution_info=info)
                    self.assertEqual(info["input_bias_gradient_fallback_reason"], "upstream_layout")
                    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

    def test_compiled_frozen_depthwise_blur_uses_shape_only_input_adjoint(self):
        from torch.fx.passes.shape_prop import ShapeProp
        from deepopt.compiler import BackendConfig, compile_fx

        layer = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        layer.weight.requires_grad_(False)
        graph = torch.fx.symbolic_trace(nn.Sequential(layer))
        reference = deepcopy(graph)
        x = torch.randn(2, 3, 7, 8, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        ShapeProp(graph).propagate(x)
        compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True))
        output, expected = compiled(x), reference(xr)
        if isinstance(output, tuple):
            output, = output
        gradient = torch.randn_like(output)
        expected.backward(gradient)
        with patch("deepopt.opt.backends.native.convolution.ops.native_backward", side_effect=AssertionError("dense fallback")):
            output.backward(gradient)
        torch.testing.assert_close(x.grad, xr.grad)

    def test_padding_shape_descriptor_uses_one_element_for_asymmetric_and_negative_geometry(self):
        from deepopt.opt.nn.padding import PaddingGeometry, padding_adjoint

        for mode in ("reflect", "replicate"):
            for pads in ((1, 2, 1, 2), (-1, 2, 1, -1), (4, 0, 0, 0), (0, 0, 0, 0)):
                for dtype in (torch.float32, torch.bfloat16):
                    with self.subTest(mode=mode, pads=pads, dtype=dtype):
                        x = torch.randn(2, 3, 4, 5, dtype=dtype, requires_grad=True)
                        output = F.pad(x, pads, mode=mode)
                        gradient = torch.randn_like(output)
                        expected = torch.autograd.grad(output, x, gradient)[0]
                        prefix = "reflection" if mode == "reflect" else "replication"
                        operator = getattr(torch.ops.aten, f"{prefix}_pad2d_backward")
                        descriptors = []
                        original = operator.default

                        def check_descriptor(g, descriptor, pad):
                            descriptors.append((tuple(descriptor.shape), descriptor.untyped_storage().nbytes()))
                            return original(g, descriptor, pad)

                        with patch.object(operator, "default", side_effect=check_descriptor):
                            actual = padding_adjoint(gradient, PaddingGeometry.from_tensor(x, pads, mode))
                        if any(pads):
                            self.assertEqual(descriptors, [(tuple(x.shape), x.element_size())])
                        else:
                            self.assertEqual(descriptors, [])
                            self.assertNotEqual(actual.untyped_storage().data_ptr(), gradient.untyped_storage().data_ptr())
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
