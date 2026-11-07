"""Quantization metadata, autograd storage/dispatch, and adapter ownership."""

from copy import deepcopy
from itertools import product
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from deepopt.opt.backends import gemm_nt
from deepopt.opt.do_nn.AOTAutograd_Func import linear_autograd_func
from deepopt.opt.do_nn.functional import do_linear, validate_fp8_plan
from deepopt.opt.do_nn.hf_conv1d_module import DOHFConv1D, get_hf_conv1d_type
from deepopt.opt.do_nn.linear_module import DOLinear
from deepopt.opt.do_nn.mha_module import DOMultiheadAttention
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.opt.quantization.tensor import QuantizedTensor
from deepopt.opt.scaling_strategy.amax import AmaxScaling
from tests.opt.fp8_support import round_fp8


def _plan(blocks=None):
    return {
        "fp8_training": {"fp8_dtypes": {
            "activation": torch.float8_e4m3fn,
            "weight": torch.float8_e4m3fn,
            "grad": torch.float8_e5m2,
        }},
        "block_shapes": dict.fromkeys(("activation", "weight", "grad"), blocks),
    }


class ScalingWhiteBoxTests(unittest.TestCase):
    def test_tensor_and_partial_block_amax_and_zero_scale(self):
        x = torch.tensor([[0., 0., 4., -8., 2.], [0., 0., 1., -2., -3.],
                          [16., -2., -6., 1., .5]])
        result = AmaxScaling((2, 2)).compute(x)
        expected_amax = torch.tensor([[0., 8., 3.], [16., 6., .5]])
        expected_scale = torch.tensor([[1., 448. / 8, 448. / 3], [448. / 16, 448. / 6, 448. / .5]])
        torch.testing.assert_close(result.scale, expected_scale)
        torch.testing.assert_close(result.scale_inv, expected_scale.reciprocal())
        self.assertEqual(result.block_shape, (2, 2))
        self.assertEqual(result.scale.shape, expected_amax.shape)
        tensorwise = AmaxScaling().compute(x, torch.float8_e5m2)
        torch.testing.assert_close(tensorwise.scale, torch.tensor([[57344. / 16]]))

    def test_scaling_rejects_invalid_shapes_and_nonfinite_values(self):
        for shape in ((0, 2), (2, -1), (True, 2), [2, 2], (2,), "tensor"):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                AmaxScaling(shape)
        for x in (torch.ones(3), torch.empty(0, 3), torch.tensor([[float("nan")]]),
                  torch.tensor([[float("inf")]])):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                AmaxScaling().compute(x)
        with self.assertRaises(TypeError):
            AmaxScaling().compute(torch.ones(2, 3, dtype=torch.float64))

    def test_fp8_values_scales_and_transpose_own_consistent_layouts(self):
        x = torch.arange(-17., 18.).reshape(5, 7).requires_grad_()
        quantized = quantize_fp8(x, block_shape=(2, 3))
        self.assertEqual(quantized.data.dtype, torch.float8_e4m3fn)
        self.assertEqual(quantized.scale_inv.shape, (3, 3))
        self.assertFalse(quantized.data.requires_grad)
        self.assertFalse(quantized.scale_inv.requires_grad)
        transposed = quantized.transpose()
        self.assertEqual(transposed.block_shape, (3, 2))
        self.assertEqual(transposed.data.shape, (7, 5))
        self.assertTrue(transposed.data.is_contiguous())
        self.assertTrue(transposed.scale_inv.is_contiguous())
        torch.testing.assert_close(transposed.scale_inv, quantized.scale_inv.T)
        torch.testing.assert_close(transposed.data.float(), quantized.data.float().T)

    def test_quantized_tensor_rejects_incompatible_metadata(self):
        values = torch.ones(3, 5).to(torch.float8_e4m3fn)
        for data, scales, blocks, error in (
            (values.float(), torch.ones(1, 1), None, TypeError),
            (values, torch.ones(1, 1, dtype=torch.float16), None, TypeError),
            (values, torch.ones(1, 1), (2, 3), ValueError),
            (values, torch.ones(2, 2), None, ValueError),
            (values.flatten(), torch.ones(1, 1), None, ValueError),
        ):
            with self.subTest(blocks=blocks, error=error), self.assertRaises(error):
                QuantizedTensor(data=data, scale_inv=scales, block_shape=blocks)

    def test_backend_dispatch_rejects_invalid_reduction_and_output(self):
        a = quantize_fp8(torch.ones(2, 3))
        wrong_k = quantize_fp8(torch.ones(4, 5))
        with self.assertRaisesRegex(ValueError, "reduction"):
            gemm_nt(a, wrong_k, backend="reference")
        for kwargs in ({"backend": "unknown"}, {"backend": "reference", "num_stages": True},
                       {"backend": "reference", "out_dtype": torch.float64}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gemm_nt(a, a, **kwargs)


class FP8AutogradWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(442)

    def test_backward_saves_only_required_quantized_operands(self):
        for needs_x, needs_w, needs_b, expected_shapes in (
            (True, True, True, [(5, 7), (3, 3), (4, 7), (2, 3)]),
            (True, False, False, [(4, 7), (2, 3)]),
            (False, True, False, [(5, 7), (3, 3)]),
            (False, False, True, []),
        ):
            with self.subTest(needs=(needs_x, needs_w, needs_b)):
                stored = []

                def pack(tensor):
                    stored.append(tensor)
                    return tensor

                x = torch.randn(5, 7, requires_grad=needs_x)
                weight = torch.randn(4, 7, requires_grad=needs_w)
                bias = torch.randn(4, requires_grad=needs_b)
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                    result = do_linear(x, weight, bias, plan=_plan((2, 3)))
                self.assertEqual([tuple(t.shape) for t in stored], expected_shapes)
                for index, tensor in enumerate(stored):
                    self.assertEqual(tensor.dtype, torch.float8_e4m3fn if index % 2 == 0 else torch.float32)
                    self.assertNotEqual(tensor.data_ptr(), x.data_ptr())
                    self.assertNotEqual(tensor.data_ptr(), weight.data_ptr())
                result.sum().backward()

    def test_backward_dispatch_uses_transposed_quantized_operands_and_original_dtypes(self):
        plan = _plan((2, 3))
        x = torch.randn(5, 7, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(4, 7, requires_grad=True)
        bias = torch.randn(4, requires_grad=True)
        with patch.object(linear_autograd_func, "gemm_nt", wraps=gemm_nt) as dispatch:
            do_linear(x, weight, bias, plan=plan).sum().backward()
        self.assertEqual(dispatch.call_count, 3)
        forward, dx, dw = dispatch.call_args_list
        self.assertIs(forward.kwargs["bias"], bias)
        self.assertEqual(dx.args[1].block_shape, (3, 2))
        self.assertEqual(dw.args[0].block_shape, (3, 2))
        self.assertEqual(dw.args[1].block_shape, (3, 2))
        self.assertEqual(dx.args[0].data.dtype, torch.float8_e5m2)
        self.assertEqual(dx.kwargs["out_dtype"], torch.bfloat16)
        self.assertEqual(dw.kwargs["out_dtype"], torch.float32)
        self.assertNotIn("bias", dx.kwargs)
        self.assertNotIn("bias", dw.kwargs)

    def test_bias_only_backward_needs_no_quantization_or_gemm(self):
        bias = torch.randn(4, requires_grad=True)
        result = do_linear(torch.randn(5, 7), torch.randn(4, 7), bias, plan=_plan())
        with patch.object(linear_autograd_func, "gemm_nt") as dispatch, \
                patch.object(linear_autograd_func, "quantize_fp8") as quantizer:
            result.sum().backward()
        dispatch.assert_not_called()
        quantizer.assert_not_called()
        torch.testing.assert_close(bias.grad, torch.full((4,), 5.))

    def test_rowwise_backward_operands_keep_native_fp8_reduction_grids(self):
        for gradient_block in ((1, 32), (32, 32)):
            for needs_input_grad in (False, True):
                with self.subTest(gradient_block=gradient_block, input_grad=needs_input_grad):
                    plan = _plan()
                    plan["block_shapes"] = {"activation": (1, 32), "weight": (32, 32),
                                             "grad": gradient_block}
                    x = torch.randn(5, 7, requires_grad=needs_input_grad)
                    weight = torch.randn(4, 7, requires_grad=True)
                    gradient = torch.randn(5, 4)
                    with patch.object(linear_autograd_func, "gemm_nt", wraps=gemm_nt) as dispatch:
                        do_linear(x, weight, plan=plan).backward(gradient)
                    for call in dispatch.call_args_list:
                        for operand in call.args[:2]:
                            self.assertTrue(operand.block_shape is None or operand.block_shape[1] % 32 == 0)
                    dweight = dispatch.call_args_list[-1]
                    qxt = round_fp8(x.T, torch.float8_e4m3fn, (1, 32))
                    qgt = (round_fp8(gradient.T, torch.float8_e5m2, (1, 32))
                           if gradient_block == (1, 32) else
                           round_fp8(gradient, torch.float8_e5m2, gradient_block).T)
                    torch.testing.assert_close(dweight.args[0].dequantize(), qgt, rtol=0, atol=0)
                    torch.testing.assert_close(dweight.args[1].dequantize(), qxt, rtol=0, atol=0)
                    # The independent oracle may retain transposed strides;
                    # FP32 GEMM accumulation order can then differ slightly.
                    torch.testing.assert_close(weight.grad, qgt @ qxt.T, rtol=1e-6, atol=1e-6)

    def test_second_order_differentiation_is_explicitly_unsupported(self):
        x = torch.randn(5, 7, requires_grad=True)
        weight = torch.randn(4, 7, requires_grad=True)
        first_grad = torch.autograd.grad(do_linear(x, weight, plan=_plan()).sum(), x, create_graph=True)[0]
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(first_grad.sum(), weight)

    def test_plan_defaults_and_nested_configuration_are_independently_owned(self):
        raw = _plan((2, 3))
        normalized = validate_fp8_plan(raw)
        self.assertEqual(normalized["fp8_training"]["backend"], "reference")
        self.assertEqual(normalized["fp8_training"]["num_stages"], 3)
        self.assertIsNone(normalized["fp8_training"]["out_dtype"])
        raw["block_shapes"]["weight"] = None
        raw["fp8_training"]["fp8_dtypes"]["grad"] = torch.float8_e4m3fn
        self.assertEqual(normalized["block_shapes"]["weight"], (2, 3))
        self.assertEqual(normalized["fp8_training"]["fp8_dtypes"]["grad"], torch.float8_e5m2)

    def test_triton_preserves_independent_formats_and_small_training_blocks(self):
        roles = ("activation", "weight", "grad")
        for formats in product((torch.float8_e4m3fn, torch.float8_e5m2), repeat=3):
            for shape in ((1, 32), (32, 1), (3, 5)):
                with self.subTest(formats=formats, shape=shape):
                    plan = _plan(shape)
                    plan["fp8_training"]["backend"] = "triton"
                    plan["fp8_training"]["fp8_dtypes"] = dict(zip(roles, formats))
                    normalized = validate_fp8_plan(plan)
                    self.assertEqual(normalized["fp8_training"]["fp8_dtypes"], dict(zip(roles, formats)))
                    self.assertEqual(normalized["block_shapes"], dict.fromkeys(roles, shape))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            do_linear(torch.ones(2, 3), torch.ones(4, 3), plan=plan)


class FP8AdaptersWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(911)

    def test_linear_owns_plan_but_preserves_parameters_and_training_flag(self):
        original = nn.Linear(7, 4).eval()
        plan = _plan((2, 3))
        module = DOLinear.from_linear(original, plan=plan)
        self.assertIs(module.weight, original.weight)
        self.assertIs(module.bias, original.bias)
        self.assertFalse(module.training)
        plan["block_shapes"]["activation"] = None
        self.assertEqual(module.block_shapes["activation"], (2, 3))
        view = module.fp8_dtypes
        view["weight"] = torch.float8_e5m2
        self.assertEqual(module.fp8_dtypes["weight"], torch.float8_e4m3fn)

    def test_attention_preserves_parameter_and_child_identity(self):
        for separate in (False, True):
            with self.subTest(separate=separate):
                original = nn.MultiheadAttention(8, 2, kdim=6 if separate else 8, vdim=10 if separate else 8)
                original.out_proj.eval()
                module = DOMultiheadAttention.from_mha(original, plan=_plan())
                self.assertTrue(module.training)
                self.assertFalse(module.out_proj.training)
                self.assertIs(module.out_proj, original.out_proj)
                self.assertEqual(tuple(module.state_dict()), tuple(original.state_dict()))
                self.assertEqual(tuple(dict(module.named_parameters())), tuple(dict(original.named_parameters())))
                for name, parameter in original.named_parameters():
                    self.assertIs(module.get_parameter(name), parameter)

    def test_attention_shape_masks_and_gradients_match_native_without_quantization(self):
        # Replace only the FP8 primitive to isolate adapter semantics from FP8 error.
        for batch_first, unbatched, need_weights, separate in (
            (True, False, True, False), (False, False, False, False),
            (True, True, True, True), (False, False, True, True),
        ):
            with self.subTest(batch_first=batch_first, unbatched=unbatched,
                              need_weights=need_weights, separate=separate):
                original = nn.MultiheadAttention(8, 2, batch_first=batch_first,
                                                 kdim=6 if separate else 8, vdim=10 if separate else 8)
                dense = deepcopy(original)
                module = DOMultiheadAttention.from_mha(original, plan=_plan())
                args = [torch.randn(2, length, width) for length, width in (
                    (3, 8), (4, original.kdim), (4, original.vdim)
                )]
                if unbatched:
                    args = [value[0] for value in args]
                elif not batch_first:
                    args = [value.transpose(0, 1) for value in args]
                actual_inputs = [value.detach().requires_grad_() for value in args]
                expected_inputs = [value.detach().clone().requires_grad_() for value in args]
                mask = torch.zeros((2 if unbatched else 4), 3, 4)
                mask[:, :, -1] = float("-inf")
                padding = torch.zeros((4,) if unbatched else (2, 4))
                padding[..., 1] = float("-inf")
                kwargs = dict(attn_mask=mask, key_padding_mask=padding, need_weights=need_weights,
                              average_attn_weights=False)
                with patch("deepopt.opt.do_nn.mha_module.do_linear",
                           side_effect=lambda x, w, b=None, **kw: F.linear(x, w, b)):
                    actual, actual_weights = module(*actual_inputs, **kwargs)
                expected, expected_weights = dense(*expected_inputs, **kwargs)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                if need_weights:
                    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-6)
                else:
                    self.assertIsNone(actual_weights)
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
                expected.backward(gradient)
                for actual_input, expected_input in zip(actual_inputs, expected_inputs):
                    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-5, atol=1e-6)
                for name, expected_param in dense.named_parameters():
                    torch.testing.assert_close(module.get_parameter(name).grad, expected_param.grad,
                                               rtol=1e-5, atol=1e-6)

    def test_packed_self_attention_projects_once_and_dropout_tracks_training(self):
        original = nn.MultiheadAttention(8, 2, dropout=.4, batch_first=True)
        module = DOMultiheadAttention.from_mha(original, plan=_plan())
        x = torch.randn(2, 4, 8)
        with patch("deepopt.opt.do_nn.mha_module.do_linear", wraps=do_linear) as projection:
            module(x, x, x)
        self.assertEqual(projection.call_count, 2)  # packed QKV, then output
        self.assertIs(projection.call_args_list[0].args[1], original.in_proj_weight)
        torch.manual_seed(1)
        first, _ = module(x, x, x)
        torch.manual_seed(2)
        second, _ = module(x, x, x)
        self.assertFalse(torch.equal(first, second))
        module.eval()
        first, _ = module(x, x, x)
        second, _ = module(x, x, x)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_attention_explicit_causal_mask_and_unsupported_variants(self):
        for kwargs in ({"add_bias_kv": True}, {"add_zero_attn": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(NotImplementedError):
                DOMultiheadAttention.from_mha(nn.MultiheadAttention(8, 2, **kwargs), plan=_plan())
        module = DOMultiheadAttention.from_mha(nn.MultiheadAttention(8, 2, batch_first=True), plan=_plan())
        x = torch.randn(2, 4, 8)
        with self.assertRaisesRegex(ValueError, "explicit attn_mask"):
            module(x, x, x, is_causal=True)
        mask = torch.ones(4, 4, dtype=torch.bool).triu(1)
        _, weights = module(x, x, x, is_causal=True, attn_mask=mask)
        self.assertTrue((weights[:, mask] == 0).all())
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4))

    def test_hf_adapter_preserves_transposed_weight_layout_and_gradient(self):
        # A narrow stand-in verifies the adapter's protocol without requiring transformers.
        # Real HF model execution lives in the optional full-model tests.
        class Conv1D(nn.Module):
            def __init__(self):
                super().__init__()
                self.nf = 4
                self.weight = nn.Parameter(torch.randn(7, 4))
                self.bias = nn.Parameter(torch.randn(4))

        provider = ModuleType("transformers.pytorch_utils")
        provider.Conv1D = Conv1D
        with patch.dict(sys.modules, {"transformers.pytorch_utils": provider}):
            original = Conv1D().eval()
            module = DOHFConv1D.from_hf_conv1d(original, plan=_plan((2, 3)))
            self.assertIs(get_hf_conv1d_type(), Conv1D)
            self.assertIs(module.weight, original.weight)
            self.assertIs(module.bias, original.bias)
            self.assertEqual(module.weight.shape, (7, 4))
            self.assertEqual(tuple(module.state_dict()), ("weight", "bias"))
            self.assertFalse(module.training)
            x = torch.randn(2, 3, 7, requires_grad=True)
            expected_x = x.detach().clone().requires_grad_()
            expected_w = original.weight.detach().clone().requires_grad_()
            expected_b = original.bias.detach().clone().requires_grad_()
            actual = module(x)
            expected = do_linear(expected_x, expected_w.T, expected_b, plan=_plan((2, 3)))
            torch.testing.assert_close(actual, expected)
            grad = torch.randn_like(actual)
            actual.backward(grad)
            expected.backward(grad)
            torch.testing.assert_close(x.grad, expected_x.grad)
            torch.testing.assert_close(original.weight.grad, expected_w.grad)
            torch.testing.assert_close(original.bias.grad, expected_b.grad)
        with self.assertRaises(TypeError):
            DOHFConv1D.from_hf_conv1d(nn.Conv1d(7, 4, 1), plan=_plan())


if __name__ == "__main__":
    unittest.main()
