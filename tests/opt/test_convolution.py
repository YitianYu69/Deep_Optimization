"""Spatial convolution contracts, independent quantization oracles and conversion."""

import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as torch_f

import deepopt
from deepopt.opt.nn import functional as deepopt_f
from deepopt.opt.policy import conv_plan, prepare_fp8_model
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.quantizer import quantize_fp8


def channel_matrix(value):
    """The codec owns source positions, rather than an unfolded patch tensor."""
    return value.movedim(1, -1).reshape(-1, value.shape[1])


def from_channel_matrix(value, shape):
    return value.reshape(shape[0], *shape[2:], shape[1]).movedim(-1, 1)


def decoded_source(value, *, group_size=17, bits=2, block_shape=None, fp8=False):
    matrix = channel_matrix(value)
    if fp8:
        quantized = quantize_fp8(matrix, block_shape=block_shape)
        packed = quantize_pack_activation(quantized.data.float(), group_size, bits=bits, block_shape=block_shape)
        restored = dequantize_unpack_activation(packed, dtype=quantized.data.dtype)
        from deepopt.opt.quantization.tensor import QuantizedTensor

        matrix = QuantizedTensor(restored, quantized.scale_inv, block_shape).dequantize()
    else:
        matrix = dequantize_unpack_activation(quantize_pack_activation(matrix, group_size, bits=bits), value.dtype)
    memory_format = (torch.channels_last if value.ndim == 4 and value.is_contiguous(memory_format=torch.channels_last)
                     else torch.channels_last_3d if value.ndim == 5 and value.is_contiguous(memory_format=torch.channels_last_3d)
                     else torch.contiguous_format)
    return from_channel_matrix(matrix, value.shape).contiguous(memory_format=memory_format)


def quantized_source(value, *, block_shape=None):
    return from_channel_matrix(quantize_fp8(channel_matrix(value), block_shape=block_shape).dequantize(), value.shape)


def settings(**kwargs):
    return deepopt.FP8Config(**{"enabled": True, "backend": "reference", "trace_mode": "modules", **kwargs})


class ConvolutionNativeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(836)

    def test_four_modes_resolve_from_shared_configuration(self):
        for fp8 in (False, True):
            for compression in (False, True):
                with self.subTest(fp8=fp8, compression=compression):
                    plan = conv_plan(config={"runtime": {
                        "fp8": {"enabled": fp8, "backend": "reference"},
                        "activation_compression": {"enabled": compression, "group_size": 17, "bits": 2},
                    }})
                    if fp8:
                        self.assertEqual(plan["fp8_training"]["backend"], "reference")
                        codec = plan["fp8_training"]["activation_compression"]
                    else:
                        self.assertFalse(plan["fp8"])
                        codec = plan["activation_compression"]
                    self.assertEqual(codec["enabled"], compression)
                    self.assertEqual(codec["group_size"], 17)
                    with self.assertRaises(TypeError):
                        codec["bits"] = 4

    def test_native_compression_forward_dx_db_exact_dw_decodes_unique_source(self):
        cases = (
            (1, (2, 4, 11), (6, 2, 3), {"stride": 2, "padding": 2, "dilation": 2, "groups": 2}),
            (2, (2, 4, 7, 9), (6, 2, 3, 2), {"stride": (2, 1), "padding": (1, 1), "groups": 2}),
            (3, (1, 4, 5, 6, 7), (4, 1, 2, 2, 2), {"padding": 1, "dilation": (1, 2, 1), "groups": 4}),
        )
        for rank, shape, weight_shape, options in cases:
            for bits in (1, 2, 4, 8):
                with self.subTest(rank=rank, bits=bits):
                    x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
                    weight = torch.randn(weight_shape, dtype=torch.bfloat16, requires_grad=True)
                    bias = torch.randn(weight_shape[0], dtype=torch.bfloat16, requires_grad=True)
                    native = getattr(torch_f, f"conv{rank}d")
                    functional = getattr(deepopt_f, f"conv{rank}d")
                    expected = native(x, weight, bias, **options)
                    gradient = torch.randn_like(expected)
                    expected_dx, expected_db = torch.autograd.grad(expected, (x, bias), gradient)
                    saved = []
                    plan = conv_plan(False, activation_compression={
                        "enabled": True, "group_size": 17, "bits": bits, "attention_bits": 1,
                    })
                    with torch.autograd.graph.saved_tensors_hooks(
                        lambda tensor: saved.append(tensor) or tensor.clone(), lambda tensor: tensor,
                    ):
                        actual = functional(x, weight, bias, plan=plan, **options)
                    actual.backward(gradient)
                    restored = decoded_source(x.detach(), bits=bits)
                    expected_dw = getattr(torch.nn.grad, f"conv{rank}d_weight")(
                        restored, weight.shape, gradient, **options)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)
                    torch.testing.assert_close(bias.grad, expected_db, rtol=0, atol=0)
                    torch.testing.assert_close(weight.grad, expected_dw, rtol=0, atol=0)
                    self.assertTrue(any(value.dtype == torch.int32 for value in saved))
                    self.assertFalse(any(value is x for value in saved))
                    expected_packed = quantize_pack_activation(channel_matrix(x.detach()), 17, bits=bits)
                    words = [value for value in saved if value.dtype == torch.int32]
                    self.assertEqual(len(words), 1)
                    torch.testing.assert_close(words[0], expected_packed.data, rtol=0, atol=0)

    def test_native_off_preserves_torch_validation_and_unbatched_same_padding(self):
        plan = conv_plan(False, activation_compression=False)
        for rank in (1, 2, 3):
            with self.subTest(rank=rank):
                x = torch.randn((2,) + (7,) * rank, requires_grad=True)
                weight = torch.randn((4, 2) + (3,) * rank, requires_grad=True)
                native = getattr(torch_f, f"conv{rank}d")
                call = getattr(deepopt_f, f"do_conv{rank}d")
                actual = call(x, weight, padding="same", plan=plan)
                torch.testing.assert_close(actual, native(x, weight, padding="same"), rtol=0, atol=0)
                self.assertEqual(actual.shape, (4,) + (7,) * rank)
                with self.assertRaises((ValueError, RuntimeError)):
                    call(x, weight, groups=3, plan=plan)

    def test_channels_last_noncontiguous_bias_only_and_inference(self):
        plan = conv_plan(False, activation_compression=True)
        for rank, memory_format in ((2, torch.channels_last), (3, torch.channels_last_3d)):
            with self.subTest(rank=rank):
                x = torch.randn((1, 3) + (5,) * rank).to(memory_format=memory_format)
                x = x.transpose(-1, -2)
                weight = torch.randn((4, 3) + (2,) * rank)
                bias = torch.randn(4, requires_grad=True)
                call = getattr(deepopt_f, f"conv{rank}d")
                actual = call(x, weight, bias, plan=plan)
                native = getattr(torch_f, f"conv{rank}d")(x, weight, bias)
                torch.testing.assert_close(actual, native, rtol=0, atol=0)
                actual.sum().backward()
                torch.testing.assert_close(bias.grad, torch.full_like(bias, actual.numel() / 4), rtol=0, atol=0)
                saved = []
                with torch.no_grad(), torch.autograd.graph.saved_tensors_hooks(
                    lambda value: saved.append(value) or value, lambda value: value,
                ):
                    call(x, weight.requires_grad_(), bias, plan=plan)
                self.assertEqual(saved, [])

    def test_cpu_autocast_fp32_master_parameters(self):
        x = torch.randn(2, 3, 7, 9, requires_grad=True)
        weight = torch.randn(5, 3, 3, 3, requires_grad=True)
        bias = torch.randn(5, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = deepopt_f.conv2d(x, weight, bias, padding=1,
                                     plan=conv_plan(False, activation_compression={"enabled": True, "group_size": 17}))
            expected = torch_f.conv2d(x, weight, bias, padding=1)
        gradient = torch.randn_like(actual)
        dx, db = torch.autograd.grad(expected, (x, bias), gradient)
        actual.backward(gradient)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, dx, rtol=0, atol=0)
        torch.testing.assert_close(bias.grad, db, rtol=0, atol=0)
        restored = decoded_source(x.detach().bfloat16())
        dw = torch.nn.grad.conv2d_weight(restored, weight.shape, gradient, padding=1)
        torch.testing.assert_close(weight.grad, dw.float(), rtol=0, atol=0)

    def test_compressed_unbatched_same_even_kernel_preserves_forward_and_dx(self):
        plan = conv_plan(False, activation_compression=True)
        for rank in (1, 2, 3):
            with self.subTest(rank=rank):
                x = torch.randn((3,) + (5,) * rank, requires_grad=True)
                weight = torch.randn((4, 3) + (2,) * rank, requires_grad=True)
                actual = getattr(deepopt_f, f"conv{rank}d")(x, weight, padding="same", plan=plan)
                expected = getattr(torch_f, f"conv{rank}d")(x, weight, padding="same")
                gradient = torch.randn_like(expected)
                expected_dx = torch.autograd.grad(expected, x, gradient)[0]
                actual.backward(gradient)
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
                torch.testing.assert_close(x.grad, expected_dx, rtol=2e-6, atol=2e-6)
                padded = torch_f.pad(x.detach().unsqueeze(0), (0, 1) * rank)
                restored = decoded_source(padded, group_size=512)
                interior = (slice(None), slice(None), *(slice(0, size) for size in x.shape[1:]))
                literal_zero_padding = torch.zeros_like(restored)
                literal_zero_padding[interior] = restored[interior]
                expected_dw = getattr(torch.nn.grad, f"conv{rank}d_weight")(
                    literal_zero_padding, weight.shape, gradient.unsqueeze(0))
                torch.testing.assert_close(weight.grad, expected_dw, rtol=2e-6, atol=2e-6)


class ConvolutionFP8ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(338)

    def test_spatial_forward_and_both_gradients_use_quantized_operands(self):
        for rank in (1, 2, 3):
            for compressed in (False, True):
                for block in (None, (2, 4)):
                    with self.subTest(rank=rank, compression=compressed, block=block):
                        x = torch.randn((1, 4) + (5,) * rank, requires_grad=True)
                        weight = torch.randn((6, 2) + (2,) * rank, requires_grad=True)
                        bias = torch.randn(6, requires_grad=True)
                        compression = {"enabled": compressed, "group_size": 17, "bits": 2}
                        plan = conv_plan(settings(block_shapes={"activation": block, "weight": None, "grad": None}),
                                         activation_compression=compression)
                        options = {"stride": 2, "padding": 1, "groups": 2}
                        call, native = getattr(deepopt_f, f"conv{rank}d"), getattr(torch_f, f"conv{rank}d")
                        with patch.object(torch_f, "unfold", side_effect=AssertionError("Do not materialize patches")):
                            actual = call(x, weight, bias, plan=plan, **options)
                        qx = quantized_source(x.detach(), block_shape=block)
                        qw = quantize_fp8(weight.detach().reshape(6, -1)).dequantize().reshape_as(weight)
                        expected = native(qx, qw, bias.detach(), **options)
                        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
                        gradient = torch.randn_like(actual)
                        actual.backward(gradient)
                        qg = quantized_source(gradient)
                        expected_dx = getattr(torch.nn.grad, f"conv{rank}d_input")(
                            x.shape, qw, qg, **options)
                        restored = decoded_source(x.detach(), block_shape=block, fp8=True) if compressed else qx
                        expected_dw = getattr(torch.nn.grad, f"conv{rank}d_weight")(
                            restored, weight.shape, qg, **options)
                        torch.testing.assert_close(x.grad, expected_dx, rtol=2e-5, atol=2e-5)
                        torch.testing.assert_close(weight.grad, expected_dw, rtol=2e-5, atol=2e-5)
                        torch.testing.assert_close(bias.grad, gradient.sum((0,) + tuple(range(2, rank + 2))),
                                                   rtol=0, atol=0)

    def test_pointwise_matches_existing_linear_for_all_dimensions(self):
        plan = conv_plan(settings(), activation_compression={"enabled": True, "group_size": 17})
        for rank in (1, 2, 3):
            with self.subTest(rank=rank):
                x = torch.randn((2, 7) + (3,) * rank, requires_grad=True)
                weight = torch.randn((5, 7) + (1,) * rank, requires_grad=True)
                bias = torch.randn(5, requires_grad=True)
                output = getattr(deepopt_f, f"conv{rank}d")(x, weight, bias, plan=plan)
                linear = deepopt_f.linear(channel_matrix(x), weight.reshape(5, 7), bias, plan=plan)
                torch.testing.assert_close(output, from_channel_matrix(linear, output.shape), rtol=0, atol=0)
                gradient = torch.randn_like(output)
                expected_gradients = torch.autograd.grad(linear, (x, weight, bias), channel_matrix(gradient))
                output.backward(gradient)
                for actual, expected in zip((x.grad, weight.grad, bias.grad), expected_gradients):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_independent_activation_weight_and_grad_scale_grids(self):
        blocks = {"activation": (2, 3), "weight": (3, 4), "grad": (4, 2)}
        x = torch.randn(1, 4, 7, 5).transpose(-1, -2).requires_grad_()
        weight = torch.randn(6, 2, 3, 2, requires_grad=True)
        options = {"padding": (1, 0), "groups": 2, "dilation": (1, 2)}
        plan = conv_plan(settings(block_shapes=blocks), activation_compression={
            "enabled": True, "group_size": 512, "bits": 4,
        })
        actual = deepopt_f.conv2d(x, weight, plan=plan, **options)
        qx = quantized_source(x.detach(), block_shape=blocks["activation"])
        qw = quantize_fp8(weight.detach().reshape(6, -1), block_shape=blocks["weight"]).dequantize().reshape_as(weight)
        expected = torch_f.conv2d(qx, qw, **options)
        gradient = torch.randn_like(actual)
        qg = quantized_source(gradient, block_shape=blocks["grad"])
        actual.backward(gradient)
        saved_x = decoded_source(x.detach(), bits=4, block_shape=blocks["activation"], fp8=True)
        expected_dw = torch.nn.grad.conv2d_weight(saved_x, weight.shape, qg, **options)
        expected_dx = torch.nn.grad.conv2d_input(x.shape, qw, qg, **options)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(x.grad, expected_dx, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(weight.grad, expected_dw, rtol=2e-5, atol=2e-5)

    def test_functional_plan_reuse_and_fake_metadata(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        plan = conv_plan(settings(), activation_compression=True)
        with FakeTensorMode():
            x = torch.empty(2, 4, 11, 9)
            weight = torch.empty(6, 2, 3, 2)
            output = deepopt_f.conv2d(x, weight, stride=(2, 1), padding=(1, 0), groups=2, plan=plan)
            self.assertEqual(output.shape, (2, 6, 6, 8))

    def test_fake_cuda_spatial_forward_and_backward_need_no_triton_or_gpu(self):
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
        from deepopt.opt.nn.autograd.convolution.fp8 import _FP8ConvolutionFunction
        from deepopt.opt.nn.convolution.geometry import ConvGeometry

        blocked = {name: None for name in (
            "triton", "deepopt.opt.backends.triton.convolution",
            "deepopt.opt.backends.triton.quantization.fp8",
            "deepopt.opt.backends.triton.quantization.activation.fp8",
            "deepopt.opt.backends.triton.quantization.activation.affine",
        )}
        for compressed in (False, True):
            with self.subTest(compressed=compressed), FakeTensorMode(), torch.no_grad(), patch.dict(sys.modules, blocked):
                x = torch.empty(2, 4, 11, 9, device="cuda:0", dtype=torch.bfloat16)
                weight = torch.empty(6, 2, 3, 2, device="cuda:0")
                bias = torch.empty(6, device="cuda:0")
                geometry = ConvGeometry.from_tensors(x, weight, stride=(2, 1), padding=(1, 0), groups=2)
                plan = conv_plan(settings(backend="triton"), activation_compression={
                    "enabled": compressed, "group_size": 17, "bits": 2,
                })
                # Drive the Function contract directly: CPU-only PyTorch builds
                # cannot create real CUDA autograd device guards for fake leaves.
                ctx = SimpleNamespace(needs_input_grad=(True, True, True))
                ctx.save_for_backward = lambda *values: setattr(ctx, "saved_tensors", values)
                output = _FP8ConvolutionFunction.forward(
                    ctx, x, weight, bias, geometry, plan, torch.bfloat16, True, None, None, True)
                gradients = _FP8ConvolutionFunction.backward(ctx, torch.empty_like(output))
                self.assertIsInstance(output, FakeTensor)
                self.assertEqual(output.shape, (2, 6, 6, 8))
                self.assertEqual(output.dtype, torch.bfloat16)
                for gradient, reference in zip(gradients[:3], (x, weight, bias)):
                    self.assertIsInstance(gradient, FakeTensor)
                    self.assertEqual(gradient.shape, reference.shape)
                    self.assertEqual(gradient.dtype, reference.dtype)
                    self.assertEqual(gradient.device, reference.device)
                self.assertTrue(all(value is None for value in gradients[3:]))
                self.assertEqual(ctx.saved_tensors[0].dtype,
                                 torch.int32 if compressed else torch.float8_e4m3fn)

    def test_delayed_scale_snapshots_survive_updates_before_backward(self):
        x = torch.randn(1, 4, 9, requires_grad=True)
        source = nn.Conv1d(4, 6, 3, padding=1)
        config = settings(scaling_recipe="delayed")
        compression = {"enabled": True, "group_size": 17, "bits": 2}
        plan = conv_plan(config, activation_compression=compression)
        with self.assertRaisesRegex(ValueError, "prepared"):
            deepopt_f.conv1d(x, source.weight, padding=1, plan=plan)
        model, _ = prepare_fp8_model(source, config, activation_compression=compression)
        model.fp8_scaling.scales.fill_(8)
        with deepopt.fp8_scaling_scope():
            output = model(x)
        model.fp8_scaling.scales.fill_(32)
        gradient = torch.randn_like(output)
        output.backward(gradient)

        def cast(value, scale):
            return (value.float() * scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() / scale

        qx = cast(channel_matrix(x.detach()), 8)
        qw = cast(model.weight.detach().reshape(6, -1), 8).reshape_as(model.weight)
        qg = from_channel_matrix(cast(channel_matrix(gradient), 32), gradient.shape)
        saved = quantize_pack_activation((qx * 8), 17, bits=2)
        restored = dequantize_unpack_activation(saved, torch.float8_e4m3fn).float() / 8
        restored = from_channel_matrix(restored, x.shape)
        torch.testing.assert_close(output, torch_f.conv1d(from_channel_matrix(qx, x.shape), qw, model.bias, padding=1),
                                   rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(x.grad, torch.nn.grad.conv1d_input(x.shape, qw, qg, padding=1), rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(model.weight.grad, torch.nn.grad.conv1d_weight(restored, model.weight.shape, qg, padding=1),
                                   rtol=2e-5, atol=2e-5)

    def test_same_padding_compressed_fp8_dw_keeps_synthetic_cells_literal_zero(self):
        for rank in (1, 2, 3):
            with self.subTest(rank=rank):
                x = torch.randn((1, 3) + (5,) * rank, requires_grad=True)
                weight = torch.randn((4, 3) + (2,) * rank, requires_grad=True)
                plan = conv_plan(settings(), activation_compression={"enabled": True, "group_size": 17})
                actual = getattr(deepopt_f, f"conv{rank}d")(x, weight, padding="same", plan=plan)
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
                padded = torch_f.pad(x.detach(), (0, 1) * rank)
                restored = decoded_source(padded, fp8=True)
                interior = (slice(None), slice(None), *(slice(0, size) for size in x.shape[2:]))
                literal_zero_padding = torch.zeros_like(restored)
                literal_zero_padding[interior] = restored[interior]
                qg = quantized_source(gradient)
                expected_dw = getattr(torch.nn.grad, f"conv{rank}d_weight")(
                    literal_zero_padding, weight.shape, qg)
                torch.testing.assert_close(weight.grad, expected_dw, rtol=2e-5, atol=2e-5)


class ConvolutionModelConversionTests(unittest.TestCase):
    def test_module_conversion_keeps_parameters_state_keys_aliases_and_padding(self):
        for rank in (1, 2, 3):
            with self.subTest(rank=rank):
                module_type = getattr(nn, f"Conv{rank}d")
                source = module_type(3, 5, 3, padding=1, padding_mode="reflect")
                source.eval()
                source_weight, source_bias = source.weight, source.bias
                state_keys = tuple(source.state_dict())
                prepared, report = prepare_fp8_model(source, settings(), activation_compression=True)
                self.assertEqual(type(prepared).__name__, f"DOConv{rank}d")
                self.assertIs(prepared.weight, source_weight)
                self.assertIs(prepared.bias, source_bias)
                self.assertFalse(prepared.training)
                self.assertEqual(tuple(prepared.state_dict()), state_keys)
                self.assertEqual(report.converted_modules, ("<root>",))
                x = torch.randn((1, 3) + (5,) * rank)
                padding = [1, 1] * rank
                padded = torch_f.pad(x, padding, mode="reflect")
                expected = getattr(deepopt_f, f"conv{rank}d")(
                    padded, prepared.weight, prepared.bias, plan=prepared.plan)
                torch.testing.assert_close(prepared(x), expected, rtol=0, atol=0)

    def test_shared_modules_and_exclusions(self):
        shared = nn.Conv2d(3, 3, 3, padding=1)
        model = nn.ModuleDict({"first": shared, "alias": shared, "excluded": nn.Conv2d(3, 4, 1)})
        original = copy.deepcopy(model.state_dict())
        prepared, report = prepare_fp8_model(model, settings(exclude_modules=("excluded",), intercept_linear=False))
        self.assertIs(prepared["first"], prepared["alias"])
        self.assertEqual(type(prepared["first"]).__name__, "DOConv2d")
        self.assertIs(type(prepared["excluded"]), nn.Conv2d)
        self.assertIn("excluded", report.excluded_modules)
        for name, value in prepared.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_fx_function_and_scoped_native_call_use_same_plan(self):
        class SpatialFunctional(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(6, 3, 3, 3))
                self.bias = nn.Parameter(torch.randn(6))

            def forward(self, x):
                return torch_f.conv2d(x, self.weight, self.bias, padding=1)

        source = SpatialFunctional()
        x = torch.randn(2, 3, 5, 7)
        prepared, report = prepare_fp8_model(source, settings(trace_mode="full"))
        self.assertTrue(report.function_calls)
        expected = deepopt_f.conv2d(x, prepared.weight, prepared.bias, padding=1,
                                   plan=conv_plan(settings()))
        torch.testing.assert_close(prepared(x), expected, rtol=0, atol=0)
        with deepopt.autocast("cpu", dtype=torch.bfloat16, fp8=settings()):
            scoped = torch_f.conv2d(x, prepared.weight, prepared.bias, padding=1)
            explicit = deepopt_f.conv2d(x, prepared.weight, prepared.bias, padding=1,
                                       plan=conv_plan(settings()))
        torch.testing.assert_close(scoped, explicit, rtol=0, atol=0)

    def test_module_execution_info_follows_scoped_policy_and_same_padding(self):
        for padding in (1, "same"):
            with self.subTest(padding=padding):
                source = nn.Conv2d(3, 4, 3, padding=padding)
                model, _ = prepare_fp8_model(source, settings(), activation_compression=False)
                x = torch.randn(1, 3, 5, 7)
                self.assertIsNone(model.execution_info)
                model(x)
                self.assertEqual(model.execution_info["forward"], "reference")
                self.assertFalse(model.execution_info["compression"])
                with deepopt.autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=True):
                    model(x)
                self.assertEqual(model.execution_info["forward"], "native")
                self.assertTrue(model.execution_info["compression"])
                with deepopt.autocast("cpu", dtype=torch.bfloat16, fp8=settings(), activation_compression=False):
                    model(x)
                self.assertEqual(model.execution_info["forward"], "reference")
                self.assertFalse(model.execution_info["compression"])
                info = model.execution_info
                info["forward"] = "tampered"
                self.assertEqual(model.execution_info["forward"], "reference")

    def test_channels_last_weight_cache_reuses_original_parameter_and_invalidates(self):
        from deepopt.opt.nn.autograd.convolution import fp8 as autograd_convolution

        for rank, memory_format in ((2, torch.channels_last), (3, torch.channels_last_3d)):
            with self.subTest(rank=rank):
                source = getattr(nn, f"Conv{rank}d")(4, 6, 2).to(memory_format=memory_format)
                parameter = source.weight
                model, _ = prepare_fp8_model(source, settings(), activation_compression=False)
                self.assertIs(model.weight, parameter)
                self.assertFalse(parameter.is_contiguous())
                x = torch.randn((1, 4) + (5,) * rank).to(memory_format=memory_format)
                weight_shape = (6, 4 * 2 ** rank)
                with patch.object(autograd_convolution, "quantize_fp8", wraps=quantize_fp8) as quantize:
                    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=True):
                        model(x)
                        model(x)
                        weight_calls = [call for call in quantize.call_args_list if tuple(call.args[0].shape) == weight_shape]
                        self.assertEqual(len(weight_calls), 1)
                        parameter.add_(0.125)
                        updated = model(x)
                        weight_calls = [call for call in quantize.call_args_list if tuple(call.args[0].shape) == weight_shape]
                        self.assertEqual(len(weight_calls), 2)
                        model.clear_fp8_cache()
                        expected = model(x)
                    torch.testing.assert_close(updated, expected, rtol=0, atol=0)
                    weight_calls = [call for call in quantize.call_args_list if tuple(call.args[0].shape) == weight_shape]
                    self.assertEqual(len(weight_calls), 3)

    def test_delayed_fx_call_sites_own_histories_and_reject_capture_before_cuda(self):
        from deepopt.cuda_graph.model import CapturedCallable
        from deepopt.opt.nn.convolution.modules import DOFunctionalConvolution
        from deepopt.opt.scaling.delayed import delayed_states

        class FunctionalConvolution(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 3, 3, 3))

            def forward(self, x):
                return torch_f.conv2d(input=x, weight=self.weight, padding=1) + torch_f.conv2d(x * 3, self.weight, padding=1)

        source = FunctionalConvolution()
        config = settings(trace_mode="full", scaling_recipe="delayed")
        model, report = prepare_fp8_model(source, config, activation_compression=True)
        states = delayed_states(model)
        call_sites = [module for module in model.modules() if isinstance(module, DOFunctionalConvolution)]
        self.assertEqual(len(states), 2)
        self.assertEqual(len(call_sites), 2)
        self.assertEqual(len(report.function_calls), 2)
        self.assertEqual(tuple(model.parameters()), (source.weight,))
        self.assertTrue(all(tuple(module.parameters()) == () for module in call_sites))
        x = torch.ones(1, 3, 5, 7, requires_grad=True)
        with deepopt.fp8_scaling_scope():
            model(x).sum().backward()
        self.assertNotEqual(states[0].scales[0].item(), states[1].scales[0].item())
        self.assertTrue(torch.isfinite(source.weight.grad).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        with patch("deepopt.cuda_graph.model.cuda_device", side_effect=AssertionError("Queried CUDA too early")):
            with self.assertRaisesRegex(ValueError, "Delayed scaling.*capture is unsupported"):
                CapturedCallable(model, (x.detach(),), device="cuda:0", warmup=1)

    def test_capture_policy_distinguishes_convolution_interception_without_cuda(self):
        from deepopt._config_resolution import ResolvedConfig
        from deepopt.cuda_graph._validation import operation_policy_spec
        from deepopt.opt._autocast import fp8_context

        original = operation_policy_spec()
        policy = {"enabled": True, "backend": "triton", "intercept_linear": False}
        with fp8_context("cuda", {**policy, "intercept_convolution": False}, config=ResolvedConfig({})):
            disabled = operation_policy_spec()
            self.assertFalse(disabled[3])
            with fp8_context("cuda", {**policy, "intercept_convolution": True}, config=ResolvedConfig({})):
                enabled = operation_policy_spec()
                self.assertTrue(enabled[3])
                self.assertEqual(enabled[:3], disabled[:3])
            self.assertEqual(operation_policy_spec(), disabled)
        self.assertEqual(operation_policy_spec(), original)

    def test_complex_prepared_convolution_keeps_native_forward_and_backward(self):
        source = nn.Conv2d(3, 4, 3, padding=1, dtype=torch.complex64)
        model, _ = prepare_fp8_model(source, settings(backend="triton"), activation_compression=True)
        x = torch.randn(1, 3, 5, 7, dtype=torch.complex64, requires_grad=True)
        expected = torch_f.conv2d(x, source.weight, source.bias, padding=1)
        expected_gradients = torch.autograd.grad(expected.abs().square().sum(), (x, source.weight, source.bias))
        actual = model(x)
        actual.abs().square().sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for gradient, reference in zip((x.grad, source.weight.grad, source.bias.grad), expected_gradients):
            torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
        self.assertEqual(model.execution_info["forward"], "native")
        self.assertFalse(model.execution_info["compression"])
        self.assertIn("real operands", model.execution_info["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
