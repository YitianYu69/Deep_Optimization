"""Real GPU validation of spatial FP8 convolution and compressed saved sources."""

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as torch_f

from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class ConvolutionFP8CUDATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(929)
        self.enterContext(torch.backends.cudnn.flags(deterministic=True))

    def test_spatial_fp8_forward_dx_dw_match_explicit_quantized_reference(self):
        from deepopt import FP8Config
        from deepopt.opt.nn import functional as deepopt_f
        from deepopt.opt.policy import conv_plan
        from deepopt.opt.quantization.quantizer import quantize_fp8
        from tests.opt.test_convolution import decoded_source, quantized_source

        cases = (
            (1, (2, 16, 19), (32, 16, 3), {"stride": 2, "padding": 2, "dilation": 2}),
            (2, (2, 16, 9, 11), (32, 8, 3, 2), {"stride": (2, 1), "padding": (1, 1), "groups": 2}),
            (3, (1, 16, 5, 7, 9), (16, 16, 2, 2, 2), {"padding": 1}),
            (2, (1, 16, 9, 7), (16, 1, 3, 3), {"padding": 2, "dilation": 2, "groups": 16}),
            (1, (2, 8, 1050), (8, 8, 3), {"padding": 1}),
        )
        for rank, shape, weight_shape, options in cases:
            for compressed in (False, True):
                with self.subTest(rank=rank, compressed=compressed):
                    cpu_x = torch.randn(shape)
                    cpu_weight = torch.randn(weight_shape)
                    cpu_bias = torch.randn(weight_shape[0])
                    x, weight, bias = [value.cuda().requires_grad_() for value in (cpu_x, cpu_weight, cpu_bias)]
                    plan = conv_plan(FP8Config(enabled=True, backend="triton", out_dtype="fp32"),
                                     activation_compression={"enabled": compressed, "group_size": 17, "bits": 2})
                    with patch.object(torch_f, "unfold", side_effect=AssertionError("No unfolded source")):
                        output = getattr(deepopt_f, f"conv{rank}d")(x, weight, bias, plan=plan, **options)
                        cpu_gradient = torch.randn(tuple(output.shape))
                        output.backward(cpu_gradient.cuda())
                    qx = quantized_source(cpu_x)
                    qw = quantize_fp8(cpu_weight.reshape(weight_shape[0], -1)).dequantize().reshape(weight_shape)
                    qg = quantized_source(cpu_gradient)
                    saved_x = decoded_source(cpu_x, fp8=True) if compressed else qx
                    expected_output = getattr(torch_f, f"conv{rank}d")(qx, qw, cpu_bias, **options)
                    expected_dx = getattr(torch.nn.grad, f"conv{rank}d_input")(shape, qw, qg, **options)
                    expected_dw = getattr(torch.nn.grad, f"conv{rank}d_weight")(
                        saved_x, weight_shape, qg, **options)
                    torch.testing.assert_close(output.cpu(), expected_output, rtol=2e-3, atol=2e-3)
                    torch.testing.assert_close(x.grad.cpu(), expected_dx, rtol=2e-3, atol=2e-3)
                    torch.testing.assert_close(weight.grad.cpu(), expected_dw, rtol=2e-3, atol=2e-3)
                    expected_db = cpu_gradient.sum((0,) + tuple(range(2, rank + 2)))
                    torch.testing.assert_close(bias.grad.cpu(), expected_db, rtol=2e-5, atol=2e-5)

    def test_groupwise_source_compression_bits_and_channel_layout(self):
        from deepopt import FP8Config
        from deepopt.opt.nn import functional as deepopt_f
        from deepopt.opt.policy import conv_plan

        x = torch.randn(2, 16, 9, 11, device="cuda", dtype=torch.bfloat16).to(memory_format=torch.channels_last)
        weight = torch.randn(32, 16, 3, 3, device="cuda", requires_grad=True)
        settings = FP8Config(enabled=True, backend="triton", block_shapes={
            "activation": (4, 8), "weight": None, "grad": None,
        })
        for bits in (1, 2, 4, 8):
            with self.subTest(bits=bits):
                plan = conv_plan(settings, activation_compression={"enabled": True, "group_size": 17, "bits": bits})
                saved = []
                weight.grad = None
                with torch.autograd.graph.saved_tensors_hooks(
                    lambda value: saved.append(value) or value, lambda value: value,
                ):
                    output = deepopt_f.conv2d(x, weight, padding=1, plan=plan)
                output.sum().backward()
                words = [value for value in saved if value.dtype == torch.int32]
                self.assertEqual(len(words), 1)
                self.assertEqual(tuple(words[0].shape), ((2 * 9 * 11 + 3) // 4 * 2, bits))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.isfinite(weight.grad).all())

    def test_native_bf16_compression_uses_native_convolution_gradients(self):
        from deepopt.opt.nn import functional as deepopt_f
        from deepopt.opt.policy import conv_plan

        plan = conv_plan(False, activation_compression=True)
        for rank in (1, 2, 3):
            with self.subTest(rank=rank):
                x = torch.randn((1, 8) + (7,) * rank, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                weight = torch.randn((16, 8) + (3,) * rank, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                actual = getattr(deepopt_f, f"conv{rank}d")(x, weight, padding=1, plan=plan)
                expected = getattr(torch_f, f"conv{rank}d")(x, weight, padding=1)
                gradient = torch.randn_like(actual)
                expected_dx = torch.autograd.grad(expected, x, gradient)[0]
                actual.backward(gradient)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)

    def test_independent_scale_grids_gather_source_weight_and_gradient_tiles(self):
        from deepopt import FP8Config
        from deepopt.opt.nn import functional as deepopt_f
        from deepopt.opt.policy import conv_plan
        from deepopt.opt.quantization.quantizer import quantize_fp8
        from tests.opt.test_convolution import decoded_source, quantized_source

        blocks = {"activation": (4, 8), "weight": (4, 8), "grad": (4, 16)}
        cpu_x = torch.randn(1, 16, 7, 9).transpose(-1, -2)
        cpu_weight = torch.randn(16, 8, 3, 2)
        x = cpu_x.cuda().requires_grad_()
        weight = cpu_weight.cuda().requires_grad_()
        options = {"padding": (1, 1), "dilation": (1, 2), "groups": 2}
        plan = conv_plan(FP8Config(enabled=True, backend="triton", out_dtype="fp32", block_shapes=blocks),
                         activation_compression={"enabled": True, "group_size": 17, "bits": 4})
        output = deepopt_f.conv2d(x, weight, plan=plan, **options)
        cpu_gradient = torch.randn(tuple(output.shape))
        output.backward(cpu_gradient.cuda())
        qx = quantized_source(cpu_x, block_shape=blocks["activation"])
        qw = quantize_fp8(cpu_weight.reshape(16, -1), block_shape=blocks["weight"]).dequantize().reshape_as(cpu_weight)
        qg = quantized_source(cpu_gradient, block_shape=blocks["grad"])
        saved_x = decoded_source(cpu_x, block_shape=blocks["activation"], bits=4, fp8=True)
        torch.testing.assert_close(output.cpu(), torch_f.conv2d(qx, qw, **options), rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(x.grad.cpu(), torch.nn.grad.conv2d_input(x.shape, qw, qg, **options),
                                   rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(weight.grad.cpu(), torch.nn.grad.conv2d_weight(saved_x, weight.shape, qg, **options),
                                   rtol=2e-3, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
