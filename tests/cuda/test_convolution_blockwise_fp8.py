"""Real CUDA arithmetic and FP8 instruction checks for block-scaled convolution."""

from contextlib import ExitStack
import re
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class BlockwiseConvolutionFP8CUDATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1771)

    def test_all_ranks_mixed_blocks_compression_groups_and_asymmetric_padding(self):
        from deepopt import FP8Config
        from deepopt.opt.nn import functional as deepopt_f
        from deepopt.opt.policy import conv_plan
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
        from deepopt.opt.quantization.quantizer import quantize_fp8
        from deepopt.opt.quantization.tensor import QuantizedTensor
        from tests.opt.test_convolution import channel_matrix, from_channel_matrix

        cases = (
            ((2, 6, 17), (10, 3, 3), {"groups": 2, "stride": 2, "padding": 2, "dilation": 2},
             ((2, 4), (4, 4), (2, 4))),
            ((1, 16, 7, 9), (32, 8, 3, 2), {"groups": 2, "stride": (2, 1), "padding": 1, "dilation": (1, 2)},
             ((8, 32), (8, 32), (4, 32))),
            ((1, 4, 4, 5, 7), (8, 4, 2, 2, 2), {"padding": 1, "dilation": (2, 1, 1)},
             ((4, 4), (4, 8), (4, 4))),
            ((1, 3, 5), (6, 1, 5), {"groups": 3, "padding": 5, "dilation": 3},
             ((1, 2), (2, 4), (1, 2))),
            ((1, 4, 5, 7), (8, 4, 2, 2), {"padding": "same"}, ((4, 4), (4, 8), (4, 4))),
        )
        for shape, weight_shape, options, blocks in cases:
            for compressed in (False, True):
                with self.subTest(shape=shape, compressed=compressed):
                    cpu_x, cpu_weight = torch.randn(shape), torch.randn(weight_shape)
                    x, weight = cpu_x.cuda().requires_grad_(), cpu_weight.cuda().requires_grad_()
                    rank = len(shape) - 2
                    if rank == 2:
                        x = x.detach().to(memory_format=torch.channels_last).requires_grad_()
                    settings = FP8Config(enabled=True, backend="triton", out_dtype="fp32", weight_dtype="e5m2",
                                         block_shapes=dict(zip(("activation", "weight", "grad"), blocks)))
                    plan = conv_plan(settings, activation_compression={"enabled": compressed, "group_size": 17})
                    output = getattr(deepopt_f, f"conv{rank}d")(x, weight, plan=plan, **options)
                    gradient = torch.randn(tuple(output.shape))
                    output.backward(gradient.cuda())
                    same = options.get("padding") == "same"
                    source = F.pad(cpu_x, (0, 1) * rank) if same else cpu_x
                    qx = quantize_fp8(channel_matrix(source), block_shape=blocks[0])
                    qw = quantize_fp8(cpu_weight.reshape(weight_shape[0], -1), block_shape=blocks[1],
                                      fp8_dtype=torch.float8_e5m2)
                    qg = quantize_fp8(channel_matrix(gradient), block_shape=blocks[2])
                    saved = qx
                    if compressed:
                        packed = quantize_pack_activation(qx.data.float(), 17, block_shape=blocks[0])
                        saved = QuantizedTensor(dequantize_unpack_activation(packed, qx.data.dtype), qx.scale_inv, blocks[0])
                    original_x = from_channel_matrix(qx.dequantize(), source.shape)
                    saved_x = from_channel_matrix(saved.dequantize(), source.shape)
                    if same:
                        mask = torch.zeros_like(saved_x)
                        interior = (slice(None), slice(None), *(slice(0, size) for size in shape[2:]))
                        mask[interior] = saved_x[interior]
                        saved_x = mask
                    native_options = {**options, "padding": 0} if same else options
                    qweight = qw.dequantize().reshape(weight_shape)
                    qgrad = from_channel_matrix(qg.dequantize(), gradient.shape)
                    expected_output = getattr(F, f"conv{rank}d")(original_x, qweight, **native_options)
                    expected_dx = getattr(torch.nn.grad, f"conv{rank}d_input")(
                        source.shape, qweight, qgrad, **native_options)
                    if same:
                        expected_dx = expected_dx[interior]
                    expected_dw = getattr(torch.nn.grad, f"conv{rank}d_weight")(
                        saved_x, weight_shape, qgrad, **native_options)
                    torch.testing.assert_close(output.cpu(), expected_output, rtol=2e-3, atol=2e-3)
                    torch.testing.assert_close(x.grad.cpu(), expected_dx, rtol=2e-3, atol=2e-3)
                    torch.testing.assert_close(weight.grad.cpu(), expected_dw, rtol=2e-3, atol=2e-3)

    def test_compiled_forward_dx_dw_use_fp8_matrix_instructions(self):
        from deepopt import FP8Config
        from deepopt.opt.backends.triton.convolution.forward import _forward_kernel
        from deepopt.opt.backends.triton.convolution.backward import _input_gradient_kernel
        from deepopt.opt.backends.triton.convolution.weight_gradient_blockwise import _blockwise_weight_gradient_kernel
        from deepopt.opt.nn import functional as deepopt_f
        from deepopt.opt.policy import conv_plan

        compiled_ptx = {}

        def capture(name, original):
            def run(*args, **kwargs):
                compiled = original(*args, **kwargs)
                compiled_ptx[name] = compiled.asm["ptx"]
                return compiled
            return run

        kernels = {"forward": _forward_kernel, "dx": _input_gradient_kernel, "dw": _blockwise_weight_gradient_kernel}
        with ExitStack() as stack:
            for name, kernel in kernels.items():
                stack.enter_context(patch.object(kernel, "run", side_effect=capture(name, kernel.run)))
            x = torch.randn(1, 16, 5, 7, device="cuda", requires_grad=True)
            weight = torch.randn(32, 16, 3, 3, device="cuda", requires_grad=True)
            plan = conv_plan(FP8Config(enabled=True, backend="triton", block_shapes={
                "activation": (4, 8), "weight": (4, 8), "grad": (4, 8),
            }), activation_compression=True)
            deepopt_f.conv2d(x, weight, padding=1, plan=plan).sum().backward()
        self.assertEqual(set(compiled_ptx), set(kernels))
        for name, ptx in compiled_ptx.items():
            with self.subTest(kernel=name):
                instructions = re.findall(r"[^\n]*(?:wgmma\.mma_async|mma\.sync|tcgen05\.mma)[^\n]*", ptx)
                self.assertTrue(any(re.search(r"e4m3|e5m2|f8f6f4", instruction) for instruction in instructions), ptx)
                self.assertFalse(any("tf32" in instruction for instruction in instructions), ptx)


if __name__ == "__main__":
    unittest.main()
