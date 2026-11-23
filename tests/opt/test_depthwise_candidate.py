"""Interpret the actual optional stencil kernels; CUDA compilation is separate."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
import torch

from deepopt.opt.backends.reference import convolution as oracle
from deepopt.opt.backends.triton.convolution.launch import (
    depthwise_eligible, depthwise_splits, geometry_arguments, operand_arguments,
)
from deepopt.opt.nn.convolution import ConvGeometry, source_matrix
from deepopt.opt.quantization.quantizer import quantize_fp8
from tests.opt.convolution_kernel_support import Pointer, kernel_namespace


class DepthwiseCandidateTests(unittest.TestCase):
    def test_preflight_operands_obey_cuda_quantizer_contract(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from tests.cuda.test_flash_attention_backend import _DEPTHWISE_BLOCK_SHAPES

        # Exercise the actual preflight cases through CUDA validation without
        # allocating GPU storage. The CPU oracle permits additional block sizes.
        with FakeTensorMode():
            x = torch.empty(2, 5, 7, 9, device='cuda')
            w = torch.empty(5, 1, 3, 3, device='cuda')
            geometry = ConvGeometry.from_tensors(x, w, stride=2, padding=1, groups=5)
            g = torch.empty(geometry.output_shape, device='cuda')
            for block in _DEPTHWISE_BLOCK_SHAPES:
                for value in (source_matrix(x), w.reshape(5, -1), source_matrix(g)):
                    with self.subTest(block=block, shape=tuple(value.shape)):
                        quantized = quantize_fp8(value, block_shape=block)
                        self.assertEqual(quantized.data.shape, value.shape)
                        self.assertEqual(quantized.data.device, value.device)
                        expected = ((1, 1) if block is None else
                                    tuple((size + b - 1) // b for size, b in zip(value.shape, block)))
                        self.assertEqual(quantized.scale_inv.shape, expected)

    def test_actual_kernel_indexing_forward_dx_dw_and_split_tails(self):
        from tests.cuda.test_flash_attention_backend import _DEPTHWISE_BLOCK_SHAPES

        torch.manual_seed(94)
        tl, kernels = kernel_namespace()
        for rank in (1, 2, 3):
            # Retain CPU-only arbitrary blocks alongside the actual CUDA cases.
            for block in (*_DEPTHWISE_BLOCK_SHAPES, (3, 2)):
                for wide in (False, True):
                    spatial = (5, 7, 6)[-rank:]
                    x = torch.randn(2, 3, *spatial)
                    w = torch.randn(3, 1, *([3] * rank))
                    geometry = ConvGeometry.from_tensors(x, w, stride=2, padding=1, dilation=1, groups=3)
                    if rank == 2:
                        geometry = replace(geometry, synthetic_padding_right=(0, 1))
                    g = torch.randn(geometry.output_shape)
                    qx, qw, qg = [quantize_fp8(v, block_shape=block)
                                   for v in (source_matrix(x), w.reshape(3, -1), source_matrix(g))]
                    self.assertTrue(depthwise_eligible(geometry))

                    def arguments(a, b):
                        values = geometry_arguments(geometry)
                        values.pop('CI'); values.pop('CO')
                        values['CHANNELS'] = values.pop('GROUPS')
                        return {**values, **operand_arguments('A', a), **operand_arguments('B', b),
                                'INDEX64': wide, 'TENSORWISE': block is None}

                    for transpose, a, expected in [(False, qx, oracle.forward(qx, qw, geometry)),
                                                    (True, qg, oracle.input_gradient(qg, qw, geometry))]:
                        output = torch.empty_like(expected)
                        for pid in range((output.numel() + 31) // 32):
                            tl.programs = (pid, 0, 0)
                            kernels['_depthwise_stencil'](*(Pointer(v) for v in
                                (a.data, qw.data, a.scale_inv, qw.scale_inv, output, output)),
                                **arguments(a, qw), TRANSPOSE=transpose, HAS_BIAS=False, BIAS_STRIDE=1, BLOCK=32)
                        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)
                        if not transpose:
                            bias = torch.randn(6)[::2]
                            for pid in range((output.numel() + 31) // 32):
                                tl.programs = (pid, 0, 0)
                                kernels['_depthwise_stencil'](*(Pointer(v) for v in
                                    (a.data, qw.data, a.scale_inv, qw.scale_inv, output, bias)),
                                    **arguments(a, qw), TRANSPOSE=False, HAS_BIAS=True, BIAS_STRIDE=2, BLOCK=32)
                            torch.testing.assert_close(output, oracle.forward(qx, qw, geometry, bias=bias), rtol=2e-5, atol=2e-5)
                    for splits in (1, 3):
                        partial = torch.empty(splits, 3, geometry.kernel_volume)
                        output = torch.empty(geometry.weight_matrix_shape)
                        for split in range(splits):
                            for tap in range(geometry.kernel_volume):
                                tl.programs = (0, tap, split)
                                kernels['_depthwise_weight_partial'](*(Pointer(v) for v in
                                    (qx.data, qg.data, qx.scale_inv, qg.scale_inv, partial)),
                                    **arguments(qx, qg), SPLITS=splits, BLOCK_R=8, BLOCK_C=4)
                        for pid in range((output.numel() + 15) // 16):
                            tl.programs = (pid, 0, 0)
                            kernels['_depthwise_weight_finish'](Pointer(partial), Pointer(output), output.numel(),
                                                               splits, 1 << (splits - 1).bit_length(), 16)
                        torch.testing.assert_close(output, oracle.weight_gradient(qx, qg, geometry), rtol=2e-5, atol=3e-5)

    def test_split_workspace_is_bounded_and_exposes_long_reductions(self):
        self.assertFalse(depthwise_eligible(SimpleNamespace(groups=1_000_000, in_channels=1_000_000,
                                                           out_channels=1_000_000, kernel_volume=25)))
        for channels, taps in ((3, 9), (1152, 25), (8192, 49)):
            splits = depthwise_splits(512 * 224 * 224, channels, taps)
            self.assertGreater(splits, 1)
            self.assertLessEqual(splits * channels * taps * 4, 32 * 1024 * 1024)
