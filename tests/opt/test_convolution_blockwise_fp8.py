"""Execute real kernel bodies on CPU against dequantized convolution oracles."""

from dataclasses import replace
from itertools import product
import unittest

import torch

from deepopt import FP8Config
from deepopt.opt.backends.reference import convolution as reference
from deepopt.opt.backends.triton.convolution.launch import geometry_arguments, operand_arguments, launch_options
from deepopt.opt.backends.triton.convolution.scales import scale_chunk_length
from deepopt.opt.nn.convolution import ConvGeometry, source_matrix
from deepopt.opt.nn.convolution.dispatch import convolution_execution
from deepopt.opt.policy import conv_plan
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.opt.quantization.tensor import QuantizedTensor
from tests.opt.convolution_kernel_support import Pointer, kernel_namespace


class ScaleSegmentTests(unittest.TestCase):
    def test_segments_cover_each_sample_and_stop_before_either_scale_changes(self):
        for a_start, a_step, a_block, b_start, b_step, b_block in product(
            (0, 3, 11), (1, 2, 5), (1, 4, 32), (0, 5), (1, 3), (2, 8),
        ):
            position = 0
            while position < 73:
                a = a_start + position * a_step
                b = b_start + position * b_step
                length = scale_chunk_length(a, a_step, a_block, b, b_step, b_block, min(32, 73 - position))
                self.assertGreater(length, 0)
                self.assertLessEqual(length, min(32, 73 - position))
                self.assertEqual(a // a_block, (a + (length - 1) * a_step) // a_block)
                self.assertEqual(b // b_block, (b + (length - 1) * b_step) // b_block)
                if length < min(32, 73 - position):
                    self.assertTrue((a + length * a_step) // a_block != a // a_block
                                    or (b + length * b_step) // b_block != b // b_block)
                position += length
            self.assertEqual(position, 73)

    def test_policy_reports_blockwise_fp8_for_forward(self):
        geometry = ConvGeometry.from_tensors(torch.empty(1, 4, 7, 9), torch.empty(6, 4, 3, 3), padding=1)
        settings = FP8Config(enabled=True, backend="triton", block_shapes={
            "activation": (1, 32), "weight": (4, 32), "grad": (1, 32),
        })
        info = convolution_execution(geometry, conv_plan(settings))
        self.assertEqual(info["forward"], "implicit_blockwise_fp8")
        self.assertEqual(info["backward"], "triton")
        self.assertIsNone(info["fallback_reason"])


class BlockwiseConvolutionKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(1669)
        self.language, self.kernels = kernel_namespace()

    def _launch(self, operation, qa, qb, geometry, *, bias=None, splits=1, index64=False, stripe_w=128):
        tensorwise = qa.scale_inv.numel() == 1 and qb.scale_inv.numel() == 1
        options = launch_options(tensorwise=tensorwise)
        args = {**geometry_arguments(geometry), **operand_arguments("A", qa), **operand_arguments("B", qb),
                "TENSORWISE": tensorwise, "INDEX64": index64,
                **{name: value for name, value in options.items() if name.startswith("BLOCK_")}}
        if operation == "forward":
            shape, m, n = geometry.grad_matrix_shape, geometry.grad_matrix_shape[0], geometry.out_channels_per_group
            kernel = "_forward_kernel"
        elif operation == "input_gradient":
            shape, m, n = geometry.input_matrix_shape, geometry.input_matrix_shape[0], geometry.in_channels_per_group
            kernel = "_input_gradient_kernel"
        else:
            shape, m = geometry.weight_matrix_shape, geometry.out_channels_per_group
            n = geometry.weight_matrix_shape[1] if tensorwise else geometry.in_channels_per_group
            kernel = "_weight_gradient_kernel" if tensorwise else "_blockwise_weight_gradient_kernel"
            args["SPLIT_K"] = splits
            if not tensorwise:
                args["STRIPE_W"] = stripe_w
        output = torch.full(((splits, *shape) if operation == "weight_gradient" else shape), float("nan"))
        pointers = [Pointer(qa.data), Pointer(qb.data), Pointer(qa.scale_inv), Pointer(qb.scale_inv), Pointer(output)]
        if operation == "forward":
            pointers.append(Pointer(output if bias is None else bias))
            args.update(HAS_BIAS=bias is not None, BIAS_STRIDE=1 if bias is None else bias.stride(0))
        elif operation == "input_gradient" and bias is not None:
            args.update(bias=Pointer(bias), HAS_BIAS=True, BIAS_STRIDE=bias.stride(0))
        tiles = (m + options["BLOCK_M"] - 1) // options["BLOCK_M"] * (
            (n + options["BLOCK_N"] - 1) // options["BLOCK_N"]) * geometry.groups
        taps = geometry.kernel_volume if operation == "weight_gradient" and not tensorwise else 1
        for pid, split in product(range(tiles * taps), range(splits)):
            self.language.programs = (pid, split, 0)
            self.kernels[kernel](*pointers, **args)
        self.assertTrue(torch.isfinite(output).all(), f"unwritten {operation} output")
        if operation == "weight_gradient":
            if splits > 1:
                reduced = torch.full(shape, float("nan"))
                for pid in range((reduced.numel() + 255) // 256):
                    self.language.programs = (pid, 0, 0)
                    self.kernels["_reduce_weight_gradient"](
                        Pointer(output), Pointer(reduced), ELEMENTS=reduced.numel(), SPLITS=splits,
                        SPLIT_BLOCK=1 << (splits - 1).bit_length(), BLOCK=256)
                return reduced
            return output[0]
        return output

    def _check(self, shape, weight_shape, options, blocks, *, compressed=False, strided=False,
               splits=1, synthetic=(), index64=False, stripe_w=128):
        x, weight = torch.randn(shape) * 7, torch.randn(weight_shape) * 0.3
        geometry = ConvGeometry.from_tensors(x, weight, **options)
        if synthetic:
            geometry = replace(geometry, synthetic_padding_right=synthetic)
        grad = torch.randn(geometry.output_shape) * 3
        qx = quantize_fp8(source_matrix(x), block_shape=blocks[0])
        qw = quantize_fp8(weight.reshape(weight_shape[0], -1), block_shape=blocks[1], fp8_dtype=torch.float8_e5m2)
        qg = quantize_fp8(source_matrix(grad), block_shape=blocks[2])
        saved_x = qx
        if compressed:
            packed = quantize_pack_activation(qx.data.float(), group_size=17, bits=2, block_shape=qx.block_shape)
            saved_x = QuantizedTensor(dequantize_unpack_activation(packed, dtype=qx.data.dtype), qx.scale_inv, qx.block_shape)
        if strided:
            qx, qw, qg, saved_x = [QuantizedTensor(q.data.T.contiguous().T, q.scale_inv.T.contiguous().T, q.block_shape)
                                      for q in (qx, qw, qg, saved_x)]
        bias = torch.randn(geometry.out_channels * 2)[::2]
        actual = self._launch("forward", qx, qw, geometry, bias=bias, index64=index64)
        dx = self._launch("input_gradient", qg, qw, geometry, index64=index64)
        dw = self._launch("weight_gradient", qg, saved_x, geometry, splits=splits, index64=index64, stripe_w=stripe_w)
        torch.testing.assert_close(actual, reference.forward(qx, qw, geometry, bias=bias), rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(dx, reference.input_gradient(qg, qw, geometry), rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(dw, reference.weight_gradient(saved_x, qg, geometry), rtol=3e-5, atol=3e-5)
        self.assertGreater(self.language.dots, 0)

    def test_mixed_scale_layouts_all_ranks_groups_stride_dilation_and_padding(self):
        cases = (
            ((2, 6, 9), (10, 3, 3), {"groups": 2, "stride": 2, "padding": 2, "dilation": 2}),
            ((1, 4, 5, 6), (6, 2, 3, 2), {"groups": 2, "stride": (2, 1), "padding": 1, "dilation": (1, 2)}),
            ((1, 2, 4, 3, 5), (3, 2, 2, 2, 2), {"stride": (1, 2, 1), "padding": (1, 0, 2), "dilation": (2, 1, 1)}),
            ((1, 3, 5), (6, 1, 5), {"groups": 3, "padding": 5, "dilation": 3}),
        )
        layouts = ((None, None, None), ((2, 4), None, None), (None, (2, 4), None),
                   (None, None, (2, 4)), ((4, 2), (2, 4), (2, 2)))
        for case, blocks in product(cases, layouts):
            with self.subTest(shape=case[0], blocks=blocks):
                self._check(*case, blocks, compressed=True, strided=True)

    def test_one_position_scales_still_use_fp8_dots_and_deterministic_splits(self):
        self._check((2, 4, 11), (6, 4, 3), {"padding": 1}, ((1, 4), (2, 4), (1, 2)), splits=3, index64=True)

    def test_synthetic_padding_stays_zero_after_lossy_decode(self):
        self._check((1, 3, 6, 8), (4, 3, 2, 2), {}, ((4, 2), (2, 4), (2, 2)),
                    compressed=True, synthetic=(1, 1), splits=3)

    def test_tiles_cross_channel_boundaries_and_tensorwise_split_k(self):
        self._check((1, 35, 5), (67, 35, 2), {"padding": 1}, ((8, 32), (16, 32), (8, 32)), splits=3)
        self._check((1, 5, 9), (7, 5, 3), {"padding": 1}, (None, None, None), splits=3)

    def test_coarse_blocks_cross_width_stripes_scale_edges_and_full_dot_tiles(self):
        for stripe_w in (4, 128):
            with self.subTest(stripe_w=stripe_w):
                self._check((1, 4, 259), (6, 4, 3), {"padding": 1, "stride": 2},
                            ((128, 4), (4, 8), (64, 4)), splits=3, stripe_w=stripe_w)
        self._check((1, 4, 9), (6, 4, 3), {"padding": 1}, ((64, 64), (64, 64), (64, 64)))


if __name__ == "__main__":
    unittest.main()
