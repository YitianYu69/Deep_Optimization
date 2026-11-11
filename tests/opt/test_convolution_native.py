"""Native spatial convolution and compressed saved-source contracts."""

import unittest

import torch
from torch.nn import functional as F

from deepopt.opt.nn.autograd.convolution.native import _NativeConvolutionFunction
from deepopt.opt.nn.convolution import ConvGeometry, restore_matrix, source_matrix
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation


class ConvolutionGeometryTests(unittest.TestCase):
    def test_source_matrix_has_spatial_rows_not_contiguous_channel_chunks(self):
        for spatial in ((7,), (3, 7), (2, 3, 7)):
            with self.subTest(spatial=spatial):
                shape = (2, 4, *spatial)
                x = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
                matrix = source_matrix(x)
                torch.testing.assert_close(matrix[0], x[(0, slice(None), *([0] * len(spatial)))])
                torch.testing.assert_close(restore_matrix(matrix, shape), x)
                strided = x[..., ::2]
                torch.testing.assert_close(restore_matrix(source_matrix(strided), strided.shape), strided)

    def test_geometry_grouping_dilation_and_padded_dimensions(self):
        geometry = ConvGeometry.from_tensors(torch.empty(2, 6, 11, 13),
                                             torch.empty(8, 3, 3, 2),
                                             stride=(2, 1), padding=(2, 0),
                                             dilation=(2, 1), groups=2)
        self.assertEqual(geometry.output_spatial, (6, 12))
        self.assertEqual(geometry.input_matrix_shape, (286, 6))
        self.assertEqual(geometry.grad_matrix_shape, (144, 8))
        self.assertEqual(geometry.weight_matrix_shape, (8, 18))
        self.assertEqual(geometry.in_channels_per_group, 3)
        self.assertEqual(geometry.out_channels_per_group, 4)
        self.assertEqual(geometry.input_spatial_3d, (1, 11, 13))
        self.assertEqual(geometry.padding_3d, (0, 2, 0))
        self.assertFalse(geometry.pointwise)
        for rank in (1, 2, 3):
            geometry = ConvGeometry.from_tensors(torch.empty(2, 4, *([5] * rank)),
                                                 torch.empty(6, 4, *([1] * rank)))
            self.assertTrue(geometry.pointwise)

    def test_invalid_geometry_fails_before_kernel_execution(self):
        x, weight = torch.empty(2, 4, 5), torch.empty(6, 4, 3)
        for arguments in ({"stride": 0}, {"padding": -1}, {"dilation": 0},
                          {"groups": True}, {"stride": (1, 1)}, {"padding": "same"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ConvGeometry.from_tensors(x, weight, **arguments)
        with self.assertRaises(ValueError):
            ConvGeometry.from_tensors(x, torch.empty(6, 4, 7))
        with self.assertRaises(ValueError):
            ConvGeometry.from_tensors(x, weight, groups=2)


class NativeConvolutionCompressionTests(unittest.TestCase):
    def _check(self, rank, *, groups=1, channels_last=False, weight_channels_last=False,
               amp=False, strided=False, stride_value=1, dilation_value=1):
        torch.manual_seed(80 + rank)
        spatial = (9,) if rank == 1 else ((7, 8) if rank == 2 else (5, 6, 7))
        x = torch.randn(2, 4, *spatial)
        if strided:
            x = x[..., ::2]
        if channels_last:
            memory_format = torch.channels_last if rank == 2 else torch.channels_last_3d
            x = x.contiguous(memory_format=memory_format)
        x = x.detach().requires_grad_()
        weight = torch.randn(6 if groups != 4 else 4, 4 // groups, *([3] * rank), requires_grad=True)
        if weight_channels_last:
            memory_format = torch.channels_last if rank == 2 else torch.channels_last_3d
            weight = weight.detach().contiguous(memory_format=memory_format).requires_grad_()
        bias = torch.randn(weight.shape[0], requires_grad=True)
        stride = (stride_value,) * rank
        padding, dilation = ((dilation_value,) * rank, (dilation_value,) * rank)
        geometry = ConvGeometry.from_tensors(x, weight, stride, padding, dilation, groups)
        convolution = (F.conv1d, F.conv2d, F.conv3d)[rank - 1]
        policy = {"enabled": True, "group_size": 17, "bits": 2}
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            output = _NativeConvolutionFunction.apply(x, weight, bias, geometry, policy)
            expected = convolution(x, weight, bias, stride, padding, dilation, groups)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        grad = torch.randn_like(output)
        expected_dx, _, expected_db = torch.autograd.grad(expected, (x, weight, bias), grad)
        restored_matrix = dequantize_unpack_activation(quantize_pack_activation(source_matrix(x.detach()).to(output.dtype),
                                                           17, bits=2), dtype=output.dtype)
        restore_format = (torch.channels_last if rank == 2 else torch.channels_last_3d
                          ) if channels_last else torch.contiguous_format
        restored = restore_matrix(restored_matrix, geometry.input_shape, restore_format)
        rounded_weight = weight.detach().to(output.dtype).requires_grad_()
        oracle_output = convolution(restored, rounded_weight, None, stride, padding, dilation, groups)
        expected_dw = torch.autograd.grad(oracle_output, rounded_weight, grad)[0].to(weight.dtype)
        output.backward(grad)
        torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)
        torch.testing.assert_close(weight.grad, expected_dw, rtol=0, atol=0)
        torch.testing.assert_close(bias.grad, expected_db, rtol=0, atol=0)

    def test_all_spatial_ranks_match_explicit_reconstruction_gradients(self):
        for rank in (1, 2, 3):
            for groups in (1, 2, 4):
                with self.subTest(rank=rank, groups=groups):
                    self._check(rank, groups=groups)

    def test_native_amp_rounding_channels_last_and_strided_sources(self):
        for rank in (1, 2, 3):
            with self.subTest(rank=rank, layout="strided"):
                self._check(rank, amp=True, strided=True)
            if rank > 1:
                with self.subTest(rank=rank, layout="channels_last"):
                    self._check(rank, amp=True, channels_last=True)

    def test_stride_dilation_and_overlapping_input_gradients(self):
        for rank in (1, 2, 3):
            with self.subTest(rank=rank):
                self._check(rank, groups=2, stride_value=2, dilation_value=2)

    def test_grouped_amp_uses_native_input_and_bias_reductions(self):
        for rank in (1, 2, 3):
            for groups in (2, 4):
                with self.subTest(rank=rank, groups=groups):
                    self._check(rank, groups=groups, amp=True)

    def test_weight_layout_is_preserved_for_native_gradient_selection(self):
        for rank in (2, 3):
            with self.subTest(rank=rank):
                self._check(rank, weight_channels_last=True, amp=True)

    def test_packed_source_is_saved_through_autograd_hooks(self):
        x = torch.randn(2, 4, 7, 8, requires_grad=True)
        weight = torch.randn(6, 4, 3, 3, requires_grad=True)
        geometry = ConvGeometry.from_tensors(x, weight, padding=1)
        saved = []

        def save(tensor):
            saved.append((tensor.dtype, tuple(tensor.shape), tensor.numel() * tensor.element_size()))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
            output = _NativeConvolutionFunction.apply(x, weight, None, geometry,
                                                      {"group_size": 512, "bits": 2})
        self.assertEqual(saved[0][0], torch.int32)
        self.assertEqual(saved[-1][:2], (weight.dtype, tuple(weight.shape)))
        self.assertLess(sum(item[2] for item in saved[:-1]), x.numel() * x.element_size())
        self.assertFalse(any(shape == tuple(x.shape) for _, shape, _ in saved))
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)

    def test_input_only_gradient_does_not_save_or_compress_source(self):
        x = torch.randn(2, 4, 9, requires_grad=True)
        weight = torch.randn(6, 4, 3)
        geometry = ConvGeometry.from_tensors(x, weight)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor,
                                                    lambda tensor: tensor):
            output = _NativeConvolutionFunction.apply(x, weight, None, geometry,
                                                      {"group_size": 512, "bits": 2})
        self.assertEqual(len(saved), 1)
        self.assertIs(saved[0], weight)
        output.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_empty_batch_does_not_require_a_codec_group(self):
        x = torch.randn(0, 4, 7, requires_grad=True)
        weight = torch.randn(6, 4, 3, requires_grad=True)
        bias = torch.randn(6, requires_grad=True)
        geometry = ConvGeometry.from_tensors(x, weight)
        output = _NativeConvolutionFunction.apply(x, weight, bias, geometry,
                                                  {"group_size": 512, "bits": 2})
        self.assertEqual(tuple(output.shape), (0, 6, 5))
        output.sum().backward()
        self.assertEqual(tuple(x.grad.shape), tuple(x.shape))
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight))
        torch.testing.assert_close(bias.grad, torch.zeros_like(bias))


if __name__ == "__main__":
    unittest.main()
