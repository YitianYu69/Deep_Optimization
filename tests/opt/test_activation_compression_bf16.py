import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from deepopt.opt.nn.functional import compressed_linear, do_linear
from deepopt.opt.quantization.activation import PackedActivation, quantize_pack_activation, dequantize_unpack_activation


class ActivationCodecTests(unittest.TestCase):
    def test_words_are_little_lane_first_and_round_ties_even(self):
        x = torch.tensor([[0.0, 0.5, 1.5, 2.5, 3.0, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]])
        packed = quantize_pack_activation(x, group_size=16)
        codes = [0, 0, 2, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]
        expected = sum(code << (index * 2) for index, code in enumerate(codes))
        expected = torch.tensor(expected, dtype=torch.int64).to(torch.int32)
        self.assertEqual(packed.data[0, 0], expected)
        torch.testing.assert_close(dequantize_unpack_activation(packed, torch.float32), torch.tensor([codes], dtype=torch.float32))

    def test_constants_tails_and_noncontiguous_input(self):
        x = torch.arange(77, dtype=torch.float32).reshape(7, 11).T
        packed = quantize_pack_activation(x, group_size=17)
        actual = dequantize_unpack_activation(packed, torch.float32)
        self.assertEqual(tuple(packed.data.shape), (5, 2))
        for start in range(0, x.numel(), 17):
            group = x.reshape(-1)[start:start + 17]
            step = (group.max() - group.min()) / 3
            expected = (torch.round((group - group.min()) / step).clamp(0, 3) * step + group.min()
                        if step > 0 else group)
            torch.testing.assert_close(actual.reshape(-1)[start:start + 17], expected)
        constant = torch.full((3, 19), -7.25)
        compressed = quantize_pack_activation(constant, 11)
        self.assertTrue(torch.all(compressed.step == 0))
        self.assertTrue(torch.all(compressed.data == 0))
        torch.testing.assert_close(dequantize_unpack_activation(compressed, torch.float32), constant)

    def test_rectangular_edge_tiles_and_direct_transpose(self):
        x = torch.arange(35, dtype=torch.float32).reshape(5, 7) - 19
        packed = quantize_pack_activation(x, block_shape=(3, 4))
        actual = dequantize_unpack_activation(packed, torch.float32)
        self.assertEqual(tuple(packed.data.shape), (4, 1))
        self.assertEqual(packed.minimum.tolist(), [-19, -15, 2, 6])
        expected = torch.empty_like(x)
        for row in range(0, 5, 3):
            for col in range(0, 7, 4):
                tile = x[row:row + 3, col:col + 4]
                step = (tile.max() - tile.min()) / 3
                expected[row:row + 3, col:col + 4] = torch.round((tile - tile.min()) / step) * step + tile.min()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(dequantize_unpack_activation(packed, torch.float32, transpose=True), expected.T)

    def test_fp32_metadata_preserves_large_bf16_range(self):
        x = torch.tensor([[-1e8, 0, 1e8, 2e8]], dtype=torch.bfloat16)
        packed = quantize_pack_activation(x)
        self.assertEqual(packed.step.dtype, torch.float32)
        self.assertEqual(packed.minimum.dtype, torch.float32)
        actual = dequantize_unpack_activation(packed, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLessEqual((actual - x.float()).abs().max(), packed.step.max() / 2)

    def test_full_finite_bf16_range_does_not_overflow_affine_math(self):
        maximum = torch.finfo(torch.bfloat16).max
        x = torch.tensor([[-maximum, 0, maximum]], dtype=torch.bfloat16)
        packed = quantize_pack_activation(x)
        self.assertTrue(torch.isfinite(packed.step).all())
        self.assertTrue(torch.isfinite(packed.minimum).all())
        actual = dequantize_unpack_activation(packed, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual[:, (0, 2)], x.float()[:, (0, 2)], rtol=2e-7, atol=0)
        self.assertLessEqual((actual - x.float()).abs().max(), packed.step.max() / 2)
        self.assertTrue(torch.isfinite(dequantize_unpack_activation(packed, torch.bfloat16)).all())

    def test_fp8_decode_clamps_rounded_affine_endpoint(self):
        packed = PackedActivation(torch.tensor([[3]], dtype=torch.int32), torch.tensor([0.125]),
                                  torch.tensor([448.0]), (1, 1), 16)
        actual = dequantize_unpack_activation(packed, torch.float8_e4m3fn)
        self.assertEqual(actual.float().item(), 448.0)

    def test_one_bit_full_bf16_range_uses_finite_half_step_metadata(self):
        maximum = torch.finfo(torch.bfloat16).max
        x = torch.tensor([[-maximum, 0, maximum]], dtype=torch.bfloat16)
        packed = quantize_pack_activation(x, bits=1)
        self.assertTrue(torch.isfinite(packed.step).all())
        self.assertTrue(torch.isfinite(packed.minimum).all())
        self.assertTrue((packed.step < 0).all())
        restored = dequantize_unpack_activation(packed, dtype=torch.float32)
        self.assertTrue(torch.isfinite(restored).all())
        torch.testing.assert_close(restored[:, (0, 2)], x.float()[:, (0, 2)], rtol=0, atol=0)
        torch.testing.assert_close(restored[:, 1], x.float()[:, 0], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(dequantize_unpack_activation(packed, dtype=torch.bfloat16)).all())

    def test_default_storage_and_validation(self):
        packed = quantize_pack_activation(torch.ones(4, 256))
        self.assertEqual(packed.bits, 2)
        self.assertEqual(packed.group_size, 512)
        self.assertEqual(tuple(packed.data.shape), (2, 32))
        self.assertEqual(sum(value.numel() * value.element_size()
                             for value in (packed.data, packed.step, packed.minimum)), 272)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(group_size=invalid), self.assertRaises(ValueError):
                quantize_pack_activation(torch.ones(2, 3), invalid)
        with self.assertRaises(ValueError):
            quantize_pack_activation(torch.ones(2, 3), block_shape=(1, 0))
        with self.assertRaises(TypeError):
            quantize_pack_activation(torch.ones(2, 3, dtype=torch.int64))
        with self.assertRaises(ValueError):
            PackedActivation(packed.data, packed.step[:1], packed.minimum, packed.shape, packed.group_size)

    def test_variable_bits_codes_storage_and_masked_tails(self):
        for bits in (1, 2, 4, 8):
            with self.subTest(bits=bits):
                qmax, lanes = (1 << bits) - 1, 32 // bits
                codes = [qmax if index % 3 == 0 else index % (qmax + 1)
                         for index in range(lanes)]
                codes[0], codes[1] = 0, qmax
                x = torch.tensor([codes], dtype=torch.float32)
                packed = quantize_pack_activation(x, lanes, bits=bits)
                expected_word = sum(code << (index * bits) for index, code in enumerate(codes))
                expected_word = torch.tensor(expected_word, dtype=torch.int64).to(torch.int32)
                self.assertEqual(packed.bits, bits)
                torch.testing.assert_close(packed.data, expected_word.reshape(1, 1))
                torch.testing.assert_close(dequantize_unpack_activation(packed, torch.float32), x)

                x = torch.arange(77, dtype=torch.float32).reshape(7, 11).T / 7 - 3
                packed = quantize_pack_activation(x, 17, bits=bits)
                self.assertEqual(tuple(packed.data.shape), (5, (17 + lanes - 1) // lanes))
                expected = torch.empty_like(x.reshape(-1))
                for start in range(0, x.numel(), 17):
                    chunk = x.reshape(-1)[start:start + 17]
                    step = (chunk.max() - chunk.min()) / qmax
                    expected[start:start + 17] = torch.round((chunk - chunk.min()) / step) * step + chunk.min()
                torch.testing.assert_close(dequantize_unpack_activation(packed, torch.float32), expected.reshape_as(x))
                torch.testing.assert_close(dequantize_unpack_activation(packed, torch.float32, transpose=True),
                                           expected.reshape_as(x).T)

                constant = torch.full((3, 19), -7.25)
                packed_constant = quantize_pack_activation(constant, 11, bits=bits)
                self.assertTrue(torch.all(packed_constant.step == 0))
                self.assertTrue(torch.all(packed_constant.data == 0))
                torch.testing.assert_close(dequantize_unpack_activation(packed_constant, torch.float32), constant)

    def test_variable_bits_rectangular_edges_fake_and_validation(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        x = torch.arange(35, dtype=torch.float32).reshape(5, 7) - 19
        for bits in (1, 2, 4, 8):
            with self.subTest(bits=bits):
                qmax = (1 << bits) - 1
                packed = quantize_pack_activation(x, block_shape=(3, 4), bits=bits)
                expected = torch.empty_like(x)
                for row in range(0, 5, 3):
                    for col in range(0, 7, 4):
                        tile = x[row:row + 3, col:col + 4]
                        step = (tile.max() - tile.min()) / qmax
                        expected[row:row + 3, col:col + 4] = torch.round((tile - tile.min()) / step) * step + tile.min()
                torch.testing.assert_close(dequantize_unpack_activation(packed, torch.float32), expected)
                with FakeTensorMode():
                    fake = quantize_pack_activation(torch.empty(5, 7), block_shape=(3, 4), bits=bits)
                    self.assertEqual(fake.bits, bits)
                    self.assertEqual(tuple(fake.data.shape), tuple(packed.data.shape))
                    self.assertEqual(tuple(dequantize_unpack_activation(fake, transpose=True).shape), (7, 5))
        for bits in (0, 3, 16, True, 2.0, None):
            with self.subTest(invalid_bits=bits), self.assertRaises(ValueError):
                quantize_pack_activation(x, bits=bits)
            with self.subTest(invalid_payload_bits=bits), self.assertRaises(ValueError):
                PackedActivation(torch.zeros(1, 1, dtype=torch.int32), torch.zeros(1), torch.zeros(1),
                                 (1, 1), 16, bits=bits)

    def test_variable_bits_fp8_codec_rounds_before_compressing(self):
        from deepopt.opt.quantization.activation_fp8 import quantize_pack_fp8_activation
        from deepopt.opt.quantization.quantizer import quantize_fp8

        x = torch.linspace(-7.5, 7.0, 77).reshape(7, 11).T
        for bits in (1, 2, 4, 8):
            for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                for block in (None, (1, 4), (3, 5)):
                    with self.subTest(bits=bits, dtype=dtype, block=block):
                        quantized, packed = quantize_pack_fp8_activation(
                            x, fp8_dtype=dtype, block_shape=block, group_size=17, bits=bits)
                        original = quantize_fp8(x, fp8_dtype=dtype, block_shape=block)
                        torch.testing.assert_close(quantized.data.float(), original.data.float(), rtol=0, atol=0)
                        torch.testing.assert_close(quantized.scale_inv, original.scale_inv, rtol=0, atol=0)
                        expected = quantize_pack_activation(original.data.float(), 17, block_shape=block, bits=bits)
                        self.assertEqual(packed.bits, bits)
                        torch.testing.assert_close(packed.data, expected.data, rtol=0, atol=0)
                        torch.testing.assert_close(packed.step, expected.step, rtol=0, atol=0)
                        torch.testing.assert_close(packed.minimum, expected.minimum, rtol=0, atol=0)

    def test_variable_bits_reach_fp8_linear_saved_operand_and_weight_gradient(self):
        from deepopt import FP8Config
        from deepopt.opt.backends import gemm_nt
        from deepopt.opt.policy import fp8_plan
        from deepopt.opt.quantization.quantizer import quantize_fp8
        from deepopt.opt.quantization.tensor import QuantizedTensor

        torch.manual_seed(82)
        for bits in (1, 2, 4, 8):
            for blocks in (None, (1, 4), (2, 3)):
                with self.subTest(bits=bits, blocks=blocks):
                    x = torch.randn(5, 7, dtype=torch.bfloat16, requires_grad=True)
                    weight = torch.randn(4, 7, requires_grad=True)
                    gradient = torch.randn(5, 4, dtype=torch.bfloat16)
                    plan = fp8_plan(FP8Config(
                        enabled=True, block_shapes={"activation": blocks, "weight": None, "grad": None}),
                        activation_compression={"enabled": True, "group_size": 17, "bits": bits})
                    output = do_linear(x, weight, plan=plan)
                    output.backward(gradient)
                    rowwise = blocks is not None and blocks[0] == 1
                    qx = quantize_fp8(x.detach().T if rowwise else x.detach(), block_shape=blocks)
                    packed = quantize_pack_activation(qx.data.float(), 17, block_shape=blocks, bits=bits)
                    restored = dequantize_unpack_activation(packed, dtype=qx.data.dtype, transpose=not rowwise)
                    inverse = qx.scale_inv if rowwise else qx.scale_inv.T
                    layout = blocks if rowwise or blocks is None else (blocks[1], blocks[0])
                    saved = QuantizedTensor(restored, inverse, layout)
                    qg = quantize_fp8(gradient).transpose(contiguous=False)
                    expected = gemm_nt(qg, saved, backend="reference", out_dtype=torch.bfloat16)
                    torch.testing.assert_close(weight.grad, expected.float(), rtol=0, atol=0)

    def test_delayed_compressed_codec_keeps_scale_snapshot_for_every_bit_width(self):
        from deepopt import fp8_scaling_scope
        from deepopt.opt.scaling.delayed import DelayedScalingState

        x = torch.linspace(-3, 5, 35).reshape(5, 7)
        for bits in (1, 2, 4, 8):
            with self.subTest(bits=bits):
                state = DelayedScalingState(
                    {}, dict.fromkeys(("activation", "weight", "grad"), torch.float8_e4m3fn), device="cpu")
                state.scales.fill_(16)
                with fp8_scaling_scope():
                    quantized, packed = state.quantize_pack_activation(x, group_size=17, bits=bits)
                expected_values = (x * 16).to(torch.float8_e4m3fn)
                expected = quantize_pack_activation(expected_values.float(), 17, bits=bits)
                self.assertEqual(packed.bits, bits)
                torch.testing.assert_close(quantized.data.float(), expected_values.float(), rtol=0, atol=0)
                torch.testing.assert_close(quantized.scale_inv, torch.tensor([[1 / 16]]), rtol=0, atol=0)
                torch.testing.assert_close(packed.data, expected.data, rtol=0, atol=0)


def native_compression_plan(group_size=512, bits=2):
    return {
        "fp8": False,
        "activation_compression": {"enabled": True, "group_size": group_size, "bits": bits},
    }


class CompressedBF16LinearTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(812)

    def test_native_forward_and_dx_with_compressed_weight_gradient(self):
        x = torch.randn(2, 3, 7, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(5, 7, dtype=torch.bfloat16, requires_grad=True)
        bias = torch.randn(5, dtype=torch.bfloat16, requires_grad=True)
        gradient = torch.randn(2, 3, 5, dtype=torch.bfloat16)
        saved = []
        restored = []
        def save(tensor):
            saved.append(tensor)
            return tensor.clone()
        def restore(tensor):
            restored.append(tensor)
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(save, restore):
            output = do_linear(x, weight, bias, plan=native_compression_plan(16))
            output.backward(gradient)
        torch.testing.assert_close(output, F.linear(x, weight, bias), rtol=0, atol=0)
        torch.testing.assert_close(x.grad, (gradient.reshape(-1, 5) @ weight).reshape(x.shape), rtol=0, atol=0)
        decoded = dequantize_unpack_activation(quantize_pack_activation(x.detach().reshape(-1, 7), 16), torch.bfloat16)
        torch.testing.assert_close(weight.grad, gradient.reshape(-1, 5).T @ decoded, rtol=0, atol=0)
        torch.testing.assert_close(bias.grad, gradient.reshape(-1, 5).float().sum(0).to(torch.bfloat16), rtol=0, atol=0)
        self.assertEqual(len(saved), 4)
        self.assertEqual(len(restored), 4)
        self.assertEqual(saved[0].dtype, torch.int32)
        self.assertFalse(any(tensor is x for tensor in saved))

    def test_cpu_autocast_and_fp32_master_parameters(self):
        x = torch.randn(9, 11, requires_grad=True)
        weight = torch.randn(6, 11, requires_grad=True)
        bias = torch.randn(6, requires_grad=True)
        grad = torch.randn(9, 6, dtype=torch.bfloat16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = do_linear(x, weight, bias, plan=native_compression_plan(32))
            reference = F.linear(x, weight, bias)
        native_dx, native_db = torch.autograd.grad(reference, (x, bias), grad)
        output.backward(grad)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(weight.grad.dtype, torch.float32)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        decoded = dequantize_unpack_activation(quantize_pack_activation(x.detach().bfloat16(), 32), torch.bfloat16)
        torch.testing.assert_close(weight.grad, (grad.T @ decoded).float(), rtol=0, atol=0)
        torch.testing.assert_close(x.grad, native_dx, rtol=0, atol=0)
        torch.testing.assert_close(bias.grad, native_db, rtol=0, atol=0)

    def test_global_bit_width_reaches_saved_native_linear_and_backward(self):
        for bits in (1, 2, 4, 8):
            with self.subTest(bits=bits):
                x = torch.randn(5, 11, dtype=torch.bfloat16, requires_grad=True)
                weight = torch.randn(7, 11, dtype=torch.bfloat16, requires_grad=True)
                gradient = torch.randn(5, 7, dtype=torch.bfloat16)
                saved = []
                with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value,
                                                            lambda value: value):
                    output = do_linear(x, weight, plan=native_compression_plan(17, bits))
                output.backward(gradient)
                decoded = dequantize_unpack_activation(quantize_pack_activation(x.detach(), 17, bits=bits), torch.bfloat16)
                torch.testing.assert_close(output, F.linear(x, weight), rtol=0, atol=0)
                torch.testing.assert_close(weight.grad, gradient.T @ decoded, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, gradient @ weight.detach(), rtol=0, atol=0)
                lanes = 32 // bits
                self.assertEqual(tuple(saved[0].shape), (4, (17 + lanes - 1) // lanes))

    def test_does_not_pack_when_weight_gradient_is_not_requested(self):
        x = torch.randn(3, 5, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(2, 5, dtype=torch.bfloat16)
        with patch("deepopt.opt.nn.autograd.native_linear.quantize_pack_activation", wraps=quantize_pack_activation) as pack:
            output = do_linear(x, weight, plan=native_compression_plan())
            output.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones(3, 2, dtype=x.dtype) @ weight, rtol=0, atol=0)
            with torch.no_grad():
                output = do_linear(x, weight.clone().requires_grad_(), plan=native_compression_plan())
            pack.assert_not_called()
        self.assertFalse(output.requires_grad)

    def test_explicit_compressed_linear_owns_dispatch_inside_deepopt_scopes(self):
        import deepopt

        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                x = torch.randn(3, 7, requires_grad=True)
                weight = torch.randn(5, 7, requires_grad=True)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    expected = F.linear(x, weight)
                saved = []
                with patch("deepopt.opt.nn.autograd.native_linear.quantize_pack_activation", wraps=quantize_pack_activation) as pack:
                    with torch.autograd.graph.saved_tensors_hooks(
                        lambda value: saved.append(value) or value, lambda value: value
                    ):
                        with deepopt.autocast("cpu", dtype=torch.bfloat16, fp8=fp8, activation_compression=True):
                            actual = compressed_linear(x, weight, group_size=16)
                        actual.sum().backward()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(pack.call_count, 1)
                self.assertEqual(len(saved), 4)
                self.assertEqual(saved[0].dtype, torch.int32)

    def test_vector_bias_only_empty_batch_and_validation(self):
        x = torch.randn(5, dtype=torch.bfloat16)
        weight = torch.randn(2, 5, dtype=torch.bfloat16)
        bias = torch.randn(2, dtype=torch.bfloat16, requires_grad=True)
        do_linear(x, weight, bias, plan=native_compression_plan()).sum().backward()
        torch.testing.assert_close(bias.grad, torch.ones_like(bias))
        empty = torch.empty(0, 5, dtype=torch.bfloat16, requires_grad=True)
        trainable_weight = weight.clone().requires_grad_()
        do_linear(empty, trainable_weight, plan=native_compression_plan()).sum().backward()
        torch.testing.assert_close(trainable_weight.grad, torch.zeros_like(weight))
        with self.assertRaises(ValueError):
            do_linear(x, torch.randn(2, 6), plan=native_compression_plan())
        with self.assertRaises(ValueError):
            do_linear(x, weight, plan=native_compression_plan(0))


if __name__ == "__main__":
    unittest.main()
