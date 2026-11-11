"""Real CUDA checks for fused FP8 casts and saved-activation compression."""

import unittest

import torch

from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class ActivationCompressionCUDATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(719)

    def test_variable_bit_widths_fused_fp8_cast_and_native_codec(self):
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
        from deepopt.opt.quantization.activation_fp8 import quantize_pack_fp8_activation
        from deepopt.opt.quantization.quantizer import quantize_fp8

        cpu = torch.randn(37, 65, dtype=torch.bfloat16).T
        value = cpu.T.contiguous().to("cuda").T
        # The large group and large tile exercise the chunked fused conversion,
        # while group 17 forces partial words for every supported bit width.
        for bits in (1, 2, 4, 8):
            for group_size, block in ((17, None), (20003, None), (512, (256, 128))):
                with self.subTest(bits=bits, group_size=group_size, block=block):
                    expected_q = quantize_fp8(cpu, block_shape=block)
                    expected = quantize_pack_activation(expected_q.data.float(), group_size,
                                               block_shape=block, bits=bits)
                    actual_q, packed = quantize_pack_fp8_activation(
                        value, block_shape=block, group_size=group_size, bits=bits, check_finite=False)
                    self.assertEqual(packed.bits, bits)
                    torch.testing.assert_close(actual_q.data.float().cpu(), expected_q.data.float(), rtol=0, atol=0)
                    torch.testing.assert_close(packed.data.cpu(), expected.data, rtol=0, atol=0)
                    torch.testing.assert_close(packed.step.cpu(), expected.step, rtol=1e-6, atol=0)
                    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                        actual = dequantize_unpack_activation(packed, dtype=dtype, transpose=True)
                        reference = dequantize_unpack_activation(expected, dtype=dtype, transpose=True)
                        torch.testing.assert_close(actual.float().cpu(), reference.float(), rtol=0, atol=0)

            packed = quantize_pack_activation(value, 17, bits=bits)
            expected = quantize_pack_activation(cpu, 17, bits=bits)
            torch.testing.assert_close(packed.data.cpu(), expected.data, rtol=0, atol=0)
            actual = dequantize_unpack_activation(packed, dtype=torch.bfloat16)
            reference = dequantize_unpack_activation(expected, dtype=torch.bfloat16)
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)

    def test_binary_codec_handles_full_bf16_range_without_metadata_overflow(self):
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation

        maximum = torch.finfo(torch.bfloat16).max
        cpu = torch.tensor([[-maximum, 0, maximum]], dtype=torch.bfloat16)
        expected = quantize_pack_activation(cpu, bits=1)
        packed = quantize_pack_activation(cpu.to("cuda"), bits=1)
        torch.testing.assert_close(packed.data.cpu(), expected.data, rtol=0, atol=0)
        torch.testing.assert_close(packed.step.cpu(), expected.step, rtol=0, atol=0)
        restored = dequantize_unpack_activation(packed, dtype=torch.bfloat16)
        torch.testing.assert_close(restored.cpu(), dequantize_unpack_activation(expected), rtol=0, atol=0)

    def test_fused_cast_and_pack_match_cpu_for_strides_tails_and_scale_blocks(self):
        from deepopt.opt.quantization.activation import quantize_pack_activation
        from deepopt.opt.quantization.activation_fp8 import quantize_pack_fp8_activation
        from deepopt.opt.quantization.quantizer import quantize_fp8

        base = torch.randn(67, 131, dtype=torch.bfloat16)
        for cpu in (base, base.T, base[1::2, 1::2], base[:1].expand(7, -1)):
            value = base.to("cuda")
            if cpu is base:
                value = value
            elif cpu.shape == base.T.shape:
                value = value.T
            elif cpu.shape[0] == 7:
                value = value[:1].expand(7, -1)
            else:
                value = value[1::2, 1::2]
            for block in (None, (1, 32), (32, 32), (128, 128)):
                for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    with self.subTest(shape=cpu.shape, stride=cpu.stride(), block=block, dtype=dtype):
                        reference = quantize_fp8(cpu, fp8_dtype=dtype, block_shape=block)
                        expected = quantize_pack_activation(reference.data.float(), group_size=512, block_shape=block)
                        actual, packed = quantize_pack_fp8_activation(
                            value, fp8_dtype=dtype, block_shape=block, group_size=512,
                            check_finite=False,
                        )
                        torch.testing.assert_close(actual.data.float().cpu(), reference.data.float(), rtol=0, atol=0)
                        torch.testing.assert_close(actual.scale_inv.cpu(), reference.scale_inv, rtol=1e-6, atol=0)
                        torch.testing.assert_close(packed.data.cpu(), expected.data, rtol=0, atol=0)
                        torch.testing.assert_close(packed.minimum.cpu(), expected.minimum, rtol=0, atol=0)
                        torch.testing.assert_close(packed.step.cpu(), expected.step, rtol=1e-6, atol=0)
                        elements = 512 if block is None else block[0] * block[1]
                        self.assertEqual(packed.data.shape[1], (elements + 15) // 16)
                        self.assertEqual(packed.block_shape, block)

    def test_constant_groups_and_unpack_directly_to_fp8_or_bf16_transpose(self):
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation

        for cpu in (torch.full((37, 65), -2.5), torch.zeros(37, 65), torch.randn(37, 65)):
            for block in (None, (1, 32), (16, 32)):
                reference = quantize_pack_activation(cpu, group_size=512, block_shape=block)
                packed = quantize_pack_activation(cpu.T.contiguous().to("cuda").T, group_size=512, block_shape=block)
                for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
                    for transpose in (False, True):
                        with self.subTest(block=block, dtype=dtype, transpose=transpose):
                            expected = dequantize_unpack_activation(reference, dtype=dtype, transpose=transpose)
                            actual = dequantize_unpack_activation(packed, dtype=dtype, transpose=transpose)
                            self.assertEqual(actual.dtype, dtype)
                            self.assertEqual(actual.shape, expected.shape)
                            torch.testing.assert_close(actual.float().cpu(), expected.float(), rtol=0, atol=0)

    def test_chunked_large_groups_and_non_power_of_two_compression_size(self):
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
        from deepopt.opt.quantization.activation_fp8 import quantize_pack_fp8_activation
        from deepopt.opt.quantization.quantizer import quantize_fp8

        cpu = torch.randn(131, 67, dtype=torch.bfloat16).T
        value = cpu.T.contiguous().to("cuda").T
        for group_size, block in ((17, None), (20003, None), (512, (256, 128))):
            for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                expected_q = quantize_fp8(cpu, fp8_dtype=dtype, block_shape=block)
                expected = quantize_pack_activation(expected_q.data.float(), group_size=group_size, block_shape=block)
                actual_q, actual = quantize_pack_fp8_activation(
                    value, fp8_dtype=dtype, group_size=group_size, block_shape=block, check_finite=False)
                torch.testing.assert_close(actual_q.data.float().cpu(), expected_q.data.float(), rtol=0, atol=0)
                torch.testing.assert_close(actual.data.cpu(), expected.data, rtol=0, atol=0)
                torch.testing.assert_close(actual.step.cpu(), expected.step, rtol=1e-6, atol=0)
                torch.testing.assert_close(actual.minimum.cpu(), expected.minimum, rtol=0, atol=0)
                decoded = dequantize_unpack_activation(actual, dtype=dtype, transpose=True)
                reference = dequantize_unpack_activation(expected, dtype=dtype, transpose=True)
                torch.testing.assert_close(decoded.float().cpu(), reference.float(), rtol=0, atol=0)

    @staticmethod
    def _make_linear(weight, bias, blocks=None, recipe="current", backward="inherit"):
        import deepopt
        from deepopt.opt.policy import prepare_fp8_model

        model = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                                device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            model.weight.copy_(weight)
            if bias is not None:
                model.bias.copy_(bias)
        settings = deepopt.FP8Config(
            enabled=True, backend="triton", trace_mode="modules", intercept_linear=False,
            block_shapes={"activation": blocks, "weight": None, "grad": None},
            scaling_recipe=recipe, backward_backend=backward,
        )
        model, _ = prepare_fp8_model(model, settings)
        return model, settings

    @staticmethod
    def _run(model, settings, x, grad, compressed):
        import deepopt

        model.zero_grad(set_to_none=True)
        x = x.detach().requires_grad_()
        with deepopt.autocast(
            "cuda", dtype=torch.bfloat16, fp8=settings, cache_enabled=False, check_finite=False,
            activation_compression={"enabled": compressed, "group_size": 512},
        ):
            output = model(x)
            output.backward(grad)
        return output.detach(), x.grad, model.weight.grad.detach().clone(), model.bias.grad.detach().clone()

    def test_fp8_linear_preserves_forward_dx_db_and_dw_matches_decoded_operand(self):
        from deepopt.opt.backends import gemm_nt
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
        from deepopt.opt.quantization.quantizer import quantize_fp8
        from deepopt.opt.quantization.tensor import QuantizedTensor

        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(96, 128, device="cuda") / 8
        bias = torch.randn(96, device="cuda")
        grad = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16) / 64
        for block, backward in ((None, "inherit"), (None, "native_fast"), ((1, 32), "inherit"),
                                ((32, 32), "inherit")):
            with self.subTest(block=block, backward=backward):
                model, settings = self._make_linear(weight, bias, block, backward=backward)
                baseline = self._run(model, settings, x, grad, False)
                compressed = self._run(model, settings, x, grad, True)
                for index in (0, 1, 3):
                    torch.testing.assert_close(compressed[index], baseline[index], rtol=0, atol=0)

                rowwise = block is not None and block[0] == 1 and block[1] > 1
                qx = quantize_fp8(x.T if rowwise else x, block_shape=block, check_finite=False)
                packed = quantize_pack_activation(qx.data.float(), group_size=512, block_shape=block)
                restored = dequantize_unpack_activation(packed, dtype=qx.data.dtype, transpose=not rowwise)
                if rowwise:
                    saved = QuantizedTensor(restored, qx.scale_inv, block)
                else:
                    layout = qx.transpose(contiguous=False)
                    saved = QuantizedTensor(restored, layout.scale_inv, layout.block_shape)
                qg = quantize_fp8(grad, check_finite=False).transpose(contiguous=True)
                expected_dw = gemm_nt(qg, saved, backend="triton" if backward == "inherit" else backward,
                                      out_dtype=torch.bfloat16, num_stages="auto")
                torch.testing.assert_close(compressed[2], expected_dw.float(), rtol=2e-2, atol=2e-2)

    def test_saved_payload_contains_packed_activation_instead_of_dense_fp8_activation(self):
        import deepopt

        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        model, settings = self._make_linear(torch.randn(96, 128, device="cuda"), torch.zeros(96, device="cuda"))
        records = []

        def record(value):
            records.append((value.dtype, tuple(value.shape)))
            return value

        with torch.autograd.graph.saved_tensors_hooks(record, lambda value: value):
            with deepopt.autocast(
                "cuda", dtype=torch.bfloat16, fp8=settings, cache_enabled=False, check_finite=False,
                activation_compression={"enabled": True, "group_size": 512},
            ):
                model(x).sum().backward()
        self.assertTrue(any(dtype == torch.int32 for dtype, _ in records))
        self.assertFalse(any(dtype == torch.float8_e4m3fn and shape in ((64, 128), (128, 64))
                             for dtype, shape in records))

    def test_delayed_scaling_state_matches_uncompressed_over_multiple_steps(self):
        weight = torch.randn(96, 128, device="cuda") / 8
        bias = torch.zeros(96, device="cuda")
        normal, settings = self._make_linear(weight, bias, recipe="delayed")
        compressed, _ = self._make_linear(weight, bias, recipe="delayed")
        for scale in (1., 3., .5):
            x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * scale
            grad = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16) / 64
            baseline = self._run(normal, settings, x, grad, False)
            result = self._run(compressed, settings, x, grad, True)
            for index in (0, 1, 3):
                torch.testing.assert_close(result[index], baseline[index], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(result[2]).all())
            for name in ("scales", "updates", "history"):
                torch.testing.assert_close(getattr(compressed.fp8_scaling, name),
                                           getattr(normal.fp8_scaling, name), rtol=0, atol=0)

    def test_bf16_compressed_linear_uses_decoded_activation_for_weight_gradient(self):
        from deepopt.opt.nn.functional import do_linear
        from deepopt.opt.policy import linear_plan
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation

        x = torch.randn(37, 65, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(29, 65, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        bias = torch.randn(29, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn(37, 29, device="cuda", dtype=torch.bfloat16)
        output = do_linear(x, weight, bias, plan=linear_plan(False, activation_compression=True))
        torch.testing.assert_close(output, torch.nn.functional.linear(x, weight, bias), rtol=0, atol=0)
        output.backward(grad)
        decoded = dequantize_unpack_activation(quantize_pack_activation(x.detach(), group_size=512), dtype=x.dtype)
        torch.testing.assert_close(weight.grad, grad.T @ decoded, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, grad @ weight.detach(), rtol=0, atol=0)
        torch.testing.assert_close(bias.grad, grad.float().sum(0).to(bias.dtype), rtol=0, atol=0)

    def test_current_fused_cast_and_unpack_can_replay_cuda_graph(self):
        from deepopt.opt.quantization.activation import dequantize_unpack_activation
        from deepopt.opt.quantization.activation_fp8 import quantize_pack_fp8_activation

        value = torch.randn(65, 131, device="cuda", dtype=torch.bfloat16)

        def run():
            quantized, packed = quantize_pack_fp8_activation(value, check_finite=False)
            decoded = dequantize_unpack_activation(packed, dtype=quantized.data.dtype, transpose=True)
            return quantized, decoded

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            quantized, decoded = run()
        for factor in (.5, 2.):
            value.mul_(factor)
            graph.replay()
            expected, expected_decoded = run()
            torch.testing.assert_close(quantized.data.float(), expected.data.float(), rtol=0, atol=0)
            torch.testing.assert_close(quantized.scale_inv, expected.scale_inv, rtol=0, atol=0)
            torch.testing.assert_close(decoded.float(), expected_decoded.float(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
