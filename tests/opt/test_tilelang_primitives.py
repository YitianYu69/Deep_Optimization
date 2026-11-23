"""Backend contract tests, including mandatory CUDA validation when requested."""

import importlib.util
from contextlib import nullcontext
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


class TileLangPrimitiveHostTests(unittest.TestCase):
    def test_imports_do_not_load_optional_compilers(self):
        script = """
import importlib, sys
for name in ('_runtime', 'quantization.fp8', 'quantization.delayed',
             'quantization.activation.affine', 'quantization.activation.boolean',
             'quantization.activation.indices', 'gemm.fp8', 'gemm.epilogue',
             'gradient', 'scaling.amax'):
    importlib.import_module('deepopt.opt.backends.tilelang.' + name)
assert 'tilelang' not in sys.modules
assert 'triton' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", script], env=os.environ.copy(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_storage_view_preserves_offset_and_stride_span(self):
        from deepopt.opt.backends.tilelang._runtime import storage_view

        original = torch.arange(80).reshape(8, 10)
        view = original[1:7:2, 2:9:3].T
        flat = storage_view(view)
        self.assertEqual(flat.data_ptr(), view.data_ptr())
        self.assertEqual(flat.storage_offset(), view.storage_offset())
        for row in range(view.shape[0]):
            for col in range(view.shape[1]):
                self.assertEqual(flat[row * view.stride(0) + col * view.stride(1)], view[row, col])

    def test_cuda_requirement_rejects_cpu(self):
        from deepopt.opt.backends.tilelang._runtime import require_cuda

        with self.assertRaisesRegex(ValueError, "real CUDA"):
            require_cuda(torch.empty(2))

    def test_rocm_is_rejected_before_cuda_runtime_or_compiler_import(self):
        from deepopt.opt.backends.tilelang import _runtime
        with patch.object(torch.version, "hip", "unsupported"), \
             patch.object(_runtime, "runtime", side_effect=AssertionError("optional compiler import")):
            with self.assertRaisesRegex(ValueError, "NVIDIA CUDA"):
                _runtime.compile_kernel(object(), "cuda:0")

    def test_codec_validation(self):
        from deepopt.opt.backends.tilelang.quantization.activation.affine import _bits_consts

        for bits, group in ((0, 32), (3, 32), (2, 0), (2, True)):
            with self.subTest(bits=bits, group=group), self.assertRaises(ValueError):
                _bits_consts(bits, group)

    def test_auxiliary_dispatch_honors_mixed_family_routes(self):
        from deepopt.kernel_policy import KernelPolicy, kernel_policy_scope
        from deepopt.opt.backends.tilelang._runtime import auxiliary_module
        from deepopt.opt.backends.tilelang.quantization import fp8

        # Direct low-level TileLang use does not inherit the default Triton route.
        self.assertIs(auxiliary_module("quantization", "quantization.fp8"), fp8)
        selected = object()
        policy = KernelPolicy("mixed", "tilelang", families=(("quantization", "triton"),))
        with kernel_policy_scope(policy), patch("deepopt.opt.backends.dispatch.backend_module", return_value=selected) as dispatch:
            self.assertIs(auxiliary_module("quantization", "quantization.fp8"), selected)
            dispatch.assert_called_once_with("quantization", "quantization.fp8")

    def test_unseen_specialization_rejects_graph_capture_before_import(self):
        from deepopt.opt.backends.tilelang import _runtime

        with patch.object(torch.cuda, "device", return_value=nullcontext()), (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True)
        ), patch.object(_runtime, "runtime", side_effect=AssertionError("compiler import")):
            with self.assertRaisesRegex(RuntimeError, "Warm this TileLang"):
                _runtime.compile_kernel(object(), "cuda:0")

    def test_compiler_uses_upstream_architecture_suffix(self):
        from deepopt.opt.backends.tilelang import _runtime

        compiler = SimpleNamespace(compile=Mock(return_value="compiled"))
        architecture = SimpleNamespace(get_target_arch=Mock(return_value="90a"))
        function = object()
        with patch.object(torch.cuda, "device", return_value=nullcontext()), (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        ), patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), (
            patch.object(_runtime, "runtime", return_value=(compiler, None))
        ), patch.object(_runtime, "import_module", return_value=architecture):
            self.assertEqual(_runtime.compile_kernel(function, "cuda:0"), "compiled")
        architecture.get_target_arch.assert_called_once_with((9, 0))
        self.assertEqual(compiler.compile.call_args.kwargs["target"], {"kind": "cuda", "arch": "sm_90a"})


class TileLangPrimitiveCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reason = None
        if not torch.cuda.is_available() or torch.version.hip is not None:
            reason = "NVIDIA CUDA is required"
        elif torch.cuda.get_device_capability() < (9, 0):
            reason = "TileLang parity target is SM90+"
        elif importlib.util.find_spec("tilelang") is None:
            reason = "optional TileLang dependency is absent"
        if reason:
            if os.environ.get("DEEPOPT_REQUIRE_TILELANG") == "1":
                raise RuntimeError("Mandatory TileLang CUDA validation unavailable: " + reason)
            raise unittest.SkipTest(reason)
        cls.device = torch.device("cuda", torch.cuda.current_device())

    def test_fp8_strided_roundtrip_and_pair(self):
        from deepopt.opt.backends.tilelang.quantization.fp8 import dequantize, quantize, quantize_tensorwise_pair
        from deepopt.opt.quantization.quantizer import quantize_fp8

        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for blocks in (None, (3, 5), (1, 1), (128, 128)):
                with self.subTest(dtype=dtype, blocks=blocks):
                    values = torch.linspace(-5, 4, 143, device=self.device).reshape(11, 13).T[1:12, :9]
                    data, inverse = quantize(values, dtype, blocks, True)
                    reference = quantize_fp8(values.cpu(), fp8_dtype=dtype, block_shape=blocks, check_finite=True)
                    torch.testing.assert_close(inverse.cpu(), reference.scale_inv, rtol=1e-6, atol=1e-7)
                    self.assertTrue(torch.equal(data.cpu().view(torch.uint8), reference.data.view(torch.uint8)))
                    torch.testing.assert_close(dequantize(data, inverse, blocks).cpu(), reference.dequantize(), rtol=1e-6, atol=1e-7)
                    if blocks is None:
                        normal, paired_inverse, transpose = quantize_tensorwise_pair(values, dtype, True)
                        self.assertEqual(normal.stride(0) % 16, 0)
                        self.assertEqual(transpose.stride(0) % 16, 0)
                        self.assertTrue(torch.equal(normal.cpu().view(torch.uint8), data.cpu().view(torch.uint8)))
                        self.assertTrue(torch.equal(transpose.cpu().view(torch.uint8), data.T.cpu().contiguous().view(torch.uint8)))
                        torch.testing.assert_close(paired_inverse, inverse)

    def test_affine_packed_bytes_and_tail(self):
        from deepopt.opt.backends.tilelang.quantization.activation.affine import quantize_pack_affine, dequantize_unpack_affine
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation

        values = torch.linspace(-1, 2, 85).reshape(5, 17).T
        for bits in (1, 2, 4, 8):
            with self.subTest(bits=bits):
                reference = quantize_pack_activation(values, group_size=23, bits=bits)
                data, step, minimum = quantize_pack_affine(values.to(self.device), 23, bits=bits)
                self.assertTrue(torch.equal(data.cpu(), reference.data))
                torch.testing.assert_close(step.cpu(), reference.step, rtol=1e-6, atol=1e-7)
                torch.testing.assert_close(minimum.cpu(), reference.minimum)
                result = dequantize_unpack_affine(data, step, minimum, values.shape, 23, dtype=torch.float32, bits=bits)
                torch.testing.assert_close(result.cpu(), dequantize_unpack_activation(reference, dtype=torch.float32))

    def test_large_parallel_amax_strides_scale_grid_and_quantized_bytes(self):
        from deepopt.opt.backends.tilelang.quantization.fp8 import quantize
        from tests.opt.test_tilelang_amax import reference_quantize

        # More than 1024 chunks forces two hierarchy levels after source reads.
        count = 1024 * 1024 + 17
        large = torch.linspace(-11, 13, 2 * count, device=self.device)[1::2].reshape(1, count)
        padded = torch.full((74, 65), float("nan"), device=self.device)
        selected = padded[1::2, ::2]
        selected.copy_(torch.linspace(-7, 9, selected.numel(), device=self.device).reshape(selected.shape))
        strided = selected.T
        for values, blocks in ((large, None), (strided, (17, 33))):
            for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                with self.subTest(shape=values.shape, blocks=blocks, dtype=dtype):
                    data, inverse = quantize(values, dtype, blocks, True)
                    expected_data, expected_inverse = reference_quantize(values.cpu(), dtype, blocks)
                    torch.testing.assert_close(inverse.cpu(), expected_inverse, rtol=0, atol=0)
                    self.assertTrue(torch.equal(data.cpu().view(torch.uint8), expected_data.view(torch.uint8)))

    def test_parallel_amax_nonfinite_priority_zero_and_epsilon(self):
        from deepopt.opt.backends.tilelang.quantization.fp8 import amax_scale
        from deepopt.opt.backends.tilelang.scaling.reduction import block_amax

        for position in (0, 1024, 2048):
            values = torch.zeros((1, 2049), device=self.device)
            values[0, 1023] = float("inf")
            values[0, position] = float("nan")
            scale, inverse = amax_scale(values, 448., None, False)
            self.assertTrue(torch.isnan(scale).all())
            self.assertTrue(torch.isnan(inverse).all())
            self.assertTrue(torch.isnan(block_amax(values, 1, 2049, minimum=0)).all())
            with self.assertRaisesRegex(ValueError, "finite inputs"):
                amax_scale(values, 448., None, True)
        values.fill_(0)
        scale, inverse = amax_scale(values, 448., None, True)
        torch.testing.assert_close(scale, torch.ones_like(scale), rtol=0, atol=0)
        torch.testing.assert_close(inverse, torch.ones_like(inverse), rtol=0, atol=0)
        values.fill_(1e-15)
        scale, inverse = amax_scale(values, 448., None, True)
        expected = torch.tensor([[448.]], device=self.device) / torch.tensor([[1e-12]], device=self.device)
        torch.testing.assert_close(scale, expected, rtol=0, atol=0)
        torch.testing.assert_close(inverse, torch.true_divide(torch.ones_like(expected), expected), rtol=0, atol=0)
        values[0, -1] = -float("inf")
        scale, inverse = amax_scale(values, 448., None, False)
        torch.testing.assert_close(scale, torch.zeros_like(scale), rtol=0, atol=0)
        self.assertTrue(torch.isposinf(inverse).all())

    def test_warm_large_quantize_graph_replay_recomputes_scale(self):
        from deepopt.opt.backends.tilelang.quantization.fp8 import quantize

        values = torch.ones((1, 65553), device=self.device)

        def execute():
            return quantize(values, torch.float8_e4m3fn, None, False)

        execute()
        execute()
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            data, inverse = execute()
        values.fill_(2)
        graph.replay()
        torch.testing.assert_close(inverse, torch.full_like(inverse, 2. / 448.), rtol=0, atol=0)
        torch.testing.assert_close(data.float(), torch.full_like(values, 448.), rtol=0, atol=0)

    def test_lossless_routing_crosses_words(self):
        from deepopt.opt.backends.tilelang.quantization.activation.indices import pack, unpack

        for bits in (0, 1, 5, 17, 31, 32, 33, 64):
            with self.subTest(bits=bits):
                count = 79
                native = torch.zeros(count, device=self.device, dtype=torch.int64) if not bits else (
                    torch.arange(count, device=self.device, dtype=torch.int64) % (1 << min(bits, 6)))
                words = count if bits > 32 else (count * bits + 31) // 32
                payload = torch.empty(words, device=self.device, dtype=torch.int64 if bits > 32 else torch.int32)
                packed = SimpleNamespace(bits=bits, data=payload)
                pack(native, packed)
                restored = torch.empty_like(native)
                unpack(packed, restored)
                self.assertTrue(torch.equal(restored, native))

    def test_boolean_noncontiguous_tail(self):
        from deepopt.opt.backends.tilelang.quantization.activation.boolean import pack_boolean_mask, unpack_boolean_mask

        values = (torch.arange(105, device=self.device).reshape(7, 15) % 3 == 0).T
        payload = pack_boolean_mask(values)
        self.assertTrue(torch.equal(unpack_boolean_mask(payload, values.shape), values))

    def test_delayed_snapshot_bias_and_observations(self):
        from deepopt.opt.backends.tilelang.quantization.delayed import convert, reduce_observations

        values = torch.linspace(-2, 3, 91, device=self.device).reshape(7, 13).T
        scale = torch.tensor([5.0], device=self.device)
        normal, transpose, snapshot, observations, bias = convert(
            values, scale, torch.float8_e4m3fn, normal=True, transposed=True,
            bias=True, observe=True, check_finite=True, autotune=False)
        scale.fill_(9.0)
        torch.testing.assert_close(snapshot, torch.tensor([5.0, 0.2], device=self.device))
        torch.testing.assert_close(bias, values.sum(0), rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.equal(normal.cpu().view(torch.uint8), transpose.T.cpu().contiguous().view(torch.uint8)))
        packed = reduce_observations([(1, observations)], 3, self.device)
        self.assertEqual(packed[1, 1], 1)
        self.assertEqual(packed[1, 0], 0)
        torch.testing.assert_close(packed[0, 1], values.abs().amax())

    def test_delayed_state_history_and_nonfinite_preservation(self):
        from deepopt.opt.backends.tilelang.quantization.delayed import update_states

        for recent in (False, True):
            with self.subTest(recent=recent):
                state = SimpleNamespace(
                    options=SimpleNamespace(history_length=3, margin=1,
                        amax_algorithm="most_recent" if recent else "max"),
                    dtypes={name: torch.float8_e4m3fn for name in ("activation", "weight", "grad")},
                    scales=torch.ones(3, device=self.device), history=torch.zeros((3, 3), device=self.device),
                    updates=torch.zeros(3, device=self.device, dtype=torch.int64),
                    generations=torch.zeros(3, device=self.device, dtype=torch.int64))
                packed = torch.tensor([[8., 0., 0.], [1., 0., 1.], [0., 0., 2.]], device=self.device)
                update_states([state], packed)
                torch.testing.assert_close(state.scales, torch.tensor([28., 1., 1.], device=self.device))
                self.assertTrue(torch.isnan(state.history[2, 0]))
                self.assertTrue(torch.equal(state.updates, torch.tensor([1, 0, 1], device=self.device)))
                packed = torch.tensor([[4., 0., 0.], [1., 0., 0.], [0., 0., 0.]], device=self.device)
                update_states([state], packed)
                self.assertEqual(state.scales[0], 56. if recent else 28.)
                self.assertEqual(state.updates[0], 2)
                self.assertEqual(state.generations[2], 1)

    def test_gemm_formats_scales_and_output_stride(self):
        from deepopt.opt.backends.tilelang.quantization.fp8 import quantize
        from deepopt.opt.backends.tilelang.gemm.fp8 import gemm_nt
        from deepopt.opt.quantization.tensor import QuantizedTensor

        left = torch.linspace(-1, 1, 35, device=self.device).reshape(5, 7)
        right = torch.linspace(-2, 1, 63, device=self.device).reshape(9, 7)
        for blocks in (None, (2, 32), (2, 3)):
            for adtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                for bdtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    with self.subTest(blocks=blocks, adtype=adtype, bdtype=bdtype):
                        qa = QuantizedTensor(*quantize(left, adtype, blocks, True), blocks)
                        qb = QuantizedTensor(*quantize(right, bdtype, blocks, True), blocks)
                        storage = torch.empty((qb.scale_inv.shape[0], qb.scale_inv.shape[1] * 2), device=self.device)
                        storage[:, ::2].copy_(qb.scale_inv)
                        qb = QuantizedTensor(qb.data.T.contiguous().T, storage[:, ::2], blocks)
                        bias = torch.linspace(-0.5, 0.5, 9, device=self.device)
                        for dtype, alignment in ((torch.float32, 4), (torch.float16, 8), (torch.bfloat16, 8)):
                            result = gemm_nt(qa, qb, out_dtype=dtype, bias=bias, autotune=False)
                            reference_a = QuantizedTensor(qa.data.cpu(), qa.scale_inv.cpu(), blocks).dequantize()
                            reference_b = QuantizedTensor(qb.data.cpu(), qb.scale_inv.cpu(), blocks).dequantize()
                            expected = (reference_a @ reference_b.T + bias.cpu()).to(dtype)
                            tolerance = 3e-5 if dtype == torch.float32 else 2e-2
                            torch.testing.assert_close(result.cpu(), expected, rtol=tolerance, atol=tolerance)
                            self.assertEqual(result.stride(0) % alignment, 0)

    def test_warm_quantize_and_gemm_cuda_graph_replay(self):
        from deepopt.opt.backends.tilelang.quantization.fp8 import quantize
        from deepopt.opt.backends.tilelang.gemm.fp8 import gemm_nt
        from deepopt.opt.quantization.tensor import QuantizedTensor

        left = torch.linspace(-1, 1, 35, device=self.device).reshape(5, 7)
        right = torch.linspace(-2, 1, 63, device=self.device).reshape(9, 7)

        def execute():
            qa = QuantizedTensor(*quantize(left, torch.float8_e4m3fn, None, False))
            qb = QuantizedTensor(*quantize(right, torch.float8_e5m2, None, False))
            return gemm_nt(qa, qb, autotune=False)

        expected = execute()
        execute()
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = execute()
        for _ in range(3):
            graph.replay()
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
