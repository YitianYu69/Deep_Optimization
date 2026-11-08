"""TMA packing, strided quantized operands, and scale-grid transposition on CUDA."""

import unittest
from unittest.mock import patch

import torch

from deepopt.opt.backends import gemm_nt
from deepopt.opt.backends._triton_autotune import AUTOTUNE_KEY
from deepopt.opt.quantization.tensor import QuantizedTensor
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON, quantize_gemm_operand as quantize_fp8


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class TritonFP8WhiteBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(271)

    def test_unchecked_cuda_scaling_has_no_python_scalar_reads_and_handles_zero(self):
        from deepopt.opt.scaling_strategy.amax import AmaxScaling

        for blocks in (None, (1, 32), (32, 32)):
            for value in (0.0, 0.25):
                x = torch.full((37, 65), value, device="cuda")
                expected = AmaxScaling(blocks).compute(x.cpu())
                with patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("GPU scalar branch")), \
                     patch.object(torch.Tensor, "item", side_effect=AssertionError("GPU item read")):
                    actual = AmaxScaling(blocks).compute(x, check_finite=False)
                torch.testing.assert_close(actual.scale.cpu(), expected.scale)
                torch.testing.assert_close(actual.scale_inv.cpu(), expected.scale_inv)

    @staticmethod
    def _split_config(split):
        import triton
        from deepopt.opt.backends._triton_autotune import set_descriptor_block_shapes

        return triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32,
                              "SPLIT_K": split, "NUM_STAGES": 2, "GROUP_SIZE_M": 1},
                             num_warps=4, num_stages=2, pre_hook=set_descriptor_block_shapes)

    def test_grouped_stage_five_launches_cover_partial_groups_and_cached_split_k(self):
        import triton
        from deepopt.opt.backends import triton_fp8
        from deepopt.opt.backends._triton_autotune import set_descriptor_block_shapes

        kernel = triton_fp8._gemm_nt_kernel
        # Nine M tiles: the last eight-row group contains just one row of tiles.
        # N has two partial tiles, so grouping changes the launch order.
        m, n, k = 133, 29, 515
        sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        splits = (1, 4) if sms > 18 else (1,)
        for blocks in ((None, None), ((1, 32), (32, 32)), ((32, 1), (32, 32))):
            a = quantize_fp8(torch.randn(m, k, device="cuda") * .1, block_shape=blocks[0])
            b = quantize_fp8(torch.randn(n, k, device="cuda") * .1, block_shape=blocks[1])
            product = a.dequantize().cpu() @ b.dequantize().cpu().T
            bias = torch.randn(n, device="cuda") * .1
            for group in (1, 8):
                for split in splits:
                    config = triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32,
                                            "NUM_STAGES": 5, "SPLIT_K": split, "GROUP_SIZE_M": group},
                                           num_warps=4, num_stages=5, pre_hook=set_descriptor_block_shapes)
                    with self.subTest(blocks=blocks, group=group, split=split), \
                         patch.object(kernel, "configs", [config]), patch.object(kernel, "cache", {}):
                        for _ in range(2):
                            bias.add_(.01)
                            actual = gemm_nt(a, b, backend="triton", bias=bias, num_stages=5)
                            torch.testing.assert_close(actual.cpu(), product + bias.cpu(), rtol=5e-4, atol=5e-4)
                        self.assertEqual(len(kernel.cache), 1)

    def test_split_k_matches_oracle_for_partial_tiles_scaling_and_output_types(self):
        from deepopt.opt.backends import triton_fp8

        kernel = triton_fp8._gemm_nt_kernel
        for blocks in ((None, None), ((1, 32), (32, 32)), ((32, 1), (32, 32))):
            a = quantize_fp8(torch.randn(13, 515, device="cuda") * .1, block_shape=blocks[0])
            b = quantize_fp8(torch.randn(37, 515, device="cuda") * .1, block_shape=blocks[1],
                             fp8_dtype=torch.float8_e5m2)
            bias = torch.randn(37, device="cuda") * .1
            for split in (2, 4, 8):
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    with self.subTest(blocks=blocks, split=split, dtype=dtype), \
                         patch.object(kernel, "configs", [self._split_config(split)]), \
                         patch.object(kernel, "cache", {}):
                        actual = gemm_nt(a, b, backend="triton", bias=bias,
                                         out_dtype=dtype, num_stages=2)
                        expected = (a.dequantize().cpu() @ b.dequantize().cpu().T + bias.cpu()).to(dtype)
                        tolerance = 1e-2 if dtype == torch.bfloat16 else 2e-3 if dtype == torch.float16 else 5e-4
                        torch.testing.assert_close(actual.cpu(), expected, rtol=tolerance, atol=tolerance)
                        # The cached production launch must overwrite all partials.
                        again = gemm_nt(a, b, backend="triton", bias=None,
                                        out_dtype=dtype, num_stages=2)
                        expected = (a.dequantize().cpu() @ b.dequantize().cpu().T).to(dtype)
                        torch.testing.assert_close(again.cpu(), expected, rtol=tolerance, atol=tolerance)

    def test_split_reduction_is_in_candidate_benchmark_and_cached_launch(self):
        from deepopt.opt.backends import triton_fp8

        kernel = triton_fp8._gemm_nt_kernel
        a = quantize_fp8(torch.randn(13, 515, device="cuda") * .1, block_shape=(1, 32))
        b = quantize_fp8(torch.randn(37, 515, device="cuda") * .1, block_shape=(32, 32))
        expected = a.dequantize().cpu() @ b.dequantize().cpu().T
        observed = []

        def finish(args, exception=None):
            triton_fp8._finish_gemm(args, exception)
            if exception is None:
                observed.append(args["SPLIT_K"])
                torch.testing.assert_close(args["output_ptr"][:, :37].cpu(), expected, rtol=5e-4, atol=5e-4)

        def bench(call, quantiles):
            call()
            # Deliberately choose split-K to exercise the cached final reduction.
            return [1.0 if observed[-1] == 4 else 2.0] * len(quantiles)

        with patch.object(kernel, "configs", [self._split_config(1), self._split_config(4)]), \
             patch.object(kernel, "cache", {}), patch.object(kernel, "post_hook", new=finish), \
             patch.object(kernel, "do_bench", new=bench), \
             patch.object(triton_fp8.torch, "empty", wraps=torch.empty) as allocate:
            for _ in range(2):
                actual = gemm_nt(a, b, backend="triton", num_stages=2)
                torch.testing.assert_close(actual.cpu(), expected, rtol=5e-4, atol=5e-4)
            self.assertEqual(kernel.best_config.kwargs["SPLIT_K"], 4)
            partial_shapes = [call.args[0] for call in allocate.call_args_list
                              if call.args and isinstance(call.args[0], tuple) and len(call.args[0]) == 3]
            self.assertEqual([shape[0] for shape in partial_shapes], [8, 4])
        self.assertEqual(sorted(observed), [1, 4])

    def test_split_k_preserves_output_guard_rows_and_stride_padding(self):
        from deepopt.opt.backends import triton_fp8

        kernel = triton_fp8._gemm_nt_kernel
        a = quantize_fp8(torch.randn(13, 515, device="cuda") * .1, block_shape=(1, 32))
        b = quantize_fp8(torch.randn(19, 515, device="cuda") * .1, block_shape=(32, 32))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            alignment = 4 if dtype == torch.float32 else 8
            stride = ((19 + alignment - 1) // alignment) * alignment
            guarded = torch.full((14, stride), 123.0, dtype=dtype, device="cuda")
            allocate = torch.empty

            def output_with_guard(size, **kwargs):
                if size == (13, stride) and kwargs.get("dtype") == dtype:
                    return guarded[:13]
                return allocate(size, **kwargs)

            with patch.object(kernel, "configs", [self._split_config(4)]), \
                 patch.object(kernel, "cache", {}), \
                 patch.object(triton_fp8.torch, "empty", side_effect=output_with_guard):
                gemm_nt(a, b, backend="triton", out_dtype=dtype, num_stages=2)
            self.assertTrue(torch.all(guarded[13] == 123).item())
            self.assertTrue(torch.all(guarded[:13, 19:] == 123).item())

    def test_classifier_autotunes_with_minimal_padding_and_logical_descriptors(self):
        from deepopt.opt.backends import triton_fp8
        from triton.runtime.autotuner import Autotuner

        self.assertIsInstance(triton_fp8._gemm_nt_kernel, Autotuner)
        for batch in (2048, 848):
            x = quantize_fp8(torch.randn(batch, 512, device="cuda") * 0.1, block_shape=(1, 32))
            w = quantize_fp8(torch.randn(10, 512, device="cuda") * 0.1, block_shape=(32, 32))
            g = quantize_fp8(torch.randn(batch, 10, device="cuda") * 0.1,
                             fp8_dtype=torch.float8_e5m2, block_shape=(32, 32))
            cases = ((x, w), (g, w.transpose()), (g.transpose(), x.transpose()))
            for a, b in cases:
                with self.subTest(batch=batch, shape=(a.data.shape, b.data.shape)):
                    expected = a.dequantize().cpu() @ b.dequantize().cpu().T
                    with patch.object(triton_fp8._gemm_nt_kernel, "run",
                                      wraps=triton_fp8._gemm_nt_kernel.run) as launch, \
                         patch.object(triton_fp8, "_pack_fp8_rows",
                                      wraps=triton_fp8._pack_fp8_rows) as pack:
                        actual = gemm_nt(a, b, backend="triton", num_stages=2)
                    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=5e-4)
                    self.assertEqual([call.args[1] for call in pack.call_args_list], [16, 16])
                    self.assertEqual(actual.stride(0), ((b.data.shape[0] + 3) // 4) * 4)
                    self.assertEqual(actual.untyped_storage().nbytes(),
                                     a.data.shape[0] * actual.stride(0) * 4)
                    self.assertEqual(launch.call_count, 1)
                    a_desc, b_desc, c_desc = launch.call_args.args[:3]
                    self.assertEqual(tuple(a_desc.shape), tuple(a.data.shape))
                    self.assertEqual(tuple(b_desc.shape), tuple(b.data.shape))
                    self.assertEqual(tuple(c_desc.shape), tuple(actual.shape))
                    self.assertEqual(launch.call_args.kwargs["PIPELINE_STAGES"], 2)
                    _, selected = self._cache_entry(triton_fp8._gemm_nt_kernel, a, b, torch.float32, 2, False)
                    self.assertEqual(selected.num_stages, 2)

    def test_partial_tma_stores_preserve_guard_rows_and_stride_padding(self):
        from deepopt.opt.backends import triton_fp8

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            a = quantize_fp8(torch.randn(13, 7, device="cuda") * 0.1, block_shape=(1, 32))
            b = quantize_fp8(torch.randn(19, 7, device="cuda") * 0.1, block_shape=(32, 32))
            alignment = 4 if dtype == torch.float32 else 8
            stride = ((19 + alignment - 1) // alignment) * alignment
            guarded = torch.full((14, stride), 123.0, dtype=dtype, device="cuda")
            allocate = torch.empty

            def output_with_guard(size, **kwargs):
                if size == (13, stride) and kwargs.get("dtype") == dtype:
                    return guarded[:13]
                return allocate(size, **kwargs)

            with patch.object(triton_fp8.torch, "empty", side_effect=output_with_guard):
                actual = gemm_nt(a, b, backend="triton", out_dtype=dtype)
            expected = (a.dequantize().cpu() @ b.dequantize().cpu().T).to(dtype)
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-2, atol=3e-4)
            self.assertTrue(torch.all(guarded[13] == 123).item())
            self.assertTrue(torch.all(guarded[:13, 19:] == 123).item())

    def test_aligned_rows_reuse_storage_and_ragged_rows_get_zero_padding(self):
        from deepopt.opt.backends.triton_fp8 import _pack_fp8_rows

        aligned = torch.randn(5, 64, device="cuda").to(torch.float8_e4m3fn)
        self.assertIs(_pack_fp8_rows(aligned, 32), aligned)
        ragged = aligned[:, :37]
        packed = _pack_fp8_rows(ragged, 32)
        self.assertEqual(tuple(packed.shape), (5, 64))
        self.assertTrue(packed.is_contiguous())
        self.assertEqual(packed.data_ptr() % 16, 0)
        torch.testing.assert_close(packed[:, :37].float(), ragged.float(), rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(packed[:, 37:].float()).item(), 0)

    def test_misaligned_and_noncontiguous_rows_are_packed_without_changing_values(self):
        from deepopt.opt.backends.triton_fp8 import _pack_fp8_rows

        storage = torch.arange(1 + 4 * 64, device="cuda").to(torch.float8_e4m3fn)
        misaligned = storage[1:].view(4, 64)
        self.assertNotEqual(misaligned.data_ptr() % 16, 0)
        strided = torch.randn(4, 128, device="cuda").to(torch.float8_e4m3fn)[:, ::2]
        for data in (misaligned, strided):
            with self.subTest(stride=data.stride(), offset=data.storage_offset()):
                packed = _pack_fp8_rows(data, 32)
                self.assertTrue(packed.is_contiguous())
                self.assertEqual(packed.data_ptr() % 16, 0)
                torch.testing.assert_close(packed.float(), data.float(), rtol=0, atol=0)

    def test_transposed_scale_grids_and_noncontiguous_storage_match_oracle(self):
        a = quantize_fp8(torch.randn(97, 65, device="cuda"), block_shape=(32, 64)).transpose()
        b = quantize_fp8(torch.randn(97, 37, device="cuda"), block_shape=(32, 32)).transpose()
        storage = torch.zeros(65, 194, device="cuda", dtype=a.data.dtype)
        storage[:, ::2].copy_(a.data)
        a_strided = QuantizedTensor(storage[:, ::2], a.scale_inv, a.block_shape)
        self.assertFalse(a_strided.data.is_contiguous())
        self.assertEqual(a_strided.block_shape, (64, 32))
        expected = a.dequantize().cpu() @ b.dequantize().cpu().T
        actual = gemm_nt(a_strided, b, backend="triton")
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=5e-4)

    def test_noncontiguous_scale_grids_match_dequantized_oracle(self):
        for a_block, b_block in (((16, 32), (32, 32)), ((3, 5), (7, 9))):
            with self.subTest(blocks=(a_block, b_block)):
                operands = []
                for rows, blocks in ((37, a_block), (29, b_block)):
                    quantized = quantize_fp8(torch.randn(rows, 99, device="cuda") * .1,
                                             block_shape=blocks)
                    scale_rows, scale_cols = quantized.scale_inv.shape
                    storage = torch.empty(scale_rows, scale_cols * 2, device="cuda")
                    scales = storage[:, ::2]
                    scales.copy_(quantized.scale_inv)
                    self.assertFalse(scales.is_contiguous())
                    operands.append(QuantizedTensor(quantized.data, scales, blocks))
                a, b = operands
                expected = a.dequantize().cpu() @ b.dequantize().cpu().T
                actual = gemm_nt(a, b, backend="triton")
                torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=5e-4)

    def test_fused_scaling_matches_cpu_for_strided_inputs_and_edge_blocks(self):
        from deepopt.opt.scaling_strategy.amax import AmaxScaling

        for dtype in (torch.float32, torch.bfloat16):
            x = torch.arange(1, 36, device="cuda", dtype=dtype).reshape(5, 7).T
            self.assertFalse(x.is_contiguous())
            for blocks in (None, (1, 32), (2, 4), (4, 8)):
                with self.subTest(dtype=dtype, blocks=blocks):
                    actual = AmaxScaling(blocks).compute(x)
                    expected = AmaxScaling(blocks).compute(x.cpu())
                    torch.testing.assert_close(actual.scale.cpu(), expected.scale, rtol=1e-5, atol=1e-7)
                    torch.testing.assert_close(actual.scale_inv.cpu(), expected.scale_inv, rtol=1e-5, atol=1e-7)

    def _cache_entry(self, kernel, a, b, out_dtype, stages, has_bias):
        import triton.language as tl

        m, k = a.data.shape
        n = b.data.shape[0]
        capability = torch.cuda.get_device_capability(a.data.device)
        expected = {
            "M": m, "N": n, "K": k,
            "A_FP8_FORMAT": str(a.data.dtype), "B_FP8_FORMAT": str(b.data.dtype),
            "OUTPUT_DTYPE": {torch.float32: tl.float32, torch.float16: tl.float16,
                             torch.bfloat16: tl.bfloat16}[out_dtype],
            "A_BLOCK_ROWS": a.block_shape[0], "A_BLOCK_COLS": a.block_shape[1],
            "B_BLOCK_ROWS": b.block_shape[0], "B_BLOCK_COLS": b.block_shape[1],
            "A_SCALE_COLS": a.scale_inv.shape[1], "B_SCALE_COLS": b.scale_inv.shape[1],
            "TENSORWISE": False,
            "NATIVE_FP8": a.block_shape[1] % 32 == 0 and b.block_shape[1] % 32 == 0,
            "HAS_BIAS": has_bias, "PIPELINE_STAGES": 0 if stages == "auto" else stages,
            "DEVICE_INDEX": a.data.device.index,
            "DEVICE_CAPABILITY": capability[0] * 10 + capability[1],
            "NUM_SMS": torch.cuda.get_device_properties(a.data.device).multi_processor_count,
        }
        matches = [(key, config) for key, config in kernel.cache.items()
                   if all(dict(zip(AUTOTUNE_KEY, key)).get(name) == value
                          for name, value in expected.items())]
        self.assertEqual(len(matches), 1, expected)
        return matches[0]

    def test_native_autotuner_reuses_winner_with_fresh_inputs_and_descriptors(self):
        from triton.runtime.autotuner import Autotuner
        from deepopt.opt.backends.triton_fp8 import _gemm_nt_kernel

        self.assertIsInstance(_gemm_nt_kernel, Autotuner)
        outputs = []
        for repetition in range(2):
            a = quantize_fp8(torch.randn(23, 101, device="cuda") * 0.2,
                             fp8_dtype=torch.float8_e4m3fn, block_shape=(1, 96))
            b = quantize_fp8(torch.randn(29, 101, device="cuda") * 0.2,
                             fp8_dtype=torch.float8_e5m2, block_shape=(32, 64))
            bias = torch.randn(29, device="cuda") * 0.1
            tensors = (a.data, a.scale_inv, b.data, b.scale_inv, bias)
            originals = [tensor.float().cpu().clone() for tensor in tensors]
            expected = a.dequantize().cpu() @ b.dequantize().cpu().T + bias.cpu()

            actual = gemm_nt(a, b, backend="triton", bias=bias, num_stages=2)
            torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=5e-4)
            for tensor, original in zip(tensors, originals):
                torch.testing.assert_close(tensor.float().cpu(), original, rtol=0, atol=0)
            key, selected = self._cache_entry(_gemm_nt_kernel, a, b, torch.float32, 2, True)
            self.assertIs(_gemm_nt_kernel.best_config, selected)
            self.assertEqual(selected.num_stages, 2)
            self.assertEqual(selected.kwargs["BLOCK_K"], 32)
            self.assertIsNotNone(selected.pre_hook)

            if repetition == 0:
                first_key, first_selected = key, selected
                cache_keys = set(_gemm_nt_kernel.cache)
                timings = _gemm_nt_kernel.configs_timings
            else:
                self.assertEqual(key, first_key)
                self.assertIs(selected, first_selected)
                self.assertEqual(set(_gemm_nt_kernel.cache), cache_keys)
                self.assertIs(_gemm_nt_kernel.configs_timings, timings)
            outputs.append(actual.cpu())
        self.assertFalse(torch.equal(*outputs))

    def test_autotuner_cache_separates_formats_outputs_shapes_scales_and_stages(self):
        from deepopt.opt.backends.triton_fp8 import _gemm_nt_kernel

        e4m3, e5m2 = torch.float8_e4m3fn, torch.float8_e5m2
        cases = (
            (21, e4m3, e5m2, torch.float32, (1, 96), (32, 64), 2),
            (21, e5m2, e4m3, torch.float32, (1, 96), (32, 64), 2),
            (21, e4m3, e5m2, torch.float16, (1, 96), (32, 64), 2),
            (22, e4m3, e5m2, torch.float32, (1, 96), (32, 64), 2),
            (21, e4m3, e5m2, torch.float32, (1, 32), (32, 64), 2),
            (21, e4m3, e5m2, torch.float32, (1, 96), (32, 64), 4),
            (21, e4m3, e5m2, torch.float32, (1, 96), (32, 64), "auto"),
            (21, e4m3, e5m2, torch.float32, (32, 1), (7, 9), 2),
        )
        case_keys = set()
        for m, a_dtype, b_dtype, out_dtype, a_block, b_block, stages in cases:
            with self.subTest(m=m, formats=(a_dtype, b_dtype), output=out_dtype,
                              blocks=(a_block, b_block), stages=stages):
                a = quantize_fp8(torch.randn(m, 99, device="cuda") * 0.2,
                                 fp8_dtype=a_dtype, block_shape=a_block)
                b = quantize_fp8(torch.randn(35, 99, device="cuda") * 0.2,
                                 fp8_dtype=b_dtype, block_shape=b_block)
                expected = a.dequantize().cpu() @ b.dequantize().cpu().T
                actual = gemm_nt(a, b, backend="triton", out_dtype=out_dtype, num_stages=stages)
                tolerance = 2e-3 if out_dtype == torch.float16 else 5e-4
                torch.testing.assert_close(actual.cpu(), expected.to(out_dtype),
                                           rtol=tolerance, atol=tolerance)
                key, selected = self._cache_entry(_gemm_nt_kernel, a, b, out_dtype, stages, False)
                self.assertNotIn(key, case_keys)
                case_keys.add(key)
                if stages == "auto":
                    self.assertIn(selected.num_stages, (1, 2, 3, 4, 5))
                else:
                    self.assertEqual(selected.num_stages, stages)
                self.assertEqual(selected.num_stages, selected.kwargs["NUM_STAGES"])
                self.assertIs(selected, _gemm_nt_kernel.best_config)
                if a_block[1] % 32 or b_block[1] % 32:
                    self.assertLessEqual(selected.kwargs["BLOCK_M"], 64)
                    self.assertLessEqual(selected.kwargs["BLOCK_N"], 64)
                    self.assertEqual(selected.kwargs["BLOCK_K"], 32)
        self.assertEqual(len(case_keys), len(cases))


if __name__ == "__main__":
    unittest.main()
