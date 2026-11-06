"""TMA packing, strided quantized operands, and scale-grid transposition on CUDA."""

import unittest

import torch

from deepopt.opt.backends import gemm_nt
from deepopt.opt.backends._triton_autotune import AUTOTUNE_KEY
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.opt.quantization.tensor import QuantizedTensor
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class TritonFP8WhiteBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(271)

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
            "HAS_BIAS": has_bias, "PIPELINE_STAGES": stages,
            "DEVICE_INDEX": a.data.device.index,
            "DEVICE_CAPABILITY": capability[0] * 10 + capability[1],
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
                self.assertEqual(selected.num_stages, stages)
                self.assertIs(selected, _gemm_nt_kernel.best_config)
                if a_block[1] % 32 or b_block[1] % 32:
                    self.assertLessEqual(selected.kwargs["BLOCK_M"], 64)
                    self.assertLessEqual(selected.kwargs["BLOCK_N"], 64)
                    self.assertEqual(selected.kwargs["BLOCK_K"], 32)
        self.assertEqual(len(case_keys), len(cases))


if __name__ == "__main__":
    unittest.main()
