"""TMA packing, strided quantized operands, and scale-grid transposition on CUDA."""

import unittest

import torch

from deepopt.opt.backends import gemm_nt
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


if __name__ == "__main__":
    unittest.main()
