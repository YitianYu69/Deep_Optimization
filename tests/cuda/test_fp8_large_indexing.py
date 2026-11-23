"""Real CUDA regressions for FP8 tensors exceeding signed 32-bit addressing."""

import unittest

import torch

from deepopt.opt.quantization.quantizer import quantize_fp8
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class LargeFP8IndexingCUDATests(unittest.TestCase):
    def require_free_memory(self, amount):
        free, _ = torch.cuda.mem_get_info()
        if free < amount:
            self.skipTest(f"This large-address regression requires {amount / 2**30:.0f} GiB free CUDA memory")

    def test_vae_sized_tensorwise_quantization_and_dequantization(self):
        self.require_free_memory(20 * 2**30)
        m, n = 64 * 224 * 224, 768
        x = torch.ones((m, n), device="cuda", dtype=torch.bfloat16)
        quantized = quantize_fp8(x, check_finite=True)
        # Check both sides of the signed-32-bit boundary and the final tile.
        indices = torch.tensor([0, 2**31 - 1, 2**31, m * n - 1], device="cuda")
        values = quantized.data.view(torch.uint8).reshape(-1)[indices].view(quantized.data.dtype).float()
        torch.testing.assert_close(values, torch.full_like(values, 448.0), rtol=0, atol=0)
        torch.testing.assert_close(quantized.scale_inv, torch.full_like(quantized.scale_inv, 1 / 448),
                                   rtol=2e-7, atol=0)
        restored = quantized.dequantize().reshape(-1)[indices]
        torch.testing.assert_close(restored, torch.ones_like(restored), rtol=2e-7, atol=0)
        torch.cuda.synchronize()

    def test_small_strided_tensor_with_large_storage_span(self):
        self.require_free_memory(6 * 2**30)
        storage = torch.empty(2**31 + 13, device="cuda", dtype=torch.bfloat16)
        x = storage.as_strided((3, 7), (2**30, 2))
        expected = torch.arange(-10, 11, dtype=torch.bfloat16).reshape(3, 7)
        x.copy_(expected.to("cuda"))
        quantized = quantize_fp8(x, check_finite=True)
        reference = quantize_fp8(expected)
        torch.testing.assert_close(quantized.data.float().cpu(), reference.data.float(), rtol=0, atol=0)
        torch.testing.assert_close(quantized.scale_inv.cpu(), reference.scale_inv, rtol=2e-7, atol=0)
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
