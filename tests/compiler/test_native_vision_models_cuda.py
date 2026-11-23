"""CUDA-only original-model gates; CPU tests cannot satisfy these tests."""

import importlib.util
import os
import unittest

import torch

from benchmarks.compiler.vision_compression_support import qualify_pair


class NativeVisionCudaTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or importlib.util.find_spec("torchvision") is None:
            if os.environ.get("DEEPOPT_REQUIRE_CUDA_MODEL_TESTS") == "1":
                self.fail("Required target job needs CUDA and torchvision")
            self.skipTest("CUDA/torchvision unavailable; target qualification remains pending")
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def check_model(self, name, profiles):
        for policy in ("torch_amp", "bf16_storage"):
            for profile in profiles:
                with self.subTest(model=name, policy=policy, profile=profile):
                    torch.compiler.reset()
                    receipt = qualify_pair(name, policy, profile, device="cuda:0", batch_size=1, updates=3)
                    self.assertTrue(receipt["passed"] and receipt["cuda_verified"])

    def test_resnet18_original_224_matrix(self):
        self.check_model("resnet18", ("off", "full-two-bit"))

    def test_efficientnet_b0_original_224_matrix(self):
        self.check_model("efficientnet-b0", ("off", "full-two-bit"))

    def test_vit_base_original_224_matrix(self):
        self.check_model("vit-base", ("off", "two-bit-compatible", "full-two-bit"))

    def test_mixed_add_stores_directly_without_full_fp32_output(self):
        from deepopt.opt.nn.primitive_state import model_add

        x = torch.ones(2, 512, 1024, dtype=torch.bfloat16, device="cuda")
        position = torch.full((1, 512, 1024), -.999, dtype=torch.float32, device="cuda")
        expected = (x.float() + position).bfloat16()
        warmup = model_add(x, position)
        del warmup
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        actual = model_add(x, position)
        torch.cuda.synchronize()
        # Exactly one BF16 result plus a small allocator rounding allowance;
        # a complete FP32 temporary would exceed this independent bound.
        self.assertLessEqual(torch.cuda.max_memory_allocated() - baseline, actual.numel() * 2 + 65536)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
