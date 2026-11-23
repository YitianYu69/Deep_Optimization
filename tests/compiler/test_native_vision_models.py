"""Original 224-pixel torchvision classifiers; never substitute tiny models."""

import importlib.util
import os
import unittest

from benchmarks.compiler.vision_compression_support import qualify_pair


class NativeVisionModelTests(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec("torchvision") is None:
            if os.environ.get("DEEPOPT_REQUIRE_MODEL_TESTS") == "1":
                self.fail("Required full-model job is missing torchvision")
            self.skipTest("torchvision unavailable; full-model coverage remains pending")
        import torch

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_original_resnet18_full_two_bit_native(self):
        receipt = qualify_pair("resnet18", "bf16_storage", "full-two-bit")
        self.assertEqual(receipt["parameter_count"], 11689512)
        self.assertTrue(receipt["passed"])

    def test_original_efficientnet_b0_full_two_bit_native(self):
        receipt = qualify_pair("efficientnet-b0", "bf16_storage", "full-two-bit")
        self.assertEqual(receipt["parameter_count"], 5288548)
        self.assertTrue(receipt["passed"])

    def test_original_vit_base_full_two_bit_native(self):
        receipt = qualify_pair("vit-base", "bf16_storage", "full-two-bit")
        self.assertEqual(receipt["parameter_count"], 86567656)
        self.assertTrue(receipt["passed"])


if __name__ == "__main__":
    unittest.main()
