"""Inspect native NCCL reducer ownership and synchronization of FP8 gradients."""

import unittest
import torch
from torch import distributed as dist

from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON
from tests.distributed.fp8_support import run_workers


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
@unittest.skipUnless(torch.cuda.device_count() >= 2 and dist.is_nccl_available(),
                     "Two NVIDIA GPUs and NCCL are required")
class FP8NCCLWhiteBoxTests(unittest.TestCase):
    def test_triton_backward_uses_native_reducer_and_no_sync(self):
        run_workers(self, white_box=True, device_type="cuda")


if __name__ == "__main__":
    unittest.main()
