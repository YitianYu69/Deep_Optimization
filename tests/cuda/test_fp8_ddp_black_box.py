"""Real Triton FP8 training and checkpoint recovery across NCCL ranks."""

import unittest
import torch
from torch import distributed as dist

from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON
from tests.distributed.fp8_support import run_workers


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
@unittest.skipUnless(torch.cuda.device_count() >= 2 and dist.is_nccl_available(),
                     "Two NVIDIA GPUs and NCCL are required")
class FP8NCCLBlackBoxTests(unittest.TestCase):
    def test_triton_updates_and_step_epoch_resume(self):
        run_workers(self, white_box=False, device_type="cuda")


if __name__ == "__main__":
    unittest.main()
