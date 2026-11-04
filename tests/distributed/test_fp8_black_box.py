"""FP8 training and durable resume through the real two-rank DDP core."""

import unittest
from torch import distributed as dist

if __package__:
    from .fp8_support import run_workers
else:
    from fp8_support import run_workers


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
class FP8DistributedBlackBoxTests(unittest.TestCase):
    def test_two_rank_updates_and_checkpoint_resume(self):
        run_workers(self, white_box=False)


if __name__ == "__main__":
    unittest.main()
