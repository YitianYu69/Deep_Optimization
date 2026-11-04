"""FP8 autograd ownership, no_sync, and reducer state on real Gloo ranks."""

import unittest
from torch import distributed as dist

if __package__:
    from .fp8_support import run_workers
else:
    from fp8_support import run_workers


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
class FP8DistributedWhiteBoxTests(unittest.TestCase):
    def test_reducer_owns_master_parameters_and_accumulates_fp8_gradients(self):
        run_workers(self, white_box=True)


if __name__ == "__main__":
    unittest.main()
