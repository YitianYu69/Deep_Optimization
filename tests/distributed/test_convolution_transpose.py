"""Installed-library transposed convolution acceptance on two Gloo ranks."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from torch import distributed as dist

from tests.distributed.launcher import run_torchrun


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
class TransposedDistributedTests(unittest.TestCase):
    def test_training_accumulation_delayed_scales_and_exact_resume(self):
        environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
        environment["OMP_NUM_THREADS"] = "1"
        with tempfile.TemporaryDirectory() as directory:
            result = run_torchrun([str(Path(__file__).with_name("_conv_transpose_worker.py")), directory],
                                  env=environment, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertEqual(rows, [{"backward_auto": auto, "synchronize": sync, "exact_resume": True}
                                for auto in (False, True) for sync in (False, True)])


if __name__ == "__main__":
    unittest.main()
