"""Registered delayed histories are never overwritten by ordinary DDP broadcasts."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from torch import distributed as dist

if __package__:
    from .launcher import run_torchrun
else:
    from launcher import run_torchrun


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
class DelayedScalingDistributedTests(unittest.TestCase):
    def test_synchronized_and_local_histories_restore_exactly_on_two_ranks(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            result = run_torchrun([str(Path(__file__).with_name("_delayed_scaling_worker.py")), directory],
                                  env={**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1"}, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertEqual(records, [{"synchronize": value, "rank_local_checkpoint": True, "exact_next_update": True}
                                   for value in (True, False)])
