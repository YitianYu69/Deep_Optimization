"""EMA initialization, updates, and validation across two real Gloo ranks."""

import json
import os
from pathlib import Path
import unittest

from torch import distributed as dist

if __package__:
    from .launcher import run_torchrun
else:
    from launcher import run_torchrun


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
class DistributedEMABlackBoxTests(unittest.TestCase):
    def test_synchronized_ema_validation_and_collective_policy_rejection(self):
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
                       "OMP_NUM_THREADS": "1", "DEEPOPT_LOGGING_LEVEL": "WARNING"}
        result = run_torchrun([str(Path(__file__).with_name("_ema_worker.py"))],
                             env=environment, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        self.assertEqual(rows, [{"world_size": 2, "updates": 2,
                                 "initial_synchronized": True, "ema_synchronized": True,
                                 "validation_restored": True, "mismatch_rejected": [0, 1]}])


if __name__ == "__main__":
    unittest.main()
