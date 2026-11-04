"""Automatic FP8 preparation and policy agreement on two real CPU/Gloo ranks."""

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
class AutomaticFP8DistributedBlackBoxTests(unittest.TestCase):
    def run_workers(self, mode):
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
                       "OMP_NUM_THREADS": "1", "DEEPOPT_LOGGING_LEVEL": "WARNING"}
        result = run_torchrun([str(Path(__file__).with_name("_fp8_automatic_worker.py")), mode],
                             env=environment, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]

    def test_trainer_runtime_and_bare_core_preserve_parameters_and_match_fp8_gradients(self):
        rows = self.run_workers("training")
        self.assertEqual([row["api"] for row in rows], ["trainer", "runtime", "core"])
        for row in rows:
            self.assertEqual(row["core"], "DDPCore")
            self.assertEqual(row["world_size"], 2)
            self.assertTrue(row["synchronized_gradients"])
            self.assertTrue(row["baseline_matches"])

    def test_rank_inconsistent_policy_fails_collectively_before_ddp_wrapping(self):
        rows = self.run_workers("mismatch")
        self.assertEqual(rows, [{"mismatch_rejected": ["runtime", "core"], "world_size": 2}])


if __name__ == "__main__":
    unittest.main()
