"""Global snapshots, explicit overrides, and agreement on two real Gloo ranks."""

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
class DistributedGlobalConfigBlackBoxTests(unittest.TestCase):
    def run_workers(self, mode):
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
                       "OMP_NUM_THREADS": "1", "DEEPOPT_LOGGING_LEVEL": "WARNING"}
        result = run_torchrun([str(Path(__file__).with_name("_global_config_worker.py")), mode],
                             env=environment, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]

    def test_global_defaults_snapshots_and_explicit_overrides_train_consistently(self):
        rows = self.run_workers("agreement")
        self.assertEqual(rows, [
            {"api": "trainer", "world_size": 2, "updates": 2, "fp8": True,
             "accumulation": 2, "broadcast_buffers": False},
            {"api": "override", "world_size": 2, "updates": 1, "fp8": False,
             "accumulation": 1, "bucket_cap_mb": 3},
            {"api": "core", "world_size": 2, "fp8": True, "bucket_cap_mb": 4},
        ])

    def test_rank_mismatches_are_reported_collectively(self):
        self.assertEqual(self.run_workers("mismatch"), [
            {"rejected": ["fp8", "ddp", "accumulation"], "world_size": 2},
        ])

    def test_invalid_rank_local_configuration_fails_before_wrapping_without_poisoning_group(self):
        for api in ("runtime", "trainer", "core"):
            with self.subTest(api=api):
                self.assertEqual(self.run_workers("invalid-" + api), [
                    {"invalid_rejected": api, "ranks": [0, 1]},
                ])


if __name__ == "__main__":
    unittest.main()
