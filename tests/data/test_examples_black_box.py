"""Execute worker examples with real spawned processes and optional recovery."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]


class WorkerExampleBlackBoxTests(unittest.TestCase):
    def run_recipe(self, name, *arguments):
        result = subprocess.run(
            [sys.executable, str(ROOT / "examples/data" / name), *arguments], cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1",
                 "DEEPOPT_LOGGING_LEVEL": "INFO"}, capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_seeded_prefetch_custom_collation_and_rebatching(self):
        result = self.run_recipe("resumable_prefetch.py", "--workers", "2")
        summary = json.loads(result.stdout.splitlines()[-1])
        self.assertTrue(summary["replay_matches"])
        self.assertEqual(summary["initial_workers"], 2)
        self.assertEqual(summary["resumed_workers"], 0)
        self.assertEqual(summary["resumed_batch_sizes"], [3, 3, 3])
        indices = summary["consumed_indices"] + summary["remaining_indices"]
        self.assertEqual(sorted(indices), list(range(13)))

    def test_index_skip_example_with_workers(self):
        result = self.run_recipe("skip_batches.py", "--workers", "2")
        self.assertIn("remaining indices: [6, 7, 8, 9]", result.stderr)
        self.assertIn("next epoch indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("torchdata"), "Install the workers extra")
    def test_stateful_worker_recovery_recipe(self):
        result = self.run_recipe("worker_resume.py")
        self.assertIn("Completed epoch:", result.stderr)


if __name__ == "__main__":
    unittest.main()
