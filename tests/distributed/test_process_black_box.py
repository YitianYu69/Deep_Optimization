"""Consume initialization metadata through the public API in fresh processes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch.distributed as dist

from deepopt import initialize_distributed, shutdown_distributed

if __package__:
    from .launcher import run_torchrun
else:
    from launcher import run_torchrun


def _worker(directory):
    context = initialize_distributed("auto")
    try:
        assert initialize_distributed("auto") == context
        record = context.as_dict()
    finally:
        shutdown_distributed()
    record["initialized_after_shutdown"] = dist.is_initialized()
    Path(directory, f"rank-{context.global_rank}.json").write_text(json.dumps(record))


class ProcessMetadataBlackBoxTests(unittest.TestCase):
    def environment(self):
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}}
        return {**environment, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}

    def test_standalone_auto_context_is_json_serializable(self):
        result = subprocess.run([sys.executable, "-c",
            "import json; from deepopt import initialize_distributed; print(json.dumps(initialize_distributed('auto').as_dict()))"],
            env=self.environment(), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        context = json.loads(result.stdout)
        self.assertEqual((context["global_rank"], context["local_rank"], context["world_size"], context["local_world_size"]),
                         (0, 0, 1, 1))
        self.assertEqual(context["device"], "cpu")
        self.assertIsNone(context["backend"])
        self.assertFalse(context["is_initialized"])
        self.assertFalse(context["is_distributed"])
        self.assertTrue(context["is_main_process"])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
    def test_torchrun_returns_each_ranks_actual_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_torchrun([str(Path(__file__).resolve()), "--write-context", directory],
                                 env=self.environment(), timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for rank in range(2):
                context = json.loads(Path(directory, f"rank-{rank}.json").read_text())
                self.assertEqual((context["global_rank"], context["local_rank"]), (rank, rank))
                self.assertEqual((context["world_size"], context["local_world_size"]), (2, 2))
                self.assertEqual(context["backend"], "gloo")
                self.assertEqual(context["device"], "cpu")
                self.assertTrue(context["is_initialized"])
                self.assertTrue(context["owns_process_group"])
                self.assertTrue(context["is_distributed"])
                self.assertEqual(context["is_main_process"], rank == 0)
                self.assertFalse(context["initialized_after_shutdown"])


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write-context":
        _worker(sys.argv[2])
    else:
        unittest.main()
