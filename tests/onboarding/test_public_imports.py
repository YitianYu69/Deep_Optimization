"""The folder layout must not change recipe imports or serialized identities."""

from importlib import import_module
import pickle
import subprocess
import sys
import unittest


class PublicImportTests(unittest.TestCase):
    def test_previous_engine_paths_resolve_to_the_same_implementations(self):
        pairs = (
            ("deepopt.engine.trainer", "deepopt.engine.training.trainer", "Trainer"),
            ("deepopt.engine.runtime", "deepopt.engine.execution.runtime", "TrainingRuntime"),
            ("deepopt.engine.metrics", "deepopt.engine.reporting.metrics", "MetricAccumulator"),
            ("deepopt.engine.checkpoint", "deepopt.engine.checkpointing.checkpoint", "restore_checkpoint"),
            ("deepopt.engine.batch_size", "deepopt.memory.planning", "prepare_batch_size"),
        )
        for old, current, name in pairs:
            with self.subTest(old=old):
                self.assertIs(import_module(old), import_module(current))
                self.assertIs(getattr(import_module(old), name), getattr(import_module(current), name))
                value = getattr(import_module(old), name)
                if isinstance(value, type):
                    self.assertEqual(value.__module__, old)
                    self.assertIs(pickle.loads(pickle.dumps(value)), value)

    def test_public_memory_exports_work_in_a_fresh_process(self):
        # Memory can be the first import, rather than being initialized by Trainer.
        code = """
from deepopt.memory import (estimate_training_memory, MemoryEstimationError,
    TrainingMemoryEstimate, ModelMemorySummary, log_model_memory,
    summarize_model_memory, prepare_batch_size)
from deepopt.memory_stats import summarize_model_memory as old_summary
from deepopt import Trainer, TrainingRuntime, Checkpoint
assert old_summary is summarize_model_memory
assert callable(estimate_training_memory) and callable(prepare_batch_size)
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_configuration_import_still_avoids_torch(self):
        result = subprocess.run([sys.executable, "-c",
            "import deepopt.config, sys; assert 'torch' not in sys.modules"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
