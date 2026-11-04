"""Inspect checkpoint representation and runtime ownership for FP8 models."""

from pathlib import Path
import tempfile
import unittest

import torch

from deepopt import Checkpoint, TrainingCore
from deepopt.engine.checkpointing.checkpoint import read_checkpoint

if __package__:
    from .test_training_black_box import make_training
else:
    from test_training_black_box import make_training


class FP8TrainingWhiteBoxTests(unittest.TestCase):
    def test_checkpoint_stores_master_weights_and_optimizer_state_without_fp8_workspaces(self):
        trainer, loader = make_training()
        parameter_ids = {id(parameter) for parameter in trainer.model.parameters()}
        self.assertIsInstance(trainer.runtime.core, TrainingCore)
        self.assertEqual(parameter_ids,
                         {id(p) for group in trainer.optimizer.param_groups for p in group["params"]})
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Checkpoint(directory, every_steps=1)
            trainer.fit(loader, epochs=1, callbacks=[checkpoint], progress_bar=False, monitor=None)
            payload = read_checkpoint(Path(directory) / "last.pt")
            self.assertEqual(set(payload["model"]), {"0.weight", "0.bias", "2.weight", "2.bias"})
            for value in payload["model"].values():
                self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(payload["optimizer"]["state"])
            self.assertEqual(parameter_ids, {id(p) for p in trainer.model.parameters()})
            for module in trainer.model.modules():
                if hasattr(module, "plan"):
                    self.assertEqual(module.plan["fp8_training"]["backend"], "reference")


if __name__ == "__main__":
    unittest.main()
