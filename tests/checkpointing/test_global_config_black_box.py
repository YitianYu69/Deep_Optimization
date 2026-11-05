"""Checkpoint cadence and recovery use the owning trainer's configuration."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from deepopt import Checkpoint, Trainer, load_config, reset_config
from deepopt.data import build_loaders
from deepopt.engine.checkpointing.checkpoint import read_checkpoint


class GlobalCheckpointBlackBoxTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_step_cadence_is_bound_to_trainer_and_explicit_zero_disables_it(self):
        configuration = {"name": "checkpoint-defaults", "seed": 3,
                         "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.01,
                                      "checkpoint_every_steps": 1, "monitor": None}}
        for explicit_disable in (False, True):
            with self.subTest(explicit_disable=explicit_disable), tempfile.TemporaryDirectory() as directory:
                load_config(configuration)
                trainer = Trainer(nn.Linear(1, 1), loss_fn=F.mse_loss)
                inputs = torch.arange(6, dtype=torch.float32).reshape(-1, 1) / 6
                loader, _, _ = build_loaders(TensorDataset(inputs, inputs * 2), None)
                load_config({**configuration, "training": {**configuration["training"],
                                                            "checkpoint_every_steps": 0}})
                checkpoint = Checkpoint(directory, **({"every_steps": 0} if explicit_disable else {}))
                trainer.fit(loader, callbacks=[checkpoint], progress_bar=False)
                self.assertEqual(checkpoint.every_steps, 0 if explicit_disable else 1)
                self.assertEqual((Path(directory) / "last_step.pt").exists(), not explicit_disable)
                payload = read_checkpoint(Path(directory) / "last.pt")
                self.assertEqual(payload["config"]["training"]["checkpoint_every_steps"], checkpoint.every_steps)


if __name__ == "__main__":
    unittest.main()
