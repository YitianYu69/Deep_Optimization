"""Snapshot metadata and legacy default normalization during restoration."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import Trainer, reset_config
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint


class GlobalCheckpointWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "last.pt"

    def trainer(self, **runtime):
        return Trainer(nn.Linear(2, 1), loss_fn=F.mse_loss,
                       config={"training": {"learning_rate": 0.01}, "runtime": runtime})

    def test_metadata_mapping_overrides_an_owned_snapshot_without_schema_restrictions(self):
        trainer = self.trainer(checks=False)
        metadata = {"recipe_note": {"version": 7}}
        save_checkpoint(self.path, trainer, metadata, generators={}, history=[])
        metadata["recipe_note"]["version"] = 9
        payload = read_checkpoint(self.path)
        self.assertEqual(payload["config"]["recipe_note"], {"version": 7})
        self.assertEqual(payload["config"]["training"]["learning_rate"], 0.01)
        self.assertFalse(payload["config"]["runtime"]["checks"])
        self.assertNotIn("recipe_note", trainer.config)

    def test_old_checkpoint_omitted_distributed_and_capture_defaults_are_equivalent(self):
        trainer = self.trainer(ddp={"find_unused_parameters": False, "broadcast_buffers": True,
                                    "gradient_as_bucket_view": True, "static_graph": False,
                                    "validation_sync_buffers": True},
                               cuda_graph={"warmup": 3, "autograd": True})
        save_checkpoint(self.path, trainer, generators={}, history=[])
        for version in (4, 5):
            with self.subTest(version=version):
                payload = read_checkpoint(self.path)
                payload["format_version"] = version
                payload["config"]["runtime"].pop("ddp")
                payload["config"]["runtime"].pop("cuda_graph")
                self.assertEqual(restore_checkpoint(payload, trainer, trainer.config, generators={}), [])

    def test_saved_compile_policy_is_checked_against_actual_trainer_on_default_resume(self):
        trainer = self.trainer(compile={"enabled": True, "backend": "eager"})
        save_checkpoint(self.path, trainer, generators={}, history=[])
        payload = read_checkpoint(self.path)
        resumed = self.trainer()
        before = {name: tensor.clone() for name, tensor in resumed.model.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "compilation"):
            restore_checkpoint(payload, resumed, generators={})
        for name, tensor in resumed.model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
