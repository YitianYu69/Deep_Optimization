"""Checkpoint metadata must reconstruct the computation that produced the weights."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import Checkpoint, Trainer
from deepopt.config import FP8Config, reset_config
from deepopt.engine.checkpointing.checkpoint import (
    _checkpoint_payload, read_checkpoint, restore_checkpoint, save_checkpoint,
)


def configuration():
    return {"name": "policy-snapshot", "seed": 1,
            "training": {"epochs": 1, "batch_size": 2, "learning_rate": .01},
            "runtime": {"device": "cpu", "fp8": {"enabled": False}}}


def make_trainer(policy=True):
    torch.manual_seed(12)
    model = nn.Sequential(nn.Linear(2, 1))
    return Trainer(model, torch.optim.SGD(model.parameters(), lr=.01),
                   loss_fn=F.mse_loss, fp8=policy, config=configuration())


class FP8PolicySnapshotTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_payload_rejects_configuration_that_disables_actual_fp8(self):
        trainer = make_trainer()
        config = configuration()
        before = deepcopy(config)
        with self.assertRaisesRegex(ValueError, "FP8"):
            _checkpoint_payload(trainer, config, generators={}, history=[])
        self.assertEqual(config, before)

    def test_conflicting_policies_do_not_overwrite_existing_checkpoint(self):
        trainer = make_trainer(FP8Config(enabled=True, grad_dtype="e5m2"))
        variants = (
            {"enabled": False}, {"backend": "triton"}, {"grad_dtype": "e4m3fn"},
            {"block_shapes": {"activation": [1, 2]}}, {"num_stages": 2},
            {"out_dtype": "bf16"}, {"trace_mode": "full"}, {"max_num_imprecise_acc": 128},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, trainer, generators={}, history=[])
            saved = path.read_bytes()
            for policy in variants:
                with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "FP8"):
                    save_checkpoint(path, trainer, {"runtime": {"fp8": policy}}, generators={}, history=[])
                self.assertEqual(path.read_bytes(), saved)

    def test_dense_trainer_cannot_save_metadata_that_enables_fp8(self):
        trainer = make_trainer(False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            with self.assertRaisesRegex(ValueError, "FP8"):
                save_checkpoint(path, trainer, {"runtime": {"fp8": {"enabled": True}}},
                                generators={}, history=[])
            self.assertFalse(path.exists())

    def test_snapshot_records_actual_policy_and_reconstructs_identical_outputs(self):
        trainer = make_trainer(FP8Config(enabled=True, grad_dtype="e5m2"))
        payload = _checkpoint_payload(trainer, {}, generators={}, history=[])
        self.assertEqual(payload["config"]["runtime"]["fp8"], asdict(trainer.runtime.fp8))
        self.assertEqual(payload["trainer_settings"]["fp8"], payload["config"]["runtime"]["fp8"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, trainer, generators={}, history=[])
            payload = read_checkpoint(path)
        reconstructed = Trainer(nn.Sequential(nn.Linear(2, 1)), loss_fn=F.mse_loss, config=payload["config"])
        reconstructed.model.load_state_dict(payload["model"])
        x = torch.tensor([[1., 100.], [2., 2.]])
        torch.testing.assert_close(reconstructed.model(x), trainer.model(x), rtol=0, atol=0)
        self.assertEqual(reconstructed.runtime.fp8, trainer.runtime.fp8)

    def test_callback_rejects_conflicting_explicit_policy(self):
        trainer = make_trainer()
        batches = [(torch.ones(2, 2), torch.zeros(2, 1))]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Checkpoint(directory, config=configuration())
            with self.assertRaisesRegex(ValueError, "FP8"):
                trainer.fit(batches, epochs=1, callbacks=[checkpoint], progress_bar=False)
            self.assertFalse((Path(directory) / "last.pt").exists())

    def test_equivalent_list_block_shapes_restore_the_recorded_runtime_policy(self):
        blocks = {"activation": (1, 2), "weight": (2, 2), "grad": (1, 2)}
        policy = FP8Config(enabled=True, block_shapes=blocks)
        trainer = make_trainer(policy)
        metadata = {"runtime": {"fp8": {"block_shapes": {name: list(shape) for name, shape in blocks.items()}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, trainer, metadata, generators={}, history=[])
            payload = read_checkpoint(path)
        restored = make_trainer(policy)
        with torch.no_grad():
            restored.model[0].weight.zero_()
        restore_checkpoint(payload, restored, metadata, generators={}, batch_size_mode="manual")
        self.assertEqual(restored.runtime.fp8, policy)
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[name], value, rtol=0, atol=0)

    def test_disabled_policy_metadata_differences_are_inactive(self):
        trainer = make_trainer(False)
        inactive = asdict(replace(trainer.runtime.fp8, backend="triton", grad_dtype="e5m2"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, trainer, {"runtime": {"fp8": inactive}}, generators={}, history=[])
            self.assertFalse(read_checkpoint(path)["config"]["runtime"]["fp8"]["enabled"])

    def test_older_policy_without_interception_field_restores_default_behavior(self):
        trainer = make_trainer()
        payload = _checkpoint_payload(trainer, {}, generators={}, history=[])
        payload["config"]["runtime"]["fp8"].pop("intercept_linear")
        payload["trainer_settings"]["fp8"].pop("intercept_linear")
        payload["config"]["runtime"]["fp8"].pop("max_num_imprecise_acc")
        payload["trainer_settings"]["fp8"].pop("max_num_imprecise_acc")
        restored = make_trainer()
        restore_checkpoint(payload, restored, payload["config"], generators={}, batch_size_mode="manual")
        self.assertTrue(restored.runtime.fp8.intercept_linear)
        self.assertEqual(restored.runtime.fp8.max_num_imprecise_acc, 32)
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(restored.model.state_dict()[name], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
