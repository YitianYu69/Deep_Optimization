"""Compatibility comparison and failure containment during checkpoint restore."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from deepopt import Trainer
from deepopt.config import reset_config
from deepopt.data.resumable import ResumableDataLoader
from deepopt.engine.checkpointing.checkpoint import (
    _resume_config_for_comparison,
    read_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from deepopt.tasks.regression import RegressionTask


class StopAfterCheckpoint(Exception):
    pass


class RestoreValidationTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "state.pt"

    def components(self):
        model = nn.Linear(1, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01),
                          task=RegressionTask(), device="cpu")
        loader = ResumableDataLoader(
            TensorDataset(torch.ones(4, 1), torch.zeros(4, 1)), batch_size=2,
            generator=torch.Generator().manual_seed(7),
        )
        return trainer, loader

    def step_payload(self):
        trainer, loader = self.components()

        def save_and_stop(current, event):
            save_checkpoint(self.path, current, generators={}, history=[],
                            kind="step", train_loader=loader)
            raise StopAfterCheckpoint()

        with self.assertRaises(StopAfterCheckpoint):
            trainer.train_epoch(loader, step_callbacks=[save_and_stop])
        return read_checkpoint(self.path)

    def assert_training_blocked(self, trainer, loader):
        self.assertFalse(trainer.at_epoch_boundary)
        self.assertFalse(trainer.at_step_boundary)
        self.assertFalse(trainer._resume_pending)
        self.assertIsNone(trainer.epoch_progress)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            trainer.train_epoch(loader)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            trainer.runtime.step()

    def test_comparison_keeps_unknown_metadata_without_mutating_inputs(self):
        config = {
            "training": {"epochs": 1, "batch_size": 4, "custom_objective": {"weights": [1, 2]}},
            "runtime": {"checks": True, "custom_runtime": {"version": 3}},
            "application": {"recipe": {"revision": 7}},
            "extension": {"values": [1, 2, 3]},
        }
        before = deepcopy(config)
        normalized = _resume_config_for_comparison(config)
        self.assertEqual(config, before)
        self.assertEqual(normalized["training"], {"custom_objective": {"weights": [1, 2]}})
        self.assertEqual(normalized["runtime"], {"custom_runtime": {"version": 3}})
        self.assertEqual(normalized["application"], config["application"])
        self.assertEqual(normalized["extension"], config["extension"])
        normalized["extension"]["values"].append(4)
        normalized["training"]["custom_objective"]["weights"].append(3)
        self.assertEqual(config, before)

    def test_comparison_accepts_legacy_defaults_but_keeps_active_policies_strict(self):
        legacy = {"training": {"learning_rate": 0.01}, "runtime": {"device": "cpu"}}
        current = {
            "training": {"learning_rate": 0.01, "epochs": 5, "batch_size": 16,
                         "checkpoint_every_steps": 3, "scheduler_interval": "epoch",
                         "lr_schedule": None, "ema": {"enabled": False, "decay": 0.8}},
            "runtime": {"device": "cpu", "checks": False, "non_blocking": True,
                        "execution_mode": "eager", "diagnostics": {"enabled": True},
                        "mode_schedule": [], "compile": {"enabled": False, "backend": "eager"},
                        "fp8": {"enabled": False},
                        "ddp": {"find_unused_parameters": False, "broadcast_buffers": True,
                                "gradient_as_bucket_view": True, "static_graph": False,
                                "validation_sync_buffers": True},
                        "cuda_graph": {"device": None, "warmup": 3, "allow_unused_input": False,
                                       "autograd": True, "capture_error_mode": "global"}},
            "logging": {"level": "DEBUG"}, "schedules": {}, "loader": {}, "experiment_identity": {},
        }
        before = deepcopy(current)
        expected = _resume_config_for_comparison(legacy)
        self.assertEqual(_resume_config_for_comparison(current), expected)
        self.assertEqual(current, before)
        changes = (
            ("training", "scheduler_interval", "step"),
            ("training", "lr_schedule", {"kind": "constant"}),
            ("training", "ema", {"enabled": True, "decay": 0.8}),
            ("runtime", "compile", {"enabled": True}),
            ("runtime", "fp8", {"enabled": True}),
            ("runtime", "ddp", {"broadcast_buffers": False}),
            ("runtime", "cuda_graph", {"warmup": 4}),
        )
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                changed = deepcopy(current)
                changed[section][key] = value
                self.assertNotEqual(_resume_config_for_comparison(changed), expected)

    def test_unknown_checkpoint_metadata_cannot_change_on_resume(self):
        source, _ = self.components()
        save_checkpoint(self.path, source, generators={}, history=[])
        payload = read_checkpoint(self.path)
        payload["config"]["extension"] = {"recipe_revision": 1}
        target, _ = self.components()
        before = deepcopy(target.model.state_dict())
        with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
            restore_checkpoint(payload, target, {"extension": {"recipe_revision": 2}}, generators={})
        self.assertTrue(target.at_epoch_boundary)
        for name, value in target.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        restore_checkpoint(payload, target, {"extension": {"recipe_revision": 1}}, generators={})
        self.assertEqual(payload["config"]["extension"], {"recipe_revision": 1})

    def test_metric_factory_failure_blocks_training_before_loader_or_model_restore(self):
        payload = self.step_payload()
        target, loader = self.components()
        with (patch.object(target, "new_metrics", side_effect=RuntimeError("metric factory failed")),
              patch.object(loader, "load_state_dict") as restore_loader,
              patch.object(target.model, "load_state_dict") as restore_model):
            with self.assertRaisesRegex(RuntimeError, "metric factory failed"):
                restore_checkpoint(payload, target, generators={}, train_loader=loader, batch_size_mode="manual")
            restore_loader.assert_not_called()
            restore_model.assert_not_called()
        self.assert_training_blocked(target, loader)
        restore_checkpoint(payload, target, generators={}, train_loader=loader, batch_size_mode="manual")
        self.assertEqual(target.train_epoch(loader)["samples"], 4)

    def test_loader_failure_after_cursor_mutation_blocks_training_until_retry(self):
        payload = self.step_payload()
        target, loader = self.components()
        load_state = loader.load_state_dict

        def fail_after_loading(state):
            self.assert_training_blocked(target, loader)
            load_state(state)
            raise RuntimeError("loader failed after cursor restore")

        with patch.object(loader, "load_state_dict", side_effect=fail_after_loading):
            with self.assertRaisesRegex(RuntimeError, "loader failed after cursor restore"):
                restore_checkpoint(payload, target, generators={}, train_loader=loader, batch_size_mode="manual")
        self.assertEqual(loader.batch_sampler.position, 2)
        self.assert_training_blocked(target, loader)
        restore_checkpoint(payload, target, generators={}, train_loader=loader, batch_size_mode="manual")
        self.assertEqual(target.train_epoch(loader)["samples"], 4)

    def test_epoch_reset_failure_blocks_training_until_retry(self):
        source, _ = self.components()
        save_checkpoint(self.path, source, generators={}, history=[])
        payload = read_checkpoint(self.path)
        target, loader = self.components()
        with patch.object(loader, "reset_epoch", side_effect=RuntimeError("epoch reset failed")):
            with self.assertRaisesRegex(RuntimeError, "epoch reset failed"):
                restore_checkpoint(payload, target, generators={}, train_loader=loader, batch_size_mode="manual")
        self.assert_training_blocked(target, loader)
        restore_checkpoint(payload, target, generators={}, train_loader=loader, batch_size_mode="manual")
        self.assertEqual(target.train_epoch(loader)["samples"], 4)


if __name__ == "__main__":
    unittest.main()
