"""Resume integration: change grouping without replaying consumed data."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepopt import Trainer
from deepopt.data.resumable import ResumableDataLoader
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.memory import estimate_training_memory


class StopAfterSave(Exception):
    pass


def regression_step(model, batch):
    inputs, targets = batch
    return {"loss": (model(inputs) - targets).square().mean(), "sample_count": len(targets)}


class ResumeBatchSizeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "state.pt"
        self.dataset = TensorDataset(torch.arange(17, dtype=torch.float32).reshape(-1, 1) / 17,
                                     torch.zeros(17, 1))

    def components(self, batch_size):
        torch.manual_seed(11)
        model = nn.Linear(1, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), step_fn=regression_step)
        loader = ResumableDataLoader(self.dataset, batch_size=batch_size,
                                     generator=torch.Generator().manual_seed(13))
        return trainer, loader

    def step_payload(self, trainer, loader):
        def save_and_stop(current, event):
            save_checkpoint(self.path, current, {}, generators={}, history=[], kind="step", train_loader=loader)
            raise StopAfterSave()

        with self.assertRaises(StopAfterSave):
            trainer.train_epoch(loader, step_callbacks=[save_and_stop])
        return read_checkpoint(self.path)

    def test_repeated_step_resume_preserves_consumed_order_and_progress(self):
        trainer, loader = self.components(4)
        first = self.step_payload(trainer, loader)
        order = first["train_loader"]["state"]["sampler"]["order"]
        self.assertEqual(first["batch_size"], 4)  # Works even without a config schema.
        trainer, loader = self.components(6)
        with self.assertLogs("deepopt.engine.checkpoint", "WARNING"):
            restore_checkpoint(first, trainer, {}, generators={}, train_loader=loader, batch_size_mode="manual")
        self.assertEqual(loader.batch_sampler.position, 4)
        self.assertEqual(trainer.epoch_progress.samples, 4)
        second = self.step_payload(trainer, loader)
        self.assertEqual(second["train_loader"]["state"]["sampler"]["position"], 10)
        self.assertEqual(second["batch_size"], 6)
        trainer, loader = self.components(3)
        restore_checkpoint(second, trainer, {}, generators={}, train_loader=loader, batch_size_mode="manual")
        torch.testing.assert_close(loader.batch_sampler.order, order, rtol=0, atol=0)
        self.assertEqual(list(iter(loader.batch_sampler)), [order[10:13].tolist(), order[13:16].tolist(),
                                                         order[16:].tolist()])
        # Inspecting the sampler above consumed it; restore again before fit.
        restore_checkpoint(second, trainer, {}, generators={}, train_loader=loader, batch_size_mode="manual")
        result = trainer.train_epoch(loader)
        self.assertEqual(result["samples"], 17)
        self.assertEqual(trainer.state.optimizer_steps, 5)
        self.assertEqual(trainer.state.completed_epochs, 1)

    def test_step_size_change_requires_loader_capability_before_state_mutation(self):
        trainer, loader = self.components(4)
        saved = self.step_payload(trainer, loader)
        target, loader = self.components(6)
        loader.supports_batch_size_change = False
        before = deepcopy(target.model.state_dict())
        with self.assertRaisesRegex(ValueError, "supports_batch_size_change"):
            restore_checkpoint(saved, target, {}, generators={}, train_loader=loader, batch_size_mode="manual")
        self.assertTrue(target.at_epoch_boundary)
        for name, value in target.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_old_step_payload_without_batch_metadata_can_resize(self):
        trainer, loader = self.components(4)
        saved = self.step_payload(trainer, loader)
        del saved["batch_size"]
        sampler = saved["train_loader"]["state"]["sampler"]
        saved["train_loader"]["state"]["format_version"] = 1
        del sampler["batch_origin"]
        trainer, loader = self.components(6)
        restore_checkpoint(saved, trainer, {}, generators={}, train_loader=loader, batch_size_mode="manual")
        self.assertEqual(trainer.resume_adjustment["old_batch_size"], 4)
        self.assertEqual(loader.batch_sampler.position, 4)

    def test_automatic_plan_applies_size_before_next_batch_and_resizes_matching_validation(self):
        trainer, loader = self.components(4)
        saved = self.step_payload(trainer, loader)
        trainer, loader = self.components(8)
        validation = DataLoader(self.dataset, batch_size=8)
        plan = {"requested_batch_size": 8, "new_batch_size": 3, "mode": "auto",
                "status": "planned", "reason": "test estimate"}
        with patch("deepopt.engine.checkpointing.checkpoint.plan_batch_size", return_value=plan) as planner:
            restore_checkpoint(saved, trainer, {}, generators={}, train_loader=loader,
                               validation_loader=validation, sample_batch_fn=lambda size: None)
        self.assertEqual(planner.call_args.kwargs["requested_batch_size"], 8)
        self.assertEqual(loader.batch_sampler.position, 4)
        self.assertEqual(loader.batch_sampler.batch_size, 3)
        self.assertEqual(validation.batch_size, 3)
        self.assertEqual(len(next(iter(loader))[0]), 3)
        self.assertEqual(trainer.resume_adjustment["old_batch_size"], 4)

    def test_failed_planning_does_not_enable_partially_restored_trainer(self):
        trainer, loader = self.components(4)
        saved = self.step_payload(trainer, loader)
        trainer, loader = self.components(4)
        with patch("deepopt.engine.checkpointing.checkpoint.plan_batch_size", side_effect=ValueError("unsupported trace")):
            with self.assertRaisesRegex(ValueError, "unsupported trace"):
                restore_checkpoint(saved, trainer, {}, generators={}, train_loader=loader)
        self.assertFalse(trainer.at_epoch_boundary)
        self.assertFalse(trainer._resume_pending)
        self.assertIsNone(trainer.resume_adjustment)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            trainer.train_epoch(loader)
        # Explicit manual restoration recovers safely without repeating training.
        restore_checkpoint(saved, trainer, {}, generators={}, train_loader=loader, batch_size_mode="manual")
        self.assertEqual(trainer.train_epoch(loader)["samples"], 17)

    def test_real_fake_tensor_plan_rebatches_without_advancing_training_state(self):
        trainer, loader = self.components(4)
        saved = self.step_payload(trainer, loader)
        trainer, loader = self.components(8)

        def sample(size):
            return torch.empty(size, 1), torch.empty(size, 1)

        budget = estimate_training_memory(trainer, sample(4)).peak_bytes_by_device["cpu"]
        restore_checkpoint(saved, trainer, {}, generators={}, train_loader=loader,
                           sample_batch_fn=sample, memory_budget_bytes=budget)
        self.assertEqual(trainer.resume_adjustment["new_batch_size"], 4)
        self.assertEqual(loader.batch_sampler.position, 4)
        self.assertEqual(trainer.state.optimizer_steps, 1)
        self.assertTrue(torch.equal(torch.get_rng_state(), saved["rng"]["torch"]))
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, saved["model"][name], rtol=0, atol=0)
        self.assertEqual(trainer.train_epoch(loader)["samples"], 17)


if __name__ == "__main__":
    unittest.main()
