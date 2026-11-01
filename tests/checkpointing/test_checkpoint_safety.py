from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from deepopt.config import config_from_dict
from deepopt.data.resumable import ResumableBatchSampler, ResumableDataLoader
from deepopt.engine import Trainer
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.optimizers import MultiOptimizer
from deepopt.run import train
from deepopt.tasks.regression import RegressionTask


class TwoLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Linear(2, 2, bias=False)

    def forward(self, inputs):
        return self.first(inputs) + 2 * self.second(inputs)


class FailingRestoreSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.01, momentum=0.9)
        self.fail_restore = False

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if self.fail_restore:
            raise RuntimeError("custom optimizer restore failed after loading")


class CheckpointSafetyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = config_from_dict({
            "name": "checkpoint-safety", "seed": 3,
            "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.01,
                         "checkpoint_every_steps": 0},
            "application": {"synthetic": {"train_samples": 5, "validation_samples": 3}},
        })
        self.batches = [(torch.ones(3, 2), torch.zeros(3, 2))]

    def make_trainer(self, *, reverse=False, multiple=False, optimizer_type=torch.optim.SGD):
        model = TwoLayers()
        layers = [model.second, model.first] if reverse else [model.first, model.second]
        if multiple:
            optimizer = MultiOptimizer({
                "first_branch": torch.optim.SGD(layers[0].parameters(), lr=0.01, momentum=0.9),
                "second_branch": torch.optim.SGD(layers[1].parameters(), lr=0.02, momentum=0.9),
            })
        else:
            parameters = [parameter for layer in layers for parameter in layer.parameters()]
            optimizer = (optimizer_type(parameters) if optimizer_type is FailingRestoreSGD else
                         optimizer_type(parameters, lr=0.01, momentum=0.9))
        return Trainer(model, optimizer, task=RegressionTask())

    def payload(self, trainer):
        history = []
        trainer.fit(self.batches, self.batches, epochs=1,
                    callbacks=[lambda _, row: history.append(row)])
        path = self.root / "state.pt"
        save_checkpoint(path, trainer, self.config, generators={}, history=history)
        return read_checkpoint(path)

    def test_resume_rejects_reordered_equal_shape_parameters_and_child_routing(self):
        for multiple in (False, True):
            with self.subTest(multiple=multiple):
                payload = self.payload(self.make_trainer(multiple=multiple))
                target = self.make_trainer(multiple=multiple, reverse=True)
                before = deepcopy(target.model.state_dict())
                with self.assertRaisesRegex(ValueError, "parameter names, order, and child routing"):
                    restore_checkpoint(payload, target, self.config, generators={})
                for name, value in target.model.state_dict().items():
                    self.assertTrue(torch.equal(before[name], value))

    def test_old_checkpoints_without_parameter_binding_are_rejected(self):
        trainer = self.make_trainer()
        payload = self.payload(trainer)
        del payload["optimizer_topology"]
        with self.assertRaisesRegex(ValueError, "no named optimizer topology"):
            restore_checkpoint(payload, trainer, self.config, generators={})

    def test_model_shape_preflight_prevents_partial_tensor_copy(self):
        payload = self.payload(self.make_trainer())
        payload["model"]["first.weight"] = torch.full((2, 2), 99.0)
        payload["model"]["second.weight"] = torch.zeros(3, 3)
        target = self.make_trainer()
        before = deepcopy(target.model.state_dict())
        with self.assertRaisesRegex(ValueError, "different shape"):
            restore_checkpoint(payload, target, self.config, generators={})
        for name, value in target.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))

    def test_failed_component_restore_blocks_training_until_successful_restore(self):
        payload = self.payload(self.make_trainer(optimizer_type=FailingRestoreSGD))
        target = self.make_trainer(optimizer_type=FailingRestoreSGD)
        target.optimizer.fail_restore = True
        target._resume_pending = True
        with self.assertRaisesRegex(RuntimeError, "restore failed after loading"):
            restore_checkpoint(payload, target, self.config, generators={})
        self.assertFalse(target.at_epoch_boundary)
        self.assertFalse(target.at_step_boundary)
        self.assertFalse(target._resume_pending)
        with self.assertRaisesRegex(RuntimeError, "restore the last checkpoint"):
            target.train_epoch(self.batches)
        with self.assertRaisesRegex(RuntimeError, "restore"):
            target.runtime.step()
        target.optimizer.fail_restore = False
        restore_checkpoint(payload, target, self.config, generators={})
        self.assertTrue(target.at_epoch_boundary)
        target.train_epoch(self.batches)
        self.assertEqual(target.state.completed_epochs, 2)

    def test_completed_resume_repairs_best_and_history_without_training(self):
        run_dir = self.root / "run"

        def fail_best(path, *args, **kwargs):
            if Path(path).name == "best.pt":
                raise OSError("simulated best write failure")
            return save_checkpoint(path, *args, **kwargs)

        with redirect_stdout(io.StringIO()), patch("deepopt.run.save_checkpoint", side_effect=fail_best):
            with self.assertRaisesRegex(OSError, "best write failure"):
                train(self.config, output=run_dir)
        last_before = (run_dir / "last.pt").read_bytes()
        self.assertFalse((run_dir / "best.pt").exists())
        self.assertEqual((run_dir / "metrics.jsonl").read_text(), "")
        with patch.object(Trainer, "fit", side_effect=AssertionError("repair must not train")):
            self.assertEqual(train(resume=run_dir / "last.pt"), run_dir.resolve())
            self.assertEqual(train(resume=run_dir / "last.pt"), run_dir.resolve())
        self.assertEqual((run_dir / "last.pt").read_bytes(), last_before)
        self.assertEqual(len((run_dir / "metrics.jsonl").read_text().splitlines()), 1)
        best, last = read_checkpoint(run_dir / "best.pt"), read_checkpoint(run_dir / "last.pt")
        self.assertEqual(best["state"], last["state"])
        for name, value in best["model"].items():
            self.assertTrue(torch.equal(value, last["model"][name]))

    def test_sampler_and_loader_reject_invalid_public_arguments(self):
        generator = torch.Generator().manual_seed(3)
        for size in (-1, True, 1.5):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "size"):
                ResumableBatchSampler(size, 2, generator)
        for batch_size in (0, -1, True, 1.5):
            with self.subTest(batch_size=batch_size), self.assertRaisesRegex(ValueError, "batch_size"):
                ResumableBatchSampler(4, batch_size, generator)
            with self.subTest(loader_batch_size=batch_size), self.assertRaisesRegex(ValueError, "batch_size"):
                ResumableDataLoader(TensorDataset(torch.ones(4, 2)), batch_size=batch_size, generator=generator)
        with self.assertRaisesRegex(ValueError, "CPU torch.Generator"):
            ResumableBatchSampler(4, 2, None)
        self.assertEqual(list(ResumableBatchSampler(0, 2, generator)), [])


if __name__ == "__main__":
    unittest.main()
