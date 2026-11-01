import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deepopt.engine import Trainer, TrainingRuntime
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.data.resumable import ResumableDataLoader
from deepopt.module import TrainingModule
from deepopt.tasks.base import BatchOutput
from deepopt.tasks.regression import RegressionTask
from deepopt.tasks.classification import ClassificationTask


class CustomModule(TrainingModule):
    """Different train/validation computations with a dictionary batch."""

    def training_step(self, batch):
        prediction = self(batch["x"])
        return BatchOutput((prediction - batch["y"]).square().mean(), len(batch["y"]), "mean")

    def validation_step(self, batch):
        prediction = self(batch["x"])
        return BatchOutput((prediction - batch["y"]).abs().mean(), len(batch["y"]), "mean")


class ModuleTests(unittest.TestCase):
    def test_wrapper_only_preserves_forward_parameters_buffers_and_state(self):
        model = nn.Linear(2, 1)
        module = TrainingModule(model)
        module.register_buffer("scale", torch.tensor(2.0))
        x = torch.ones(3, 2)
        self.assertTrue(torch.equal(module(input=x), model(x)))
        self.assertEqual(list(module.parameters()), list(model.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(module.state_dict(), path)
            restored = TrainingModule(nn.Linear(2, 1))
            restored.register_buffer("scale", torch.tensor(0.0))
            restored.load_state_dict(torch.load(path, weights_only=True))
            self.assertTrue(torch.equal(restored(x), module(x)))
            self.assertEqual(restored.scale.item(), 2.0)
        with self.assertRaisesRegex(NotImplementedError, "Override training_step"):
            module.training_step(x)

    def test_separate_custom_steps_run_through_fit(self):
        module = CustomModule(nn.Linear(2, 1))
        optimizer = torch.optim.SGD(module.parameters(), lr=0.02)
        trainer = Trainer(module, optimizer)
        batches = [{"x": torch.ones(3, 2), "y": torch.ones(3, 1) * 5}]
        before = module(batches[0]["x"]).detach().clone()
        rows = []
        trainer.fit(batches, batches, epochs=1, callbacks=[lambda trainer, row: rows.append(row)])
        expected_validation = (module(batches[0]["x"]) - batches[0]["y"]).abs().mean().item()
        self.assertAlmostEqual(rows[0]["validation"]["loss"], expected_validation)
        self.assertFalse(torch.equal(before, module(batches[0]["x"])))
        self.assertEqual(trainer.state.optimizer_steps, 1)

    def test_manual_loop_matches_fit_with_uneven_accumulation(self):
        torch.manual_seed(4)
        auto = TrainingModule(nn.Linear(2, 1), task=RegressionTask())
        manual = copy.deepcopy(auto)
        dataset = TensorDataset(torch.randn(11, 2), torch.randn(11, 1))
        loader = DataLoader(dataset, batch_size=4)
        trainer = Trainer(auto, torch.optim.SGD(auto.parameters(), lr=0.1), accumulation_steps=2)
        runtime = TrainingRuntime(manual, torch.optim.SGD(manual.parameters(), lr=0.1))
        trainer.fit(loader, loader, epochs=2)
        for _ in range(2):
            manual.train()
            runtime.zero_grad()
            samples = batches = 0
            for batch in loader:
                with runtime.autocast():
                    output = manual.training_step(runtime.prepare_batch(batch))
                runtime.backward(output.loss_sum())
                samples += output.sample_count
                batches += 1
                if batches == 2:
                    runtime.step(samples=samples)
                    samples = batches = 0
            if batches:
                runtime.step(samples=samples)
        for actual, expected in zip(manual.parameters(), auto.parameters()):
            self.assertTrue(torch.equal(actual, expected))

    def test_manual_scalar_loss_and_multiple_backward_calls(self):
        model = nn.Linear(2, 1)
        expected = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        reference = torch.optim.SGD(expected.parameters(), lr=0.1)
        runtime = TrainingRuntime(model, optimizer)
        x = torch.ones(3, 2)
        output = model(x)
        runtime.backward(output.square().mean(), retain_graph=True)
        runtime.backward(output.mean())
        runtime.step()  # Scalar losses already reduced by the user.
        other = expected(x)
        (other.square().mean() + other.mean()).backward()
        reference.step()
        for actual, target in zip(model.parameters(), expected.parameters()):
            torch.testing.assert_close(actual, target)
            self.assertIsNone(actual.grad)

    def test_runtime_rejects_invalid_normalization_before_updating(self):
        model = nn.Linear(2, 1)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.1))
        runtime.backward(model(torch.ones(1, 2)).sum())
        before = model.weight.detach().clone()
        for samples in (0, -1, True, 0.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                runtime.step(samples)
        self.assertTrue(torch.equal(before, model.weight))

    def test_conflicting_step_ownership_is_rejected(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, "Provide loss_fn or step_fn"):
            Trainer(model, optimizer)
        with self.assertRaisesRegex(ValueError, "Pass the task to TrainingModule"):
            Trainer(TrainingModule(model), optimizer, task=RegressionTask())

    def test_wrapped_task_type_is_validated_on_restore(self):
        module = TrainingModule(nn.Linear(2, 2), task=RegressionTask())
        trainer = Trainer(module, torch.optim.SGD(module.parameters(), lr=0.1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, trainer, {}, generators={}, history=[])
            module.task = ClassificationTask()
            with self.assertRaisesRegex(ValueError, "same component types"):
                restore_checkpoint(read_checkpoint(path), trainer, {}, generators={})

    def test_module_step_checkpoint_matches_uninterrupted_training(self):
        dataset = TensorDataset(torch.arange(22).float().reshape(11, 2) / 22,
                                torch.ones(11, 1))

        def setup():
            torch.manual_seed(12)
            module = TrainingModule(nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.3), nn.Linear(4, 1)),
                                    task=RegressionTask())
            optimizer = torch.optim.AdamW(module.parameters(), lr=0.02)
            trainer = Trainer(module, optimizer, accumulation_steps=2)
            generator = torch.Generator().manual_seed(5)
            loader = ResumableDataLoader(dataset, batch_size=4, generator=generator)
            return trainer, loader, {"train": generator}

        full, loader, _ = setup()
        full.fit(loader, DataLoader(dataset, batch_size=4), epochs=2)
        interrupted, loader, generators = setup()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_step.pt"

            def stop(trainer, event):
                save_checkpoint(path, trainer, {}, generators=generators, history=[],
                                kind="step", train_loader=loader)
                raise InterruptedError()

            with self.assertRaises(InterruptedError):
                interrupted.fit(loader, [], epochs=2, step_callbacks=[stop])
            resumed, loader, generators = setup()
            restore_checkpoint(read_checkpoint(path), resumed, {}, generators=generators, train_loader=loader)
            resumed.fit(loader, DataLoader(dataset, batch_size=4), epochs=2)
            self.assertEqual(full.state, resumed.state)
            for expected, actual in zip(full.model.parameters(), resumed.model.parameters()):
                self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
