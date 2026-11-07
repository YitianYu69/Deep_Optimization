"""Automatic FP8 preparation through public regular-runtime training APIs."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import Checkpoint, Trainer, TrainingCore, TrainingModule, TrainingRuntime
from deepopt.config import FP8Config
from deepopt.opt import transform_model
from deepopt.opt.policy import fp8_plan
from tests.state_support import assert_state_equal


class SharedProjections(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(7, 7)
        self.alias = self.projection
        self.head = nn.Linear(7, 7, bias=False)
        self.head.weight = self.projection.weight

    def forward(self, inputs):
        return self.projection(inputs) + self.alias(inputs) + F.linear(inputs, self.head.weight)


def training_recipe(fp8=True):
    torch.manual_seed(531)
    model = nn.Sequential(nn.Linear(7, 9), nn.Tanh(), nn.Linear(9, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, fp8=fp8)
    batches = [(torch.randn(4, 7), torch.randn(4, 3)) for _ in range(2)]
    return trainer, batches


class AutomaticFP8CoreBlackBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(531)

    def test_original_root_linear_closure_uses_fp8_and_backward_outlives_compute_scope(self):
        original = nn.Linear(7, 3)
        dense = deepcopy(original)
        reference, _ = transform_model(deepcopy(original), plan=fp8_plan(True))
        original_parameters = tuple(original.parameters())
        optimizer = torch.optim.SGD(original_parameters, lr=0.03)
        baseline_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
        runtime = TrainingRuntime(original, optimizer, fp8=True)
        inputs = torch.randn(4, 7)
        actual = runtime.compute(inputs, lambda batch: original(batch))
        expected = reference(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, dense(inputs)))
        self.assertTrue(all(a is b for a, b in zip(original_parameters, runtime.model.parameters())))
        runtime.backward(actual.square().mean())
        expected.square().mean().backward()
        for a, b in zip(runtime.model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        runtime.step()
        baseline_optimizer.step()
        for a, b in zip(runtime.model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_trainer_enabled_and_disabled_match_explicit_fp8_and_dense_updates(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                torch.manual_seed(543)
                original = nn.Sequential(nn.Linear(7, 9), nn.Tanh(), nn.Linear(9, 3))
                baseline = deepcopy(original)
                if enabled:
                    baseline, _ = transform_model(baseline, plan=fp8_plan(True))
                baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.02)
                trainer = Trainer(original, torch.optim.SGD(original.parameters(), lr=0.02),
                                  loss_fn=F.mse_loss, fp8=enabled)
                batches = [(torch.randn(4, 7), torch.randn(4, 3)) for _ in range(2)]
                trainer.train_epoch(batches)
                for inputs, targets in batches:
                    baseline_optimizer.zero_grad(set_to_none=True)
                    F.mse_loss(baseline(inputs), targets).backward()
                    baseline_optimizer.step()
                for a, b in zip(trainer.model.parameters(), baseline.parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                self.assertIs(trainer.model, original)
                self.assertEqual(trainer.state.optimizer_steps, 2)

    def test_existing_optimizer_state_shared_modules_and_tied_parameters_survive_preparation(self):
        model = SharedProjections()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        inputs, targets = torch.randn(4, 7), torch.randn(4, 7)
        F.mse_loss(model(inputs), targets).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        parameters = tuple(model.parameters())
        states = {parameter: optimizer.state[parameter] for parameter in parameters}
        saved_optimizer = deepcopy(optimizer.state_dict())
        reference, _ = transform_model(deepcopy(model), plan=fp8_plan(True))
        baseline_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.002)
        baseline_optimizer.load_state_dict(saved_optimizer)

        runtime = TrainingRuntime(model, optimizer, fp8=FP8Config(enabled=True))
        self.assertIs(runtime.model, model)
        self.assertIs(model.projection, model.alias)
        self.assertIs(model.projection.weight, model.head.weight)
        for a, b in zip(parameters, model.parameters()):
            self.assertIs(a, b)
            self.assertIs(optimizer.state[a], states[a])
        assert_state_equal(self, optimizer.state_dict(), saved_optimizer)

        actual = runtime.compute(inputs, lambda batch: model(batch))
        expected = reference(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        runtime.backward(F.mse_loss(actual, targets))
        runtime.step()
        F.mse_loss(expected, targets).backward()
        baseline_optimizer.step()
        assert_state_equal(self, model.state_dict(), reference.state_dict())
        assert_state_equal(self, optimizer.state_dict(), baseline_optimizer.state_dict())

    def test_training_module_keeps_custom_step_and_bare_core_trains_fp8(self):
        class Recipe(TrainingModule):
            def training_step(self, batch):
                inputs, targets = batch
                return F.mse_loss(self(inputs), targets)

        model = Recipe(nn.Linear(7, 3))
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.02), fp8=True)
        self.assertIs(trainer.model, model)
        self.assertIsInstance(trainer.model, Recipe)
        batches = [(torch.randn(4, 7), torch.randn(4, 3))]
        self.assertTrue(torch.isfinite(torch.tensor(trainer.train_epoch(batches)["loss"])).item())

        source = nn.Linear(7, 3)
        expected, _ = transform_model(deepcopy(source), plan=fp8_plan(True))
        optimizer = torch.optim.SGD(source.parameters(), lr=0.02)
        baseline_optimizer = torch.optim.SGD(expected.parameters(), lr=0.02)
        core = TrainingCore(source, torch.device("cpu"), fp8=True)
        inputs, targets = batches[0]
        loss = core.compute((inputs, targets), lambda batch: F.mse_loss(core.model(batch[0]), batch[1]))
        core.backward(loss)
        optimizer.step()
        F.mse_loss(expected(inputs), targets).backward()
        baseline_optimizer.step()
        for a, b in zip(core.model.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_direct_checkpoint_restores_same_policy_and_rejects_policy_changes(self):
        reference, reference_batches = training_recipe()
        reference.fit(reference_batches, epochs=2, progress_bar=False)
        with tempfile.TemporaryDirectory() as directory:
            trainer, batches = training_recipe()
            trainer.fit(batches, epochs=1, callbacks=[Checkpoint(directory)], progress_bar=False)
            resumed, resumed_batches = training_recipe()
            restored = Checkpoint(directory)
            restored.restore(resumed, resumed_batches)
            resumed.fit(resumed_batches, epochs=2, callbacks=[restored], progress_bar=False)
            assert_state_equal(self, resumed.model.state_dict(), reference.model.state_dict())
            assert_state_equal(self, resumed.optimizer.state_dict(), reference.optimizer.state_dict())
            self.assertEqual(resumed.state.optimizer_steps, 4)
            self.assertTrue((Path(directory) / "last.pt").exists())
            for policy in (False, FP8Config(enabled=True, grad_dtype="e5m2")):
                with self.subTest(policy=policy):
                    incompatible, incompatible_batches = training_recipe(policy)
                    before = deepcopy(incompatible.model.state_dict())
                    with self.assertRaisesRegex(ValueError, "settings|FP8|fp8|configuration"):
                        Checkpoint(directory).restore(incompatible, incompatible_batches)
                    assert_state_equal(self, incompatible.model.state_dict(), before)


if __name__ == "__main__":
    unittest.main()
