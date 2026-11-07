"""Global defaults form owned training recipes and preserve explicit algorithms."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import Checkpoint, Trainer
from deepopt.config import config_from_dict, get_config, load_config, reset_config
from deepopt.engine.checkpointing.checkpoint import read_checkpoint
from deepopt.engine.strategies import ClipGradValue, MicrobatchMean
from deepopt.run import train as run_training
from deepopt.tasks.regression import RegressionTask
from tests.state_support import assert_state_equal
from tests.training.config_support import make_batches, make_config


class GlobalTrainingBlackBoxTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.enterContext(redirect_stdout(io.StringIO()))
        torch.manual_seed(586)

    def test_auto_optimizer_and_fit_use_snapshot_after_global_replacement(self):
        load_config(make_config())
        model = nn.Linear(2, 1)
        baseline = deepcopy(model)
        trainer = Trainer(model, loss_fn=F.mse_loss)
        baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
        replacement = make_config()
        replacement["name"] = "later-defaults"
        replacement["training"].update(epochs=5, learning_rate=0.8, optimizer="adamw")
        load_config(replacement)
        batches = make_batches()
        trainer.fit(batches, progress_bar=False)
        for _ in range(2):
            for inputs, targets in batches:
                baseline_optimizer.zero_grad(set_to_none=True)
                F.mse_loss(baseline(inputs), targets).backward()
                baseline_optimizer.step()
        self.assertIsInstance(trainer.optimizer, torch.optim.SGD)
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 0.05)
        self.assertEqual((trainer.state.completed_epochs, trainer.state.optimizer_steps), (2, 6))
        for parameter, expected in zip(model.parameters(), baseline.parameters()):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        self.assertEqual(trainer.config["name"], "global-training")
        self.assertEqual(get_config().name, "later-defaults")

    def test_learning_rate_only_partial_builds_optimizer_but_fit_needs_epochs(self):
        trainer = Trainer(nn.Linear(2, 1), loss_fn=F.mse_loss,
                          config={"training": {"learning_rate": 0.03}})
        self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 0.03)
        self.assertNotIn("epochs", trainer.config["training"])
        self.assertNotIn("batch_size", trainer.config["training"])
        with self.assertRaisesRegex(ValueError, "epochs"):
            trainer.fit(make_batches(), progress_bar=False)
        trainer.fit(make_batches(), epochs=1, progress_bar=False)
        self.assertEqual(trainer.state.optimizer_steps, 3)
        with self.assertRaisesRegex(ValueError, "optimizer"):
            Trainer(nn.Linear(2, 1), loss_fn=F.mse_loss)

    def test_existing_optimizer_groups_and_state_keep_their_own_learning_rates(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        optimizer = torch.optim.AdamW([
            {"params": model[0].parameters(), "lr": 0.03, "weight_decay": 0.0},
            {"params": model[1].parameters(), "lr": 0.07, "weight_decay": 0.1},
        ])
        inputs, targets = make_batches()[0]
        F.mse_loss(model(inputs), targets).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        before = deepcopy(optimizer.state_dict())
        state_objects = {parameter: optimizer.state[parameter] for parameter in model.parameters()}
        baseline = deepcopy(model)
        baseline_optimizer = torch.optim.AdamW([
            {"params": baseline[0].parameters()}, {"params": baseline[1].parameters()},
        ])
        baseline_optimizer.load_state_dict(deepcopy(before))
        configured = make_config()
        configured["training"]["learning_rate"] = 0.9
        load_config(configured)
        trainer = Trainer(model, optimizer, loss_fn=F.mse_loss)
        self.assertIs(trainer.optimizer, optimizer)
        self.assertIsNone(trainer.scheduler)  # lr_gamma=1 does not invent a scheduler.
        for parameter, state in state_objects.items():
            self.assertIs(optimizer.state[parameter], state)
        assert_state_equal(self, optimizer.state_dict(), before)
        trainer.train_epoch(make_batches())
        for inputs, targets in make_batches():
            baseline_optimizer.zero_grad(set_to_none=True)
            F.mse_loss(baseline(inputs), targets).backward()
            baseline_optimizer.step()
        assert_state_equal(self, model.state_dict(), baseline.state_dict())
        assert_state_equal(self, optimizer.state_dict(), baseline_optimizer.state_dict())

    def test_custom_loss_step_task_and_strategies_keep_their_numerical_contract(self):
        for interface in ("loss", "step", "task"):
            with self.subTest(interface=interface):
                configured = make_config()
                configured["training"].update(epochs=1, accumulation_steps=2, grad_clip_norm=0.5)
                load_config(configured)
                torch.manual_seed(599)
                model = nn.Linear(2, 1)
                baseline = deepcopy(model)
                strategy, gradients = MicrobatchMean(), ClipGradValue(0.1)

                def objective(prediction, targets):
                    return F.l1_loss(prediction, targets) + 0.01 * prediction.square().mean()

                def step(current, batch):
                    return {"loss": objective(current(batch[0]), batch[1])}

                if interface == "loss":
                    options = {"loss_fn": objective}
                elif interface == "step":
                    options = {"step_fn": step}
                else:
                    options = {"task": RegressionTask(loss_fn=objective)}
                trainer = Trainer(model, grad_clip_norm=None, loss_strategy=strategy,
                                  gradient_strategy=gradients, **options)
                self.assertIs(trainer.loss_strategy, strategy)
                self.assertIs(trainer.runtime.gradient_strategy, gradients)
                batches = make_batches()
                trainer.fit(batches, progress_bar=False)
                baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
                for start in range(0, len(batches), 2):
                    window = batches[start:start + 2]
                    baseline_optimizer.zero_grad(set_to_none=True)
                    for inputs, targets in window:
                        objective(baseline(inputs), targets).backward()
                    for parameter in baseline.parameters():
                        parameter.grad.div_(len(window))
                    nn.utils.clip_grad_value_(baseline.parameters(), 0.1)
                    baseline_optimizer.step()
                self.assertEqual(trainer.state.optimizer_steps, 2)
                for parameter, expected in zip(model.parameters(), baseline.parameters()):
                    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)

    def test_checkpoint_without_config_binds_to_trainer_not_later_global_defaults(self):
        configured = make_config()
        configured["training"]["epochs"] = 1
        load_config(configured)
        trainer = Trainer(nn.Linear(2, 1), loss_fn=F.mse_loss)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Checkpoint(directory)
            later = make_config()
            later["name"] = "unrelated-run"
            later["training"].update(epochs=9, learning_rate=0.8)
            later["runtime"]["fp8"] = {"enabled": True}
            load_config(later)
            trainer.fit(make_batches(), callbacks=[checkpoint], progress_bar=False)
            saved = read_checkpoint(Path(directory) / "last.pt")
        self.assertEqual(saved["config"]["name"], "global-training")
        self.assertEqual(saved["config"]["training"]["learning_rate"], 0.05)
        self.assertEqual(saved["config"]["training"]["epochs"], 1)
        self.assertFalse(saved["config"]["runtime"]["fp8"]["enabled"])
        self.assertEqual(saved["state"]["optimizer_steps"], 3)
        self.assertEqual(get_config().name, "unrelated-run")

    def test_managed_resume_uses_checkpoint_recipe_instead_of_current_global(self):
        configured = make_config()
        configured["application"] = {"synthetic": {
            "input_dim": 2, "num_classes": 2, "train_samples": 8, "validation_samples": 4,
        }}
        configured["training"]["batch_size"] = 4
        original = config_from_dict(configured)  # Pure construction, no installation.
        load_config(original)
        with tempfile.TemporaryDirectory() as directory:
            output = run_training(output=Path(directory) / "resumed", epochs=1,
                                  batch_size_mode="manual", progress_bar=False)
            later = deepcopy(configured)
            later["name"] = "unrelated-run"
            later["seed"] = 999
            later["training"].update(learning_rate=0.9, optimizer="adamw")
            later["application"]["synthetic"].update(input_dim=5, num_classes=3)
            load_config(later)
            run_training(resume=output / "last.pt", epochs=2,
                         batch_size_mode="manual", progress_bar=False)
            uninterrupted = run_training(original, output=Path(directory) / "reference",
                                         batch_size_mode="manual", progress_bar=False)
            saved, expected = read_checkpoint(output / "last.pt"), read_checkpoint(uninterrupted / "last.pt")
        self.assertEqual(saved["config"]["name"], "global-training")
        self.assertEqual(saved["config"]["application"]["synthetic"]["input_dim"], 2)
        for key in ("model", "optimizer", "scheduler", "state"):
            assert_state_equal(self, saved[key], expected[key])
        self.assertEqual(get_config().name, "unrelated-run")


if __name__ == "__main__":
    unittest.main()
