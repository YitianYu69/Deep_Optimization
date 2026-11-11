"""Ownership and precedence at trainer/runtime/configuration boundaries."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import Trainer, TrainingRuntime
from deepopt.config import get_config, load_config, reset_config
from deepopt.opt.nn.linear import DOLinear
from deepopt.optimizer_factory import build_optimizer
from tests.training.config_support import make_batches, make_config


class GlobalTrainingWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)
        torch.manual_seed(607)

    def test_explicit_false_none_and_scalar_options_override_configured_defaults(self):
        configured = make_config()
        configured["training"].update(accumulation_steps=4, grad_clip_norm=0.4, lr_gamma=0.8)
        configured["schedules"] = {"aux": {"type": "linear_warmup", "start": 0.0, "end": 1.0, "duration": 2}}
        load_config(configured)
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.07)
        with patch("deepopt.scheduling.lr.build_scheduler", side_effect=AssertionError("Explicit None must disable auto scheduling")):
            trainer = Trainer(
                model, optimizer, loss_fn=F.mse_loss,
                config={"training": {"accumulation_steps": 3, "grad_clip_norm": 0.2}},
                accumulation_steps=2, checks=False, grad_clip_norm=None, scheduler=None, schedules=None,
            )
        self.assertEqual(trainer.accumulation_steps, 2)
        self.assertFalse(trainer.runtime._checks_spec)
        self.assertIsNone(trainer.grad_clip_norm)
        self.assertIsNone(trainer.runtime.gradient_strategy)
        self.assertIsNone(trainer.scheduler)
        self.assertIsNone(trainer.schedules)
        self.assertIs(trainer.optimizer, optimizer)
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 0.07)
        self.assertEqual(trainer.config["training"]["accumulation_steps"], 2)
        self.assertIsNone(trainer.config["training"]["grad_clip_norm"])
        self.assertFalse(trainer.config["runtime"]["checks"])
        snapshot = trainer.config
        snapshot["training"]["accumulation_steps"] = 99
        self.assertEqual(trainer.config["training"]["accumulation_steps"], 2)
        self.assertEqual(get_config().training.accumulation_steps, 4)
        self.assertTrue(get_config().runtime.checks)

    def test_supplied_runtime_is_authoritative_despite_new_global_runtime_settings(self):
        configured = make_config()
        configured["runtime"].update(checks=False, execution_mode="max_speed")
        load_config(configured)
        model = nn.Linear(2, 1)
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.03))
        later = make_config()
        later["runtime"].update(checks=True, execution_mode="analyze", fp8={"enabled": True})
        load_config(later)
        trainer = Trainer(runtime, loss_fn=F.mse_loss)
        self.assertIs(trainer.runtime, runtime)
        self.assertIs(trainer.model, model)
        self.assertIs(type(model), nn.Linear)
        self.assertFalse(runtime.fp8.enabled)
        self.assertFalse(runtime._checks_spec)
        self.assertEqual(runtime.execution_mode, "max_speed")
        self.assertFalse(trainer.config["runtime"]["fp8"]["enabled"])
        self.assertTrue(get_config().runtime.fp8.enabled)
        with self.assertRaisesRegex(ValueError, "runtime|Runtime"):
            Trainer(runtime, loss_fn=F.mse_loss, config={"runtime": {"checks": True}})
        with self.assertRaisesRegex(ValueError, "runtime|Runtime"):
            Trainer(runtime, loss_fn=F.mse_loss, config={"runtime": {"fp8": {"enabled": True}}})
        training_only = Trainer(runtime, loss_fn=F.mse_loss, config={"training": {"epochs": 1}})
        self.assertIs(training_only.runtime, runtime)
        self.assertFalse(training_only.config["runtime"]["fp8"]["enabled"])

    def test_fit_accepts_only_lifecycle_overrides_and_explicit_epochs_take_priority(self):
        load_config(make_config())
        trainer = Trainer(nn.Linear(2, 1), loss_fn=F.mse_loss)
        batches = make_batches()
        trainer.fit(batches, batches, epochs=1, progress_bar=False,
                    config={"training": {"epochs": 3, "monitor": None, "monitor_mode": "max"}})
        self.assertEqual(trainer.state.completed_epochs, 1)
        self.assertIsNone(trainer.state.monitor)
        self.assertEqual(trainer.state.monitor_mode, "max")
        saved = deepcopy(trainer.model.state_dict())
        for override in ({"runtime": {"checks": False}}, {"training": {"learning_rate": 0.9}},
                         {"training": {"accumulation_steps": 3}}):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "fit config"):
                trainer.fit(batches, config=override, progress_bar=False)
        self.assertEqual(trainer.state.completed_epochs, 1)
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)

    def test_global_fp8_and_compile_prepare_before_automatic_optimizer_ownership(self):
        configured = make_config()
        configured["runtime"].update(fp8={"enabled": True}, compile={"enabled": True, "backend": "eager"})
        load_config(configured)
        model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))
        parameters = tuple(model.parameters())
        events = []

        def compiler(function, **options):
            self.assertIsInstance(model[0], DOLinear)
            self.assertIsInstance(model[2], DOLinear)
            events.append("compile")
            return function

        def optimizer_factory(actual, settings):
            self.assertIs(actual, model)
            self.assertIsNotNone(actual._compiled_call_impl)
            self.assertTrue(all(a is b for a, b in zip(parameters, actual.parameters())))
            events.append("optimizer")
            # Even a callback changing process defaults cannot alter this build.
            replacement = make_config()
            replacement["name"] = "new-global-during-construction"
            replacement["runtime"]["checks"] = False
            load_config(replacement)
            return build_optimizer(actual, settings)

        with patch("torch.compile", side_effect=compiler), \
                patch("deepopt.optimizer_factory.build_optimizer", new=optimizer_factory):
            trainer = Trainer(model, loss_fn=F.mse_loss)
        self.assertEqual(events, ["compile", "optimizer"])
        self.assertTrue(trainer.runtime.fp8.enabled)
        self.assertTrue(trainer.runtime._checks_spec)
        self.assertIsNotNone(trainer.runtime.transformation_report)
        self.assertEqual(trainer.config["name"], "global-training")
        self.assertEqual(get_config().name, "new-global-during-construction")
        self.assertTrue(all(a is b for a, b in zip(parameters, trainer.model.parameters())))

    def test_global_lr_and_coefficient_schedules_follow_optimizer_updates(self):
        configured = make_config()
        configured["training"].update(epochs=1, learning_rate=0.1, lr_gamma=0.5, scheduler_interval="step")
        configured["schedules"] = {"weight": {"type": "linear_warmup", "start": 0.0, "end": 1.0, "duration": 2}}
        load_config(configured)
        seen = []

        def step(model, batch):
            weight = trainer.schedules["weight"]
            seen.append(weight)
            return F.mse_loss(model(batch[0]), batch[1]) * weight

        trainer = Trainer(nn.Linear(2, 1), step_fn=step)
        trainer.fit(make_batches(), progress_bar=False)
        self.assertIsInstance(trainer.scheduler, torch.optim.lr_scheduler.ExponentialLR)
        self.assertEqual(trainer.scheduler.last_epoch, 3)
        self.assertEqual(trainer.schedules.completed_update, 3)
        self.assertEqual(seen, [0.0, 1.0, 1.0])
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 0.0125)


if __name__ == "__main__":
    unittest.main()
