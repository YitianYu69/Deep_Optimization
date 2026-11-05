"""Explicit arguments are applied before merged configuration validation."""

import unittest

import torch
from torch import nn

from deepopt import Trainer, TrainingCore, TrainingRuntime, load_config, reset_config
from deepopt.config import FP8Config


def configuration(runtime):
    return {"name": "precedence", "seed": 7,
            "training": {"epochs": 1, "batch_size": 2, "learning_rate": .1},
            "runtime": runtime}


class ExplicitPrecedenceTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_trainer_explicit_none_clears_inherited_mode_before_validation(self):
        load_config(configuration({"execution_mode": "analyze"}))
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1),
                          loss_fn=nn.functional.mse_loss, execution_mode=None,
                          config={"runtime": {"diagnostics": {"mode": "balanced"}}})
        self.assertEqual(trainer.execution_mode, "balanced")
        self.assertIsNone(trainer.config["runtime"]["execution_mode"])
        trainer.train_epoch([(torch.ones(2, 2), torch.zeros(2, 1))])
        self.assertEqual(trainer.state.optimizer_steps, 1)

    def test_runtime_explicit_none_clears_inherited_mode_before_validation(self):
        load_config(configuration({"execution_mode": "analyze"}))
        model = nn.Linear(2, 1)
        runtime = TrainingRuntime(model, execution_mode=None,
                                  config={"runtime": {"diagnostics": {"mode": "balanced"}}})
        self.assertEqual(runtime.execution_mode, "balanced")
        self.assertIsNone(runtime.config["runtime"]["execution_mode"])

    def test_training_explicit_fields_override_invalid_per_call_values(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1),
                          loss_fn=nn.functional.mse_loss, accumulation_steps=2, scheduler_interval="step",
                          config={"training": {"accumulation_steps": 0, "scheduler_interval": "invalid"}})
        trainer.train_epoch([(torch.ones(2, 2), torch.zeros(2, 1))] * 3)
        self.assertEqual(trainer.state.optimizer_steps, 2)
        self.assertEqual(trainer.scheduler_interval, "step")

    def test_explicit_fp8_policy_replaces_inherited_conflict_before_validation(self):
        load_config(configuration({"fp8": {"enabled": True, "backend": "triton"}}))
        selected = FP8Config(enabled=True, backend="reference", grad_dtype="e5m2")
        partial = {"runtime": {"fp8": {"grad_dtype": "e5m2"}}}
        for kind in ("trainer", "runtime", "core"):
            with self.subTest(kind=kind):
                model = nn.Linear(2, 1)
                if kind == "trainer":
                    owner = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1),
                                    loss_fn=nn.functional.mse_loss, fp8=selected, config=partial).runtime
                elif kind == "runtime":
                    owner = TrainingRuntime(model, fp8=selected, config=partial)
                else:
                    owner = TrainingCore(model, "cpu", fp8=selected, config=partial)
                self.assertEqual(owner.fp8, selected)
                output = owner.model(torch.ones(2, 2))
                self.assertTrue(torch.isfinite(output).all())

    def test_explicit_fit_fields_override_invalid_per_call_values(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1), loss_fn=nn.functional.mse_loss)
        trainer.fit([(torch.ones(2, 2), torch.zeros(2, 1))], epochs=1,
                    monitor=None, monitor_mode="min", progress_bar=False,
                    config={"training": {"epochs": 0, "monitor": "samples", "monitor_mode": "invalid"}})
        self.assertEqual(trainer.state.completed_epochs, 1)
        self.assertIsNone(trainer.state.monitor)

    def test_fit_still_rejects_non_fit_configuration_changes(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=.1), loss_fn=nn.functional.mse_loss)
        with self.assertRaisesRegex(ValueError, "fit config may only"):
            trainer.fit([(torch.ones(2, 2), torch.zeros(2, 1))], epochs=1,
                        progress_bar=False, config={"runtime": {"checks": False}})


if __name__ == "__main__":
    unittest.main()
