from contextlib import redirect_stdout
from dataclasses import replace
from functools import partial
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from deepopt.config import config_from_dict
from deepopt.engine.checkpointing.checkpoint import read_checkpoint
from deepopt.baseline import build_baseline_experiment
from deepopt.optim import build_optimizer
from deepopt.run import train


class OptimizerFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self):
        return config_from_dict({
            "name": "custom-optimizer", "seed": 10,
            "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.01, "optimizer": "custom"},
            "application": {"synthetic": {"train_samples": 13, "validation_samples": 7}},
        })

    def test_apply_func_receives_model_parameters_and_settings_once(self):
        model = nn.Linear(3, 2)
        settings = self.config().training
        calls = []

        def apply(parameters, config):
            parameters = list(parameters)
            calls.append((parameters, config))
            return torch.optim.RMSprop(parameters, lr=config.learning_rate)

        optimizer = build_optimizer(model, settings, apply_func=apply)
        self.assertIsInstance(optimizer, torch.optim.RMSprop)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][1], settings)
        self.assertEqual([id(p) for p in calls[0][0]], [id(p) for p in model.parameters()])
        self.assertEqual(optimizer.param_groups[0]["lr"], settings.learning_rate)

    def test_none_uses_configured_builtin(self):
        model = nn.Linear(3, 2)
        for name, cls in [("adamw", torch.optim.AdamW), ("sgd", torch.optim.SGD)]:
            settings = replace(self.config().training, optimizer=name)
            self.assertIsInstance(build_optimizer(model, settings, apply_func=None), cls)

    def test_invalid_factory_result_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must return a torch.optim.Optimizer"):
            build_optimizer(nn.Linear(3, 2), self.config().training, apply_func=lambda parameters, config: None)

    def test_custom_requires_factory(self):
        with self.assertRaisesRegex(ValueError, "requires an apply_func"):
            build_optimizer(nn.Linear(3, 2), self.config().training)

    def test_apply_func_optimizer_trains_and_resumes_with_state(self):
        calls = []

        def apply(parameters, config):
            calls.append(True)
            return torch.optim.RMSprop(parameters, lr=config.learning_rate, momentum=0.5)

        experiment = partial(build_baseline_experiment, optimizer_factory=partial(build_optimizer, apply_func=apply))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            run = Path(directory) / "run"
            train(self.config(), output=run, experiment_factory=experiment)
            first = read_checkpoint(run / "last.pt")
            self.assertTrue(first["optimizer"]["state"])
            self.assertTrue(first["components"]["optimizer"].endswith("RMSprop"))
            train(resume=run / "last.pt", epochs=2, experiment_factory=experiment)
            last = read_checkpoint(run / "last.pt")
            self.assertEqual(last["state"]["completed_epochs"], 2)
            self.assertGreater(last["state"]["optimizer_steps"], first["state"]["optimizer_steps"])
            self.assertTrue(any(not torch.equal(first["model"][key], last["model"][key]) for key in first["model"]))
        self.assertEqual(len(calls), 2)  # One construction per invocation, never per batch.


if __name__ == "__main__":
    unittest.main()
