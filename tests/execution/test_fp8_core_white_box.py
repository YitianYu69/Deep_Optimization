"""Automatic preparation ownership, policy visibility, and factory ordering."""

from unittest.mock import patch
import unittest

import torch
from torch import fx, nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from deepopt import Trainer, TrainingCore, TrainingRuntime
from deepopt.config import FP8Config, config_from_dict
from deepopt.experiment import build_experiment
from deepopt.opt.nn.linear import DOLinear


class AutomaticFP8CoreWhiteBoxTests(unittest.TestCase):
    def test_runtime_and_regular_core_expose_normalized_policy_and_report(self):
        model = nn.Sequential(nn.Linear(7, 3))
        runtime = TrainingRuntime(model, fp8={"enabled": True})
        self.assertIs(runtime.model, model)
        self.assertIs(runtime.core.model, model)
        self.assertIsInstance(runtime.fp8, FP8Config)
        self.assertTrue(runtime.fp8.enabled)
        self.assertEqual(runtime.fp8.trace_mode, "modules")
        self.assertEqual(runtime.core.fp8, runtime.fp8)
        self.assertIsNotNone(runtime.transformation_report)
        self.assertEqual(runtime.transformation_report, runtime.core.transformation_report)
        self.assertIsInstance(model[0], DOLinear)
        parameter = model[0].weight
        converted = model[0]
        second_core = TrainingCore(model, torch.device("cpu"), fp8=True)
        self.assertIs(second_core.model[0], converted)
        self.assertIs(second_core.model[0].weight, parameter)

    def test_disabled_policy_leaves_model_and_report_unchanged(self):
        model = nn.Linear(7, 3)
        runtime = TrainingRuntime(model, fp8=False)
        self.assertIs(runtime.model, model)
        self.assertIs(type(runtime.model), nn.Linear)
        self.assertFalse(runtime.fp8.enabled)
        self.assertIsNone(runtime.transformation_report)

    def test_reusing_prepared_model_retains_policy_and_rejects_disabled_metadata(self):
        runtime = TrainingRuntime(nn.Sequential(nn.Linear(7, 3)), fp8=True)
        model = runtime.model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        rebuilt = Trainer(model, optimizer, loss_fn=F.mse_loss)
        self.assertTrue(rebuilt.runtime.fp8.enabled)
        self.assertIs(rebuilt.runtime.transformation_report, runtime.transformation_report)
        from deepopt.engine.checkpointing.checkpoint import _trainer_settings
        self.assertTrue(_trainer_settings(rebuilt)["fp8"]["enabled"])
        for disabled in (False, {"enabled": False}, FP8Config()):
            with self.subTest(fp8=disabled), self.assertRaisesRegex(ValueError, "FP8 settings differ"):
                TrainingRuntime(model, optimizer, fp8=disabled)

    def test_explicit_full_mode_exposes_the_returned_graph_root(self):
        model = nn.Sequential(nn.Linear(7, 3))
        parameters = tuple(model.parameters())
        runtime = TrainingRuntime(model, fp8={"enabled": True, "trace_mode": "full"})
        self.assertIsInstance(runtime.model, fx.GraphModule)
        self.assertIs(runtime.model, runtime.core.model)
        self.assertTrue(all(a is b for a, b in zip(parameters, runtime.model.parameters())))
        self.assertEqual(runtime.transformation_report.trace_mode, "full")

    def test_configured_factory_transforms_before_compile_and_optimizer(self):
        model = nn.Sequential(nn.Linear(7, 3))
        parameter = model[0].weight
        events = []
        dataset = TensorDataset(torch.ones(4, 7), torch.zeros(4, 3))
        config = config_from_dict({"name": "automatic-fp8", "seed": 17,
            "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.01},
            "runtime": {"fp8": {"enabled": True},
                        "compile": {"enabled": True, "backend": "eager"}}})

        def compiler(function, **options):
            self.assertIsInstance(model[0], DOLinear)
            self.assertIs(model[0].weight, parameter)
            events.append("compile")
            return function

        def make_optimizer(actual, settings):
            self.assertIsInstance(actual[0], DOLinear)
            self.assertIs(actual[0].weight, parameter)
            events.append("optimizer")
            return torch.optim.SGD(actual.parameters(), lr=0.01)

        def make_trainer(actual, optimizer, **options):
            events.append("trainer")
            return Trainer(actual, optimizer, **options)

        with patch("torch.compile", side_effect=compiler):
            trainer, _, _, _ = build_experiment(
                config, model_factory=lambda: model, dataset_factory=lambda: (dataset, None),
                optimizer_factory=make_optimizer, scheduler_factory=lambda *args: None,
                trainer_factory=make_trainer, loss_fn=F.mse_loss,
            )
        self.assertEqual(events, ["compile", "optimizer", "trainer"])
        self.assertTrue(trainer.runtime.fp8.enabled)
        self.assertIs(trainer.model[0].weight, parameter)


if __name__ == "__main__":
    unittest.main()
