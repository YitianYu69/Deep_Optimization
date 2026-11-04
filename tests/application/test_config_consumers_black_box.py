"""Public factories produce equivalent results from typed and dictionary settings."""

from copy import deepcopy
from dataclasses import asdict
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from deepopt.config import CompileConfig, TrainingConfig
from deepopt.data.synthetic import SyntheticDataSettings, build_datasets
from deepopt.engine.execution.compilation import compile_model
from deepopt.optimizer_factory import build_optimizer
from deepopt.scheduling.lr import build_scheduler


class ConfigConsumerBlackBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(821)

    def test_dictionary_optimizers_match_typed_training_updates(self):
        for kind in ("sgd", "adamw"):
            with self.subTest(optimizer=kind):
                typed = TrainingConfig(epochs=2, batch_size=3, learning_rate=.02,
                                       optimizer=kind, momentum=.8, weight_decay=.01)
                raw = asdict(typed)
                saved = deepcopy(raw)
                expected_model = nn.Linear(4, 2)
                actual_model = deepcopy(expected_model)
                expected_optimizer = build_optimizer(expected_model, typed)
                actual_optimizer = build_optimizer(actual_model, raw)
                self.assertIs(type(actual_optimizer), type(expected_optimizer))
                batches = [(torch.randn(3, 4), torch.randn(3, 2)) for _ in range(3)]
                for inputs, targets in batches:
                    for model, optimizer in ((actual_model, actual_optimizer),
                                             (expected_model, expected_optimizer)):
                        optimizer.zero_grad(set_to_none=True)
                        F.mse_loss(model(inputs), targets).backward()
                        optimizer.step()
                for actual, expected in zip(actual_model.parameters(), expected_model.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(raw, saved)

    def test_dictionary_schedules_match_typed_learning_rate_sequences(self):
        policies = (
            None,
            {"type": "constant"},
            {"type": "step_decay", "step_size": 2, "gamma": .5},
            {"type": "exponential_decay", "gamma": .7},
            {"type": "linear_warmup", "warmup_steps": 3, "start_factor": .1},
            {"type": "cosine_decay", "total_steps": 6, "eta_min": .01},
            {"type": "warmup_cosine", "warmup_steps": 3, "total_steps": 6, "start_factor": .1},
            {"type": "piecewise", "milestones": [2, 4], "gamma": .5},
        )

        def sequence(config):
            optimizer = build_optimizer(nn.Linear(1, 1), config)
            scheduler = build_scheduler(optimizer, config)
            rates = []
            for _ in range(6):
                rates.append(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
            return type(scheduler), rates, scheduler.state_dict()

        for policy in policies:
            with self.subTest(policy=policy):
                typed = TrainingConfig(epochs=2, batch_size=3, learning_rate=.2,
                                       optimizer="sgd", momentum=0., lr_schedule=policy,
                                       lr_gamma=.8 if policy is None else 1.)
                raw = asdict(typed)
                saved = deepcopy(raw)
                self.assertEqual(sequence(raw), sequence(typed))
                self.assertEqual(raw, saved)

    def test_synthetic_dictionary_matches_typed_data_without_consuming_global_rng(self):
        typed = SyntheticDataSettings(train_samples=9, validation_samples=5, input_dim=3, num_classes=2)
        raw = asdict(typed)
        saved = deepcopy(raw)
        global_rng = torch.get_rng_state().clone()
        actual = build_datasets(raw, seed=13)
        expected = build_datasets(typed, seed=13)
        self.assertEqual([len(dataset) for dataset in actual], [9, 5])
        for actual_split, expected_split in zip(actual, expected):
            for actual_tensor, expected_tensor in zip(actual_split.tensors, expected_split.tensors):
                torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
        self.assertEqual(raw, saved)
        different = build_datasets(raw, seed=14)
        self.assertFalse(torch.equal(actual[0].tensors[0], different[0].tensors[0]))

    def test_dictionary_compilation_matches_typed_forward_and_backward(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        typed = CompileConfig(enabled=True, backend="eager", fullgraph=True)
        raw = asdict(typed)
        saved = deepcopy(raw)
        expected_model = nn.Linear(4, 2)
        actual_model = deepcopy(expected_model)
        self.assertIs(compile_model(actual_model, raw), actual_model)
        self.assertIs(compile_model(expected_model, typed), expected_model)
        actual_x = torch.randn(3, 4, requires_grad=True)
        expected_x = actual_x.detach().clone().requires_grad_()
        actual, expected = actual_model(actual_x), expected_model(expected_x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(actual_x.grad, expected_x.grad, rtol=0, atol=0)
        for actual_parameter, expected_parameter in zip(actual_model.parameters(), expected_model.parameters()):
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)
        self.assertEqual(raw, saved)


if __name__ == "__main__":
    unittest.main()
