from dataclasses import replace
import unittest

import torch
from torch import nn

from deepopt import Trainer
from deepopt.config import TrainingConfig, ConfigError, config_from_dict
from deepopt.optimizer_factory import build_optimizer
from deepopt.scheduling.lr import build_scheduler


class LRPolicyTests(unittest.TestCase):
    def applied(self, policy, steps=5):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.)
        scheduler = build_scheduler(optimizer, TrainingConfig(1, 1, 1., lr_schedule=policy))
        rates = []
        for _ in range(steps):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        return rates

    def test_native_policy_sequences_and_warmup_endpoints(self):
        cases = [({"type": "constant"}, [1.] * 5),
                 ({"type": "step_decay", "step_size": 2, "gamma": .5}, [1., 1., .5, .5, .25]),
                 ({"type": "exponential_decay", "gamma": .5}, [1., .5, .25, .125, .0625]),
                 ({"type": "linear_warmup", "warmup_steps": 3, "start_factor": .1}, [.1, .55, 1., 1., 1.]),
                 ({"type": "linear_warmup", "warmup_steps": 1, "start_factor": .1}, [1.] * 5),
                 ({"type": "cosine_decay", "total_steps": 5}, [1., .8535533905932737, .5, .14644660940672627, 0.]),
                 ({"type": "warmup_cosine", "warmup_steps": 3, "total_steps": 5, "start_factor": .1},
                  [.1, .55, 1., .5, 0.]),
                 ({"type": "piecewise", "milestones": [2, 4], "gamma": .5}, [1., 1., .5, .5, .25])]
        for policy, expected in cases:
            with self.subTest(policy=policy):
                for actual, target in zip(self.applied(policy), expected):
                    self.assertAlmostEqual(actual, target)

    def test_existing_imports_and_legacy_default_are_preserved(self):
        from deepopt.optim import build_optimizer as old_optimizer, build_scheduler as old_scheduler
        self.assertIs(old_optimizer, build_optimizer)
        self.assertIs(old_scheduler, build_scheduler)
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.)
        scheduler = build_scheduler(optimizer, TrainingConfig(1, 1, 1., lr_gamma=.5))
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)

    def test_lr_timing_is_training_only_for_step_and_epoch(self):
        for interval, expected in (("step", .25), ("epoch", .5)):
            model = nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=1.)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.5)
            trainer = Trainer(model, optimizer, scheduler=scheduler, scheduler_interval=interval,
                              loss_fn=nn.functional.mse_loss, accumulation_steps=2)
            batches = [(torch.ones(1, 1), torch.zeros(1, 1))] * 3
            trainer.train_epoch(batches)
            self.assertEqual(optimizer.param_groups[0]["lr"], expected)
            previous = scheduler.state_dict()
            trainer.evaluate(batches)
            self.assertEqual(scheduler.state_dict(), previous)

    def test_invalid_or_ambiguous_lr_configuration(self):
        raw = {"name": "lr", "seed": 1, "training": {"epochs": 1, "batch_size": 1, "learning_rate": 1.}}
        for policy in ({"type": "plateau"}, {"type": "linear_warmup", "warmup_steps": 0},
                       {"type": "cosine_decay"}, {"type": "piecewise", "milestones": [2, 2]},
                       {"type": "warmup_cosine", "warmup_steps": 2, "total_steps": 2},
                       {"type": "constant", "bindings": {"lr": "x"}}):
            with self.subTest(policy=policy), self.assertRaises(ConfigError):
                config_from_dict({**raw, "training": {**raw["training"], "lr_schedule": policy}})
        with self.assertRaisesRegex(ConfigError, "Choose"):
            config_from_dict({**raw, "training": {**raw["training"], "lr_gamma": .9,
                                                  "lr_schedule": {"type": "constant"}}})


if __name__ == "__main__":
    unittest.main()
