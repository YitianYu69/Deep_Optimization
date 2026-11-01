from contextlib import redirect_stdout
import copy
from functools import partial
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from deepopt.config import config_from_dict
from deepopt.engine import Trainer
from deepopt.engine.execution.precision import PrecisionPolicy
from deepopt.baseline import build_baseline_experiment
from deepopt.optimizers import MultiOptimizer
from deepopt.run import train
from deepopt.engine.checkpointing.checkpoint import read_checkpoint
from deepopt.tasks.classification import ClassificationTask


def apply_optimizer(model, settings):
    return MultiOptimizer({
        "first_layer": torch.optim.AdamW(model.layers[0].parameters(), lr=settings.learning_rate),
        "last_layer": torch.optim.SGD(model.layers[3].parameters(), lr=settings.learning_rate * 3, momentum=0.9),
    })


class MultiOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, epochs=3):
        return config_from_dict({
            "name": "multi", "seed": 17,
            "training": {"epochs": epochs, "batch_size": 4, "learning_rate": 0.01,
                         "optimizer": "custom", "lr_gamma": 0.8, "accumulation_steps": 2},
            "application": {"synthetic": {"train_samples": 19, "validation_samples": 7}},
        })

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_wrapper_matches_independent_optimizers_and_routes_layers(self):
        torch.manual_seed(5)
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        reference = copy.deepcopy(model)
        wrapped = MultiOptimizer({
            "first": torch.optim.AdamW(model[0].parameters(), lr=0.01),
            "last": torch.optim.SGD(model[2].parameters(), lr=0.1, momentum=0.5),
        })
        independent = [torch.optim.AdamW(reference[0].parameters(), lr=0.01),
                       torch.optim.SGD(reference[2].parameters(), lr=0.1, momentum=0.5)]
        inputs, targets = torch.randn(11, 3), torch.randint(0, 2, (11,))
        for _ in range(3):
            wrapped.zero_grad()
            for child in independent:
                child.zero_grad()
            nn.functional.cross_entropy(model(inputs), targets).backward()
            nn.functional.cross_entropy(reference(inputs), targets).backward()
            wrapped.step()
            for child in independent:
                child.step()
            self.assert_nested_equal(model.state_dict(), reference.state_dict())
        for name, child in zip(wrapped.optimizers, independent):
            self.assert_nested_equal(wrapped.optimizers[name].state_dict(), child.state_dict())

    def test_overlap_and_missing_model_parameters_are_rejected(self):
        model = nn.Linear(3, 2)
        with self.assertRaisesRegex(ValueError, "share or repeat"):
            MultiOptimizer({"a": torch.optim.SGD(model.parameters(), lr=0.1),
                            "b": torch.optim.AdamW(model.parameters())})
        wrapper = MultiOptimizer({"only_weight": torch.optim.SGD([model.weight], lr=0.1)})
        with self.assertRaisesRegex(ValueError, "Every trainable"):
            Trainer(model, wrapper, task=ClassificationTask())

    def test_scheduler_reaches_children_before_and_after_restore(self):
        trainer, loader, _, _ = build_baseline_experiment(self.config(), optimizer_factory=apply_optimizer)
        trainer.train_epoch(loader)
        optimizer = trainer.optimizer
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [0.008, 0.024])
        saved = copy.deepcopy(optimizer.state_dict())
        optimizer.load_state_dict(saved)
        self.assertIs(optimizer.param_groups[0], optimizer.optimizers["first_layer"].param_groups[0])
        trainer.train_epoch(loader)
        self.assertAlmostEqual(optimizer.optimizers["last_layer"].param_groups[0]["lr"], 0.0192)

    def test_checkpoint_resume_matches_uninterrupted_for_both_children(self):
        factory = partial(build_baseline_experiment, optimizer_factory=apply_optimizer)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            full, resumed = Path(directory) / "full", Path(directory) / "resumed"
            train(self.config(3), output=full, experiment_factory=factory)
            train(self.config(1), output=resumed, experiment_factory=factory)
            train(resume=resumed / "last.pt", epochs=3, experiment_factory=factory)
            a, b = read_checkpoint(full / "last.pt"), read_checkpoint(resumed / "last.pt")
            for key in ("model", "optimizer", "scheduler", "state", "rng", "generators"):
                self.assert_nested_equal(a[key], b[key])
            for child in b["optimizer"]["optimizers"].values():
                self.assertTrue(child["state_dict"]["state"])
            self.assertEqual(len(b["history"][-1]["learning_rates"]), 2)

    def test_nonfinite_gradient_in_one_child_aborts_all_updates(self):
        trainer, loader, _, _ = build_baseline_experiment(self.config(), optimizer_factory=apply_optimizer)
        model_before = copy.deepcopy(trainer.model.state_dict())
        scheduler_before = copy.deepcopy(trainer.scheduler.state_dict())
        handle = trainer.model.layers[3].weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        self.addCleanup(handle.remove)
        with self.assertRaisesRegex(FloatingPointError, "Non-finite gradient"):
            trainer.train_epoch(loader)
        self.assert_nested_equal(trainer.model.state_dict(), model_before)
        self.assert_nested_equal(trainer.scheduler.state_dict(), scheduler_before)
        self.assertEqual(trainer.state.optimizer_steps, 0)
        self.assertTrue(all(not child.state for child in trainer.optimizer.optimizers.values()))

    def test_restore_rejects_changed_child_type_and_partition(self):
        trainer, _, _, _ = build_baseline_experiment(self.config(), optimizer_factory=apply_optimizer)
        state = copy.deepcopy(trainer.optimizer.state_dict())
        state["optimizers"]["last_layer"]["type"] = "torch.optim.AdamW"
        with self.assertRaisesRegex(ValueError, "type"):
            trainer.optimizer.load_state_dict(state)
        state = copy.deepcopy(trainer.optimizer.state_dict())
        state["optimizers"]["last_layer"]["layout"][0][0]["shape"] = [999]
        with self.assertRaisesRegex(ValueError, "layout"):
            trainer.optimizer.load_state_dict(state)

    def test_group_changes_require_rebuilding_wrapper(self):
        trainer, _, _, _ = build_baseline_experiment(self.config(), optimizer_factory=apply_optimizer)
        child = trainer.optimizer.optimizers["last_layer"]
        child.add_param_group({"params": [nn.Parameter(torch.ones(2))]})
        with self.assertRaisesRegex(ValueError, "groups changed"):
            trainer.optimizer.step()

    def test_fp16_is_rejected_before_any_cuda_access(self):
        with self.assertRaisesRegex(ValueError, "fp16 is unsupported"):
            PrecisionPolicy(torch.device("cuda"), "fp16")
        config = self.config()
        from dataclasses import asdict
        raw = asdict(config)
        raw["runtime"] = {"device": "cuda", "precision": "fp16"}
        with self.assertRaisesRegex(ValueError, "fp16 is unsupported"):
            config_from_dict(raw)


if __name__ == "__main__":
    unittest.main()
