"""Configured compiler training, memory analysis and checkpoint reconstruction."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from deepopt.compiler import compilation_report
from deepopt.config import config_from_dict
from deepopt.engine.checkpointing.checkpoint import read_checkpoint
from deepopt.experiment import build_experiment
from deepopt.memory import estimate_training_memory
from deepopt.run import train


def configuration(*, compiled=True, fp8=False, epochs=1):
    return config_from_dict(
        {
            "name": "compiler",
            "seed": 19,
            "training": {
                "epochs": epochs,
                "batch_size": 2,
                "learning_rate": 0.01,
                "optimizer": "sgd",
                "accumulation_steps": 2,
            },
            "runtime": {
                "execution_mode": "max_speed",
                "fp8": {"enabled": fp8},
                "compile": {
                    "enabled": compiled,
                    "backend": "deepopt",
                    "dynamic": False,
                },
            },
        }
    )


def recipe(config):
    x = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
    dataset = TensorDataset(x, x.sum(dim=1, keepdim=True))
    return build_experiment(
        config,
        model_factory=lambda: nn.Sequential(
            nn.Linear(4, 8), nn.Sigmoid(), nn.Linear(8, 1)
        ),
        dataset_factory=lambda: (dataset, dataset),
        loss_fn=nn.functional.mse_loss,
    )


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_training_accumulation_partial_batch_and_fake_estimation(self):
        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                torch.compiler.reset()
                compiled, loader, _, _ = recipe(configuration(fp8=fp8))
                eager, reference_loader, _, _ = recipe(
                    configuration(compiled=False, fp8=fp8)
                )
                before = deepcopy(compiled.model.state_dict())
                # Sampling through the shuffled loader advances its generator.
                # Use the fixed dataset batch so both training orders match.
                sample = tuple(tensor[:2] for tensor in loader.dataset.tensors)
                estimate = estimate_training_memory(compiled, sample)
                self.assertGreater(estimate.peak_bytes_by_device["cpu"], 0)
                self.assertEqual(compilation_report(compiled.model).captured_graphs, 0)
                for name, value in compiled.model.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)
                compiled.train_epoch(loader)
                eager.train_epoch(reference_loader)
                self.assertEqual(compiled.state.optimizer_steps, 2)
                for name, value in compiled.model.state_dict().items():
                    torch.testing.assert_close(
                        value, eager.model.state_dict()[name], rtol=0, atol=0
                    )
                self.assertGreater(
                    compilation_report(compiled.model).captured_graphs, 0
                )

    def test_checkpoint_resume_reconstructs_compilation_and_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = configuration(fp8=True)
            train(
                config,
                output=root / "resumed",
                experiment_factory=recipe,
                batch_size_mode="manual",
                progress_bar=False,
            )
            torch.compiler.reset()
            train(
                resume=root / "resumed/last.pt",
                epochs=2,
                experiment_factory=recipe,
                batch_size_mode="manual",
                progress_bar=False,
            )
            full = asdict(config)
            full["training"]["epochs"] = 2
            torch.compiler.reset()
            train(
                config_from_dict(full),
                output=root / "full",
                experiment_factory=recipe,
                batch_size_mode="manual",
                progress_bar=False,
            )
            actual = read_checkpoint(root / "resumed/last.pt")
            expected = read_checkpoint(root / "full/last.pt")
            self.assertEqual(
                actual["config"]["runtime"]["compile"]["backend"], "deepopt"
            )
            for name, value in actual["model"].items():
                torch.testing.assert_close(
                    value, expected["model"][name], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
