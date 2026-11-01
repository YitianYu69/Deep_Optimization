from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from deepopt.baseline import build_baseline_experiment
from deepopt.config import config_from_dict
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.experiment import build_experiment
from deepopt.tasks.regression import RegressionTask


class UncountedStream(IterableDataset):
    """An epoch can end naturally without exposing its length to the core."""

    def __init__(self, stop):
        self.stop = stop

    def __len__(self):
        raise AssertionError("Training must not ask this stream for its length")

    def __iter__(self):
        for index in range(self.stop):
            yield torch.tensor([index / 10, 1.0]), torch.tensor([index / 10 + 1.0])


class DataFactoryTests(unittest.TestCase):
    def config(self, **options):
        return config_from_dict({"name": "user-data", "seed": 8,
                                 "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.01,
                                              "checkpoint_every_steps": 0}, **options})

    def test_actual_datasets_define_sizes_and_features_without_data_config(self):
        generator = torch.Generator().manual_seed(4)
        train_x = torch.randn(11, 7, generator=generator)
        valid_x = torch.randn(5, 7, generator=generator)
        datasets = (TensorDataset(train_x, train_x.sum(1, keepdim=True)),
                    TensorDataset(valid_x, valid_x.sum(1, keepdim=True)))
        calls = []

        def make_datasets():
            calls.append(True)
            return datasets

        config = self.config()
        self.assertNotIn("data", asdict(config))
        self.assertEqual(config.application, {})
        trainer, train, validation, _ = build_experiment(
            config, model_factory=lambda: nn.Linear(7, 1), dataset_factory=make_datasets,
            task_factory=RegressionTask)
        rows = []
        trainer.fit(train, validation, epochs=1, callbacks=[lambda _, row: rows.append(row)])
        self.assertEqual(calls, [True])
        self.assertEqual(rows[0]["train"]["samples"], 11)
        self.assertEqual(rows[0]["validation"]["samples"], 5)
        self.assertEqual(trainer.state.optimizer_steps, 3)
        self.assertNotIn("accuracy", rows[0]["validation"])

    def test_finite_streams_need_no_length_or_prescribed_sample_count(self):
        def make_loaders(train, validation, *, batch_size, seed):
            return (DataLoader(train, batch_size=batch_size),
                    DataLoader(validation, batch_size=batch_size), {})

        trainer, train, validation, _ = build_experiment(
            self.config(), model_factory=lambda: nn.Linear(2, 1),
            dataset_factory=lambda: (UncountedStream(9), UncountedStream(3)),
            loader_factory=make_loaders, task_factory=RegressionTask)
        rows = []
        trainer.fit(train, validation, epochs=2, callbacks=[lambda _, row: rows.append(row)])
        self.assertEqual([row["train"]["samples"] for row in rows], [9, 9])
        self.assertEqual([row["validation"]["samples"] for row in rows], [3, 3])
        self.assertEqual(trainer.state.optimizer_steps, 6)

    def test_application_data_identity_is_saved_and_checked_on_resume(self):
        config = self.config(application={"dataset": {"revision": "v1", "split": "fold-a"}})
        dataset = TensorDataset(torch.ones(3, 2), torch.zeros(3, 1))
        trainer, _, _, generators = build_experiment(
            config, model_factory=lambda: nn.Linear(2, 1), dataset_factory=lambda: (dataset, dataset),
            task_factory=RegressionTask)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, trainer, config, generators=generators, history=[])
            payload = read_checkpoint(path)
        self.assertEqual(payload["config"]["application"], config.application)
        changed = replace(config, application={"dataset": {"revision": "v2", "split": "fold-a"}})
        with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
            restore_checkpoint(payload, trainer, changed, generators=generators)

    def test_synthetic_constraints_belong_to_demo_recipe_only(self):
        # The generic parser doesn't know whether a user's 'num_classes' value
        # is a class count, an identifier, or a field they interpret differently.
        arbitrary = self.config(application={"my_task": {"num_classes": 1, "input_dim": "variable"}})
        self.assertEqual(arbitrary.application["my_task"]["input_dim"], "variable")
        for options in ({"num_classes": 1}, {"train_samples": 0}, {"input_dim": True}, {"typo": 3}):
            with self.subTest(options=options):
                config = self.config(application={"synthetic": options})
                with self.assertRaises(ValueError):
                    build_baseline_experiment(config)


if __name__ == "__main__":
    unittest.main()
