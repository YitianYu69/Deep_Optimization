from dataclasses import asdict
from functools import partial
import unittest

import torch
from torch import nn

from deepopt.config import ConfigError, config_from_dict
from deepopt.data.synthetic import SyntheticDataSettings, build_datasets
from deepopt.experiment import build_experiment
from deepopt.engine import Trainer
from deepopt.engine.strategies import MicrobatchMean
from deepopt.module import TrainingModule
from deepopt.tasks.classification import ClassificationTask


class ModelFactoryTests(unittest.TestCase):
    def config(self):
        return config_from_dict({"name": "user-model", "seed": 2,
                                 "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.01},
                                 "application": {"synthetic": {"train_samples": 9, "validation_samples": 5}}})

    def dataset_factory(self):
        config = self.config()
        settings = SyntheticDataSettings.from_dict(config.application["synthetic"])
        return partial(build_datasets, settings, config.seed)

    def test_core_config_has_no_model_schema(self):
        config = self.config()
        self.assertNotIn("model", asdict(config))
        with self.assertRaisesRegex(ConfigError, "unknown fields: model"):
            config_from_dict({**asdict(config), "model": {"architecture": "user-defined"}})
        with self.assertRaisesRegex(TypeError, "model_factory"):
            build_experiment(config)
        with self.assertRaisesRegex(TypeError, "dataset_factory"):
            build_experiment(config, model_factory=lambda: nn.Linear(16, 4))

    def test_zero_argument_factory_owns_all_architecture_settings(self):
        calls = []
        user_settings = {"widths": [11, 7], "activation": nn.SiLU}

        def make_model():
            calls.append(True)
            first, last = user_settings["widths"]
            return nn.Sequential(nn.Linear(16, first), user_settings["activation"](),
                                 nn.Linear(first, last), nn.Linear(last, 4))

        trainer, train, validation, _ = build_experiment(
            self.config(), model_factory=make_model, dataset_factory=self.dataset_factory(),
            task_factory=partial(ClassificationTask, num_classes=4))
        trainer.fit(train, validation, epochs=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(trainer.state.completed_epochs, 1)
        self.assertEqual(trainer.model[0].out_features, 11)

    def test_module_factory_keeps_task_inside_user_wrapper(self):
        def factory():
            return TrainingModule(nn.Linear(16, 4), task=ClassificationTask(num_classes=4))
        trainer, train, validation, _ = build_experiment(
            self.config(), model_factory=factory, dataset_factory=self.dataset_factory())
        trainer.fit(train, validation, epochs=1)
        self.assertIsNone(trainer.task)
        self.assertIn("accuracy", trainer.evaluate(validation))

    def test_trainer_factory_supplies_fresh_strategies_without_changing_core_config(self):
        def make_trainer(model, optimizer, **kwargs):
            return Trainer(model, optimizer, loss_strategy=MicrobatchMean(), **kwargs)

        factory = partial(build_experiment, model_factory=lambda: nn.Linear(16, 4),
                          dataset_factory=self.dataset_factory(),
                          task_factory=ClassificationTask, trainer_factory=make_trainer)
        first, train, validation, _ = factory(self.config())
        second, _, _, _ = factory(self.config())
        self.assertIsInstance(first.loss_strategy, MicrobatchMean)
        self.assertIsNot(first.loss_strategy, second.loss_strategy)
        first.fit(train, validation, epochs=1)
        self.assertEqual(first.state.completed_epochs, 1)


if __name__ == "__main__":
    unittest.main()
