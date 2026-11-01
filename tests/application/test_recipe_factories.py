from dataclasses import asdict
from functools import partial
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchmetrics import MeanAbsoluteError

from deepopt import LossReport, LossResult, Trainer
from deepopt.config import ConfigError, config_from_dict
from deepopt.engine.checkpointing.checkpoint import read_checkpoint
from deepopt.experiment import build_experiment
from deepopt.run import train


class RecipeFactoryTests(unittest.TestCase):
    def config(self, **training):
        return config_from_dict({
            "name": "functional-recipe", "seed": 8,
            "training": {"epochs": 1, "batch_size": 2, "learning_rate": .01,
                         "optimizer": "sgd", "checkpoint_every_steps": 0, **training},
        })

    @staticmethod
    def data():
        inputs = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10
        dataset = TensorDataset(inputs, inputs.sum(dim=1, keepdim=True))
        return dataset, dataset

    def test_loss_recipe_needs_no_task_or_special_trainer_factory(self):
        trainer, loader, validation, _ = build_experiment(
            self.config(), model_factory=lambda: nn.Linear(2, 1), dataset_factory=self.data,
            loss_fn=nn.functional.mse_loss, metrics={"mae": MeanAbsoluteError()},
        )
        trainer.fit(loader, validation, epochs=1)
        self.assertEqual(trainer.state.optimizer_steps, 3)
        self.assertIn("mae", trainer.evaluate(validation))
        self.assertIsNone(trainer.task)

    def test_custom_step_and_validation_and_metric_routes_are_forwarded(self):
        validation_calls = []

        def step(model, batch):
            inputs, targets = batch
            predictions = model(inputs)
            return {"loss": (predictions - targets).square().mean(), "prediction": predictions}

        def validate(model, batch):
            validation_calls.append(True)
            return step(model, batch)

        trainer, loader, validation, _ = build_experiment(
            self.config(), model_factory=lambda: nn.Linear(2, 1), dataset_factory=self.data,
            step_fn=step, validation_step_fn=validate, metrics={"mae": MeanAbsoluteError()},
            metric_inputs=lambda result, batch: (result["prediction"], batch[1]),
        )
        trainer.fit(loader, validation, epochs=1)
        self.assertEqual(len(validation_calls), 3)

    def test_omitted_options_preserve_existing_partial_trainer_factories(self):
        trainer, loader, _, _ = build_experiment(
            self.config(), model_factory=lambda: nn.Linear(2, 1), dataset_factory=self.data,
            trainer_factory=partial(Trainer, loss_fn=nn.functional.mse_loss),
        )
        trainer.train_epoch(loader)
        self.assertEqual(trainer.state.optimizer_steps, 3)

    def test_default_settings_do_not_expand_existing_trainer_factory_keywords(self):
        def make_trainer(model, optimizer, *, device, precision, checks, accumulation_steps,
                         grad_clip_norm, task, scheduler, metric_factory):
            return Trainer(model, optimizer, device=device, precision=precision, checks=checks,
                           accumulation_steps=accumulation_steps, grad_clip_norm=grad_clip_norm,
                           task=task, scheduler=scheduler, metric_factory=metric_factory,
                           loss_fn=nn.functional.mse_loss)

        trainer, loader, _, _ = build_experiment(
            self.config(), model_factory=lambda: nn.Linear(2, 1), dataset_factory=self.data,
            trainer_factory=make_trainer,
        )
        trainer.train_epoch(loader)
        self.assertEqual(trainer.state.optimizer_steps, 3)

    def test_functional_configured_run_saves_and_resumes_without_task_wrappers(self):
        factory = partial(build_experiment, model_factory=lambda: nn.Linear(2, 1),
                          dataset_factory=self.data, loss_fn=nn.functional.mse_loss)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            train(self.config(), output=run_dir, experiment_factory=factory)
            train(resume=run_dir / "last.pt", epochs=2, experiment_factory=factory)
            saved = read_checkpoint(run_dir / "last.pt")
        self.assertEqual(saved["state"]["completed_epochs"], 2)
        self.assertEqual(saved["state"]["optimizer_steps"], 6)
        self.assertEqual(len(saved["history"]), 2)

    def test_scheduler_interval_validation_and_factory_forwarding(self):
        self.assertEqual(self.config().training.scheduler_interval, "epoch")
        for interval, steps in (("epoch", 1), ("step", 3)):
            with self.subTest(interval=interval):
                config = self.config(scheduler_interval=interval)
                self.assertEqual(config_from_dict(asdict(config)), config)
                trainer, loader, _, _ = build_experiment(
                    config, model_factory=lambda: nn.Linear(2, 1), dataset_factory=self.data,
                    loss_fn=nn.functional.mse_loss,
                )
                trainer.train_epoch(loader)
                self.assertEqual(trainer.scheduler.last_epoch, steps)
        for invalid in (None, 1, True, "batch", []):
            with self.subTest(interval=invalid), self.assertRaisesRegex(ConfigError, "scheduler_interval"):
                self.config(scheduler_interval=invalid)

    def test_advanced_loss_types_are_available_without_internal_imports(self):
        value = torch.tensor(2., requires_grad=True)
        result = LossResult.from_mean(value, weight=3)
        self.assertIs(result.value, value)
        self.assertIsInstance(result.report, LossReport)


if __name__ == "__main__":
    unittest.main()
