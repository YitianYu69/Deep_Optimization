from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from deepopt.baseline import build_baseline_experiment
from deepopt.config import ConfigError, config_from_dict, _read_config as load_config, reset_config
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.experiment import build_experiment
from deepopt.tasks.regression import RegressionTask


def settings(checks=True):
    return config_from_dict({
        "name": "observer-config", "seed": 17,
        "training": {"epochs": 1, "batch_size": 3, "learning_rate": 0.01},
        "runtime": {"checks": checks},
        "application": {"synthetic": {"train_samples": 8, "validation_samples": 4,
                                      "input_dim": 2, "num_classes": 2}},
    })


class ObserverConfigTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_checks_default_on_and_roundtrip_as_a_boolean(self):
        raw = asdict(settings())
        raw.pop("runtime")
        self.assertIs(config_from_dict(raw).runtime.checks, True)
        for enabled in (False, True):
            with self.subTest(checks=enabled):
                config = settings(enabled)
                self.assertIs(config.runtime.checks, enabled)
                self.assertEqual(config_from_dict(asdict(config)), config)

    def test_toml_can_disable_runtime_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('name = "checks"\nseed = 1\n[training]\nepochs = 1\n'
                            'batch_size = 2\nlearning_rate = 0.01\n[runtime]\nchecks = false\n')
            self.assertIs(load_config(path).runtime.checks, False)

    def test_checks_reject_nonboolean_configuration_values(self):
        for value in (0, 1, "false", None, [], {}, 0.5):
            with self.subTest(value=value):
                raw = asdict(settings())
                raw["runtime"]["checks"] = value
                with self.assertRaisesRegex(ConfigError, "runtime.checks must be a boolean"):
                    config_from_dict(raw)

    def test_experiment_forwards_checks_to_the_trainer_factory(self):
        dataset = TensorDataset(torch.ones(2, 2), torch.zeros(2, 1))
        for enabled in (True, False):
            with self.subTest(checks=enabled):
                captured = {}
                marker = object()

                def trainer_factory(model, optimizer, **options):
                    captured.update(options)
                    return marker

                trainer, _, _, _ = build_experiment(
                    settings(enabled), model_factory=lambda: nn.Linear(2, 1),
                    dataset_factory=lambda: (dataset, dataset), task_factory=RegressionTask,
                    trainer_factory=trainer_factory,
                )
                self.assertIs(trainer, marker)
                self.assertIs(captured["checks"], enabled)

    def test_resume_accepts_changed_diagnostic_toggle_and_continues_training(self):
        for enabled in (True, False):
            with self.subTest(original_checks=enabled), tempfile.TemporaryDirectory() as directory:
                original_config = settings(enabled)
                trainer, loader, validation, generators = build_baseline_experiment(original_config)
                history = []
                trainer.fit(loader, validation, epochs=1,
                            callbacks=[lambda current, row: history.append(row)])
                path = Path(directory) / "last.pt"
                save_checkpoint(path, trainer, original_config, generators=generators, history=history)
                saved = read_checkpoint(path)
                self.assertNotIn("checks", saved["trainer_settings"])
                requested = settings(not enabled)
                requested = replace(requested, training=replace(requested.training, epochs=2))
                restored, loader, validation, generators = build_baseline_experiment(requested)
                restored_history = restore_checkpoint(saved, restored, requested,
                                                       generators=generators, train_loader=loader)
                self.assertEqual(restored_history, history)
                restored.fit(loader, validation, epochs=2)
                self.assertEqual(restored.state.completed_epochs, 2)

    def test_changing_checks_does_not_relax_other_resume_configuration(self):
        original = settings(True)
        trainer, _, _, generators = build_baseline_experiment(original)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initial.pt"
            save_checkpoint(path, trainer, original, generators=generators, history=[])
            saved = read_checkpoint(path)
        disabled = settings(False)
        changed_settings = (
            replace(disabled, training=replace(disabled.training, learning_rate=0.02)),
            replace(disabled, runtime=replace(disabled.runtime, deterministic=False)),
        )
        for requested in changed_settings:
            with self.subTest(runtime=requested.runtime, batch_size=requested.training.batch_size):
                restored, loader, _, generators = build_baseline_experiment(requested)
                before = {name: value.clone() for name, value in restored.model.state_dict().items()}
                with self.assertRaisesRegex(ValueError, "configuration differs"):
                    restore_checkpoint(saved, restored, requested, generators=generators, train_loader=loader)
                for name, value in restored.model.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
