"""EMA validation must not change the live optimization trajectory."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from deepopt.config import ConfigError, config_from_dict, load_config, reset_config
from deepopt.data import build_loaders
from deepopt.engine import Trainer
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from deepopt.memory import estimate_training_memory
from deepopt.run import evaluate as evaluate_checkpoint
from deepopt.tasks.regression import RegressionTask


class BufferedScalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1))
        self.register_buffer("offset", torch.zeros(1))
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

    def forward(self, inputs):
        return inputs @ self.weight + self.offset


class EMAValidationTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)
        self.training = [(torch.ones(2, 1), torch.ones(2, 1))]
        self.validation = [(torch.ones(2, 1), torch.zeros(2, 1))]

    def trainer(self, *, enabled=True, decay=0.5, accumulation_steps=1, model=None,
                validation_step_fn=None):
        model = nn.Linear(1, 1, bias=False) if model is None else model
        with torch.no_grad():
            model.weight.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        computation = ({"task": RegressionTask()} if validation_step_fn is None else
                       {"loss_fn": nn.MSELoss(), "validation_step_fn": validation_step_fn})
        return Trainer(model, optimizer, device="cpu", accumulation_steps=accumulation_steps,
                       config={"training": {"ema": {"enabled": enabled, "decay": decay}}},
                       **computation)

    def test_validation_uses_averaged_weights_and_preserves_optimizer_parameters(self):
        trainer = self.trainer()
        parameter = trainer.model.weight
        storage = parameter.data_ptr()
        trainer.train_epoch(self.training * 2)
        # SGD: 0 -> .5 -> .75; EMA: 0 -> .25 -> .5.
        torch.testing.assert_close(parameter, torch.tensor([[0.75]]))
        before = deepcopy(trainer.model.state_dict())
        for _ in range(2):
            self.assertAlmostEqual(trainer.evaluate(self.validation)["loss"], 0.25)
            self.assertIs(trainer.model.weight, parameter)
            self.assertIs(trainer.optimizer.param_groups[0]["params"][0], parameter)
            self.assertEqual(parameter.data_ptr(), storage)
            torch.testing.assert_close(trainer.model.weight, before["weight"])
            self.assertEqual(trainer.state.optimizer_steps, 2)
            self.assertEqual(trainer.ema.updates, 2)
        trainer.train_epoch(self.training)
        # The next optimizer step starts from .75, not the validation weight .5.
        torch.testing.assert_close(parameter, torch.tensor([[0.875]]))
        self.assertAlmostEqual(trainer.evaluate(self.validation)["loss"], 0.6875 ** 2)

    def test_averaging_occurs_once_per_full_or_partial_accumulation_window(self):
        trainer = self.trainer(accumulation_steps=2)
        trainer.train_epoch(self.training * 3)
        self.assertEqual(trainer.state.optimizer_steps, 2)
        self.assertEqual(trainer.ema.updates, 2)
        torch.testing.assert_close(trainer.model.weight, torch.tensor([[0.75]]))
        self.assertAlmostEqual(trainer.evaluate(self.validation)["loss"], 0.25)

    def test_fit_selects_best_epoch_using_ema_validation(self):
        trainer = self.trainer()
        rows = []
        trainer.fit(self.training, self.training, epochs=1,
                    callbacks=[lambda _, row: rows.append(row)])
        # Validation must report (.25 - 1)^2, not the live model's (.5 - 1)^2.
        self.assertAlmostEqual(trainer.state.best_score, 0.5625)
        self.assertAlmostEqual(rows[0]["validation"]["loss"], 0.5625)
        torch.testing.assert_close(trainer.model.weight, torch.tensor([[0.5]]))

    def test_validation_uses_current_buffers_without_averaging_integer_buffers(self):
        trainer = self.trainer(model=BufferedScalar())
        trainer.train_epoch(self.training * 2)
        with torch.no_grad():
            trainer.model.offset.fill_(3)
            trainer.model.count.fill_(7)
        before = deepcopy(trainer.model.state_dict())
        self.assertAlmostEqual(trainer.evaluate(self.validation)["loss"], 3.5 ** 2)
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_validation_exception_restores_live_weights_and_module_modes(self):
        observed = []

        def fail_validation(model, batch):
            observed.append(model.weight.detach().clone())
            model.offset.add_(5)
            model.count.add_(1)
            raise RuntimeError("validation deliberately failed")

        trainer = self.trainer(model=BufferedScalar(), validation_step_fn=fail_validation)
        trainer.train_epoch(self.training * 2)
        with torch.no_grad():
            trainer.model.offset.fill_(3)
            trainer.model.count.fill_(7)
        trainer.model.train()
        with self.assertRaisesRegex(RuntimeError, "validation deliberately failed"):
            trainer.evaluate(self.validation)
        torch.testing.assert_close(observed[0], torch.tensor([[0.5]]))
        torch.testing.assert_close(trainer.model.weight, torch.tensor([[0.75]]))
        self.assertTrue(trainer.model.training)
        torch.testing.assert_close(trainer.model.offset, torch.tensor([3.0]))
        self.assertEqual(trainer.model.count.item(), 7)
        self.assertEqual(trainer.state.optimizer_steps, 2)
        self.assertEqual(trainer.ema.updates, 2)

    def test_failed_optimizer_step_does_not_advance_the_average(self):
        trainer = self.trainer()
        trainer.train_epoch(self.training)
        before = deepcopy(trainer.ema.shadow)
        with patch.object(trainer.optimizer, "step", side_effect=RuntimeError("optimizer failed")):
            with self.assertRaisesRegex(RuntimeError, "optimizer failed"):
                trainer.train_epoch(self.training)
        self.assertEqual(trainer.state.optimizer_steps, 1)
        self.assertEqual(trainer.ema.updates, 1)
        for name, value in trainer.ema.shadow.items():
            torch.testing.assert_close(value, before[name])

    def test_disabled_ema_validates_the_live_model(self):
        trainer = self.trainer(enabled=False)
        self.assertIsNone(trainer.ema)
        trainer.train_epoch(self.training * 2)
        self.assertAlmostEqual(trainer.evaluate(self.validation)["loss"], 0.75 ** 2)
        torch.testing.assert_close(trainer.model.weight, torch.tensor([[0.75]]))

    def test_zero_decay_tracks_the_latest_optimizer_update(self):
        trainer = self.trainer(decay=0)
        trainer.train_epoch(self.training * 2)
        self.assertAlmostEqual(trainer.evaluate(self.validation)["loss"], 0.75 ** 2)

    def test_low_precision_parameters_keep_float32_averages_through_checkpoint_resume(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                trainer = self.trainer(model=nn.Linear(1, 1, bias=False).to(dtype=dtype))
                self.assertEqual(trainer.model.weight.dtype, dtype)
                self.assertEqual(trainer.ema.shadow["weight"].dtype, torch.float32)
                batches = [(inputs.to(dtype), targets.to(dtype)) for inputs, targets in self.training]
                trainer.train_epoch(batches * 2)
                torch.testing.assert_close(trainer.ema.shadow["weight"], torch.tensor([[0.5]]))
                self.assertEqual(trainer.model.weight.dtype, dtype)
                path = Path(directory) / "low_precision.pt"
                save_checkpoint(path, trainer, generators={}, history=[{"epoch": 1}])
                payload = read_checkpoint(path)
                self.assertEqual(payload["ema"]["shadow"]["weight"].dtype, torch.float32)
                restored = self.trainer(model=nn.Linear(1, 1, bias=False).to(dtype=dtype))
                restore_checkpoint(payload, restored, generators={})
                self.assertEqual(restored.model.weight.dtype, dtype)
                self.assertEqual(restored.ema.shadow["weight"].dtype, torch.float32)
                self.assertEqual(restored.ema.updates, 2)
                restored.train_epoch(batches)
                torch.testing.assert_close(restored.ema.shadow["weight"], torch.tensor([[0.6875]]))
                self.assertEqual(restored.model.weight.dtype, dtype)

    def test_configured_checkpoint_evaluation_restores_and_uses_saved_ema(self):
        config = config_from_dict({"name": "ema-evaluation", "seed": 0,
                                   "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.25,
                                                "ema": {"enabled": True, "decay": 0.5}}})
        constructed = []

        def experiment_factory(settings):
            model = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.zero_()
            current = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.25),
                              task=RegressionTask(), config=settings)
            constructed.append(current)
            return current, self.training * 2, self.validation, {}

        trainer, training, validation, generators = experiment_factory(config)
        trainer.train_epoch(training)
        expected = trainer.evaluate(validation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ema.pt"
            save_checkpoint(path, trainer, generators=generators, history=[{"epoch": 1}])
            result = evaluate_checkpoint(path, device="cpu", experiment_factory=experiment_factory)
        self.assertEqual(len(constructed), 2)
        self.assertEqual(result["epoch"], 1)
        self.assertEqual(result["validation"], expected)
        self.assertAlmostEqual(result["validation"]["loss"], 0.25)
        evaluated = constructed[-1]
        self.assertEqual(evaluated.ema.updates, 2)
        torch.testing.assert_close(evaluated.ema.shadow["weight"], torch.tensor([[0.5]]))
        torch.testing.assert_close(evaluated.model.weight, torch.tensor([[0.75]]))

    def test_memory_estimation_includes_shadows_without_mutating_ema(self):
        ordinary, averaged = self.trainer(enabled=False), self.trainer()
        for trainer in (ordinary, averaged):
            trainer.train_epoch(self.training)
        before = deepcopy(averaged.ema.shadow)
        live_before = deepcopy(averaged.model.state_dict())
        ordinary_report = estimate_training_memory(ordinary, self.training[0])
        averaged_report = estimate_training_memory(averaged, self.training[0])
        for field in ("initial_bytes_by_device", "persistent_bytes_by_device", "peak_bytes_by_device"):
            with self.subTest(field=field):
                self.assertEqual(getattr(averaged_report, field)["cpu"] - getattr(ordinary_report, field)["cpu"],
                                 averaged.ema.storage_bytes)
        self.assertEqual(averaged.ema.updates, 1)
        self.assertEqual(averaged.state.optimizer_steps, 1)
        for name, value in averaged.ema.shadow.items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for name, value in averaged.model.state_dict().items():
            torch.testing.assert_close(value, live_before[name], rtol=0, atol=0)

    def test_checkpoint_restores_average_and_continues_the_same_trajectory(self):
        trainer = self.trainer()
        trainer.train_epoch(self.training * 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ema.pt"
            save_checkpoint(path, trainer, generators={}, history=[{"epoch": 1}])
            payload = read_checkpoint(path)
        self.assertEqual(payload["ema"]["updates"], 2)
        torch.testing.assert_close(payload["ema"]["shadow"]["weight"], torch.tensor([[0.5]]))
        restored = self.trainer()
        restore_checkpoint(payload, restored, generators={})
        self.assertEqual(restored.state.optimizer_steps, 2)
        self.assertEqual(restored.ema.updates, 2)
        self.assertAlmostEqual(restored.evaluate(self.validation)["loss"], 0.25)
        torch.testing.assert_close(restored.model.weight, torch.tensor([[0.75]]))
        for current in (trainer, restored):
            current.train_epoch(self.training)
        torch.testing.assert_close(restored.model.weight, trainer.model.weight)
        self.assertEqual(restored.state.optimizer_steps, trainer.state.optimizer_steps)
        self.assertAlmostEqual(restored.evaluate(self.validation)["loss"], 0.6875 ** 2)
        self.assertAlmostEqual(restored.evaluate(self.validation)["loss"],
                               trainer.evaluate(self.validation)["loss"])

    def test_step_checkpoint_captures_the_completed_average_before_callbacks(self):
        dataset = TensorDataset(torch.ones(6, 1), torch.ones(6, 1))
        uninterrupted = self.trainer(accumulation_steps=2)
        uninterrupted.train_epoch(self.training * 3)
        trainer = self.trainer(accumulation_steps=2)
        loader, _, generators = build_loaders(dataset, None, batch_size=2, seed=7, resumable=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step.pt"

            def save_then_stop(current, event):
                save_checkpoint(path, current, generators=generators, history=[],
                                kind="step", train_loader=loader)
                raise RuntimeError("stop after saving")

            with self.assertRaisesRegex(RuntimeError, "stop after saving"):
                trainer.train_epoch(loader, step_callbacks=[save_then_stop])
            payload = read_checkpoint(path)
        self.assertEqual(payload["ema"]["updates"], 1)
        torch.testing.assert_close(payload["ema"]["shadow"]["weight"], torch.tensor([[0.25]]))
        restored = self.trainer(accumulation_steps=2)
        loader, _, generators = build_loaders(dataset, None, batch_size=2, seed=7, resumable=True)
        restore_checkpoint(payload, restored, generators=generators, train_loader=loader)
        restored.train_epoch(loader)
        self.assertEqual(restored.state.optimizer_steps, 2)
        self.assertEqual(restored.ema.updates, 2)
        torch.testing.assert_close(restored.model.weight, uninterrupted.model.weight)
        self.assertAlmostEqual(restored.evaluate(self.validation)["loss"],
                               uninterrupted.evaluate(self.validation)["loss"])

    def test_checkpoint_rejects_changed_ema_policy_before_loading_weights(self):
        trainer = self.trainer()
        trainer.train_epoch(self.training)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ema.pt"
            save_checkpoint(path, trainer, generators={}, history=[{"epoch": 1}])
            payload = read_checkpoint(path)
        for enabled, decay in ((False, 0.5), (True, 0.9)):
            with self.subTest(enabled=enabled, decay=decay):
                restored = self.trainer(enabled=enabled, decay=decay)
                before = restored.model.weight.detach().clone()
                with self.assertRaises(ValueError):
                    restore_checkpoint(payload, restored, generators={})
                torch.testing.assert_close(restored.model.weight, before)
                self.assertEqual(restored.state.optimizer_steps, 0)

    def test_missing_or_inconsistent_ema_state_is_rejected_before_loading(self):
        trainer = self.trainer()
        trainer.train_epoch(self.training)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ema.pt"
            save_checkpoint(path, trainer, generators={}, history=[{"epoch": 1}])
            payload = read_checkpoint(path)
        missing = deepcopy(payload)
        del missing["ema"]
        inconsistent = deepcopy(payload)
        inconsistent["ema"]["updates"] = 0
        for invalid in (missing, inconsistent):
            with self.subTest(missing="ema" not in invalid):
                restored = self.trainer()
                with self.assertRaisesRegex(ValueError, "EMA|ema"):
                    restore_checkpoint(invalid, restored, generators={})
                torch.testing.assert_close(restored.model.weight, torch.zeros(1, 1))
                self.assertEqual(restored.ema.updates, 0)

    def test_disabled_ema_can_restore_legacy_checkpoint_without_ema_fields(self):
        trainer = self.trainer(enabled=False)
        trainer.train_epoch(self.training)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            save_checkpoint(path, trainer, generators={}, history=[{"epoch": 1}])
            payload = read_checkpoint(path)
        payload.pop("ema", None)
        payload["config"]["training"].pop("ema", None)
        restored = self.trainer(enabled=False)
        restore_checkpoint(payload, restored, generators={})
        self.assertIsNone(restored.ema)
        torch.testing.assert_close(restored.model.weight, trainer.model.weight)
        self.assertAlmostEqual(restored.evaluate(self.validation)["loss"], 0.25)


class EMAConfigurationTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)
        self.raw = {"name": "ema-test", "seed": 0,
                    "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.25}}

    def config(self, ema):
        return config_from_dict({**self.raw, "training": {**self.raw["training"], "ema": ema}})

    def test_defaults_and_toml_roundtrip(self):
        defaults = asdict(config_from_dict(self.raw))["training"]["ema"]
        self.assertEqual(defaults, {"enabled": False, "decay": 0.999})
        self.assertEqual(asdict(self.config({"enabled": True}))["training"]["ema"],
                         {"enabled": True, "decay": 0.999})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ema.toml"
            path.write_text('''name = "ema-test"
seed = 0
[training]
epochs = 1
batch_size = 2
learning_rate = 0.25
[training.ema]
enabled = true
decay = 0.5
''', encoding="utf-8")
            loaded = load_config(path)
        self.assertEqual(asdict(loaded)["training"]["ema"], {"enabled": True, "decay": 0.5})
        self.assertEqual(config_from_dict(asdict(loaded)), loaded)

    def test_invalid_ema_settings_fail_at_configuration_loading(self):
        invalid = [None, True, [], "enabled", {"enabled": 1}, {"enabled": "true"},
                   {"decay": True}, {"decay": -0.1}, {"decay": 1.0},
                   {"decay": float("nan")}, {"decay": float("inf")},
                   {"decay": "0.9"}, {"unknown": True}]
        for ema in invalid:
            with self.subTest(ema=ema), self.assertRaisesRegex(ConfigError, "ema"):
                self.config(ema)


if __name__ == "__main__":
    unittest.main()
