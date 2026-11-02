"""Configured compilation with real Torch backends, training and recovery."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from deepopt.config import config_from_dict, load_config
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint
from deepopt.experiment import build_experiment
from deepopt.memory import estimate_training_memory
from deepopt.run import evaluate, train


def make_config(enabled=True, backend="aot_eager", **settings):
    return config_from_dict({"name": "compiled", "seed": 9,
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": .01, "optimizer": "sgd"},
        "runtime": {"execution_mode": "max_speed", "compile": {
            "enabled": enabled, "backend": backend, "fullgraph": True, **settings}}})


def recipe(config):
    x = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10
    dataset = TensorDataset(x, x.sum(dim=1, keepdim=True))
    return build_experiment(config, model_factory=lambda: nn.Linear(2, 1),
                            dataset_factory=lambda: (dataset, dataset), loss_fn=nn.functional.mse_loss)


class CompileBlackBoxTests(unittest.TestCase):
    def setUp(self):
        # Independent tests must not consume each other's Dynamo cache budget.
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_real_backends_match_eager_updates_with_a_partial_final_batch(self):
        for backend in ("eager", "aot_eager"):
            with self.subTest(backend=backend):
                compiled, batches, _, _ = recipe(make_config(backend=backend))
                eager, reference_batches, _, _ = recipe(make_config(enabled=False))
                compiled.train_epoch(batches)
                eager.train_epoch(reference_batches)
                self.assertIs(type(compiled.model), nn.Linear)
                self.assertEqual(list(compiled.model.state_dict()), ["weight", "bias"])
                for actual, expected in zip(compiled.model.parameters(), eager.model.parameters()):
                    torch.testing.assert_close(actual, expected)
                self.assertEqual(compiled.state.optimizer_steps, eager.state.optimizer_steps)

    def test_fake_memory_estimation_does_not_compile_or_mutate_live_model(self):
        trainer, batches, _, _ = recipe(make_config())
        function = trainer.model._compiled_call_impl
        before = deepcopy(trainer.model.state_dict())
        estimate = estimate_training_memory(trainer, next(iter(batches)))
        self.assertGreater(estimate.peak_bytes_by_device["cpu"], 0)
        self.assertIs(trainer.model._compiled_call_impl, function)
        self.assertIsNone(deepcopy(trainer.model)._compiled_call_impl)
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        trainer.train_epoch(batches)

    def test_toml_train_checkpoint_resume_and_evaluate(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config_file = Path(directory, "compile.toml")
            config_file.write_text('''name = "compiled"
seed = 9
[training]
epochs = 1
batch_size = 2
learning_rate = 0.01
optimizer = "sgd"
[runtime]
execution_mode = "max_speed"
[runtime.compile]
enabled = true
backend = "aot_eager"
fullgraph = true
dynamic = true
''')
            config = load_config(config_file)
            output = Path(directory, "run")
            train(config, output=output, experiment_factory=recipe, batch_size_mode="manual", progress_bar=False)
            first = read_checkpoint(output / "last.pt")
            self.assertTrue(first["config"]["runtime"]["compile"]["enabled"])
            self.assertEqual(set(first["model"]), {"weight", "bias"})
            train(resume=output / "last.pt", epochs=2, experiment_factory=recipe,
                  batch_size_mode="manual", progress_bar=False)
            resumed = read_checkpoint(output / "last.pt")
            full = asdict(config)
            full["training"]["epochs"] = 2
            uninterrupted = Path(directory, "uninterrupted")
            train(config_from_dict(full), output=uninterrupted, experiment_factory=recipe,
                  batch_size_mode="manual", progress_bar=False)
            expected = read_checkpoint(uninterrupted / "last.pt")
            for name, value in resumed["model"].items():
                torch.testing.assert_close(value, expected["model"][name], rtol=0, atol=0)
            self.assertEqual(evaluate(output / "last.pt", experiment_factory=recipe)["epoch"], 2)

    def test_old_checkpoint_without_compile_table_still_restores(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config = make_config(enabled=False)
            output = Path(directory, "run")
            train(config, output=output, experiment_factory=recipe, batch_size_mode="manual", progress_bar=False)
            payload = read_checkpoint(output / "last.pt")
            payload["config"]["runtime"].pop("compile")
            trainer, loader, valid, generators = recipe(config)
            restore_checkpoint(payload, trainer, config, train_loader=loader, validation_loader=valid,
                               generators=generators, batch_size_mode="manual")
            self.assertEqual(trainer.state.completed_epochs, 1)

    def test_resume_rejects_changed_enabled_compiler_settings(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory, "run")
            train(make_config(), output=output, experiment_factory=recipe, batch_size_mode="manual", progress_bar=False)
            changed = asdict(make_config(backend="eager"))
            changed["training"]["epochs"] = 2
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                train(config_from_dict(changed), resume=output / "last.pt", experiment_factory=recipe,
                      batch_size_mode="manual", progress_bar=False)

if __name__ == "__main__":
    unittest.main()
