"""Checkpoint recipe discovery terminates through compiled forward facades."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

import torch

import deepopt
from deepopt.checkpoint import Checkpoint
from deepopt.compiler import compilation_report
from deepopt.config import CompileConfig
from deepopt.data import build_loaders
from deepopt.engine.execution.compilation import compile_model
from deepopt.engine.checkpointing.checkpoint import read_checkpoint
from deepopt.operations import operation_recipes
from examples.vae.mbconv_vae import MBConvVAE_V2
from examples.vae.train_vae import make_vae_step
from tests.checkpointing.test_registered_operation_recipe import _Model, operation


class _ForwardFacade:
    def __init__(self, original):
        self.original = original

    def __call__(self, *args, **kwargs):
        raise AssertionError("Recipe discovery must never execute a numerical facade")


class _StopAfterSavedUpdate(Checkpoint):
    def on_optimizer_step(self, trainer, event):
        super().on_optimizer_step(trainer, event)
        if event["optimizer_step"] == 1:
            raise _SavedUpdate()


class _SavedUpdate(RuntimeError):
    pass


class CompiledOperationRecipeTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.compiler.reset()
        self.addCleanup(torch.set_num_threads, self.threads)
        self.addCleanup(torch.compiler.reset)
        self.addCleanup(deepopt.reset_config)
        self.addCleanup(deepopt.shutdown_distributed)

    def test_deep_facades_and_bound_method_cycle_preserve_declared_schema(self):
        model = _Model(operation("deep-forward"))
        expected = operation_recipes(model)
        forward = model.forward
        for _ in range(1500):
            forward = _ForwardFacade(forward)
        model.forward = forward
        # The numerical facade can refer back to the same model through its
        # bound original method. Discovery must normalize that method identity.
        self.assertEqual(operation_recipes(model), expected)

    def test_compiled_vae_recipe_can_be_discovered_before_first_forward(self):
        model = MBConvVAE_V2(base_channels=2, max_channels=8, latent_channels=1, blocks_per_stage=2)
        expected = operation_recipes(model)
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
            options={"require_full_aot": True, "use_native_autodiff": True}))
        self.assertEqual(operation_recipes(model), expected)

    def test_same_helper_on_different_owners_discovers_both_operation_schemas(self):
        model = torch.nn.ModuleList([_Model(operation("first-owner")), _Model(operation("second-owner"))])
        self.assertEqual([row["schema"] for row in operation_recipes(model)], ["first-owner", "second-owner"])

    def test_real_compiled_vae_step_checkpoint_and_restore_continue_training(self):
        raw = {
            "training": {"epochs": 1, "batch_size": 2, "learning_rate": .001,
                "checkpoint_every_steps": 1, "monitor": None},
            "runtime": {"device": "cpu", "precision": "fp32", "bf16_policy": "torch_amp",
                "checks": False, "activation_compression": {"enabled": True, "bits": 2,
                    "group_size": 1024, "norm_xhat_recenter": True},
                "compile": {"enabled": True, "backend": "deepopt", "dynamic": False, "fullgraph": True,
                    "options": {"use_native_autodiff": True, "require_full_aot": True,
                        "share_compatible_packs": True}}},
            "schedules": {"beta": {"type": "linear_warmup", "start": 0., "end": 1e-6,
                "duration": 10000, "interval": "update"}}, "logging": {"level": "ERROR"}}
        images = torch.rand(4, 3, 16, 16, generator=torch.Generator().manual_seed(27)) * 2 - 1

        def components():
            model = MBConvVAE_V2(base_channels=2, max_channels=8, latent_channels=1, blocks_per_stage=2)
            expected_recipe = operation_recipes(model)
            optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
            schedules = deepopt.ScheduleController.from_config(raw["schedules"])
            trainer = deepopt.Trainer(model, optimizer, step_fn=make_vae_step(schedules),
                                      schedules=schedules, config=deepcopy(raw))
            loader, _, _ = build_loaders(images, images[:2], batch_size=2, seed=27,
                resumable=True, num_workers=0, persistent_workers=False)
            return trainer, loader, expected_recipe

        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            directory = Path(temporary)
            trainer, loader, expected_recipe = components()
            checkpoint = _StopAfterSavedUpdate(directory)
            with self.assertRaises(_SavedUpdate):
                trainer.fit(loader, epochs=1, callbacks=[checkpoint], progress_bar=False)
            payload = read_checkpoint(directory / "last_step.pt")
            self.assertEqual(payload["state"]["optimizer_steps"], 1)
            self.assertEqual(payload["kind"], "step")
            self.assertEqual(operation_recipes(trainer.model), expected_recipe)
            report = compilation_report(trainer.model)
            self.assertGreater(report.graphs[0].native_autodiff_operations, 0)
            self.assertFalse(report.fallback_reasons)
            expected = deepcopy(payload["model"])

            restored, restored_loader, restored_recipe = components()
            self.assertEqual(restored_recipe, expected_recipe)
            restored_checkpoint = Checkpoint(directory)
            restored_checkpoint.restore(restored, restored_loader)
            for name, value in restored.model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            self.assertEqual(restored.state.optimizer_steps, 1)
            restored.fit(restored_loader, epochs=1, callbacks=[restored_checkpoint], progress_bar=False)
            final = read_checkpoint(directory / "last.pt")
            self.assertEqual(final["state"]["optimizer_steps"], 2)
            self.assertEqual(final["state"]["completed_epochs"], 1)
            self.assertTrue(all(torch.isfinite(value).all() for value in final["model"].values()))


if __name__ == "__main__":
    unittest.main()
