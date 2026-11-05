from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanSquaredError

from deepopt import Checkpoint, EveryNSteps, LossResult, Trainer
from deepopt.config import config_from_dict, _read_config as load_config, reset_config
from deepopt.data import ResumableDataLoader, build_loaders
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, save_checkpoint
from deepopt.experiment import build_experiment
from deepopt.memory import prepare_batch_size
from deepopt.run import train, evaluate


def components(resumable=False, accumulation=1):
    torch.manual_seed(7)
    inputs = torch.arange(18, dtype=torch.float32).reshape(9, 2) / 18
    data = TensorDataset(inputs, inputs.sum(dim=1, keepdim=True))
    generator = torch.Generator().manual_seed(17)
    loader = (ResumableDataLoader(data, batch_size=2, generator=generator) if resumable else
              DataLoader(data, batch_size=2, shuffle=True, generator=generator))
    model = nn.Linear(2, 1)
    return Trainer(model, torch.optim.AdamW(model.parameters(), lr=0.02), loss_fn=F.mse_loss,
                   accumulation_steps=accumulation), loader


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        stdout, stderr = redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())
        stdout.__enter__(); stderr.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)
        self.addCleanup(stderr.__exit__, None, None, None)

    def config(self, **training):
        return config_from_dict({"name": "onboarding", "seed": 7,
            "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.02, **training}})

    def test_plain_loader_configured_run_and_validation_free_resume(self):
        def factory(config):
            trainer, loader = components()
            return trainer, loader, None, {"train": loader.generator}
        run = train(self.config(), output=self.root / "run", experiment_factory=factory, progress_bar=False)
        payload = read_checkpoint(run / "last.pt")
        self.assertIsNone(payload["state"]["monitor"])
        self.assertEqual(payload["history"][0]["validation"], {})
        self.assertFalse((run / "best.pt").exists())
        self.assertFalse((run / "last_step.pt").exists())
        train(resume=run / "last.pt", epochs=2, experiment_factory=factory, progress_bar=False)
        self.assertEqual(read_checkpoint(run / "last.pt")["state"]["completed_epochs"], 2)
        with self.assertRaisesRegex(ValueError, "validation loader"):
            evaluate(run / "last.pt", experiment_factory=factory)

    def test_explicit_step_interval_still_rejects_ordinary_loader_before_updates(self):
        t, loader = components()
        with self.assertRaisesRegex(ValueError, "resumable loader"):
            train(self.config(), output=self.root / "run", checkpoint_every_steps=2,
                  experiment_factory=lambda config: (t, loader, None, {}), progress_bar=False)
        self.assertEqual(t.state.optimizer_steps, 0)
        self.assertFalse((self.root / "run").exists())

    def test_none_false_and_toml_monitor_disable_selection(self):
        for value in (None, False):
            config = self.config(monitor=value)
            self.assertIsNone(config.training.monitor)
            self.assertEqual(config_from_dict(asdict(config)), config)
        path = self.root / "config.toml"
        path.write_text('name="test"\nseed=7\n[training]\nepochs=1\nbatch_size=2\nlearning_rate=0.01\nmonitor=false\n')
        self.assertIsNone(load_config(path).training.monitor)
        data = TensorDataset(torch.ones(3, 2), torch.zeros(3, 1))
        t, loader, val, _ = build_experiment(self.config(), model_factory=lambda: nn.Linear(2, 1),
                                            dataset_factory=lambda: (data, None), loss_fn=F.mse_loss)
        self.assertIsNone(val)
        self.assertIs(type(loader), DataLoader)

    def test_checkpoint_callback_owns_history_and_restores_exact_epoch(self):
        reference, reference_loader = components()
        reference.fit(reference_loader, epochs=2, progress_bar=False)
        t, loader = components()
        callback = Checkpoint(self.root / "run")
        t.fit(loader, epochs=1, callbacks=[callback], progress_bar=False)
        self.assertEqual(len(callback.history), 1)
        self.assertFalse((callback.directory / "best.pt").exists())
        resumed, resumed_loader = components()
        restored = Checkpoint(callback.directory)
        restored.restore(resumed, resumed_loader)
        resumed.fit(resumed_loader, epochs=2, callbacks=[restored], progress_bar=False)
        for a, b in zip(reference.model.parameters(), resumed.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(len(restored.history), 2)
        self.assertEqual(len((callback.directory / "metrics.jsonl").read_text().splitlines()), 2)
        self.assertEqual(read_checkpoint(callback.directory / "last.pt")["state"]["optimizer_steps"], 10)

    def test_best_checkpoint_and_no_accidental_directory_reuse(self):
        t, loader = components()
        callback = Checkpoint(self.root / "run")
        t.fit(loader, loader, epochs=1, callbacks=[callback], progress_bar=False)
        self.assertTrue((callback.directory / "best.pt").exists())
        other, other_loader = components()
        with self.assertRaisesRegex(ValueError, "already contains"):
            other.fit(other_loader, epochs=1, callbacks=[Checkpoint(callback.directory)], progress_bar=False)

    def test_restore_new_batch_size_without_repeating_saved_metadata(self):
        for resumable in (False, True):
            with self.subTest(resumable=resumable):
                t, loader = components(resumable=resumable)
                config = {"training": {"batch_size": 2}, "recipe": "regression"}
                callback = Checkpoint(self.root / str(resumable), config=config)
                t.fit(loader, epochs=1, callbacks=[callback], progress_bar=False)
                resumed, new_loader = components(resumable=resumable)
                from deepopt.memory.planning import set_loader_batch_size
                set_loader_batch_size(new_loader, 3)
                restored = Checkpoint(callback.directory)
                restored.restore(resumed, new_loader, path="last.pt")
                self.assertEqual(restored.config["training"]["batch_size"], 3)
                self.assertEqual(config["training"]["batch_size"], 2)
                self.assertEqual(read_checkpoint(callback.directory / "last.pt")["batch_size"], 2)
                resumed.fit(new_loader, epochs=2, callbacks=[restored], progress_bar=False)
                self.assertEqual(read_checkpoint(callback.directory / "last.pt")["config"]["training"]["batch_size"], 3)
                # Explicit contradictory settings still fail before mutation.
                other, other_loader = components(resumable=resumable)
                with self.assertRaisesRegex(ValueError, "configuration batch size"):
                    Checkpoint(callback.directory, config={"training": {"batch_size": 3}}).restore(other, other_loader)
                self.assertEqual(other.state.optimizer_steps, 0)

    def test_checkpoint_selection_reads_only_one_tensor_payload(self):
        t, loader = components(resumable=True)
        callback = Checkpoint(self.root / "selection", every_steps=1)
        t.fit(loader, epochs=1, callbacks=[callback], progress_bar=False)
        resumed, restored_loader = components(resumable=True)
        real_load = torch.load
        reads = []
        def inspect_load(path, **kwargs):
            payload = real_load(path, **kwargs)
            reads.append((Path(path).name, kwargs))
            if kwargs.get("map_location") == "meta":
                self.assertTrue(kwargs["mmap"])
                self.assertTrue(all(tensor.is_meta for tensor in payload["model"].values()))
            return payload
        with patch("torch.load", side_effect=inspect_load):
            Checkpoint(callback.directory).restore(resumed, restored_loader)
        self.assertEqual([name for name, kw in reads if kw["map_location"] == "cpu"], ["last.pt"])
        self.assertEqual(len([kw for _, kw in reads if kw["map_location"] == "meta"]), 2)
        self.assertEqual(resumed.state.completed_epochs, 1)
        with self.assertRaisesRegex(ValueError, "older"):
            Checkpoint(callback.directory).restore(resumed, restored_loader, path="last_step.pt")

    def test_step_restore_reuses_metadata_and_retains_cursor_with_new_size(self):
        t, loader = components(resumable=True)
        callback = Checkpoint(self.root / "step-resize", every_steps=2,
                              config={"training": {"batch_size": 2}})
        def interrupt(trainer, event):
            raise RuntimeError("interrupt after save")
        with self.assertRaisesRegex(RuntimeError, "interrupt"):
            t.fit(loader, epochs=1, callbacks=[callback],
                  step_callbacks=[EveryNSteps(interrupt, 2)], progress_bar=False)
        saved_position = loader.batch_sampler.position
        resumed, new_loader = components(resumable=True)
        new_loader.set_batch_size(3)
        restored = Checkpoint(callback.directory)
        restored.restore(resumed, new_loader)
        self.assertEqual(new_loader.batch_sampler.position, saved_position)
        self.assertEqual(new_loader.batch_sampler.batch_size, 3)
        self.assertEqual(restored.config["training"]["batch_size"], 3)
        resumed.fit(new_loader, epochs=1, callbacks=[restored], progress_bar=False)
        self.assertEqual(restored.history[0]["train"]["samples"], 9)
        self.assertEqual(resumed.state.optimizer_steps, 4)

    def test_auto_restore_records_selected_size_in_callback_metadata(self):
        t, loader = components()
        callback = Checkpoint(self.root / "auto", config={"training": {"batch_size": 2}})
        t.fit(loader, epochs=1, callbacks=[callback], progress_bar=False)
        resumed, new_loader = components()
        restored = Checkpoint(callback.directory)
        plan = {"requested_batch_size": 2, "new_batch_size": 1, "mode": "auto", "status": "planned",
                "report": {}, "memory_budget_bytes": 1024}
        with patch("deepopt.engine.checkpointing.checkpoint.plan_batch_size", return_value=plan):
            restored.restore(resumed, new_loader, batch_size_mode="auto")
        self.assertEqual(new_loader.batch_size, 1)
        self.assertEqual(restored.config["training"]["batch_size"], 1)

    def test_drop_last_helper_and_last_update_resume_match_training(self):
        def make():
            t, original = components()
            train_loader, validation, _ = build_loaders(original.dataset, original.dataset,
                batch_size=2, seed=7, resumable=True, drop_last=True)
            self.assertEqual(len(train_loader), 4)
            self.assertEqual(len(validation), 4)
            return t, train_loader
        reference, reference_loader = make()
        reference.fit(reference_loader, epochs=2, progress_bar=False)
        t, loader = make()
        callback = Checkpoint(self.root / "drop", every_steps=4)
        def stop(trainer, event):
            raise RuntimeError("interrupt at last full batch")
        with self.assertRaisesRegex(RuntimeError, "interrupt"):
            t.fit(loader, epochs=2, callbacks=[callback], step_callbacks=[EveryNSteps(stop, 4)], progress_bar=False)
        resumed, resumed_loader = make()
        restored = Checkpoint(callback.directory)
        restored.restore(resumed, resumed_loader)
        resumed.fit(resumed_loader, epochs=2, callbacks=[restored], progress_bar=False)
        self.assertEqual(restored.history[0]["train"]["samples"], 8)
        for a, b in zip(reference.model.parameters(), resumed.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_every_steps_counts_updates_and_only_synchronizes_when_due(self):
        t, loader = components(resumable=True, accumulation=2)
        callback = Checkpoint(self.root / "run", every_steps=2)
        with patch.object(t.runtime, "synchronize") as sync:
            t.fit(loader, epochs=2, callbacks=[callback], progress_bar=False)
        # Epoch start/end (4) plus updates 2,4,6 (3); not every microbatch/update.
        self.assertEqual(sync.call_count, 7)
        payload = read_checkpoint(callback.directory / "last_step.pt")
        self.assertEqual(payload["state"]["optimizer_steps"], 6)
        self.assertEqual(payload["state"]["completed_epochs"], 1)
        self.assertEqual(len(payload["history"]), 1)
        resumed, resumed_loader = components(resumable=True, accumulation=2)
        Checkpoint(callback.directory).restore(resumed, resumed_loader)
        self.assertEqual(resumed.state.completed_epochs, 2)  # Last epoch wins the tie.

    def test_step_interruption_restores_history_and_partial_window(self):
        reference, reference_loader = components(resumable=True, accumulation=2)
        reference.fit(reference_loader, epochs=1, progress_bar=False)
        t, loader = components(resumable=True, accumulation=2)
        callback = Checkpoint(self.root / "run", every_steps=2)
        def save_then_interrupt(path, *args, **kwargs):
            save_checkpoint(path, *args, **kwargs)
            if kwargs.get("kind") == "step":
                raise RuntimeError("simulated interruption")
        with patch("deepopt.checkpoint.save_checkpoint", side_effect=save_then_interrupt):
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                t.fit(loader, epochs=1, callbacks=[callback], progress_bar=False)
        resumed, resumed_loader = components(resumable=True, accumulation=2)
        restored = Checkpoint(callback.directory, every_steps=2)
        restored.restore(resumed, resumed_loader)
        resumed.fit(resumed_loader, epochs=1, callbacks=[restored], progress_bar=False)
        for a, b in zip(reference.model.parameters(), resumed.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(restored.history[0]["train"]["samples"], 9)

    def test_every_n_steps_callable_preserves_normal_callback_path(self):
        t, loader = components()
        seen = []
        t.fit(loader, epochs=1, progress_bar=False,
              step_callbacks=[EveryNSteps(lambda trainer, event: seen.append(event["optimizer_step"]), 2)])
        self.assertEqual(seen, [2, 4])
        for value in (True, 0, -1):
            with self.assertRaises(ValueError):
                EveryNSteps(lambda *_: None, value)
        for value in (True, -1):
            with self.assertRaises(ValueError):
                Checkpoint(self.root, every_steps=value)

    def test_ordinary_and_explicit_weighted_reporting_are_distinct(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        loader = DataLoader(TensorDataset(torch.zeros(3, 1), torch.tensor([[0.], [0.], [10.]])), batch_size=2)
        t = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.01), loss_fn=F.mse_loss,
                    metrics={"mse": MeanSquaredError()})
        result = t.evaluate(loader)
        self.assertEqual(result["loss"], 50)
        def weighted(predictions, targets):
            return LossResult.from_mean(F.mse_loss(predictions, targets), weight=targets.numel())
        weighted_trainer = Trainer(model, t.optimizer, loss_fn=weighted)
        self.assertAlmostEqual(weighted_trainer.evaluate(loader)["loss"], 100/3)
        self.assertAlmostEqual(result["mse"], 100/3, places=5)

    def test_simple_batch_preparation_uses_supplied_loader(self):
        t, loader = components()
        plan = prepare_batch_size(t, loader, mode="manual")
        self.assertEqual(plan["new_batch_size"], 2)
        expected = {"requested_batch_size": 2, "new_batch_size": 1, "mode": "auto", "status": "planned",
                    "report": {}, "memory_budget_bytes": 1024}
        with patch("deepopt.memory.planning.plan_batch_size", return_value=expected):
            plan = prepare_batch_size(t, loader)
        self.assertEqual(loader.batch_size, 1)
        self.assertEqual(plan["new_batch_size"], 1)


if __name__ == "__main__":
    unittest.main()
