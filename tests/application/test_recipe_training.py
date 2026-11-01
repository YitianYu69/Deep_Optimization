import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from deepopt import Trainer
from deepopt.data.resumable import ResumableDataLoader
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint


class RecipeTrainingTests(unittest.TestCase):
    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for name in expected:
                self.assert_nested_equal(actual[name], expected[name])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def setup_resume(self, *, interval="step", one_cycle=True):
        torch.manual_seed(81)
        data_rng = torch.Generator().manual_seed(82)
        dataset = TensorDataset(torch.randn(13, 2, dtype=torch.double, generator=data_rng),
                                torch.randn(13, 1, dtype=torch.double, generator=data_rng))
        model = nn.Sequential(nn.Linear(2, 5), nn.Dropout(0.25), nn.Tanh(), nn.Linear(5, 1)).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = (torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.02, total_steps=9)
                     if one_cycle else torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9))
        trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, scheduler=scheduler,
                          scheduler_interval=interval, accumulation_steps=2)
        generator = torch.Generator().manual_seed(83)
        loader = ResumableDataLoader(dataset, batch_size=3, generator=generator)
        return trainer, loader, {"train": generator}

    def test_train_only_fit_emits_empty_validation_without_best_selection(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), loss_fn=F.mse_loss,
                          accumulation_steps=2)
        rows = []
        batches = [(torch.ones(2, 2), torch.zeros(2, 1))] * 3
        with patch.object(trainer, "evaluate", side_effect=AssertionError("unexpected evaluation")):
            state = trainer.fit(batches, epochs=2, callbacks=[lambda current, row: rows.append(row)])
        self.assertEqual(state.completed_epochs, 2)
        self.assertEqual(state.optimizer_steps, 4)
        self.assertIsNone(state.monitor)
        self.assertIsNone(state.best_score)
        self.assertEqual(state.best_epoch, 0)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["validation"], {})
            self.assertIn("loss", row["train"])
            self.assertEqual(row["train"]["samples"], 6)

    def test_nonempty_monitor_without_validation_fails_before_training(self):
        model = nn.Linear(2, 1)
        trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), loss_fn=F.mse_loss)
        before = copy.deepcopy(model.state_dict())
        initial_state = copy.deepcopy(trainer.state)
        for monitor in ("loss", "accuracy"):
            with self.subTest(monitor=monitor), patch.object(trainer, "train_epoch") as train_epoch:
                with self.assertRaisesRegex(ValueError, "validation loader"):
                    trainer.fit([], epochs=1, monitor=monitor)
                train_epoch.assert_not_called()
                self.assertEqual(trainer.state, initial_state)
                self.assertTrue(trainer.at_epoch_boundary)
                self.assert_nested_equal(model.state_dict(), before)

    def test_validation_keeps_default_loss_monitor_and_explicit_none_disables_only_selection(self):
        batches = [(torch.ones(2, 1), torch.zeros(2, 1))]
        for disable_monitor in (False, True):
            with self.subTest(disable_monitor=disable_monitor):
                model = nn.Linear(1, 1)
                trainer = Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), loss_fn=F.mse_loss)
                rows = []
                options = {"monitor": None} if disable_monitor else {}
                trainer.fit(batches, batches, epochs=1, callbacks=[lambda current, row: rows.append(row)],
                            **options)
                self.assertIn("loss", rows[0]["validation"])
                if disable_monitor:
                    self.assertIsNone(trainer.state.best_score)
                else:
                    self.assertEqual(trainer.state.monitor, "loss")
                    self.assertEqual(trainer.state.best_score, rows[0]["validation"]["loss"])

    def test_step_schedulers_match_manual_updates_with_partial_accumulation_windows(self):
        for scheduler_kind in ("step_lr", "one_cycle"):
            with self.subTest(scheduler=scheduler_kind):
                torch.manual_seed(84)
                model = nn.Linear(2, 1).double()
                reference = copy.deepcopy(model)
                batches = [(torch.randn(count, 2, dtype=torch.double), torch.randn(count, 1, dtype=torch.double))
                           for count in (3, 1, 2)]
                optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
                expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05, momentum=0.9)

                def create_scheduler(current_optimizer):
                    if scheduler_kind == "one_cycle":
                        return torch.optim.lr_scheduler.OneCycleLR(current_optimizer, max_lr=0.05, total_steps=4)
                    return torch.optim.lr_scheduler.StepLR(current_optimizer, step_size=1, gamma=0.8)

                scheduler, expected_scheduler = create_scheduler(optimizer), create_scheduler(expected_optimizer)
                trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, scheduler=scheduler,
                                  scheduler_interval="step", accumulation_steps=2)
                seen, expected_lrs = [], []

                def record(current, event):
                    self.assertTrue(current.at_step_boundary)
                    self.assertEqual(current.scheduler.last_epoch, event["optimizer_step"])
                    seen.append(current.optimizer.param_groups[0]["lr"])

                trainer.fit(batches, epochs=2, step_callbacks=[record])
                for _ in range(2):
                    for start in range(0, len(batches), 2):
                        losses = [F.mse_loss(reference(inputs), targets)
                                  for inputs, targets in batches[start:start + 2]]
                        expected_optimizer.zero_grad()
                        (sum(losses) / len(losses)).backward()
                        expected_optimizer.step()
                        expected_scheduler.step()
                        expected_lrs.append(expected_optimizer.param_groups[0]["lr"])
                self.assertEqual(seen, expected_lrs)
                self.assertEqual(trainer.state.optimizer_steps, 4)
                self.assertEqual(scheduler.last_epoch, 4)
                for actual, expected in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                self.assert_nested_equal(scheduler.state_dict(), expected_scheduler.state_dict())

    def test_optimizer_failure_does_not_advance_step_scheduler(self):
        class FailingSGD(torch.optim.SGD):
            def step(self, closure=None):
                raise RuntimeError("optimizer update failed")

        model = nn.Linear(1, 1)
        optimizer = FailingSGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        trainer = Trainer(model, optimizer, loss_fn=F.mse_loss, scheduler=scheduler, scheduler_interval="step")
        scheduler_before = copy.deepcopy(scheduler.state_dict())
        with self.assertRaisesRegex(RuntimeError, "optimizer update failed"):
            trainer.fit([(torch.ones(2, 1), torch.zeros(2, 1))], epochs=1)
        self.assertEqual(trainer.state.optimizer_steps, 0)
        self.assert_nested_equal(scheduler.state_dict(), scheduler_before)
        self.assertFalse(trainer.at_step_boundary)

    def test_train_only_step_resume_is_exact_with_per_update_scheduler(self):
        full, loader, generators = self.setup_resume()
        expected_rows = []
        full.fit(loader, epochs=3, callbacks=[lambda current, row: expected_rows.append(row)])
        expected_rng = torch.get_rng_state().clone()
        expected_generator = generators["train"].get_state().clone()
        for stop_step in (1, 3, 4, 9):
            with self.subTest(stop_step=stop_step), tempfile.TemporaryDirectory() as directory:
                interrupted, loader, generators = self.setup_resume()
                rows = []
                checkpoint = Path(directory) / "last_step.pt"

                def stop(current, event):
                    if event["optimizer_step"] == stop_step:
                        save_checkpoint(checkpoint, current, {}, generators=generators, history=rows,
                                        kind="step", train_loader=loader)
                        raise InterruptedError("interrupt at completed update")

                with self.assertRaises(InterruptedError):
                    interrupted.fit(loader, epochs=3, callbacks=[lambda current, row: rows.append(row)],
                                    step_callbacks=[stop])
                saved = read_checkpoint(checkpoint)
                self.assertEqual(saved["scheduler"]["last_epoch"], stop_step)
                self.assertEqual(saved["trainer_settings"]["scheduler_interval"], "step")
                resumed, loader, generators = self.setup_resume()
                rows = restore_checkpoint(saved, resumed, {}, generators=generators, train_loader=loader)
                resumed.fit(loader, epochs=3, callbacks=[lambda current, row: rows.append(row)])
                self.assert_nested_equal(resumed.model.state_dict(), full.model.state_dict())
                self.assert_nested_equal(resumed.optimizer.state_dict(), full.optimizer.state_dict())
                self.assert_nested_equal(resumed.scheduler.state_dict(), full.scheduler.state_dict())
                self.assertEqual(resumed.state, full.state)
                self.assertIsNone(resumed.state.best_score)
                self.assertEqual(resumed.state.best_epoch, 0)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                torch.testing.assert_close(generators["train"].get_state(), expected_generator, rtol=0, atol=0)
                self.assertEqual(len(rows), 3)
                for actual, expected in zip(rows, expected_rows):
                    self.assertEqual(actual["epoch"], expected["epoch"])
                    self.assertEqual(actual["optimizer_steps"], expected["optimizer_steps"])
                    self.assertEqual(actual["learning_rate"], expected["learning_rate"])
                    self.assertEqual(actual["learning_rates"], expected["learning_rates"])
                    self.assertEqual(actual["validation"], {})
                    self.assertEqual(actual["train"]["loss"], expected["train"]["loss"])

    def test_scheduler_interval_mismatch_rejected_before_restore_mutation(self):
        for original_interval, changed_interval in (("epoch", "step"), ("step", "epoch")):
            with self.subTest(original=original_interval), tempfile.TemporaryDirectory() as directory:
                original, loader, generators = self.setup_resume(interval=original_interval, one_cycle=False)
                rows = []
                original.fit(loader, epochs=1, callbacks=[lambda current, row: rows.append(row)])
                checkpoint = Path(directory) / "last.pt"
                save_checkpoint(checkpoint, original, {}, generators=generators, history=rows)
                saved = read_checkpoint(checkpoint)
                changed, loader, generators = self.setup_resume(interval=changed_interval, one_cycle=False)
                before = copy.deepcopy(changed.model.state_dict())
                state_before = copy.deepcopy(changed.state)
                with patch.object(changed.runtime, "invalidate") as invalidate:
                    with self.assertRaisesRegex(ValueError, "trainer settings"):
                        restore_checkpoint(saved, changed, {}, generators=generators, train_loader=loader)
                    invalidate.assert_not_called()
                self.assert_nested_equal(changed.model.state_dict(), before)
                self.assertEqual(changed.state, state_before)
                self.assertTrue(changed.at_epoch_boundary)

    def test_old_epoch_scheduler_metadata_and_config_resume_with_default_interval(self):
        full, loader, generators = self.setup_resume(interval="epoch", one_cycle=False)
        full.fit(loader, epochs=2)
        with tempfile.TemporaryDirectory() as directory:
            original, loader, generators = self.setup_resume(interval="epoch", one_cycle=False)
            rows = []
            original.fit(loader, epochs=1, callbacks=[lambda current, row: rows.append(row)])
            checkpoint = Path(directory) / "last.pt"
            save_checkpoint(checkpoint, original, {"training": {"epochs": 1}},
                            generators=generators, history=rows)
            saved = read_checkpoint(checkpoint)
            self.assertNotIn("scheduler_interval", saved["trainer_settings"])
            resumed, loader, generators = self.setup_resume(interval="epoch", one_cycle=False)
            restore_checkpoint(saved, resumed, {"training": {"epochs": 2, "scheduler_interval": "epoch"}},
                               generators=generators, train_loader=loader)
            resumed.fit(loader, epochs=2)
            self.assert_nested_equal(resumed.model.state_dict(), full.model.state_dict())
            self.assert_nested_equal(resumed.scheduler.state_dict(), full.scheduler.state_dict())
            self.assertEqual(resumed.state, full.state)
            self.assertEqual(resumed.scheduler.last_epoch, 2)

    def test_old_step_progress_without_epoch_start_rates_remains_compatible_for_epoch_scheduler(self):
        full, loader, generators = self.setup_resume(interval="epoch", one_cycle=False)
        expected_rows = []
        full.fit(loader, epochs=3, callbacks=[lambda current, row: expected_rows.append(row)])
        with tempfile.TemporaryDirectory() as directory:
            interrupted, loader, generators = self.setup_resume(interval="epoch", one_cycle=False)
            checkpoint = Path(directory) / "last_step.pt"

            def stop(current, event):
                save_checkpoint(checkpoint, current, {}, generators=generators, history=[],
                                kind="step", train_loader=loader)
                raise InterruptedError("first update")

            with self.assertRaises(InterruptedError):
                interrupted.fit(loader, epochs=3, step_callbacks=[stop])
            saved = read_checkpoint(checkpoint)
            saved["epoch_progress"].pop("learning_rates")
            resumed, loader, generators = self.setup_resume(interval="epoch", one_cycle=False)
            rows = restore_checkpoint(saved, resumed, {}, generators=generators, train_loader=loader)
            resumed.fit(loader, epochs=3, callbacks=[lambda current, row: rows.append(row)])
            self.assert_nested_equal(resumed.model.state_dict(), full.model.state_dict())
            self.assert_nested_equal(resumed.scheduler.state_dict(), full.scheduler.state_dict())
            self.assertEqual([row["learning_rates"] for row in rows],
                             [row["learning_rates"] for row in expected_rows])
            self.assertNotIn("learning_rates", saved["epoch_progress"])

    def test_step_scheduler_resume_requires_valid_epoch_start_rate_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            interrupted, loader, generators = self.setup_resume()
            checkpoint = Path(directory) / "last_step.pt"

            def stop(current, event):
                save_checkpoint(checkpoint, current, {}, generators=generators, history=[],
                                kind="step", train_loader=loader)
                raise InterruptedError("first update")

            with self.assertRaises(InterruptedError):
                interrupted.fit(loader, epochs=3, step_callbacks=[stop])
            saved = read_checkpoint(checkpoint)
            for rates in (None, [], [0.1, 0.2], "0.1"):
                with self.subTest(rates=rates):
                    malformed = copy.deepcopy(saved)
                    malformed["epoch_progress"]["learning_rates"] = rates
                    resumed, loader, generators = self.setup_resume()
                    with self.assertRaisesRegex(ValueError, "within-epoch checkpoint progress"):
                        restore_checkpoint(malformed, resumed, {}, generators=generators, train_loader=loader)
                    self.assertFalse(resumed.at_epoch_boundary)

    def test_invalid_scheduler_interval_and_metric_scheduler_fail_at_construction(self):
        for interval in (None, "batch", True):
            with self.subTest(interval=interval):
                model = nn.Linear(1, 1)
                with self.assertRaisesRegex(ValueError, "scheduler_interval"):
                    Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), loss_fn=F.mse_loss,
                            scheduler_interval=interval)
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        with self.assertRaisesRegex(ValueError, "no required arguments"):
            Trainer(model, optimizer, loss_fn=F.mse_loss, scheduler=scheduler, scheduler_interval="step")


if __name__ == "__main__":
    unittest.main()
