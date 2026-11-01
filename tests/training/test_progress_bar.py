import builtins
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from deepopt import Trainer
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint


class RecordingBar:
    def __init__(self, **options):
        self.options = options
        self.n = options["initial"]
        self.increments = []
        self.postfixes = []
        self.closes = 0

    def update(self, count):
        self.n += count
        self.increments.append(count)

    def set_postfix(self, values, *, refresh):
        self.postfixes.append((dict(values), refresh))

    def close(self):
        self.closes += 1


class IterableBatches:
    def __init__(self):
        self.iterations = 0

    def __len__(self):
        raise AssertionError("Progress must not ask for a loader length")

    def __iter__(self):
        self.iterations += 1
        yield torch.ones(2, 1), torch.zeros(2, 1)


class ProgressBarTests(unittest.TestCase):
    def trainer(self):
        model = nn.Linear(1, 1)
        return Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1), loss_fn=nn.MSELoss())

    @contextmanager
    def capture_bars(self):
        bars = []

        def create(**options):
            bar = RecordingBar(**options)
            bars.append(bar)
            return bar

        package, auto = ModuleType("tqdm"), ModuleType("tqdm.auto")
        package.__path__ = []
        auto.tqdm = Mock(side_effect=create)
        package.auto = auto
        with patch.dict(sys.modules, {"tqdm": package, "tqdm.auto": auto}):
            yield bars, auto.tqdm

    def test_default_bar_counts_complete_epochs_and_reuses_reported_metrics(self):
        trainer, train_loader, validation_loader = self.trainer(), IterableBatches(), IterableBatches()
        rows, callback_counts = [], []
        with self.capture_bars() as (bars, factory):
            def record(current, row):
                rows.append(row)
                callback_counts.append(bars[0].n)

            trainer.fit(train_loader, validation_loader, epochs=2, callbacks=[record])
        factory.assert_called_once_with(total=2, initial=0, desc="Training", unit="epoch", dynamic_ncols=True)
        bar = bars[0]
        self.assertEqual(bar.increments, [1, 1])
        self.assertEqual(bar.n, 2)
        self.assertEqual(bar.closes, 1)
        self.assertEqual(callback_counts, [0, 1])
        self.assertEqual(train_loader.iterations, 2)
        self.assertEqual(validation_loader.iterations, 2)
        for (postfix, refresh), row in zip(bar.postfixes, rows):
            self.assertEqual(postfix, {"train/loss": row["train"]["loss"], "val/loss": row["validation"]["loss"]})
            self.assertFalse(refresh)

    def test_disabled_bar_does_not_import_tqdm_or_format_progress(self):
        trainer = self.trainer()
        original_import = builtins.__import__

        def import_without_tqdm(name, *args, **kwargs):
            if name == "tqdm" or name.startswith("tqdm."):
                raise AssertionError("Disabled progress imported tqdm")
            return original_import(name, *args, **kwargs)

        with self.capture_bars() as (bars, factory), patch("builtins.__import__", side_effect=import_without_tqdm):
            trainer.fit(IterableBatches(), epochs=2, progress_bar=False)
        factory.assert_not_called()
        self.assertEqual(bars, [])
        self.assertEqual(trainer.state.completed_epochs, 2)

    def test_train_only_graceful_stop_keeps_target_total_and_omits_validation_metrics(self):
        trainer, loader = self.trainer(), IterableBatches()

        def stop(current, row):
            current.request_stop()

        with self.capture_bars() as (bars, factory):
            trainer.fit(loader, epochs=5, callbacks=[stop])
        self.assertEqual(bars[0].options["total"], 5)
        self.assertEqual(bars[0].n, 1)
        self.assertEqual(bars[0].closes, 1)
        self.assertEqual(loader.iterations, 1)
        self.assertEqual(set(bars[0].postfixes[0][0]), {"train/loss"})

    def test_restored_completed_epochs_supply_initial_progress_without_skipping_work(self):
        original, rows = self.trainer(), []
        original.fit(IterableBatches(), epochs=1, callbacks=[lambda current, row: rows.append(row)],
                     progress_bar=False)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "last.pt"
            save_checkpoint(checkpoint, original, {}, generators={}, history=rows)
            resumed = self.trainer()
            restore_checkpoint(read_checkpoint(checkpoint), resumed, {}, generators={})
        loader = IterableBatches()
        with self.capture_bars() as (bars, factory):
            resumed.fit(loader, epochs=3)
        factory.assert_called_once_with(total=3, initial=1, desc="Training", unit="epoch", dynamic_ncols=True)
        self.assertEqual(bars[0].increments, [1, 1])
        self.assertEqual(bars[0].n, 3)
        self.assertEqual(bars[0].closes, 1)
        self.assertEqual(loader.iterations, 2)
        self.assertEqual(resumed.state.completed_epochs, 3)

    def test_failed_validation_or_callbacks_close_without_counting_the_epoch(self):
        for phase in ("validation", "callback"):
            with self.subTest(phase=phase):
                trainer = self.trainer()

                def fail(current, row):
                    raise RuntimeError("epoch failed")

                with self.capture_bars() as (bars, factory):
                    if phase == "validation":
                        with patch.object(trainer, "evaluate", side_effect=RuntimeError("epoch failed")):
                            with self.assertRaisesRegex(RuntimeError, "epoch failed"):
                                trainer.fit(IterableBatches(), IterableBatches(), epochs=2)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "epoch failed"):
                            trainer.fit(IterableBatches(), epochs=2, callbacks=[fail])
                self.assertEqual(bars[0].n, 0)
                self.assertEqual(bars[0].closes, 1)
                self.assertEqual(bars[0].postfixes, [])
                self.assertFalse(trainer.at_epoch_boundary)

    def test_keyboard_interrupt_still_closes_progress(self):
        trainer = self.trainer()

        def interrupt(current, row):
            raise KeyboardInterrupt()

        with self.capture_bars() as (bars, factory):
            with self.assertRaises(KeyboardInterrupt):
                trainer.fit(IterableBatches(), epochs=2, callbacks=[interrupt])
        self.assertEqual(bars[0].n, 0)
        self.assertEqual(bars[0].closes, 1)
        self.assertFalse(trainer.at_epoch_boundary)

    def test_invalid_progress_flag_fails_before_training_or_bar_creation(self):
        trainer = self.trainer()
        for flag in (None, 0, 1, "yes"):
            with self.subTest(progress_bar=flag), self.capture_bars() as (bars, factory):
                with self.assertRaisesRegex(ValueError, "progress_bar"):
                    trainer.fit(IterableBatches(), epochs=1, progress_bar=flag)
                factory.assert_not_called()
                self.assertEqual(trainer.state.completed_epochs, 0)
                self.assertEqual(trainer.state.optimizer_steps, 0)

    def test_real_bar_uses_stderr_and_keeps_json_callback_stdout_readable(self):
        try:
            import tqdm.auto
        except ImportError:
            self.skipTest("tqdm dependency is not installed in the test environment")
        stdout, stderr = io.StringIO(), io.StringIO()
        trainer = self.trainer()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            trainer.fit(IterableBatches(), epochs=1,
                        callbacks=[lambda current, row: print(json.dumps(row))])
        lines = stdout.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["epoch"], 1)
        self.assertIn("Training", stderr.getvalue())
        self.assertIn("1/1", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
