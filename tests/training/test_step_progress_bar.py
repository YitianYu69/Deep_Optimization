"""Step progress observes committed updates without becoming a synchronized callback."""

import builtins
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from deepopt import Checkpoint, Trainer
from deepopt.data import ResumableDataLoader
from deepopt.experiment import build_experiment
from deepopt.run import train
from tests.training import test_progress_bar as epoch_tests


class InterruptedRun(Exception):
    pass


class StepProgressBarTests(unittest.TestCase):
    capture_bars = epoch_tests.ProgressBarTests.capture_bars

    def trainer(self, *, accumulation_steps=1, metrics=None):
        model = nn.Linear(1, 1)
        return Trainer(model, torch.optim.SGD(model.parameters(), lr=0.1),
                       loss_fn=nn.functional.mse_loss, accumulation_steps=accumulation_steps, metrics=metrics)

    @staticmethod
    def batches(count):
        return [(torch.ones(2, 1), torch.zeros(2, 1)) for _ in range(count)]

    def loader(self, count, batch_size):
        data = TensorDataset(torch.ones(count, 1), torch.zeros(count, 1))
        return ResumableDataLoader(data, batch_size=batch_size,
                                   generator=torch.Generator().manual_seed(7))

    def test_default_bar_advances_with_updates_before_epoch_callbacks(self):
        trainer = self.trainer(accumulation_steps=2)
        rows, counts = [], []
        with self.capture_bars() as (bars, factory):
            def record(current, row):
                counts.append((bars[-1].n, current.state.optimizer_steps))
                rows.append(row)

            trainer.fit(self.batches(5), epochs=2, callbacks=[record])
        self.assertEqual(factory.call_count, 2)
        self.assertEqual([bar.options["total"] for bar in bars], [3, 3])
        self.assertEqual([bar.options["initial"] for bar in bars], [0, 0])
        self.assertEqual([bar.options["unit"] for bar in bars], ["step", "step"])
        self.assertIn("1/2", bars[0].options["desc"])
        self.assertIn("2/2", bars[1].options["desc"])
        self.assertEqual(counts, [(3, 3), (3, 6)])
        for bar, row in zip(bars, rows):
            self.assertEqual(bar.increments, [1, 1, 1])
            self.assertEqual(bar.closes, 1)
            self.assertEqual(bar.postfixes[-1][0]["train/loss"], row["train"]["loss"])

    def test_explicit_step_mode_shows_validation_and_saving_phases(self):
        trainer, phases = self.trainer(), []
        original_evaluate = trainer.evaluate
        with self.capture_bars() as (bars, _):
            def evaluate(loader):
                phases.append(bars[0].desc)
                return original_evaluate(loader)

            with patch.object(trainer, "evaluate", side_effect=evaluate):
                trainer.fit(self.batches(2), self.batches(1), epochs=1, progress_bar="step",
                            callbacks=[lambda *_: phases.append(bars[0].desc)])
        self.assertIn("Validation", phases[0])
        self.assertIn("Finalizing", phases[1])
        self.assertEqual(bars[0].n, 2)
        self.assertEqual(bars[0].closes, 1)
        self.assertIn("val/loss", bars[0].postfixes[-1][0])

    def test_unknown_length_loader_is_not_sized_or_iterated_ahead(self):
        trainer, loader = self.trainer(), epoch_tests.IterableBatches()
        with self.capture_bars() as (bars, _):
            trainer.fit(loader, epochs=2)
        self.assertEqual(loader.iterations, 2)
        self.assertEqual(len(bars), 2)
        self.assertTrue(all(bar.options["total"] is None for bar in bars))
        self.assertTrue(all(bar.n == 1 and bar.closes == 1 for bar in bars))

    def test_progress_does_not_compute_metrics_between_configured_reports(self):
        trainer = self.trainer(accumulation_steps=2)
        synchronized_at = []
        with self.capture_bars(), \
                patch.object(trainer.runtime, "synchronize",
                             side_effect=lambda: synchronized_at.append(trainer.state.optimizer_steps)), \
                patch.object(trainer._epochs.metrics, "compute", wraps=trainer._epochs.metrics.compute) as compute:
            trainer.fit(self.batches(5), epochs=1, progress_bar_every_steps=10)
        self.assertEqual(synchronized_at, [0, 3])
        self.assertEqual(compute.call_count, 1)

    def test_interval_displays_all_running_metrics_and_flushes_the_final_partial_interval(self):
        trainer = self.trainer(metrics={"mae": MeanAbsoluteError(), "mse": MeanSquaredError()})
        summaries, callback_counts = [], []
        compute = trainer._epochs.metrics.compute

        def summarize(*args, **kwargs):
            result = compute(*args, **kwargs)
            summaries.append((trainer.state.optimizer_steps, result))
            return result

        with self.capture_bars() as (bars, _), \
                patch.object(trainer._epochs.metrics, "compute", side_effect=summarize):
            trainer.fit(self.batches(5), self.batches(1), epochs=1, progress_bar_every_steps=2,
                        step_callbacks=[lambda *_: callback_counts.append(bars[0].n)])
        self.assertEqual(callback_counts, [0, 2, 2, 4, 4])
        self.assertEqual([step for step, _ in summaries], [2, 4, 5, 5])
        self.assertEqual(bars[0].increments, [2, 2, 1])
        expected_keys = {"train/loss", "train/mae", "train/mse", "optimizer_step"}
        first, second, tail = [postfix for postfix, _ in bars[0].postfixes[:3]]
        self.assertEqual(set(first), expected_keys)
        for postfix, (_, values) in zip((first, second, tail), summaries[:3]):
            for name in ("loss", "mae", "mse"):
                self.assertEqual(postfix[f"train/{name}"], values[name])
        self.assertEqual(bars[0].n, 5)
        self.assertTrue({"val/loss", "val/mae", "val/mse"}.issubset(bars[0].postfixes[-1][0]))

    def test_interval_counts_optimizer_updates_under_accumulation(self):
        trainer = self.trainer(accumulation_steps=2)
        observed = []
        with self.capture_bars() as (bars, _):
            trainer.fit(self.batches(5), epochs=1, progress_bar_every_steps=2,
                        step_callbacks=[lambda *_: observed.append(bars[0].n)])
        self.assertEqual(observed, [0, 2, 2])
        self.assertEqual(bars[0].increments, [2, 1])
        self.assertEqual(bars[0].n, 3)

    def test_large_interval_keeps_unknown_length_and_interrupted_counts_accurate(self):
        trainer, loader = self.trainer(), epoch_tests.IterableBatches()
        with self.capture_bars() as (bars, _):
            trainer.fit(loader, epochs=1, progress_bar_every_steps=50)
        self.assertIsNone(bars[0].options["total"])
        self.assertEqual(bars[0].increments, [1])

        def interrupt(*_):
            raise InterruptedRun()

        trainer = self.trainer()
        with self.capture_bars() as (bars, _), self.assertRaises(InterruptedRun):
            trainer.fit(self.batches(3), epochs=1, progress_bar_every_steps=50, step_callbacks=[interrupt])
        self.assertEqual(bars[0].increments, [1])
        self.assertEqual(bars[0].n, 1)
        self.assertEqual(bars[0].closes, 1)

    def test_invalid_interval_fails_before_training_or_progress_side_effects(self):
        trainer = self.trainer()
        for interval in (None, 0, -1, True, False, 1.0, "10", [], {}):
            with self.subTest(interval=interval), self.capture_bars() as (bars, factory):
                with self.assertRaisesRegex(ValueError, "progress_bar_every_steps"):
                    trainer.fit(self.batches(1), epochs=1, progress_bar_every_steps=interval)
                factory.assert_not_called()
                self.assertEqual(trainer.state.optimizer_steps, 0)

    def test_non_main_rank_does_not_import_tqdm_or_create_bars(self):
        trainer = self.trainer()
        original_import = builtins.__import__

        def reject_tqdm(name, *args, **kwargs):
            if name == "tqdm" or name.startswith("tqdm."):
                raise AssertionError("Non-main rank imported tqdm")
            return original_import(name, *args, **kwargs)

        with self.capture_bars() as (bars, factory), \
                patch("deepopt.engine.training.trainer.distributed.is_main_process", return_value=False), \
                patch("builtins.__import__", side_effect=reject_tqdm):
            trainer.fit(epoch_tests.IterableBatches(), epochs=1)
        factory.assert_not_called()
        self.assertEqual(bars, [])
        self.assertEqual(trainer.state.optimizer_steps, 1)

    def test_step_resume_with_rebatching_uses_committed_prefix_and_remaining_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.trainer(accumulation_steps=2)
            loader = self.loader(17, 2)
            checkpoint = Checkpoint(directory, every_steps=1)

            def interrupt(current, event):
                if event["optimizer_step"] == 2:
                    raise InterruptedRun()

            with self.assertRaises(InterruptedRun):
                original.fit(loader, epochs=1, callbacks=[checkpoint], step_callbacks=[interrupt], progress_bar=False)
            resumed = self.trainer(accumulation_steps=2)
            loader = self.loader(17, 3)
            checkpoint = Checkpoint(directory, every_steps=1)
            checkpoint.restore(resumed, loader)
            with self.capture_bars() as (bars, _):
                resumed.fit(loader, epochs=2, callbacks=[checkpoint], progress_bar_every_steps=3)
        self.assertEqual(bars[0].options["initial"], 2)
        self.assertEqual(bars[0].options["total"], 4)  # Two saved plus two remaining updates.
        self.assertEqual(bars[0].increments, [2])  # Flush the short resumed interval.
        self.assertEqual(bars[0].n, 4)
        self.assertEqual(bars[1].options["initial"], 0)
        self.assertEqual(bars[1].options["total"], 3)  # Six microbatches at the new size.
        self.assertEqual(bars[1].increments, [3])
        self.assertEqual(resumed.state.optimizer_steps, 7)

    def test_exhausted_step_checkpoint_does_not_create_a_fake_update(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.trainer(accumulation_steps=2)
            loader = self.loader(5, 2)
            checkpoint = Checkpoint(directory, every_steps=1)

            def interrupt(current, event):
                if event["optimizer_step"] == 2:
                    raise InterruptedRun()

            with self.assertRaises(InterruptedRun):
                original.fit(loader, epochs=1, callbacks=[checkpoint], step_callbacks=[interrupt], progress_bar=False)
            resumed = self.trainer(accumulation_steps=2)
            loader = self.loader(5, 2)
            checkpoint = Checkpoint(directory, every_steps=1)
            checkpoint.restore(resumed, loader)
            with self.capture_bars() as (bars, _):
                resumed.fit(loader, epochs=1, callbacks=[checkpoint])
        self.assertEqual(bars[0].options["initial"], 2)
        self.assertEqual(bars[0].options["total"], 2)
        self.assertEqual(bars[0].increments, [])
        self.assertEqual(bars[0].closes, 1)
        self.assertEqual(resumed.state.optimizer_steps, 2)

    def test_update_callback_failure_keeps_committed_step_and_closes_bar(self):
        trainer = self.trainer()

        def interrupt(*_):
            raise InterruptedRun()

        with self.capture_bars() as (bars, _), self.assertRaises(InterruptedRun):
            trainer.fit(self.batches(3), epochs=1, step_callbacks=[interrupt])
        self.assertEqual(bars[0].n, 1)
        self.assertEqual(bars[0].closes, 1)
        self.assertEqual(trainer.state.completed_epochs, 0)
        self.assertEqual(trainer.state.optimizer_steps, 1)

    def test_validation_or_saving_failure_closes_step_bar(self):
        for phase in ("validation", "callback"):
            with self.subTest(phase=phase):
                trainer = self.trainer()

                def fail(*_):
                    raise InterruptedRun()

                with self.capture_bars() as (bars, _), self.assertRaises(InterruptedRun):
                    if phase == "validation":
                        with patch.object(trainer, "evaluate", side_effect=fail):
                            trainer.fit(self.batches(1), self.batches(1), epochs=1)
                    else:
                        trainer.fit(self.batches(1), epochs=1, callbacks=[fail])
                self.assertEqual(bars[0].n, 1)
                self.assertEqual(bars[0].closes, 1)

    def test_real_step_bar_keeps_stdout_json_readable(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        trainer = self.trainer()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            trainer.fit(self.batches(3), epochs=1,
                        callbacks=[lambda _, row: print(json.dumps(row))])
        self.assertEqual(json.loads(stdout.getvalue())["optimizer_steps"], 3)
        self.assertIn("1/1", stderr.getvalue())
        self.assertIn("3/3", stderr.getvalue())
        self.assertIn("step", stderr.getvalue())

    def test_configured_runner_accepts_both_named_progress_modes(self):
        dataset = TensorDataset(torch.ones(4, 1), torch.zeros(4, 1))
        factory = partial(build_experiment, model_factory=lambda: nn.Linear(1, 1),
                          dataset_factory=lambda: (dataset, None), loss_fn=nn.functional.mse_loss)
        config = {"name": "progress-modes", "seed": 7,
                  "training": {"epochs": 1, "batch_size": 2, "learning_rate": 0.1},
                  "logging": {"level": "ERROR", "file_level": "ERROR"}}
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("step", "epoch"):
                with self.subTest(mode=mode), self.capture_bars() as (bars, _), redirect_stdout(io.StringIO()):
                    output = Path(directory) / mode
                    train(config, output=output, experiment_factory=factory,
                          batch_size_mode="manual", progress_bar=mode, progress_bar_every_steps=10)
                self.assertTrue((output / "last.pt").exists())
                self.assertEqual(bars[0].options["unit"], mode)
                self.assertEqual(bars[0].increments, [2] if mode == "step" else [1])


if __name__ == "__main__":
    unittest.main()
