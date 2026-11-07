from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from functools import partial
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from deepopt.cli import main
from deepopt.baseline import baseline_sample_batch, build_baseline_experiment
from deepopt.config import ConfigError, config_from_dict, _read_config as load_config, reset_config
from deepopt.engine.checkpointing.checkpoint import read_checkpoint, restore_checkpoint
from deepopt.run import train


class ResumeCLIBatchSizeTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config_path = self.root / "config.toml"
        self.config_path.write_text('''name = "batch-size-cli"
seed = 7
[training]
epochs = 1
batch_size = 7
learning_rate = 0.02
checkpoint_every_steps = 0
[application.synthetic]
train_samples = 17
validation_samples = 9
''')

    def cli(self, *args, expected_code=0):
        result = subprocess.run(
            [sys.executable, "-m", "deepopt", *map(str, args)],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        self.assertEqual(result.returncode, expected_code, result.stderr)
        return result

    def test_epoch_resume_changes_batch_size_without_config_and_keeps_stdout_json(self):
        output = self.root / "run"
        first = self.cli("train", "--config", self.config_path, "--output", output,
                         "--batch-size", 4, "--no-progress")
        first_row = json.loads(first.stdout)
        self.assertEqual(first_row["batch_size"], 4)
        self.assertEqual(first_row["optimizer_steps"], 5)
        self.assertEqual(first_row["train"]["samples"], 17)
        self.assertEqual(json.loads((output / "config.json").read_text())["training"]["batch_size"], 4)

        resumed = self.cli("train", "--resume", output / "last.pt", "--epochs", 2,
                           "--batch-size", 6, "--batch-size-mode", "manual", "--no-progress")
        row = json.loads(resumed.stdout)
        self.assertEqual(row["epoch"], 2)
        self.assertEqual(row["batch_size"], 6)
        self.assertEqual(row["optimizer_steps"], 8)  # Five old batches, then three new batches.
        self.assertEqual(row["train"]["samples"], 17)
        self.assertEqual(row["validation"]["samples"], 9)
        self.assertIn("batch", resumed.stderr.lower())
        self.assertIn("4", resumed.stderr)
        self.assertIn("6", resumed.stderr)
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["batch_size"] for row in rows], [4, 6])
        self.assertEqual(json.loads((output / "config.json").read_text())["training"]["batch_size"], 6)
        payload = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(payload["config"]["training"]["batch_size"], 6)
        self.assertEqual(payload["state"]["completed_epochs"], 2)
        adjustments = [json.loads(line) for line in (output / "resume.jsonl").read_text().splitlines()]
        self.assertEqual(len(adjustments), 1)
        self.assertEqual(adjustments[0]["old_batch_size"], 4)
        self.assertEqual(adjustments[0]["new_batch_size"], 6)
        self.assertEqual(adjustments[0]["mode"], "manual")
        self.assertIn("status", adjustments[0])

        unchanged = self.cli("train", "--resume", output / "last.pt", "--epochs", 3,
                             "--batch-size-mode", "manual", "--no-progress")
        self.assertEqual(json.loads(unchanged.stdout)["optimizer_steps"], 11)
        adjustments = [json.loads(line) for line in (output / "resume.jsonl").read_text().splitlines()]
        self.assertEqual(len(adjustments), 2)
        self.assertEqual(adjustments[1]["old_batch_size"], 6)
        self.assertEqual(adjustments[1]["new_batch_size"], 6)

    def test_default_progress_preserves_json_stdout(self):
        result = self.cli("train", "--config", self.config_path, "--output", self.root / "progress")
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["epoch"], 1)
        self.assertEqual(rows[0]["optimizer_steps"], 3)
        self.assertNotIn("\x1b", result.stdout)

    def test_automatic_resume_skips_memory_planning_on_cpu_and_uses_requested_size(self):
        output = self.root / "automatic"
        self.cli("train", "--config", self.config_path, "--output", output, "--no-progress")
        resumed = self.cli("train", "--resume", output / "last.pt", "--epochs", 2,
                           "--batch-size", 5, "--no-progress")
        row = json.loads(resumed.stdout)
        self.assertEqual(row["optimizer_steps"], 7)  # Three original batches plus four at size five.
        self.assertEqual(row["train"]["samples"], 17)
        self.assertIn("skip", resumed.stderr.lower())
        self.assertEqual(json.loads((output / "config.json").read_text())["training"]["batch_size"], 5)
        adjustment = json.loads((output / "resume.jsonl").read_text())
        self.assertEqual(adjustment["mode"], "auto")
        self.assertEqual(adjustment["new_batch_size"], 5)
        self.assertEqual(adjustment["status"], "skipped")
        self.assertIn(adjustment["reason"], resumed.stderr)

    def test_cli_forwards_progress_toggle_and_optional_batch_override(self):
        cases = (([], True, None, "auto"),
                 (["--no-progress", "--batch-size", "5", "--batch-size-mode", "manual"], False, 5, "manual"))
        for flags, enabled, batch_size, mode in cases:
            with self.subTest(flags=flags), patch("deepopt.run.train", return_value=self.root) as run:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    main(["train", "--resume", str(self.root / "last.pt"), *flags])
                self.assertIs(run.call_args.kwargs["progress_bar"], enabled)
                self.assertEqual(run.call_args.kwargs["batch_size"], batch_size)
                self.assertEqual(run.call_args.kwargs["batch_size_mode"], mode)

    def test_invalid_batch_overrides_fail_before_building_components_or_creating_output(self):
        config = load_config(self.config_path)
        for value in (0, -1, True, 1.5, "4"):
            with self.subTest(value=value):
                factory = Mock()
                output = self.root / "invalid"
                with self.assertRaisesRegex(ConfigError, "batch_size must be an integer"):
                    train(config, output=output, batch_size=value, progress_bar=False, experiment_factory=factory)
                factory.assert_not_called()
                self.assertFalse(output.exists())

    def test_cli_invalid_batch_size_has_readable_error_and_no_artifacts(self):
        output = self.root / "invalid"
        result = self.cli("train", "--config", self.config_path, "--output", output,
                          "--batch-size", 0, "--no-progress", expected_code=2)
        self.assertIn("batch_size must be an integer", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse(output.exists())

    def test_progress_toggle_requires_boolean_before_any_run_side_effects(self):
        factory = Mock()
        with self.assertRaisesRegex(ValueError, "progress_bar must be a boolean"):
            train(load_config(self.config_path), output=self.root / "invalid", progress_bar="false",
                  experiment_factory=factory)
        factory.assert_not_called()
        self.assertFalse((self.root / "invalid").exists())

    def test_batch_size_mode_rejects_unknown_choices_before_building_components(self):
        factory = Mock()
        with self.assertRaisesRegex(ValueError, "batch_size_mode must be auto or manual"):
            train(load_config(self.config_path), output=self.root / "invalid", batch_size_mode="guess",
                  experiment_factory=factory)
        factory.assert_not_called()
        self.assertFalse((self.root / "invalid").exists())

    def test_baseline_sample_factory_uses_only_its_recipe_shapes_in_fake_mode(self):
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
        config = load_config(self.config_path)
        config = replace(config, application={"synthetic": {"input_dim": 5, "num_classes": 3}})
        with FakeTensorMode():
            inputs, targets = baseline_sample_batch(config, 4)
        self.assertIsInstance(inputs, FakeTensor)
        self.assertIsInstance(targets, FakeTensor)
        self.assertEqual(inputs.shape, (4, 5))
        self.assertEqual(targets.shape, (4,))
        self.assertEqual(inputs.dtype, torch.float32)
        self.assertEqual(targets.dtype, torch.long)

    def test_memory_planning_arguments_validate_before_building_components(self):
        factory = Mock()
        for value in (0, -1, True, 1.5, "1024"):
            with self.subTest(memory_budget_bytes=value):
                with self.assertRaisesRegex(ValueError, "memory_budget_bytes must be a positive integer"):
                    train(load_config(self.config_path), output=self.root / "invalid",
                          memory_budget_bytes=value, experiment_factory=factory)
        with self.assertRaisesRegex(TypeError, "sample_batch_fn must be callable"):
            train(load_config(self.config_path), output=self.root / "invalid",
                  sample_batch_fn=42, experiment_factory=factory)
        factory.assert_not_called()
        self.assertFalse((self.root / "invalid").exists())

    def test_baseline_auto_resume_uses_fake_sample_factory_and_persists_selected_size(self):
        from deepopt.memory import estimate_training_memory
        output = self.root / "planned"
        with redirect_stdout(io.StringIO()):
            train(load_config(self.config_path), output=output, batch_size=8, progress_bar=False)
        payload = read_checkpoint(output / "last.pt")
        saved_config = config_from_dict(payload["config"])
        trainer, loader, validation, generators = build_baseline_experiment(saved_config)
        restore_checkpoint(payload, trainer, saved_config, generators=generators,
                           train_loader=loader, validation_loader=validation, batch_size_mode="manual")
        # Set a deterministic CPU budget at the four-sample estimate. The
        # automatic path must reject eight by its estimate and select four.
        peaks = {size: estimate_training_memory(trainer, baseline_sample_batch(saved_config, size))
                 .peak_bytes_by_device["cpu"] for size in (8, 4)}
        self.assertGreater(peaks[8], peaks[4])
        with redirect_stdout(io.StringIO()), self.assertLogs("deepopt.engine.checkpoint", level="WARNING"):
            train(resume=output / "last.pt", epochs=2, memory_budget_bytes=peaks[4], progress_bar=False)
        decision = json.loads((output / "resume.jsonl").read_text())
        self.assertEqual(decision["status"], "planned")
        self.assertEqual([candidate["batch_size"] for candidate in decision["candidates"]], [8, 4])
        self.assertEqual(decision["new_batch_size"], 4)
        self.assertEqual(decision["validation_batch_size"], 4)
        self.assertEqual(json.loads((output / "config.json").read_text())["training"]["batch_size"], 4)
        checkpoint = read_checkpoint(output / "last.pt")
        self.assertEqual(checkpoint["config"]["training"]["batch_size"], 4)
        self.assertEqual(checkpoint["batch_size"], 4)
        self.assertEqual(checkpoint["state"]["optimizer_steps"], 8)  # Three old batches plus five at size four.

    def test_fresh_baseline_plans_before_training_and_records_selected_size(self):
        from deepopt.memory import estimate_training_memory
        config = load_config(self.config_path)
        config = replace(config, training=replace(config.training, batch_size=8))
        trainer, _, _, _ = build_baseline_experiment(config)
        peaks = {size: estimate_training_memory(trainer, baseline_sample_batch(config, size))
                 .peak_bytes_by_device["cpu"] for size in (8, 4)}
        self.assertGreater(peaks[8], peaks[4])
        output = self.root / "fresh-planned"
        with redirect_stdout(io.StringIO()):
            train(config, output=output, memory_budget_bytes=peaks[4], progress_bar=False)
        decision = json.loads((output / "batch_size.jsonl").read_text())
        self.assertEqual(decision["event"], "start")
        self.assertEqual(decision["status"], "planned")
        self.assertEqual(decision["new_batch_size"], 4)
        self.assertEqual(decision["validation_batch_size"], 4)
        self.assertEqual(json.loads((output / "config.json").read_text())["training"]["batch_size"], 4)
        checkpoint = read_checkpoint(output / "last.pt")
        self.assertEqual(checkpoint["batch_size"], 4)
        self.assertEqual(checkpoint["state"]["optimizer_steps"], 5)
        self.assertFalse((output / "resume.jsonl").exists())

    def test_custom_recipe_needs_sample_factory_before_creating_output(self):
        output = self.root / "unsupported"
        with self.assertRaisesRegex(ValueError, "requires sample_batch_fn"):
            train(load_config(self.config_path), output=output, memory_budget_bytes=1_000_000,
                  experiment_factory=partial(build_baseline_experiment), progress_bar=False)
        self.assertFalse(output.exists())

    def test_completed_epoch_repair_skips_planning_and_preserves_checkpoint(self):
        output = self.root / "complete"
        with redirect_stdout(io.StringIO()):
            train(load_config(self.config_path), output=output, batch_size_mode="manual", progress_bar=False)
        checkpoint = output / "last.pt"
        original_bytes = checkpoint.read_bytes()
        sample = Mock(side_effect=AssertionError("Completed runs must not request a sample batch"))
        stdout = io.StringIO()
        with redirect_stdout(stdout), self.assertLogs("deepopt.run", level="WARNING"):
            train(resume=checkpoint, epochs=1, memory_budget_bytes=1, sample_batch_fn=sample,
                  progress_bar=False)
        sample.assert_not_called()
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(checkpoint.read_bytes(), original_bytes)
        decision = json.loads((output / "resume.jsonl").read_text())
        self.assertEqual(decision["mode"], "auto")
        self.assertEqual(decision["status"], "skipped")
        self.assertIn("already complete", decision["reason"])
        self.assertEqual(len((output / "batch_size.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
