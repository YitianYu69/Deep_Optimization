"""Configured FP8 training, durable resume, evaluation, and legacy compatibility."""

from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from deepopt.baseline import build_baseline_experiment
from deepopt.config import CompileConfig, FP8Config, _read_config as load_config, reset_config
from deepopt.engine.checkpointing.checkpoint import read_checkpoint
from deepopt.opt.do_nn.linear_module import DOLinear
from deepopt.run import evaluate, train


class ConfiguredFP8RunTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = load_config(Path(__file__).resolve().parents[2] / "configs" / "fp8.toml")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def run_training(self, config=None, **kwargs):
        return train(config, batch_size_mode="manual", progress_bar=False, **kwargs)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assert_nested_equal(value, reference)
        else:
            self.assertEqual(actual, expected)

    def test_published_config_train_resume_and_evaluate_match_uninterrupted(self):
        partial = self.run_training(self.config, output=self.root / "resumed", epochs=1)
        first = read_checkpoint(partial / "last.pt")
        expected_policy = asdict(self.config.runtime.fp8)
        self.assertEqual(first["config"]["runtime"]["fp8"], expected_policy)
        self.assertEqual(first["trainer_settings"]["fp8"], expected_policy)
        self.assertEqual(first["state"]["optimizer_steps"], 2)
        self.assertTrue((partial / "best.pt").exists())

        # Reconstruct exclusively from the saved configuration, as the CLI does.
        self.run_training(resume=partial / "last.pt", epochs=2)
        resumed = read_checkpoint(partial / "last.pt")
        uninterrupted = self.run_training(self.config, output=self.root / "uninterrupted")
        reference = read_checkpoint(uninterrupted / "last.pt")
        for key in ("model", "optimizer", "scheduler", "state"):
            self.assert_nested_equal(resumed[key], reference[key])
        self.assertEqual(resumed["state"]["optimizer_steps"], 4)
        self.assertEqual(len((partial / "metrics.jsonl").read_text().splitlines()), 2)
        snapshot = json.loads((partial / "config.json").read_text())
        # JSON represents tuple-valued policy fields (including exclusions) as
        # lists; checkpoint metadata above retains the typed policy containers.
        self.assertEqual(snapshot["runtime"]["fp8"], json.loads(json.dumps(expected_policy)))

        best = read_checkpoint(partial / "best.pt")
        result = evaluate(partial / "best.pt", device="cpu")
        self.assertEqual(result["epoch"], best["state"]["completed_epochs"])
        self.assertEqual(result["validation"]["accuracy"], best["state"]["best_score"])
        self.assertTrue(torch.isfinite(torch.tensor(result["validation"]["loss"])))

    def test_changed_enabled_fp8_policy_is_rejected_without_overwriting_checkpoint(self):
        path = self.run_training(self.config, output=self.root / "original", epochs=1) / "last.pt"
        original_bytes = path.read_bytes()
        changed = replace(self.config, runtime=replace(
            self.config.runtime, fp8=replace(self.config.runtime.fp8, grad_dtype="e5m2")))
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            self.run_training(changed, resume=path, epochs=2)
        self.assertEqual(path.read_bytes(), original_bytes)

    def test_legacy_dense_checkpoint_without_fp8_table_still_resumes(self):
        dense = replace(self.config, runtime=replace(self.config.runtime, fp8=FP8Config()))
        output = self.run_training(dense, output=self.root / "legacy", epochs=1)
        path = output / "last.pt"
        saved = read_checkpoint(path)
        saved["config"]["runtime"].pop("fp8")
        self.assertNotIn("fp8", saved["trainer_settings"])
        torch.save(saved, path)

        self.run_training(resume=path, epochs=2)
        resumed = read_checkpoint(path)
        reference_output = self.run_training(dense, output=self.root / "dense-reference")
        reference = read_checkpoint(reference_output / "last.pt")
        self.assertFalse(resumed["config"]["runtime"]["fp8"]["enabled"])
        self.assertNotIn("fp8", resumed["trainer_settings"])
        self.assert_nested_equal(resumed["model"], reference["model"])
        self.assert_nested_equal(resumed["optimizer"], reference["optimizer"])

    def test_actual_eager_compiler_runs_after_fp8_preparation(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        configured = replace(self.config, runtime=replace(
            self.config.runtime,
            compile=CompileConfig(enabled=True, backend="eager", fullgraph=False),
        ))
        trainer, loader, _, _ = build_baseline_experiment(configured)
        self.assertIsNotNone(trainer.model._compiled_call_impl)
        self.assertTrue(any(isinstance(module, DOLinear) for module in trainer.model.modules()))
        self.assertTrue(trainer.runtime.fp8.enabled)
        self.assertIs(trainer.runtime.transformation_report, trainer.model._deepopt_fp8_report)
        before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
        result = trainer.train_epoch(loader)
        self.assertEqual(trainer.state.optimizer_steps, 2)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
        self.assertTrue(any(not torch.equal(before[name], parameter)
                            for name, parameter in trainer.model.named_parameters()))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in trainer.model.parameters()))


if __name__ == "__main__":
    unittest.main()
