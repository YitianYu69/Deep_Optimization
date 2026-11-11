"""Run documented CPU recipes without downloads or persistent output directories."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ExampleBlackBoxTests(unittest.TestCase):
    def run_example(self, relative_path, *arguments, timeout=60):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src"))),
               "OMP_NUM_THREADS": "1", "DEEPOPT_LOGGING_LEVEL": "INFO"}
        result = subprocess.run([sys.executable, str(ROOT / relative_path), *map(str, arguments)],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        self.assertEqual(result.returncode, 0, f"{relative_path} {arguments}\n{result.stdout}\n{result.stderr}")
        return result

    def test_base_cpu_recipes_and_documented_modes(self):
        commands = [
            ("training/regression.py",), ("training/multimodal.py",),
            ("training/train_functional.py", "--mode", "supervised", "--epochs", "1"),
            ("training/train_functional.py", "--mode", "custom", "--epochs", "1"),
            *(("training/train_module.py", "--mode", mode) for mode in ("fit", "manual", "wrapper")),
            ("optimization/multiple_optimizers.py",), ("optimization/alternating_updates.py",),
            *(("optimization/train_strategies.py", "--api", api, "--weighting", weighting)
              for api in ("fit", "manual") for weighting in ("samples", "microbatches")),
            ("optimization/fp8_training.py",),
            ("optimization/fp8_training.py", "--trace-mode", "modules", "--block-size", "32"),
            ("metrics/custom_metric.py",), ("metrics/metric_routing.py",),
            ("metrics/loss_reporting.py",), ("metrics/weighted_tokens.py",),
            ("scheduling/coefficients.py",),
            ("scheduling/learning_rate.py", "--interval", "step"),
            ("scheduling/learning_rate.py", "--interval", "epoch"),
            ("memory/model_memory.py",), ("memory/estimate_memory.py",), ("memory/batch_size_modes.py",),
            ("observability/callbacks.py",), ("observability/train_observers.py", "--mode", "all"),
            ("data/skip_batches.py",), ("data/resumable_prefetch.py",),
            ("execution/compile_training.py",), ("execution/execution_modes.py",),
            ("configured/read_config.py",), ("configured/read_config.py", "--config", "configs/scheduling.toml"),
            ("configured/global_config.py",),
            ("cuda/graphs.py", "--help"),
            ("distributed/ddp_training.py", "--help"),
        ]
        for command in commands:
            with self.subTest(command=command):
                result = self.run_example("examples/" + command[0], *command[1:])
                self.assertTrue((result.stdout + result.stderr).strip(),
                                f"Example produced no result: {command}")
                if command[0] == "configured/global_config.py":
                    summary = json.loads(result.stdout.splitlines()[-1])
                    self.assertEqual(summary["first"]["updates"], 4)
                    self.assertEqual(summary["second"]["updates"], 2)
                    self.assertEqual(summary["first"]["batch_size"], 4)
                    self.assertEqual(summary["second"]["batch_size"], 8)
                    self.assertFalse(summary["first"]["checks"])
                    self.assertIsNone(summary["first"]["monitor"])
                    self.assertTrue(summary["global_reset"])

    def test_checkpoint_and_logging_recipes_use_explicit_or_temporary_artifacts(self):
        for script in ("checkpoint_resume.py", "checkpoint_resume_low_level.py"):
            for kind in ("step", "epoch"):
                with self.subTest(script=script, kind=kind):
                    result = self.run_example(f"examples/checkpointing/{script}", "--kind", kind)
                    self.assertTrue(result.stdout.strip())
        result = self.run_example("examples/checkpointing/checkpoint_resume.py",
                                  "--kind", "step", "--resume-batch-size", "4")
        self.assertEqual(json.loads(result.stdout.splitlines()[-1])["completed_epochs"], 2)
        with tempfile.TemporaryDirectory() as directory:
            self.run_example("examples/observability/logging_recipe.py", "--log-dir", directory, "--epochs", "1")
            self.assertTrue(list(Path(directory).glob("*.log")))

    def test_configured_recipes_resume_saved_settings(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            for name in ("train_regression", "train_multi_optimizer"):
                with self.subTest(recipe=name):
                    output = Path(directory, name)
                    script = f"examples/configured/{name}.py"
                    self.run_example(script, "--output", output, "--epochs", "1", "--no-progress")
                    before = torch.load(output / "last.pt", weights_only=True)
                    self.run_example(script, "--resume", output / "last.pt", "--epochs", "2", "--no-progress")
                    after = torch.load(output / "last.pt", weights_only=True)
                    self.assertEqual(after["state"]["completed_epochs"], 2)
                    self.assertGreater(after["state"]["optimizer_steps"], before["state"]["optimizer_steps"])
                    self.assertEqual(after["config"]["application"], before["config"]["application"])
                    self.assertEqual(after["config"]["training"]["learning_rate"],
                                     before["config"]["training"]["learning_rate"])

    def test_readme_quickstart_is_self_contained(self):
        text = (ROOT / "README.md").read_text()
        blocks = re.findall(r"```python\n(.*?)```", text, re.S)
        quickstart = blocks[0] + "\nassert trainer.state.completed_epochs == 2\nassert results['samples'] == 16\n"
        fp8 = next(block for block in blocks if "from deepopt import FP8Config" in block)
        scripts = {
            "quickstart": quickstart,
            "fp8": quickstart + fp8 + "\nassert report.converted_modules\n"
                   "assert trainer.state.completed_epochs == 2\n"
                   "assert torch.isfinite(torch.tensor(trainer.evaluate(validation_loader)['loss']))\n",
        }
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1"}
        for name, script in scripts.items():
            with self.subTest(recipe=name), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, "-c", script], cwd=directory, env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_progressive_cli_stages_save_resume_and_plan(self):
        import torch
        script = "examples/progressive_training.py"
        self.run_example(script, "--stage", "basic", "--epochs", "1")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "progressive")
            for stage, epochs in (("logging", "1"), ("checkpoint", "1"), ("resume", "2")):
                self.run_example(script, "--stage", stage, "--output", output, "--epochs", epochs)
            saved = torch.load(output / "checkpoints/last.pt", weights_only=True)
            self.assertEqual(saved["state"]["completed_epochs"], 2)
            self.assertTrue((output / "checkpoints/best.pt").exists())
            self.assertTrue(list((output / "logs").glob("*.log")))
            planned = Path(directory, "auto")
            self.run_example(script, "--stage", "auto", "--output", planned, "--epochs", "1",
                             "--memory-budget-bytes", "100000000")
            saved = torch.load(planned / "checkpoints/last.pt", weights_only=True)
            self.assertEqual(saved["state"]["completed_epochs"], 1)
            self.assertEqual(saved["config"]["training"]["batch_size"], 8)

    def test_documented_fp8_regular_core_recipes(self):
        text = (ROOT / "docs/fp8/README.md").read_text()
        section = text.split("## Train with the regular engine core\n", 1)[1].split("\n## ", 1)[0]
        trainer, manual, bare = re.findall(r"```python\n(.*?)```", section, re.S)
        setup, trainer_loop = trainer.split("runtime = TrainingRuntime(", 1)
        setup += "\nbefore = [parameter.detach().clone() for parameter in model.parameters()]\n"
        checks = (
            "\nassert report.converted_modules == ('0', '1', '2')\n"
            "assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in model.parameters())\n"
            "assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))\n"
        )
        recipes = {
            "trainer": "runtime = TrainingRuntime(" + trainer_loop + "\nassert trainer.state.optimizer_steps == 12\n",
            "runtime": manual + "\nassert runtime.diagnostic_state()['completed_updates'] == 12\n",
            "bare_core": bare,
        }
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1"}
        for name, loop in recipes.items():
            core = "core" if name == "bare_core" else "runtime.core"
            script = setup + loop + checks + f"\nassert type({core}).__name__ == 'TrainingCore'\n"
            with self.subTest(api=name), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run([sys.executable, "-c", script], cwd=directory, env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
