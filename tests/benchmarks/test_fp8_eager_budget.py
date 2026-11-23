"""Eager benchmark budgeting/progress contracts; fake clocks are not GPU timing."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/fp8/benchmark_fp8_eager.py"
SPEC = importlib.util.spec_from_file_location("fp8_eager_budget_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class Clock:
    def __init__(self, cost):
        self.now, self.cost, self.calls = 0., cost, 0

    def __call__(self):
        return self.now

    def operation(self):
        self.calls += 1
        self.now += self.cost

    def event(self, **kwargs):
        clock = self

        class Event:
            def record(self):
                self.timestamp = clock.now

            def elapsed_time(self, other):
                return (other.timestamp - self.timestamp) * 1000

        return Event()


class EagerBudgetTests(unittest.TestCase):
    def test_fixed_configuration_is_default_and_autotuning_is_opt_in(self):
        for options, expected in (([], False), (["--autotune"], True), (["--no-autotune"], False)):
            with self.subTest(options=options):
                args = benchmark.parser().parse_args(options)
                self.assertEqual(args.autotune, expected)
                self.assertEqual(args.time_budget, 2.)
                args.deepopt_backend = "reference"
                case = benchmark.LinearCase(args, "deepopt", (32, 16, 48), "forward", device="cpu")
                self.assertEqual(case.configuration["autotune"], expected)
                self.assertEqual(case.model.plan["fp8_training"]["autotune"], expected)

    def test_slow_calls_cap_warmup_repeats_and_iterations(self):
        args = benchmark.parser().parse_args([])
        clock = Clock(1.)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, Event=clock.event))
        with patch.object(benchmark, "perf_counter", new=clock):
            result = benchmark.measure(clock.operation, args, fake_torch)
        self.assertEqual(result["sampling"], {
            "requested_warmup": 10, "requested_iterations": 50, "requested_repeats": 5,
            "actual_warmup": 1, "actual_iterations_per_repeat": [1], "actual_repeats": 1,
            "time_budget_seconds": 2., "budget_exhausted": True, "elapsed_seconds": 2.})
        self.assertEqual(clock.calls, 2)
        self.assertEqual(result["wall_ms"]["samples"], [1000.])
        self.assertEqual(result["cuda_stream_ms"]["samples"], [1000.])

    def test_zero_budget_runs_all_requested_samples(self):
        args = benchmark.parser().parse_args(["--time-budget", "0", "--warmup", "2",
                                              "--iterations", "3", "--repeats", "4"])
        clock = Clock(.25)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, Event=clock.event))
        with patch.object(benchmark, "perf_counter", new=clock):
            result = benchmark.measure(clock.operation, args, fake_torch)
        self.assertEqual(clock.calls, 14)
        self.assertEqual(result["sampling"]["actual_warmup"], 2)
        self.assertEqual(result["sampling"]["actual_iterations_per_repeat"], [3, 3, 3, 3])
        self.assertFalse(result["sampling"]["budget_exhausted"])
        self.assertEqual(result["sampling"]["elapsed_seconds"], 3.5)
        self.assertEqual(len(result["wall_ms"]["samples"]), 4)

    def test_single_call_can_exceed_soft_target_but_is_reported(self):
        args = benchmark.parser().parse_args(["--time-budget", "0.01"])
        clock = Clock(3.)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, Event=clock.event))
        with patch.object(benchmark, "perf_counter", new=clock):
            result = benchmark.measure(clock.operation, args, fake_torch)
        self.assertEqual(result["sampling"]["actual_iterations_per_repeat"], [1])
        self.assertEqual(result["sampling"]["elapsed_seconds"], 6.)
        self.assertTrue(result["sampling"]["budget_exhausted"])

    def test_invalid_budgets_fail_before_worker_launch(self):
        for value in ("-1", "nan", "inf"):
            with self.subTest(value=value), patch.object(sys, "argv", [str(SCRIPT), "--time-budget", value]), (
                patch.object(benchmark, "run")
            ) as run, patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as failure:
                benchmark.main()
            self.assertEqual(failure.exception.code, 2)
            run.assert_not_called()

    def test_worker_reports_cold_stages_before_calls_and_checkpoints_each_stage(self):
        original_matmul, original_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", original_matmul)
        self.addCleanup(setattr, torch.backends.cudnn, "allow_tf32", original_cudnn)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "deepopt" / "result.json"
            args = benchmark.parser().parse_args(["--_worker", "deepopt", "--_checkpoint", str(checkpoint),
                "--shapes", "32x16x48", "--modes", "forward", "--warmup", "1", "--iterations", "1", "--repeats", "1"])
            progress = io.StringIO()
            stages = []

            def step():
                stages.append(json.loads(checkpoint.read_text())["cases"][0]["stage"])
                self.assertIn("first launch", progress.getvalue())

            def check():
                stages.append(json.loads(checkpoint.read_text())["cases"][0]["stage"])
                return {"errors_vs_fp32": {"output": {"finite": True}}}

            case = SimpleNamespace(model=object(), configuration={"autotune": False}, step=Mock(side_effect=step),
                                   check=Mock(side_effect=check))
            timing = {"wall_ms": {"median": 1., "samples": [1.]}, "cuda_stream_ms": {"median": 1., "samples": [1.]},
                      "sampling": {"actual_iterations_per_repeat": [1], "actual_repeats": 1}}
            with patch("sys.stderr", new=progress), patch.object(torch.cuda, "is_available", return_value=True), (
                patch.object(torch.cuda, "set_device")
            ), patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), (
                patch.object(torch.cuda, "get_device_name", return_value="fake GPU")
            ), patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "empty_cache"), (
                patch.object(benchmark, "package_info", return_value={"version": "test", "path": "test"})
            ), patch.object(benchmark, "LinearCase", return_value=case), patch.object(benchmark, "measure", return_value=timing):
                result = benchmark.worker(args)
            self.assertEqual(result["status"], "ok", result)
            self.assertEqual(stages, ["first_launch", "validation"])
            self.assertEqual(case.step.call_count, 1)
            self.assertEqual(case.check.call_count, 1)
            row = json.loads(checkpoint.read_text())["cases"][0]
            self.assertEqual(row["stage"], "complete")
            self.assertEqual(row["timing"], timing)
            self.assertIn("setup_wall_ms", row["cold_start"])
            self.assertIn("first_launch_wall_ms", row["cold_start"])

    def test_nonprofile_worker_crash_recovers_partial_case_and_inherits_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            options = ["--implementations", "deepopt", "--output", str(output)]
            args = benchmark.parser().parse_args(options)

            def launch(command, **kwargs):
                self.assertIsNone(kwargs["stderr"])
                self.assertEqual(kwargs["stdout"], subprocess.PIPE)
                self.assertNotIn("capture_output", kwargs)
                checkpoint = Path(command[command.index("--_checkpoint") + 1])
                partial = {"implementation": "deepopt", "status": "running", "cases": [
                    {"shape_m_k_n": [32, 16, 48], "mode": "forward", "status": "running", "stage": "first_launch",
                     "cold_start": {"setup_wall_ms": 1.}}]}
                benchmark.write_json(checkpoint, partial)
                return SimpleNamespace(returncode=-11, stdout="", stderr=None)

            with patch.object(sys, "argv", [str(SCRIPT), *options]), (
                patch.object(benchmark.subprocess, "run", side_effect=launch)
            ), patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                report = benchmark.run(args)
            self.assertEqual(report["results"][0]["status"], "error")
            row = report["results"][0]["cases"][0]
            self.assertEqual(row["cold_start"]["setup_wall_ms"], 1.)
            self.assertIn("first_launch", row["error"])
            self.assertEqual(report["progress"]["completed_implementations"], 1)
            self.assertIsNone(report["progress"]["active_implementation"])
            self.assertEqual(json.loads(output.read_text()), json.loads(json.dumps(report)))


if __name__ == "__main__":
    unittest.main()
