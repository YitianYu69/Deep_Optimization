"""Training experiment controls and reports are testable without CUDA."""

from contextlib import nullcontext
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from deepopt import TrainingRuntime
from deepopt.engine.training.ema import ExponentialMovingAverage


spec = importlib.util.spec_from_file_location(
    "fp8_training_benchmark", Path(__file__).resolve().parents[2] / "benchmarks/fp8/benchmark_fp8_training.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class TrainingBenchmarkTests(unittest.TestCase):
    def test_default_comparison_selects_delayed_and_exposes_its_recipe(self):
        args = benchmark._parser().parse_args(["--history-length", "17", "--amax-algorithm", "most_recent", "--margin", "1"])
        self.assertEqual(args.cases, ["bf16-runtime", "fp8-prepared", "fp8-delayed"])
        policy = benchmark._fp8_policy(args, benchmark.CASES["fp8-delayed"])
        self.assertEqual(policy["scaling_recipe"], "delayed")
        self.assertEqual(policy["delayed_scaling"], {"history_length": 17, "amax_algorithm": "most_recent", "margin": 1, "reduce_amax": True})
        current = benchmark._fp8_policy(args, benchmark.CASES["fp8-prepared"])
        self.assertNotIn("scaling_recipe", current)
        self.assertTrue(all(value is None for value in policy["block_shapes"].values()))

    def test_fingerprint_includes_delayed_numerical_metadata(self):
        from deepopt import FP8Config
        from deepopt.opt.policy import prepare_fp8_model

        model, _ = prepare_fp8_model(nn.Linear(5, 3), FP8Config(enabled=True, scaling_recipe="delayed"))
        before = benchmark._state_fingerprint(model, torch)
        self.assertEqual(before, benchmark._state_fingerprint(copy.deepcopy(model), torch))
        model.fp8_scaling.history[0, 0] = 7
        self.assertNotEqual(before, benchmark._state_fingerprint(model, torch))

    def test_help_does_not_import_optional_libraries(self):
        original = __import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("torch", "timm", "deepopt", "torchmetrics"):
                raise AssertionError("CLI must not import " + name)
            return original(name, *args, **kwargs)

        with patch.object(sys, "argv", ["benchmark", "--help"]), patch("builtins.__import__", new=guarded), \
             patch("sys.stdout", new=io.StringIO()), self.assertRaises(SystemExit) as raised:
            benchmark.main()
        self.assertEqual(raised.exception.code, 0)

    def test_suite_persists_partial_reports_and_uses_current_interpreter(self):
        with tempfile.TemporaryDirectory() as directory:
            args = benchmark._parser().parse_args(["--output", directory, "--cases", "bf16-runtime", "fp8-prepared",
                                                  "--no-fill-uninitialized-memory"])
            calls = []

            def worker(command, log_path):
                calls.append(command)
                payload = json.loads(command[-1])
                self.assertFalse(payload["settings"]["fill_uninitialized_memory"])
                self.assertTrue(payload["settings"]["deterministic"])
                previous = json.loads((Path(directory) / "summary.json").read_text())
                self.assertEqual(len(previous["results"]), len(calls) - 1)
                result = {"status": "ok", "run_id": payload["run_id"], "case": payload["case"],
                          "batch_size": payload["batch_size"], "initial_state_sha256": "same",
                          "timing": {"wall_ms_per_update": {"median": 20 if len(calls) == 1 else 10}}}
                benchmark._write_json(Path(payload["directory"]) / "result.json", result)
                return 0

            before = list(sys.path)
            with patch.object(benchmark, "_run_child", side_effect=worker), \
                 patch("sys.stderr", new=io.StringIO()), patch("sys.stdout", new=io.StringIO()):
                report = benchmark._run(args)
            self.assertEqual(before, sys.path)
            self.assertEqual(report, json.loads((Path(directory) / "summary.json").read_text()))
            self.assertEqual(report["comparisons"][0]["speedup"], 2)
            for command in calls:
                self.assertEqual(command[0], sys.executable)
                self.assertNotIn("--source", command)

    def test_old_reports_are_not_accepted_for_failed_or_empty_workers(self):
        for returncode in (0, 9):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as directory:
                args = benchmark._parser().parse_args(["--output", directory, "--cases", "bf16-native"])
                benchmark._write_json(Path(directory) / "bf16-native-b512" / "result.json",
                                      {"status": "ok", "run_id": "old"})
                with patch.object(benchmark, "_run_child", return_value=returncode), \
                     patch("sys.stderr", new=io.StringIO()), patch("sys.stdout", new=io.StringIO()):
                    result = benchmark._run(args)
                self.assertEqual(result["results"][0]["status"], "error")

    def test_mismatched_initial_weights_do_not_produce_a_speedup_claim(self):
        rows = [dict(status="ok", case=name, batch_size=512, initial_state_sha256=name,
                     timing={"wall_ms_per_update": {"median": 2.0}})
                for name in ("bf16-runtime", "fp8-prepared")]
        comparison = benchmark._comparisons(rows)[0]
        self.assertFalse(comparison["same_initial_state"])
        self.assertIsNone(comparison["speedup"])

    def test_nonfinite_sampled_loss_does_not_produce_a_speedup_claim(self):
        rows = [dict(status="ok", case=name, batch_size=512, initial_state_sha256="same",
                     timing={"wall_ms_per_update": {"median": 2.0}, "finite_sampled_losses": finite})
                for name, finite in (("bf16-native", True), ("bf16-runtime", False))]
        comparison = benchmark._comparisons(rows)[0]
        self.assertFalse(comparison["finite_sampled_losses"])
        self.assertIsNone(comparison["speedup"])

    def test_stats_retain_distribution_and_percentile(self):
        result = benchmark._stats([5, 1, 3, 2, 4])
        self.assertEqual(result["median"], 3)
        self.assertEqual(result["p95"], 5)
        self.assertEqual(result["samples"], [5, 1, 3, 2, 4])

    def test_allocation_filling_can_be_disabled_without_disabling_deterministic_algorithms(self):
        args = benchmark._parser().parse_args(["--no-fill-uninitialized-memory"])
        self.assertTrue(args.deterministic)
        self.assertFalse(args.fill_uninitialized_memory)
        fake = SimpleNamespace(use_deterministic_algorithms=Mock(),
                               utils=SimpleNamespace(deterministic=SimpleNamespace(fill_uninitialized_memory=True)),
                               backends=SimpleNamespace(cudnn=SimpleNamespace(), cuda=SimpleNamespace(matmul=SimpleNamespace())))
        benchmark._configure_determinism(args, fake)
        fake.use_deterministic_algorithms.assert_called_once_with(True)
        self.assertFalse(fake.utils.deterministic.fill_uninitialized_memory)
        self.assertTrue(fake.backends.cudnn.deterministic)

    def test_component_recorder_does_not_assign_cuda_time_to_loader_fetch(self):
        event = Mock()
        event.elapsed_time.return_value = 4.0
        fake = SimpleNamespace(cuda=SimpleNamespace(Event=Mock(return_value=event)))
        recorder = benchmark.PhaseRecorder(fake)
        with patch.object(benchmark.time, "perf_counter", side_effect=[0, .002, 1, 1.003]):
            with recorder.phase("fetch", cuda=False):
                pass
            with recorder.phase("backward"):
                pass
        result = recorder.summary()
        self.assertIsNone(result["fetch"]["cuda_stream_ms"])
        self.assertAlmostEqual(result["fetch"]["host_submission_ms"]["median"], 2)
        self.assertEqual(result["backward"]["cuda_stream_ms"]["median"], 4)
        self.assertEqual(fake.cuda.Event.call_count, 2)

    def test_profile_normalizes_cuda_activity_without_adding_nested_cpu_time(self):
        cpu, gpu = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
        profiler = SimpleNamespace(
            key_averages=lambda: [SimpleNamespace(device_type=cpu, key="aten::linear", count=2,
                                                  self_cpu_time_total=1000, cpu_time_total=9000)],
            events=lambda: [SimpleNamespace(device_type=gpu, name="gemm", device_time_total=2000),
                            SimpleNamespace(device_type=gpu, name="gemm", device_time_total=4000),
                            SimpleNamespace(device_type=gpu, name="experiment.forward_loss",
                                            device_time_total=99000, is_user_annotation=True)])
        result = benchmark._profile_summary(profiler, torch, steps=2, limit=1)
        self.assertEqual(result["cuda_activity_sum_ms_per_update"], 3)
        self.assertEqual(result["top_cuda_activities"][0]["count"], 2)
        self.assertEqual(result["top_cpu_ops"][0]["self_cpu_ms_per_update"], .5)
        profiler.events = lambda: []
        self.assertIsNone(benchmark._profile_summary(profiler, torch, 2, 1)["cuda_activity_sum_ms_per_update"])

    def _session(self, runtime):
        torch.manual_seed(123)
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        session = benchmark.TrainingSession.__new__(benchmark.TrainingSession)
        session.torch, session.device = torch, torch.device("cpu")
        session.model = model
        session.optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        session.runtime = (TrainingRuntime(model, session.optimizer, device="cpu", precision="fp32",
                                           fp8=False, execution_mode="max_speed") if runtime else None)
        session.metric = Mock()
        session.source = SimpleNamespace(next=lambda: (torch.randn(5, 3), torch.tensor([0, 1, 0, 1, 0])))
        session.ema = ExponentialMovingAverage(model, .95)
        session.last_loss = session.recorder = None
        session.annotate = False
        return session

    def test_training_step_really_updates_weights_and_ema_for_native_and_runtime(self):
        for runtime in (False, True):
            with self.subTest(runtime=runtime):
                session = self._session(runtime)
                before = copy.deepcopy(session.model.state_dict())
                with patch.object(torch, "autocast", return_value=nullcontext()):
                    session.step()
                self.assertTrue(any(not torch.equal(before[name], value)
                                    for name, value in session.model.state_dict().items()))
                self.assertTrue(all(parameter.grad is None for parameter in session.model.parameters()))
                self.assertEqual(session.ema.updates, 1)
                session.metric.update.assert_called_once()
                self.assertFalse(session.last_loss.requires_grad)
                if runtime:
                    self.assertEqual(session.runtime.mode_controller.completed_updates, 1)

    def test_coverage_hooks_preserve_updates_and_are_removed(self):
        session = self._session(runtime=False)
        with patch.object(torch, "autocast", return_value=nullcontext()), patch.object(torch.cuda, "synchronize"):
            result = benchmark._coverage(session, torch)
        self.assertEqual([row["module"] for row in result["module_calls"]], ["0", "2"])
        self.assertTrue(result["saved_storage_bytes_by_kind_dtype"])
        self.assertTrue(all(not module._forward_pre_hooks for module in session.model.modules()))
        json.dumps(result, allow_nan=False)

    def test_observer_captures_actual_fp8_projections_without_changing_backward(self):
        session = self._session(runtime=False)
        session.runtime = TrainingRuntime(session.model, session.optimizer, device="cpu", precision="fp32",
                                          fp8=True, execution_mode="max_speed")
        session.model = session.runtime.model
        session.ema = ExponentialMovingAverage(session.model, .95)
        with patch.object(torch.cuda, "synchronize"):
            result = benchmark._coverage(session, torch)
        self.assertIsNone(result["projection_observation_error"])
        self.assertEqual([row["weight"] for row in result["fp8_projection_calls"]], ["0.weight", "2.weight"])
        self.assertEqual(session.runtime.mode_controller.completed_updates, 1)
        self.assertTrue(all(parameter.grad is None for parameter in session.model.parameters()))

    def test_interception_ablation_rejects_different_fp8_coverage(self):
        rows = [dict(status="ok", case=name, batch_size=512, initial_state_sha256="same",
                     timing={"wall_ms_per_update": {"median": 2.0}},
                     coverage={"fp8_projection_calls": [{"weight": weight}]})
                for name, weight in (("fp8-intercept", "head.weight"), ("fp8-prepared", "fc1.weight"))]
        result = benchmark._comparisons(rows)[0]
        self.assertFalse(result["same_fp8_projection_calls"])
        self.assertIsNone(result["speedup"])

    def test_initial_state_fingerprint_detects_parameter_mutations(self):
        model = nn.Linear(3, 2)
        before = benchmark._state_fingerprint(model, torch)
        self.assertEqual(before, benchmark._state_fingerprint(copy.deepcopy(model), torch))
        with torch.no_grad():
            model.weight.add_(1)
        self.assertNotEqual(before, benchmark._state_fingerprint(model, torch))

    def test_real_cpu_constructor_runs_with_synthetic_data_and_library_ema(self):
        args = benchmark._parser().parse_args(["--device", "cpu", "--image-size", "2"])
        factory = SimpleNamespace(create_model=lambda *args, **kwargs: nn.Sequential(nn.Flatten(), nn.Linear(12, 10)))
        with patch.dict(sys.modules, {"timm": factory}):
            session = benchmark.TrainingSession(args, benchmark.CASES["bf16-native"], 2, torch)
        before = benchmark._state_fingerprint(session.model, torch)
        with patch.object(torch, "autocast", return_value=nullcontext()):
            session.step()
        self.assertNotEqual(before, benchmark._state_fingerprint(session.model, torch))
        self.assertEqual(session.ema.updates, 1)
        self.assertTrue(torch.isfinite(session.metric.compute()))

    def test_scheduled_profile_exports_trace_after_its_separate_warmup(self):
        session = self._session(runtime=False)
        real_profile = torch.profiler.profile
        adapter = SimpleNamespace(ProfilerActivity=torch.profiler.ProfilerActivity, schedule=torch.profiler.schedule,
                                  profile=lambda **kwargs: real_profile(**{
                                      **kwargs, "activities": [torch.profiler.ProfilerActivity.CPU]}))
        proxy = SimpleNamespace(profiler=adapter, cuda=SimpleNamespace(synchronize=Mock()), autograd=torch.autograd)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(torch, "autocast", return_value=nullcontext()):
            result = benchmark._profile(session, SimpleNamespace(profile_steps=2, profile_rows=10, trace=True),
                                        proxy, Path(directory))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["profiled_updates"], 2)
            self.assertEqual(result["cuda_activity_status"], "unavailable")
            self.assertTrue((Path(directory) / "training-trace.json").is_file())
        self.assertEqual(session.ema.updates, 3)
        self.assertFalse(session.annotate)

    def test_failure_after_timing_preserves_completed_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {"case": "bf16-native", "batch_size": 512, "directory": directory, "run_id": "current"}

            def fail(payload):
                benchmark._write_json(Path(directory) / "result.json",
                                      {"status": "running", "run_id": "current", "timing": {"test": 123}})
                raise RuntimeError("late measurement failed")

            with patch.object(sys, "argv", ["benchmark", "--_worker", json.dumps(payload)]), \
                 patch.object(benchmark, "_worker", side_effect=fail), self.assertRaisesRegex(RuntimeError, "late"):
                benchmark.main()
            result = json.loads((Path(directory) / "result.json").read_text())
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["timing"], {"test": 123})
            self.assertIn("late measurement failed", result["traceback"])

    def test_cifar_requires_data_without_starting_an_experiment(self):
        with patch.object(sys, "argv", ["benchmark", "--input-mode", "cifar10"]), \
             patch("sys.stderr", new=io.StringIO()), patch.object(benchmark, "_run") as run, \
             self.assertRaises(SystemExit) as raised:
            benchmark.main()
        self.assertEqual(raised.exception.code, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
