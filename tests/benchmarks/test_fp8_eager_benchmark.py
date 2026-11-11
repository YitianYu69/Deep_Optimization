"""Portable eager harness checks; CPU reference execution is not GPU validation."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/fp8/benchmark_fp8_eager.py"
spec = importlib.util.spec_from_file_location("fp8_eager_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class EagerBenchmarkTests(unittest.TestCase):
    def test_trace_attribution_uses_launch_thread_and_excludes_gpu_annotation_ranges(self):
        def event(category, name, timestamp, duration, thread, **args):
            return dict(ph="X", cat=category, name=name, ts=timestamp, dur=duration, pid=1, tid=thread, args=args)

        trace = {"traceEvents": [
            event("user_annotation", "linear.forward", 100, 200, 1),
            event("user_annotation", "deepopt.quantize.activation", 110, 30, 1),
            event("user_annotation", "deepopt.gemm.forward", 150, 40, 1),
            event("cuda_driver", "cuLaunchKernelEx", 120, 2, 1, correlation=11),
            event("cuda_driver", "cuLaunchKernelEx", 160, 2, 1, correlation=12),
            event("kernel", "quantize", 125, 5, 7, correlation=11),
            event("kernel", "gemm", 195, 10, 7, correlation=12),
            event("gpu_user_annotation", "deepopt.gemm.forward", 160, 9000, 7),
            event("user_annotation", "linear.backward", 1000, 1000, 1),
            event("user_annotation", "deepopt.gemm.weight_gradient", 1200, 100, 2),
            event("cuda_runtime", "cudaLaunchKernel", 1250, 2, 2, correlation=21),
            event("kernel", "gemm", 2500, 7, 7, correlation=21),
        ]}
        report = benchmark.trace_scope_summary(trace, 1)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["cuda_activity_sum_us_per_step"], 22)
        self.assertEqual(report["cuda_activity_count_per_step"], 3)
        scopes = {row["name"]: row for row in report["scopes"]}
        self.assertEqual(scopes["deepopt.quantize.activation"]["cuda_us_per_step"], 5)
        self.assertEqual(scopes["deepopt.gemm.forward"]["cuda_us_per_step"], 10)
        self.assertEqual(scopes["linear.forward"]["cuda_us_per_step"], 15)
        self.assertEqual(scopes["linear.backward"]["cuda_us_per_step"], 7)
        self.assertEqual(scopes["deepopt.gemm.weight_gradient"]["cuda_us_per_step"], 7)

    def test_trace_attribution_keeps_unmatched_work_and_does_not_guess_between_concurrent_phases(self):
        trace = {"traceEvents": [
            dict(ph="X", cat="user_annotation", name="linear.forward", ts=100, dur=100, pid=1, tid=1),
            dict(ph="X", cat="user_annotation", name="linear.backward", ts=100, dur=100, pid=1, tid=3),
            dict(ph="X", cat="cuda_driver", name="cuLaunchKernelEx", ts=150, dur=2, pid=1, tid=2, args={"correlation": 10}),
            dict(ph="X", cat="kernel", name="gemm", ts=300, dur=8, pid=0, tid=7, args={"correlation": 10}),
            dict(ph="X", cat="kernel", name="missing launch", ts=320, dur=4, pid=0, tid=7, args={"correlation": 20}),
        ]}
        report = benchmark.trace_scope_summary(trace, 2)
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["cuda_activity_sum_us_per_step"], 6)
        self.assertEqual(report["unattributed_us_per_step"], 6)
        self.assertTrue(all(row["cuda_us_per_step"] is None for row in report["scopes"]))

    def test_profiled_forward_preserves_no_grad_execution_and_has_no_backward_ranges(self):
        args = benchmark.parser().parse_args(["--profile-steps", "1"])
        args.deepopt_backend = "reference"
        case = benchmark.LinearCase(args, "deepopt", (32, 16, 48), "forward", device="cpu")
        expected = case.step()
        with tempfile.TemporaryDirectory() as directory:
            report = benchmark.diagnostics(case, args, Path(directory))
        self.assertEqual(report["status"], "ok", report)
        scopes = {row["name"] for row in report["torch_profiler"]["scopes"]}
        self.assertIn("linear.forward", scopes)
        self.assertNotIn("linear.backward", scopes)
        self.assertNotIn("linear.zero_grad", scopes)
        manifest = report["torch_profiler"]["deepopt_ranges"]
        self.assertEqual([row["role"] for row in manifest["first_step_gemms"]], ["forward"])
        self.assertEqual([row["role"] for row in manifest["first_step_quantization"]], ["activation", "weight"])
        actual = case.step()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in case.model.parameters()))

    def test_cpu_profiles_capture_stages_preserve_gradients_and_restore_library_aliases(self):
        from deepopt.opt.nn.autograd import linear as linear_autograd_func

        args = benchmark.parser().parse_args(["--profile", "--profile-steps", "2", "--bias"])
        args.deepopt_backend = "reference"
        aliases = {name: getattr(linear_autograd_func, name) for name in
                   ("gemm_nt", "quantize_fp8", "_quantize_tensorwise_pair", "_quantize_tensorwise_transpose")}
        for implementation in ("bf16", "deepopt"):
            with self.subTest(implementation=implementation), tempfile.TemporaryDirectory() as directory:
                case = benchmark.LinearCase(args, implementation, (32, 16, 48), "forward-backward", device="cpu")
                expected_output = case.step().detach().clone()
                expected_gradients = [case.x.grad.clone(), *(p.grad.clone() for p in case.model.parameters())]
                report = benchmark.diagnostics(case, args, Path(directory))
                self.assertEqual(report["status"], "ok", report)
                device = report["torch_profiler"]
                self.assertEqual(device["cuda_activity_status"], "not_requested")
                self.assertIsNone(device["cuda_activity_sum_us_per_step"])
                scopes = {row["name"]: row for row in device["scopes"]}
                for name in ("linear.zero_grad", "linear.forward", "linear.backward"):
                    self.assertEqual(scopes[name]["calls_per_step"], 1)
                if implementation == "deepopt":
                    for role in ("forward", "input_gradient", "weight_gradient"):
                        self.assertEqual(scopes["deepopt.gemm." + role]["calls_per_step"], 1)
                    self.assertEqual([row["role"] for row in device["deepopt_ranges"]["first_step_quantization"]],
                                     ["activation", "weight", "grad"])
                self.assertIn("traceEvents", json.loads(Path(device["trace"]).read_text()))
                self.assertGreater(Path(report["python"]["pstats"]).stat().st_size, 0)
                self.assertIn("function calls", Path(report["python"]["text"]).read_text())
                torch.testing.assert_close(case.step(), expected_output, rtol=0, atol=0)
                for actual, expected in zip([case.x.grad, *(p.grad for p in case.model.parameters())], expected_gradients):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for name, function in aliases.items():
                    self.assertIs(getattr(linear_autograd_func, name), function)

    def test_cuda_activity_summary_excludes_annotations_and_does_not_add_nested_ranges(self):
        cpu, gpu = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
        forward = SimpleNamespace(id=10, device_type=cpu, name="linear.forward", cpu_parent=None)
        gemm = SimpleNamespace(id=20, device_type=cpu, name="deepopt.gemm.forward", cpu_parent=forward)
        quantize = SimpleNamespace(id=30, device_type=cpu, name="deepopt.quantize.activation", cpu_parent=forward)
        profiler = SimpleNamespace(
            events=lambda: [forward, gemm, quantize,
                            SimpleNamespace(device_type=gpu, name="gemm", device_time_total=200, linked_correlation_id=20),
                            SimpleNamespace(device_type=gpu, name="quantize", device_time_total=40, linked_correlation_id=30),
                            SimpleNamespace(device_type=gpu, name="deepopt.gemm.forward", device_time_total=9000,
                                            linked_correlation_id=20, is_user_annotation=True)],
            key_averages=lambda: [SimpleNamespace(device_type=cpu, key="linear.forward", count=2,
                                                  self_cpu_time_total=60, cpu_time_total=600, device_time_total=9240),
                                  SimpleNamespace(device_type=cpu, key="deepopt.gemm.forward", count=2,
                                                  self_cpu_time_total=40, cpu_time_total=400, device_time_total=9200)],
        )
        report = benchmark.profile_summary(profiler, torch, 2)
        self.assertEqual(report["cuda_activity_sum_us_per_step"], 120)
        self.assertEqual(report["cuda_activity_count_per_step"], 1)
        self.assertEqual([row["name"] for row in report["cuda_activities"]], ["gemm", "quantize"])
        self.assertEqual(report["scopes"][0]["cuda_us_per_step"], 120)
        self.assertEqual(report["scopes"][1]["cuda_us_per_step"], 100)
        self.assertEqual(report["cuda_scope_attribution_status"], "complete")
        profiler.events = lambda: [SimpleNamespace(device_type=gpu, name="gemm", device_time_total=200)]
        report = benchmark.profile_summary(profiler, torch, 2)
        self.assertEqual(report["cuda_activity_sum_us_per_step"], 100)
        self.assertEqual(report["cuda_scope_attribution_status"], "unavailable")
        self.assertTrue(all(row["cuda_us_per_step"] is None for row in report["scopes"]))

    def test_diagnostic_failure_keeps_the_independent_python_profile(self):
        args = benchmark.parser().parse_args(["--profile-steps", "1"])
        case = benchmark.LinearCase(args, "bf16", (32, 16, 48), "forward-backward", device="cpu")
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(benchmark, "cuda_profile", side_effect=RuntimeError("CUPTI unavailable")):
            report = benchmark.diagnostics(case, args, Path(directory))
            self.assertEqual(report["status"], "error")
            self.assertIn("CUPTI unavailable", report["torch_profiler"]["error"])
            self.assertEqual(report["python"]["status"], "ok")
            self.assertTrue(Path(report["python"]["pstats"]).is_file())

    def test_temporary_installed_library_ranges_are_restored_after_an_exception(self):
        from deepopt.opt.nn.autograd import linear as linear_autograd_func

        args = benchmark.parser().parse_args([])
        args.deepopt_backend = "reference"
        case = benchmark.LinearCase(args, "deepopt", (32, 16, 48), "forward-backward", device="cpu")
        original = linear_autograd_func.gemm_nt
        with self.assertRaisesRegex(RuntimeError, "diagnostic failed"):
            with benchmark.deepopt_ranges(case):
                self.assertIsNot(linear_autograd_func.gemm_nt, original)
                raise RuntimeError("diagnostic failed")
        self.assertIs(linear_autograd_func.gemm_nt, original)
        self.assertTrue(all(row["finite"] for row in case.check()["errors_vs_fp32"].values()))

    def test_profile_bundle_recovers_checkpointed_timing_after_worker_crash_and_excludes_old_runs(self):
        def launch(command, **kwargs):
            implementation = command[command.index("--_worker") + 1]
            root = Path(command[-1])
            result = {"implementation": implementation, "status": "ok", "cases": []}
            if implementation == "deepopt":
                result["cases"].append({"shape_m_k_n": [32, 16, 48], "mode": "forward-backward", "status": "ok",
                                        "timing": {"wall_ms": {"median": .4}, "cuda_stream_ms": {"median": .39}}})
                benchmark.write_json(root / implementation / "result.json", result)
                return SimpleNamespace(returncode=-11, stdout="", stderr="profiler crashed\n")
            return SimpleNamespace(returncode=0, stdout=benchmark.RESULT_PREFIX + json.dumps(result), stderr="")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "profiles"
            parent.mkdir()
            (parent / "old.trace.json").write_text("stale")
            output = root / "report.json"
            arguments = [str(SCRIPT), "--profile", "--profile-dir", str(parent), "--output", str(output)]
            with patch.object(sys, "argv", arguments), patch.object(benchmark.subprocess, "run", side_effect=launch), \
                 patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(benchmark.main(), 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["results"][1]["status"], "error")
            self.assertEqual(report["results"][1]["cases"][0]["timing"]["wall_ms"]["median"], .4)
            with zipfile.ZipFile(report["profile_archive"]) as archive:
                self.assertIn("summary.json", archive.namelist())
                self.assertIn("benchmark.py", archive.namelist())
                self.assertIn("profiles/deepopt/result.json", archive.namelist())
                self.assertFalse(any("old.trace" in name for name in archive.namelist()))

    def test_native_selection_preserves_triton_forward_and_uses_the_supported_backward_policy(self):
        for backend in ("native_fast", "native_precise"):
            with self.subTest(backend=backend):
                args = benchmark.parser().parse_args(["--deepopt-backward-backend", backend])
                case = benchmark.LinearCase(args, "deepopt", (32, 16, 48), "forward-backward", device="cpu")
                self.assertEqual(case.configuration["backend"], "triton")
                self.assertEqual(case.configuration["backward_backend"], backend)
                self.assertEqual(case.model.plan["fp8_training"]["backward_backend"], backend)
                with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as failure:
                    benchmark.parser().parse_args(["--deepopt-backend", backend])
                self.assertEqual(failure.exception.code, 2)

    def test_forward_does_not_build_a_graph_or_gradients(self):
        args = benchmark.parser().parse_args([])
        for implementation in ("bf16", "deepopt"):
            with self.subTest(implementation=implementation):
                args.deepopt_backend = "reference"
                case = benchmark.LinearCase(args, implementation, (32, 16, 48), "forward", device="cpu")
                output = case.step()
                self.assertEqual(output.shape, (32, 48))
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertFalse(output.requires_grad)
                self.assertFalse(case.model.training)
                self.assertTrue(all(parameter.grad is None for parameter in case.model.parameters()))

    def test_training_uses_identical_data_and_resets_all_gradients_between_calls(self):
        args = benchmark.parser().parse_args(["--bias"])
        baseline = benchmark.LinearCase(args, "bf16", (32, 16, 48), "forward-backward", device="cpu")
        for fp8_format in ("e4m3fn", "hybrid"):
            with self.subTest(format=fp8_format):
                args.format, args.deepopt_backend = fp8_format, "reference"
                case = benchmark.LinearCase(args, "deepopt", (32, 16, 48), "forward-backward", device="cpu")
                for value, expected in ((case.x, baseline.x), (case.grad, baseline.grad),
                                        (case.source.weight, baseline.source.weight), (case.source.bias, baseline.source.bias)):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                initial_weight = case.source.weight.detach().clone()
                case.step()
                first = [case.x.grad.clone(), *(parameter.grad.clone() for parameter in case.model.parameters())]
                case.step()
                for actual, expected in zip([case.x.grad, *(parameter.grad for parameter in case.model.parameters())], first):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(case.source.weight, initial_weight, rtol=0, atol=0)
                report = case.check()
                self.assertEqual(report["gradient_dtypes"], {"input": "torch.bfloat16", "weight": "torch.float32"})
                for error in report["errors_vs_fp32"].values():
                    self.assertTrue(error["finite"])
                    self.assertLess(error["relative_l2"], 0.15)

    def test_weight_cache_is_disabled_during_calls_and_ambient_policy_is_restored(self):
        args = benchmark.parser().parse_args([])
        args.deepopt_backend = "reference"
        case = benchmark.LinearCase(args, "deepopt", (32, 16, 48), "forward-backward", device="cpu")
        observed = []
        handle = case.model.register_forward_pre_hook(lambda *unused: observed.append(torch.is_autocast_cache_enabled()))
        self.addCleanup(handle.remove)
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=True):
            case.step()
            case.step()
            self.assertTrue(torch.is_autocast_cache_enabled())
        self.assertEqual(observed, [False, False])

    def test_worker_failure_does_not_drop_other_libraries_or_the_json_report(self):
        calls = []

        def launch(command, **kwargs):
            calls.append((command, kwargs))
            implementation = command[-1]
            result = {"implementation": implementation, "status": "ok", "cases": []}
            if implementation == "torchao":
                return SimpleNamespace(returncode=1, stdout="", stderr="incompatible torchao extension\n")
            return SimpleNamespace(returncode=0, stdout=benchmark.RESULT_PREFIX + json.dumps(result), stderr="")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch.object(sys, "argv", [str(SCRIPT), "--output", str(output)]), \
                 patch.object(benchmark.subprocess, "run", side_effect=launch), \
                 patch("sys.stdout", new=io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(benchmark.main(), 1)
            report = json.loads(output.read_text())
        self.assertEqual([row["implementation"] for row in report["results"]], list(benchmark.IMPLEMENTATIONS))
        self.assertEqual([row["status"] for row in report["results"]], ["ok", "ok", "error", "ok"])
        self.assertIn("incompatible torchao", report["results"][2]["error"])
        for command, kwargs in calls:
            self.assertEqual(command[0], sys.executable)
            self.assertNotIn("cwd", kwargs)
            self.assertNotIn("env", kwargs)

    def test_copied_script_runs_with_installed_deepopt_from_an_unrelated_directory(self):
        code = """
import runpy
import sys
scope = runpy.run_path(sys.argv[1], run_name="portable_benchmark")
args = scope["parser"]().parse_args([])
args.deepopt_backend = "reference"
case = scope["LinearCase"](args, "deepopt", (32, 16, 48), "forward-backward", device="cpu")
assert all(error["finite"] for error in case.check()["errors_vs_fp32"].values())
from pathlib import Path
args.profile_steps = 1
diagnostics = scope["diagnostics"](case, args, Path(sys.argv[1]).parent / "profiles")
assert diagnostics["status"] == "ok", diagnostics
assert Path(diagnostics["torch_profiler"]["trace"]).is_file()
print("installed library check passed")
"""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory, "benchmark.py")
            target.write_text(SCRIPT.read_text())
            completed = subprocess.run([sys.executable, "-I", "-c", code, str(target)],
                                       cwd=directory, capture_output=True, text=True, timeout=60)
            help_result = subprocess.run([sys.executable, "-I", str(target), "--help"],
                                         cwd=directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("installed library check passed", completed.stdout)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--implementations", help_result.stdout)

    def test_cli_rejects_shapes_that_cannot_use_all_three_fp8_paths(self):
        for shape in ("0x16x16", "16x10x16", "16x16", "-16x16x16", "16x16xtext"):
            with self.subTest(shape=shape), patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    benchmark.parser().parse_args(["--shapes=" + shape])
                self.assertEqual(failure.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
