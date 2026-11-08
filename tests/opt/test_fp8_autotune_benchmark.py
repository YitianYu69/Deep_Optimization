"""Benchmark operands follow production autograd; search diagnostics retain each key."""

import importlib.util
from contextlib import nullcontext
import json
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from deepopt.config import FP8Config, fp8_config_from_dict
from deepopt.opt.backends import gemm_nt
from deepopt.opt.do_nn.functional import do_linear, validate_fp8_plan
from deepopt.opt.do_nn.AOTAutograd_Func import linear_autograd_func


spec = importlib.util.spec_from_file_location(
    "fp8_autotune_benchmark", Path(__file__).resolve().parents[2] / "benchmarks/benchmark_fp8_autotune.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class FP8AutotuneBenchmarkTests(unittest.TestCase):
    def test_conversion_and_bias_scopes_measure_actual_operations(self):
        args = SimpleNamespace(batches=[3], input_features=7, output_features=5, implementation="fp8",
                               scaling="tensorwise", scale_k=32, num_stages="auto", scope="quantize-pair")
        measured = []

        def measure(operation, *unused, **kwargs):
            measured.append(operation())
            return {}

        with patch.object(benchmark, "_measure", new=measure):
            rows = benchmark._measure_batches(args, torch, "cpu", benchmark._benchmark_plan(args, torch), do_linear, gemm_nt)
            self.assertEqual(len(rows), 3)
            for normal, transposed in measured:
                torch.testing.assert_close(normal.data.T.float(), transposed.data.float(), rtol=0, atol=0)
            measured.clear()
            args.scope = "bias-gradient"
            rows = benchmark._measure_batches(args, torch, "cpu", {}, do_linear, gemm_nt)
        self.assertEqual(rows[0]["operation"], "bias_gradient")
        self.assertEqual(measured[0].shape, (5,))
        self.assertEqual(measured[0].dtype, torch.float32)

    def test_fixed_tile_is_validated_before_launch_and_restored_after_scope(self):
        from unittest.mock import Mock
        from deepopt.opt.backends import _triton_autotune as policy

        args = dict.fromkeys(policy.AUTOTUNE_KEY, 1)
        args.update(M=1024, N=1024, K=512, NATIVE_FP8=True, TENSORWISE=True,
                    A_BLOCK_ROWS=1024, B_BLOCK_ROWS=1024, A_BLOCK_COLS=512, B_BLOCK_COLS=512,
                    PIPELINE_STAGES=0, NUM_SMS=132)
        original, configs, cache = Mock(), [], {}
        kernel = SimpleNamespace(arg_names=list(args), run=original, configs=configs, cache=cache)
        triton = SimpleNamespace(Config=lambda kwargs, **options: SimpleNamespace(kwargs=kwargs, **options))
        modules = {"triton": triton, "deepopt.opt.backends.triton_fp8": SimpleNamespace(_gemm_nt_kernel=kernel)}
        with patch.dict(sys.modules, modules), patch.object(policy, "prune_gemm_configs", wraps=policy.prune_gemm_configs) as prune:
            with benchmark._fixed_gemm_config((128, 128, 64, 8, 3, 1, 1)):
                kernel.run(**args)
                kernel.run(**args)
            self.assertEqual(prune.call_count, 1)
            self.assertEqual(original.call_count, 2)
            self.assertIs(kernel.configs, configs)
            self.assertIs(kernel.cache, cache)
            self.assertIs(kernel.run, original)
            with benchmark._fixed_gemm_config((128, 128, 128, 8, 3, 1, 1)), \
                 self.assertRaisesRegex(ValueError, "No Triton autotune"):
                kernel.run(**{**args, "A_BLOCK_COLS": 96})
        self.assertEqual(original.call_count, 2)

    def test_host_submission_profile_is_separate_and_cleans_gradients(self):
        from unittest.mock import Mock

        prepare, operation = Mock(), Mock()
        result = benchmark._host_profile(operation, SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock())),
                                         "cuda:0", SimpleNamespace(profile_iterations=2, profile_rows=5), prepare)
        self.assertEqual(operation.call_count, 2)
        self.assertEqual(prepare.call_count, 2)
        self.assertLessEqual(len(result["top_functions"]), 5)
        json.dumps(result, allow_nan=False)

    def test_fp64_diagnostics_separate_quantization_from_accumulation_in_chunks(self):
        from deepopt.opt.quantization.quantizer import quantize_fp8

        left, right = torch.randn(7, 13), torch.randn(5, 13)
        for block_shape in (None, (2, 4)):
            a, b = (quantize_fp8(value, block_shape=block_shape) for value in (left, right))
            bias = torch.randn(5)
            # Match the oracle's direct FP64 scale application, avoiding a
            # dequantize-to-FP32 rounding step in the reference.
            aq = benchmark._dequantized_slice(a, slice(0, 7), 0, 13, torch)
            bq = benchmark._dequantized_slice(b, slice(0, 5), 0, 13, torch)
            expected = aq @ bq.T + bias.double()
            result = benchmark._numerical_breakdown(expected, a, b, bias, (left, right, bias), torch,
                                                    chunk_rows=3, chunk_k=4)
            self.assertLess(result["accumulation_and_output_cast"]["max_abs"], 1e-12)
            self.assertGreater(result["quantization"]["relative_l2"], 0)
            result = benchmark._numerical_breakdown(expected + 1, a, b, bias, (left, right, bias), torch,
                                                    chunk_rows=3, chunk_k=4)
            self.assertAlmostEqual(result["accumulation_and_output_cast"]["max_abs"], 1, places=12)
            json.dumps(result, allow_nan=False)

    def test_vit_default_covers_training_and_validation_tails_and_rejects_removed_flag(self):
        with patch.object(sys, "argv", ["benchmark", "--preset", "vit"]), \
             patch.object(benchmark, "_run", return_value={}) as run, patch("sys.stdout", new=io.StringIO()):
            benchmark.main()
        self.assertEqual(run.call_args.args[0].batches, [512, 336, 272])
        with patch.object(sys, "argv", ["benchmark", "--max-num-imprecise-acc", "32"]), \
             patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            benchmark.main()

    def test_numerical_input_patterns_cover_zero_magnitudes_and_cancellation(self):
        for pattern in ("random", "zeros", "mixed-magnitude", "cancellation"):
            value = benchmark._input_matrix(7, 13, torch=torch, device="cpu", dtype=torch.bfloat16,
                                            pattern=pattern, role="activation")
            self.assertEqual(value.shape, (7, 13))
            self.assertEqual(value.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(value).all())
            if pattern == "zeros":
                self.assertEqual(value.count_nonzero(), 0)
            if pattern == "cancellation":
                torch.testing.assert_close(value[:, :12:2], -value[:, 1:12:2])

    def test_event_timings_are_normalized_per_operation(self):
        from unittest.mock import Mock

        event = Mock()
        event.elapsed_time.side_effect = [12.0, 18.0]
        cuda = SimpleNamespace(device=lambda device: nullcontext(), Event=lambda **kwargs: event,
                               synchronize=Mock())
        operation = Mock()
        result = benchmark._event_times(operation, SimpleNamespace(cuda=cuda), "cuda:0",
                                        SimpleNamespace(repeats=2, iterations=3))
        self.assertEqual(result, {"median_ms": 5.0, "samples_ms": [4.0, 6.0]})
        self.assertEqual(operation.call_count, 6)
        self.assertEqual(event.synchronize.call_count, 2)

    def test_memory_measures_extra_peak_above_clean_baseline_and_cleans_up(self):
        from unittest.mock import Mock

        order = []
        cuda = SimpleNamespace(synchronize=Mock(), memory_allocated=Mock(side_effect=[100, 160]),
                               memory_reserved=Mock(return_value=200), max_memory_allocated=Mock(return_value=180),
                               max_memory_reserved=Mock(return_value=260), reset_peak_memory_stats=Mock())
        result = benchmark._memory_measurement(lambda: order.append("operation"), SimpleNamespace(cuda=cuda),
                                               "cuda:0", lambda: order.append("prepare"))
        self.assertEqual(order, ["prepare", "operation", "prepare"])
        self.assertEqual(result["extra_peak_allocated_bytes"], 80)
        self.assertEqual(result["extra_peak_reserved_bytes"], 60)
        self.assertEqual(result["allocated_with_output_and_gradients_bytes"], 160)
        cuda.reset_peak_memory_stats.assert_called_once_with("cuda:0")

    def test_profiler_keeps_cpu_and_cuda_activity_totals_separate(self):
        cpu, gpu = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
        profiler = SimpleNamespace(
            key_averages=lambda: [
                SimpleNamespace(device_type=cpu, key="aten::linear", count=2,
                                self_cpu_time_total=2000, cpu_time_total=8000),
                SimpleNamespace(device_type=cpu, key="cudaLaunchKernel", count=4,
                                self_cpu_time_total=3000, cpu_time_total=3000)],
            events=lambda: [SimpleNamespace(device_type=gpu, name="gemm", device_time_total=1000),
                            SimpleNamespace(device_type=gpu, name="gemm", device_time_total=2000),
                            SimpleNamespace(device_type=gpu, name="copy", device_time_total=1000),
                            SimpleNamespace(device_type=cpu, name="aten::linear", device_time_total=9000)])
        result = benchmark._profile_summary(profiler, torch, iterations=2, limit=1)
        self.assertEqual(result["top_cuda_activities"], [{"name": "gemm", "count": 2, "total_ms_per_call": 1.5}])
        self.assertEqual(result["cuda_activity_sum_ms_per_call"], 2.0)
        self.assertEqual(result["cuda_activity_count_per_call"], 1.5)
        self.assertEqual(result["top_cpu_ops"][0]["name"], "cudaLaunchKernel")
        self.assertEqual(result["cuda_runtime_ops"][0]["count"], 4)
        json.dumps(result, allow_nan=False)

    def test_profile_summary_accepts_real_cpu_profiler_and_marks_missing_cuda(self):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            torch.ones(3, 3) @ torch.ones(3, 3)
        result = benchmark._profile_summary(profiler, torch, iterations=1, limit=20)
        self.assertTrue(any(row["name"] == "aten::mm" for row in result["top_cpu_ops"]))
        self.assertEqual(result["cuda_activity_status"], "unavailable")
        self.assertIsNone(result["cuda_activity_sum_ms_per_call"])

    def test_profiler_failure_is_reported_and_cleans_gradients(self):
        from unittest.mock import Mock

        prepare = Mock()
        profiler = SimpleNamespace(ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
                                   profile=Mock(side_effect=RuntimeError("CUPTI unavailable")))
        fake_torch = SimpleNamespace(profiler=profiler, cuda=SimpleNamespace(synchronize=Mock()))
        result = benchmark._profile_measurement(Mock(), fake_torch, "cuda:0",
                                                SimpleNamespace(profile_iterations=2), prepare)
        self.assertEqual(result, {"status": "unavailable", "error": "CUPTI unavailable"})
        self.assertEqual(prepare.call_count, 2)

    def test_detailed_measurements_follow_timing_and_precede_graph_capture(self):
        from unittest.mock import Mock

        args = SimpleNamespace(warmups=1, iterations=3, repeats=2, detailed=True, scope="gemm")
        order = []
        operation, prepare = Mock(), Mock()
        with patch.object(benchmark, "_event_times", side_effect=lambda *args: order.append("events") or {}), \
             patch.object(benchmark, "_memory_measurement", side_effect=lambda *args: order.append("memory") or {}), \
             patch.object(benchmark, "_profile_measurement", side_effect=lambda *args: order.append("profile") or {}), \
             patch.object(benchmark, "_graph_measurement", side_effect=lambda *args: order.append("graph") or {}), \
             patch.object(benchmark, "perf_counter", side_effect=[0, 1, 2, 2.03, 3, 3.03]):
            result = benchmark._measure(operation, SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock())),
                                        "cuda:0", args, prepare=prepare)
        self.assertEqual(order, ["events", "memory", "profile", "graph"])
        self.assertEqual(operation.call_count, 8)
        self.assertAlmostEqual(result["warm_median_ms"], 10.0)
        self.assertEqual(result["first_use_seconds"], 1)

    def test_graph_capture_warms_then_replays_and_reports_unsupported_capture(self):
        from unittest.mock import Mock

        stream, graph, operation = Mock(), Mock(), Mock()
        cuda = SimpleNamespace(device=lambda device: nullcontext(), Stream=Mock(return_value=stream),
                               current_stream=Mock(), stream=lambda stream: nullcontext(),
                               CUDAGraph=Mock(return_value=graph), graph=lambda *args, **kwargs: nullcontext())
        with patch.object(benchmark, "_event_times", return_value={"median_ms": 0.1}) as timing:
            result = benchmark._graph_measurement(operation, SimpleNamespace(cuda=cuda), "cuda:0", object())
        self.assertEqual(result, {"status": "ok", "median_ms": 0.1})
        self.assertEqual(operation.call_count, 4)
        graph.replay.assert_called_once()
        self.assertEqual(timing.call_args.args[0], graph.replay)
        cuda.Stream.side_effect = RuntimeError("capture not supported")
        result = benchmark._graph_measurement(operation, SimpleNamespace(cuda=cuda), "cuda:0", object())
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("capture not supported", result["error"])

    def test_prepared_bf16_gemms_do_not_build_autograd_graphs(self):
        x = torch.randn(3, 7, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(5, 7, requires_grad=True)
        bias = torch.randn(5, requires_grad=True)
        grad = torch.randn(3, 5, dtype=torch.bfloat16)
        for _, left, right, product_bias, dtype in benchmark._dense_products(x, weight, bias, grad):
            result = torch.nn.functional.linear(left, right, product_bias).to(dtype)
            self.assertFalse(result.requires_grad)

    def test_fp32_inputs_preserve_gradient_dtype_in_both_scopes(self):
        args = SimpleNamespace(batches=[3], input_features=7, output_features=5, input_dtype="fp32",
                               implementation="bf16", accuracy=True, scope="linear")
        observed = []

        def measure(operation, *unused, **kwargs):
            result = operation()
            observed.append(result.dtype)
            return {}

        with patch.object(benchmark, "_measure", new=measure):
            rows = benchmark._measure_batches(args, torch, "cpu", {}, do_linear, gemm_nt)
            args.scope = "gemm"
            rows += benchmark._measure_batches(args, torch, "cpu", {}, do_linear, gemm_nt)
        self.assertEqual(observed, [torch.bfloat16, torch.bfloat16, torch.float32, torch.float32])
        self.assertTrue(all(row["errors_vs_fp32"] for row in rows))

    def test_output_file_contains_full_combined_report_and_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "results" / "fp8-benchmark.json"
            command = ["benchmark", "--implementation", "both", "--scope", "both",
                       "--reuse-cache", "--output", str(destination)]

            def worker(command, **kwargs):
                result = {"implementation": command[command.index("--implementation") + 1],
                          "scope": command[command.index("--scope") + 1],
                          "environment": {"deepopt_version": "test"},
                          "timings": [{"warm_median_ms": 1.25,
                                       "cuda_events": {"median_ms": 1.1},
                                       "profile": {"status": "unavailable", "error": "test error"}}],
                          "autotune_selections": [{"tile": {"BLOCK_K": 64}}]}
                return SimpleNamespace(returncode=0, stderr="",
                                       stdout=benchmark._RESULT_PREFIX + json.dumps(result) + "\n")

            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(sys, "argv", command), patch.object(benchmark.subprocess, "run", side_effect=worker), \
                 patch("sys.stdout", new=stdout), patch("sys.stderr", new=stderr):
                benchmark.main()
            saved = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(saved, json.loads(stdout.getvalue()))
            self.assertEqual(len(saved["results"]), 4)
            self.assertNotIn("output", saved["settings"])
            self.assertTrue(all(row["timings"] and row["autotune_selections"] for row in saved["results"]))
            self.assertTrue(all(row["timings"][0]["cuda_events"]["median_ms"] == 1.1
                                for row in saved["results"]))
            self.assertIn(str(destination.resolve()), stderr.getvalue())

    def test_failed_benchmark_preserves_existing_output_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "previous.json"
            previous = '{"previous": true}\n'
            destination.write_text(previous, encoding="utf-8")
            with patch.object(sys, "argv", ["benchmark", "--output", str(destination)]), \
                 patch.object(benchmark, "_run", side_effect=SystemExit(7)), \
                 self.assertRaises(SystemExit) as raised:
                benchmark.main()
            self.assertEqual(raised.exception.code, 7)
            self.assertEqual(destination.read_text(encoding="utf-8"), previous)

    def test_workers_use_active_interpreter_and_environment_without_source_overrides(self):
        command = ["benchmark", "--implementation", "both", "--scope", "both", "--reuse-cache",
                   "--detailed", "--profile-iterations", "3", "--profile-rows", "7"]
        with patch.object(sys, "argv", command), patch.object(benchmark, "_run", return_value={}) as run, \
             patch("sys.stdout", new=io.StringIO()):
            benchmark.main()
        args = run.call_args.args[0]
        self.assertFalse(hasattr(args, "source"))
        args.output = Path("parent-only-result.json")
        calls = []

        def worker(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=0, stderr="", stdout=benchmark._RESULT_PREFIX + "{}\n")

        with patch.object(benchmark.subprocess, "run", side_effect=worker), \
             patch.dict(benchmark.os.environ, {"PYTHONPATH": "caller-library-path", "CONDA_PREFIX": "caller-env"}), \
             patch("sys.stderr", new=io.StringIO()):
            result = benchmark._run(args)
        self.assertEqual(len(result["results"]), 4)
        combinations = []
        for command, options in calls:
            self.assertEqual(command[0], sys.executable)
            self.assertNotIn("--source", command)
            self.assertNotIn("--output", command)
            self.assertIn("--detailed", command)
            self.assertEqual(command[command.index("--profile-iterations") + 1], "3")
            self.assertEqual(command[command.index("--profile-rows") + 1], "7")
            self.assertNotIn("cwd", options)
            self.assertEqual(options["env"]["PYTHONPATH"], "caller-library-path")
            self.assertEqual(options["env"]["CONDA_PREFIX"], "caller-env")
            combinations.append((command[command.index("--implementation") + 1],
                                 command[command.index("--scope") + 1]))
        self.assertEqual(combinations, [("fp8", "linear"), ("fp8", "gemm"),
                                        ("bf16", "linear"), ("bf16", "gemm")])
        self.assertNotIn("source", result["settings"])
        self.assertNotIn("output", result["settings"])
        json.dumps(result, allow_nan=False)

    def test_worker_reports_imported_library_without_modifying_import_path(self):
        import deepopt

        command = ["benchmark", "--implementation", "bf16", "--scope", "linear"]
        with patch.object(sys, "argv", command), patch.object(benchmark, "_run", return_value={}) as run, \
             patch("sys.stdout", new=io.StringIO()):
            benchmark.main()
        args = run.call_args.args[0]
        original_path = list(sys.path)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "set_device"), \
             patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), \
             patch.object(torch.cuda, "get_device_name", return_value="test GPU"), \
             patch.object(torch.cuda, "synchronize"), patch.object(torch, "empty"), \
             patch("importlib.metadata.version", return_value="0.1.test"), \
             patch.object(benchmark, "_measure_batches", return_value=[]):
            result = benchmark._worker(args)
        self.assertEqual(sys.path, original_path)
        self.assertEqual(result["environment"]["deepopt_source"], str(Path(deepopt.__file__).resolve()))
        self.assertEqual(result["environment"]["deepopt_version"], "0.1.test")
        self.assertNotIn("source", result)

    def test_nonfinite_accuracy_results_are_json_safe(self):
        result = benchmark._error_metrics(torch.tensor([float("nan")]), torch.ones(1), torch)
        self.assertFalse(result["finite"])
        self.assertIsNone(result["relative_l2"])
        json.dumps(result, allow_nan=False)

    def test_tensorwise_benchmark_has_no_block_scaling_or_forced_accumulation(self):
        args = SimpleNamespace(scaling="tensorwise", scale_k=32, num_stages="auto",
                               grad_format="e4m3fn")
        plan = benchmark._benchmark_plan(args, torch)
        self.assertEqual(plan["block_shapes"], dict.fromkeys(("activation", "weight", "grad")))
        self.assertNotIn("max_num_imprecise_acc", plan["fp8_training"])
        self.assertEqual(plan["fp8_training"]["fp8_dtypes"]["grad"], torch.float8_e4m3fn)

    def test_vit_preset_matches_real_forward_and_weight_gradient_shapes(self):
        args = SimpleNamespace(preset="vit", batches=[512], tokens_per_sample=197, output_features=10)
        self.assertEqual(benchmark._projection_cases(args),
                         [("qkv", 100864, 768, 2304), ("attention", 100864, 768, 768),
                          ("fc1", 100864, 768, 3072), ("fc2", 100864, 3072, 768),
                          ("head", 512, 768, 10)])

    def test_bf16_component_and_autograd_comparisons_report_numerical_errors(self):
        args = SimpleNamespace(batches=[3], input_features=7, output_features=5,
                               implementation="bf16", accuracy=True, scope="linear")

        def measure(operation, *unused, **kwargs):
            operation()
            return {}

        with patch.object(benchmark, "_measure", new=measure):
            rows = benchmark._measure_batches(args, torch, "cpu", {}, do_linear, gemm_nt)
            self.assertEqual(set(rows[0]["errors_vs_fp32"]), {"forward", "input_gradient", "weight_gradient"})
            args.scope = "gemm"
            rows = benchmark._measure_batches(args, torch, "cpu", {}, do_linear, gemm_nt)
        self.assertEqual([row["M_N_K"] for row in rows], [[3, 5, 7], [3, 7, 5], [5, 7, 3]])
        self.assertTrue(all(row["errors_vs_fp32"]["finite"] for row in rows))

    def test_benchmark_cli_accepts_dense_shapes_stage_five_and_wider_scales(self):
        command = ["benchmark", "--batches", "1024", "--input-features", "2048",
                   "--output-features", "1024", "--scale-k", "128", "--num-stages", "5"]
        with patch.object(sys, "argv", command), patch.object(benchmark, "_run", return_value={}) as run, \
             patch("sys.stdout", new=io.StringIO()):
            benchmark.main()
        args = run.call_args.args[0]
        self.assertEqual((args.batches, args.input_features, args.output_features, args.scale_k, args.num_stages),
                         ([1024], 2048, 1024, 128, 5))

    def test_benchmark_uses_requested_dimensions_for_both_scopes(self):
        args = SimpleNamespace(batches=[3], input_features=7, output_features=5,
                               scope="linear", num_stages=5)
        plan = {"fp8_training": {"backend": "reference", "fp8_dtypes": {
            role: torch.float8_e4m3fn for role in ("activation", "weight", "grad")}},
            "block_shapes": dict.fromkeys(("activation", "weight", "grad"))}

        def measure(operation, *unused, **kwargs):
            operation()
            return {}

        with patch.object(benchmark, "_measure", new=measure), \
             patch.object(benchmark, "_prepare_gemm_products", wraps=benchmark._prepare_gemm_products) as capture:
            rows = benchmark._measure_batches(args, torch, "cpu", plan, do_linear, gemm_nt)
            args.scope = "gemm"
            # The GEMM benchmark dispatch normally selects Triton; use the
            # reference oracle to exercise its operands without a CUDA device.
            def reference(a, b, **kwargs):
                return gemm_nt(a, b, **{**kwargs, "backend": "reference"})

            rows += benchmark._measure_batches(args, torch, "cpu", plan, do_linear, reference)
        x, weight, bias, grad, _ = capture.call_args.args
        self.assertEqual([tuple(value.shape) for value in (x, weight, bias, grad)],
                         [(3, 7), (5, 7), (5,), (3, 5)])
        self.assertEqual([row["operation"] for row in rows],
                         ["forward_backward", "forward", "input_gradient", "weight_gradient"])

    def test_quantize_scope_measures_independent_transpose_views(self):
        args = SimpleNamespace(batches=[3], input_features=7, output_features=5, scope="quantize")
        plan = {"fp8_training": {"fp8_dtypes": dict.fromkeys(
            ("activation", "weight", "grad"), torch.float8_e4m3fn)},
            "block_shapes": {"activation": (1, 32), "weight": (32, 32), "grad": (1, 32)}}

        def measure(operation, *unused, **kwargs):
            self.assertIsNotNone(operation())
            return {}

        for scope in ("quantize", "scale", "dequantize"):
            args.scope = scope
            with self.subTest(scope=scope), patch.object(benchmark, "_measure", new=measure):
                rows = benchmark._measure_batches(args, torch, "cpu", plan, do_linear, gemm_nt)
            self.assertEqual([row["operation"] for row in rows],
                             ["activation", "weight", "gradient", "activation_transposed", "gradient_transposed"])
            self.assertEqual([row["shape"] for row in rows[-2:]], [[7, 3], [5, 3]])
            self.assertEqual([row["stride"] for row in rows[-2:]], [[1, 7], [1, 5]])

    def test_capture_matches_actual_autograd_for_both_orientation_policies(self):
        for activation, grad_blocks in (((1, 32), (32, 32)), ((32, 32), (1, 32)), (None, None)):
            plan = {
                "fp8_training": {"backend": "reference", "out_dtype": torch.bfloat16,
                                 "fp8_dtypes": {"activation": torch.float8_e4m3fn,
                                                "weight": torch.float8_e4m3fn, "grad": torch.float8_e5m2}},
                "block_shapes": {"activation": activation, "weight": (32, 32), "grad": grad_blocks},
            }
            x = torch.randn(37, 65, dtype=torch.bfloat16, requires_grad=True)
            weight = torch.randn(10, 65, requires_grad=True)
            bias = torch.randn(10, requires_grad=True)
            grad = torch.randn(37, 10, dtype=torch.bfloat16)
            actual = []
            original = linear_autograd_func.gemm_nt

            def record(a, b, **kwargs):
                actual.append((a, b, kwargs.get("bias"), kwargs["out_dtype"]))
                return original(a, b, **kwargs)

            with patch.object(linear_autograd_func, "gemm_nt", new=record):
                do_linear(x, weight, bias, plan=plan).backward(grad)
            captured = benchmark._prepare_gemm_products(x, weight, bias, grad, plan)
            self.assertEqual([row[0] for row in captured], ["forward", "input_gradient", "weight_gradient"])
            for expected, (_, *observed) in zip(actual, captured):
                for left, right in zip(expected[:2], observed[:2]):
                    self.assertEqual(left.block_shape, right.block_shape)
                    self.assertEqual(left.data.dtype, right.data.dtype)
                    torch.testing.assert_close(left.data.float(), right.data.float(), rtol=0, atol=0)
                    torch.testing.assert_close(left.scale_inv, right.scale_inv, rtol=0, atol=0)
                self.assertIs(expected[2], observed[2])
                self.assertEqual(expected[3], observed[3])
            self.assertIsNone(x.grad)
            self.assertIsNone(weight.grad)
            self.assertIsNone(bias.grad)
            if activation == (1, 32):
                self.assertEqual(captured[2][2].block_shape, (1, 32))

    def test_records_each_key_failed_trials_and_restores_tuner(self):
        good = SimpleNamespace(kwargs={"BLOCK_M": 16, "SPLIT_K": 4}, num_warps=2, num_stages=2)
        bad = SimpleNamespace(kwargs={"BLOCK_M": 32, "SPLIT_K": 1}, num_warps=4, num_stages=3)

        class Tuner:
            keys = ("M",)
            arg_names = ("M", "a_ptr")

            def __init__(self):
                self.nargs = {"M": 32, "a_ptr": SimpleNamespace(dtype="fp8")}
                self.cache = {}

            def _bench(self, *args, config, **kwargs):
                return [0.2, 0.1, 0.3] if config is good else [float("inf")] * 3

        tuner = Tuner()
        modules = {"deepopt.opt.backends.triton_fp8": SimpleNamespace(_gemm_nt_kernel=tuner),
                   "deepopt.opt.backends._triton_autotune": SimpleNamespace(AUTOTUNE_KEY=("M",))}
        with patch.dict(sys.modules, modules):
            with benchmark._record_autotune_trials() as trials:
                for m in (32, 64):
                    tuner.nargs["M"] = m
                    tuner._bench(config=good)
                    tuner._bench(config=bad)
                    tuner.cache[m, "fp8"] = good
            self.assertNotIn("_bench", tuner.__dict__)
            records = benchmark._autotune_diagnostics(trials)
        self.assertEqual([row["key"]["M"] for row in records], [32, 64])
        for row in records:
            self.assertEqual(len(row["candidates"]), 2)
            self.assertEqual(row["candidates"][0]["quantiles_ms"], [0.2, 0.1, 0.3])
            self.assertEqual(row["candidates"][1]["quantiles_ms"], [None] * 3)
            self.assertEqual(row["candidates"][1]["status"], "unavailable")
        json.dumps(records, allow_nan=False)

    def test_auto_and_explicit_stages_round_trip_through_public_configuration(self):
        self.assertEqual(FP8Config().num_stages, "auto")
        for stages in ("auto", 1, 2, 3, 4, 5):
            parsed = fp8_config_from_dict({"num_stages": stages})
            self.assertEqual(parsed.num_stages, stages)
            plan = {"fp8_training": {"num_stages": stages, "fp8_dtypes": {
                role: torch.float8_e4m3fn for role in ("activation", "weight", "grad")}},
                "block_shapes": dict.fromkeys(("activation", "weight", "grad"))}
            self.assertEqual(validate_fp8_plan(plan)["fp8_training"]["num_stages"], stages)
            from deepopt.opt.quantization.quantizer import quantize_fp8
            operand = quantize_fp8(torch.ones(2, 3))
            torch.testing.assert_close(gemm_nt(operand, operand, backend="reference", num_stages=stages),
                                       torch.full((2, 2), 3.0))
