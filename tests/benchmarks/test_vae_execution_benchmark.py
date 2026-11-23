"""Comparable eager/native measurements without a numerical-oracle route."""

from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.vae import benchmark_execution
from examples.vae import native_execution
from tests.benchmarks.test_vae_native_execution import _receipt


def _worker_args(receipt, output, mode, *, cpu=False, updates=2):
    arguments = ["--from-result", str(receipt), "--output", str(output),
        "--execution-mode", mode, "--benchmark", "--warmup", "1", "--updates", str(updates)]
    if cpu:
        arguments += ["--device", "cpu", "--batch-size", "1", "--synthetic", "--optimizer", "sgd"]
    return native_execution.parser().parse_args(arguments)


def _successful_worker(plan, mode, *, trial=1, identity=None):
    """Explicit metadata fixture for ratio/identity rules; no CUDA is executed."""
    native = mode == "native"
    wall_ms, cuda_ms = (10., 8.) if native else (20., 16.)
    allocated, reserved = (500, 1000) if native else (1000, 2000)
    cold_allocated, cold_reserved = (4000, 8000) if native else (3000, 6000)
    identity = identity or {"version": native_execution.EXPECTED_VERSION,
        "source_sha256": "1" * 64, "files_hashed": 457, "editable": False}
    memory = {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved}
    startup = {"peak_allocated_bytes": cold_allocated, "peak_reserved_bytes": cold_reserved}
    return {**deepcopy(plan), "trial": trial, "execution_mode": mode, "status": "completed",
        "execution_completed": True, "warmup_completed": 1, "updates_completed": 2,
        "native_ownership_verified": native, "eager_ownership_verified": not native,
        "runtime_config": deepcopy(plan["settings"]),
        "installed": identity, "installed_after": deepcopy(identity), "installed_stable": True,
        "source_stable": True, "helper_source_stable": True, "input_stable": True,
        "provenance": {"source_sha256": "2" * 64, "model_source_sha256": "3" * 64,
                       "optimizer_source_sha256": "4" * 64},
        "provenance_after": {"source_sha256": "2" * 64, "model_source_sha256": "3" * 64,
                             "optimizer_source_sha256": "4" * 64},
        "helper_sha256": {"examples/vae/native_execution.py": "5" * 64},
        "helper_sha256_after": {"examples/vae/native_execution.py": "5" * 64},
        "model": {"initial_state_sha256": "6" * 64, "parameters": 32317999,
                  "options": deepcopy(native_execution.ORIGINAL_MODEL)},
        "optimizer": {"name": "custom", "initial_state_sha256": "9" * 64},
        "input": {"sha256": "7" * 64, "sha256_after": "7" * 64,
                  "shape": [128, 3, 224, 224], "dtype": "torch.float32", "requires_grad": False,
                  "mode": "first_dataset_batch_repeated", "stride": [150528, 50176, 224, 1],
                  "contiguous": True, "logical_bytes": 77070336, "storage_bytes": 77070336},
        "software": {"torch": "2.15.0+cu132", "torch_git": "8" * 40, "cuda": "13.2", "cudnn": 92600,
            "deterministic_algorithms": False, "deterministic_warn_only": False,
            "cudnn_benchmark": True, "cudnn_deterministic": False,
            "gpu": {"name": "NVIDIA H200", "uuid": "unit-test-GPU", "capability": [9, 0],
                    "total_memory": 150156607488}},
        "compiler_report_summary": {"captured_graphs": int(native), "runtime_fallbacks": 0,
            "native_autodiff_operations": 412 if native else 0, "native_autodiff_regions": 2 if native else 0,
            "legacy_backward_operations": []},
        "measurement": {"valid": True, "invalid_reasons": [], "measured_updates": 2, "warmup_updates": 1,
            "timing_scope": "device_resident_train_step_including_loss_backward_optimizer_schedule",
            "wall_seconds": wall_ms * 2 / 1000, "wall_ms_per_update": wall_ms,
            "cuda_event_ms_per_update": cuda_ms, "images_per_second": 128 * 1000 / wall_ms,
            "native_graph_count_before": int(native), "native_graph_count_after": int(native),
            "memory": memory, "postchecks": {"finite_losses": True, "finite_parameters": True},
            "samples": [{"update": number, "host_submission_seconds": wall_ms / 1000,
                         "cuda_event_ms": cuda_ms, "loss": .5} for number in (2, 3)]},
        "memory": {"startup": startup, "steady": memory, "total": deepcopy(startup)},
        "memory_segments": {"startup": deepcopy(startup), "measurement": deepcopy(memory),
                            "postchecks": deepcopy(memory), "total": deepcopy(startup)},
        "capacity_status": "fits_requested_geometry", "physical_target_executed": True}


class VAEExecutionBenchmarkPlanningTests(unittest.TestCase):
    def test_exported_entry_points_parse_on_target_python311(self):
        for module in (native_execution, benchmark_execution):
            with self.subTest(module=module.__name__):
                ast.parse(Path(module.__file__).read_text(), feature_version=(3, 11))

    def test_worker_eager_and_native_keep_one_recipe_and_normalize_recorded_compile_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, raw = _receipt(directory)
            original = receipt.read_bytes()
            plans = {mode: native_execution.execution_plan(_worker_args(receipt,
                Path(directory) / f"{mode}.json", mode)) for mode in ("eager", "native")}
            self.assertEqual(receipt.read_bytes(), original)
        for mode, plan in plans.items():
            self.assertEqual(plan["target_settings"], raw)
            self.assertEqual(plan["requested_input_shape"], [128, 3, 224, 224])
            self.assertEqual(plan["settings"]["runtime"]["activation_compression"],
                             raw["runtime"]["activation_compression"])
            self.assertEqual(plan["settings"]["runtime"]["activation_memory"],
                             raw["runtime"]["activation_memory"])
            self.assertEqual(plan["settings"]["runtime"]["bf16_policy"], "bf16_storage")
            self.assertFalse(plan["numerical_comparisons_performed"])
            compiler = plan["settings"]["runtime"]["compile"]
            self.assertIsNone(compiler["mode"])
            self.assertEqual(compiler["enabled"], mode == "native")
            if mode == "native":
                self.assertTrue(compiler["fullgraph"])
                self.assertTrue(compiler["options"]["require_full_aot"])
                self.assertTrue(compiler["options"]["use_native_autodiff"])
            else:
                self.assertEqual(compiler["options"], {})
        eager, native = (deepcopy(plans[mode]["settings"]) for mode in ("eager", "native"))
        eager["runtime"].pop("compile")
        native["runtime"].pop("compile")
        self.assertEqual(eager, native)

    def test_worker_benchmark_dry_run_is_standard_library_only(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _receipt(directory)
            for mode in ("eager", "native"):
                command = [sys.executable, "-S", str(Path(native_execution.__file__).resolve()),
                    "--from-result", str(receipt), "--execution-mode", mode, "--benchmark", "--dry-run"]
                completed = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                plan = json.loads(completed.stdout)
                self.assertEqual(plan["execution_mode"], mode)
                self.assertTrue(plan["benchmark"])
                self.assertEqual(plan["requested_input_shape"], [128, 3, 224, 224])

    def test_supervisor_dry_run_alternates_order_without_importing_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _receipt(directory)
            command = [sys.executable, "-S", str(Path(benchmark_execution.__file__).resolve()),
                "--from-result", str(receipt), "--trials", "2", "--dry-run"]
            completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        plan = json.loads(completed.stdout)
        self.assertEqual(plan["order"], [["eager", "native"], ["native", "eager"]])
        self.assertEqual(plan["requested_input_shape"], [128, 3, 224, 224])
        self.assertFalse(plan["numerical_comparisons_performed"])

    def test_benchmark_rejects_missing_warmup_and_diagnostic_runtime_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, raw = _receipt(directory)
            args = _worker_args(receipt, Path(directory) / "worker.json", "native")
            args.warmup = 0
            with self.assertRaisesRegex(ValueError, "warmup"):
                native_execution.execution_plan(args)
            args.warmup = 1
            for mode in ("analyze", "balanced"):
                with self.subTest(mode=mode):
                    changed = deepcopy(raw)
                    changed["runtime"]["execution_mode"] = mode
                    receipt.write_text(json.dumps({"settings": changed}))
                    with self.assertRaises(ValueError):
                        native_execution.execution_plan(args)


class VAEExecutionBenchmarkAggregationTests(unittest.TestCase):
    def workers(self, directory):
        receipt, _ = _receipt(directory)
        return [_successful_worker(native_execution.execution_plan(_worker_args(receipt,
            Path(directory) / f"{mode}.json", mode)), mode) for mode in ("eager", "native")]

    def test_ratios_use_measured_window_and_preserve_cold_start_peaks(self):
        with tempfile.TemporaryDirectory() as directory:
            workers = self.workers(directory)
            # Later postchecks must not become the measured-window peak.
            workers[0]["memory"]["steady"] = {"peak_allocated_bytes": 900000, "peak_reserved_bytes": 1800000}
            workers[1]["memory"]["steady"] = {"peak_allocated_bytes": 990000, "peak_reserved_bytes": 1980000}
            summary = benchmark_execution.aggregate_trials(workers)
        self.assertTrue(summary["valid"], summary["invalid_reasons"])
        self.assertEqual(summary["ratios"]["runtime_speedup_eager_over_native"]["median"], 2.)
        self.assertEqual(summary["ratios"]["steady_allocated_reduction_percent"]["median"], 50.)
        self.assertEqual(summary["ratios"]["steady_reserved_reduction_percent"]["median"], 50.)
        self.assertAlmostEqual(summary["ratios"]["total_allocated_reduction_percent"]["median"], -100 / 3)
        self.assertAlmostEqual(summary["ratios"]["total_reserved_reduction_percent"]["median"], -100 / 3)
        self.assertEqual(summary["modes"]["native"]["metrics"]["startup_peak_allocated_bytes"]["median"], 4000)
        self.assertEqual(summary["modes"]["eager"]["metrics"]["startup_peak_allocated_bytes"]["median"], 3000)

    def test_required_memory_audit_rejects_missing_receipts_without_changing_legacy_default(self):
        from tests.benchmarks.test_vae_memory_comparison import audited_row

        with tempfile.TemporaryDirectory() as directory:
            workers = self.workers(directory)
            self.assertTrue(benchmark_execution.aggregate_trials(workers)["valid"])
            required = benchmark_execution.aggregate_trials(workers, require_memory_audit=True)
            self.assertFalse(required["valid"])
            self.assertFalse(required["ratios"])
            self.assertIn("missing_or_disabled", " ".join(required["invalid_reasons"]))
            for index, row in enumerate(workers):
                row["memory_audit"] = audited_row(arm=row["execution_mode"], pid=100 + index)["memory_audit"]
            self.assertTrue(benchmark_execution.aggregate_trials(workers, require_memory_audit=True)["valid"])
            workers[0].pop("memory_audit")
            incomplete = benchmark_execution.aggregate_trials(workers, require_memory_audit=True)
            self.assertFalse(incomplete["valid"])
            self.assertFalse(incomplete["ratios"])
            self.assertEqual(incomplete["modes"]["eager"]["successful_trials"], 0)
            self.assertEqual(incomplete["modes"]["native"]["successful_trials"], 1)
            self.assertEqual(incomplete["modes"]["native"]["metrics"]["wall_ms_per_update"]["median"], 10.)

    def test_identity_mismatch_refuses_ratio(self):
        changes = ((("model", "initial_state_sha256"), "a" * 64),
                   (("input", "sha256"), "b" * 64),
                   (("optimizer", "initial_state_sha256"), "d" * 64),
                   (("installed", "source_sha256"), "c" * 64),
                   (("software", "torch"), "other-Torch"),
                   (("software", "gpu", "uuid"), "other-GPU"),
                   (("settings", "runtime", "bf16_policy"), "torch_amp"),
                   (("settings", "runtime", "activation_memory", "multiply_backward"), "materialized"))
        for path, value in changes:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as directory:
                workers = self.workers(directory)
                target = workers[1]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                summary = benchmark_execution.aggregate_trials(workers)
                self.assertFalse(summary["valid"])
                self.assertTrue(summary["invalid_reasons"])
                self.assertFalse(summary["paired_trials"][0]["valid"])
                self.assertFalse(summary["ratios"])

    def test_wrong_execution_owner_refuses_ratio(self):
        for mode in ("eager", "native"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                workers = self.workers(directory)
                row = next(row for row in workers if row["execution_mode"] == mode)
                row["settings"]["runtime"]["compile"]["enabled"] = mode == "eager"
                summary = benchmark_execution.aggregate_trials(workers)
                self.assertFalse(summary["valid"])
                self.assertFalse(summary["ratios"])

    def test_failed_or_oom_route_preserves_the_successful_peer(self):
        for capacity in ("cuda_oom", "not_established"):
            with self.subTest(capacity=capacity), tempfile.TemporaryDirectory() as directory:
                workers = self.workers(directory)
                workers[0].update(status="failed", execution_completed=False, capacity_status=capacity,
                                  error="OutOfMemoryError: requested batch" if capacity == "cuda_oom" else "RuntimeError: failed")
                workers[0]["measurement"].update(valid=False, invalid_reasons=["worker_failed"])
                summary = benchmark_execution.aggregate_trials(workers)
                self.assertFalse(summary["valid"])
                self.assertFalse(summary["ratios"])
                self.assertEqual(summary["modes"]["eager"]["successful_trials"], 0)
                self.assertEqual(summary["modes"]["native"]["successful_trials"], 1)
                self.assertEqual(summary["modes"]["native"]["metrics"]["wall_ms_per_update"]["median"], 10.)

    def test_cuda_comparison_requires_event_and_all_allocator_peak_metrics(self):
        paths = (("measurement", "cuda_event_ms_per_update"),
                 ("measurement", "memory", "peak_allocated_bytes"),
                 ("memory", "startup", "peak_reserved_bytes"),
                 ("memory", "total", "peak_allocated_bytes"))
        for path in paths:
            with self.subTest(path=path), tempfile.TemporaryDirectory() as directory:
                workers = self.workers(directory)
                target = workers[1]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = None
                summary = benchmark_execution.aggregate_trials(workers)
                self.assertFalse(summary["valid"])
                self.assertFalse(summary["ratios"])
                self.assertTrue(any("missing_positive_CUDA_metric" in reason
                                    for reason in summary["invalid_reasons"]))

    def test_any_invalid_trial_suppresses_headline_ratios_but_preserves_valid_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            workers = self.workers(directory)
            second = deepcopy(workers)
            for row in second:
                row["trial"] = 2
            second[1]["input"]["sha256"] = "a" * 64
            summary = benchmark_execution.aggregate_trials(workers + second)
        self.assertFalse(summary["valid"])
        self.assertFalse(summary["ratios"])
        self.assertTrue(summary["paired_trials"][0]["valid"])
        self.assertEqual(summary["paired_trials"][0]["ratios"]["runtime_speedup_eager_over_native"], 2.)
        self.assertFalse(summary["paired_trials"][1]["valid"])


class VAEExecutionBenchmarkWorkerCPUTests(unittest.TestCase):
    def setUp(self):
        import torch
        from importlib.metadata import version

        # These CPU fixtures exercise worker contracts, not one historical
        # qualification wheel; the arm-manifest pinning tests keep that gate.
        expected = patch.object(native_execution, "EXPECTED_VERSION", version("deep-optimization"))
        expected.start()
        self.addCleanup(expected.stop)
        # Execution plans install process defaults for the real command path;
        # restore them so an in-process fixture cannot affect later tests.
        from deepopt import reset_config
        reset_config()
        self.addCleanup(reset_config)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_real_native_timed_window_excludes_scans_observers_and_hashes(self):
        from deepopt.compiler.runtime import current_native_boundary_observer
        from examples.vae import benchmark_vae as benchmark

        active = False
        events, steps = [], []
        original_measure = native_execution._benchmark_updates
        original_finite = native_execution._finite
        original_fingerprint = benchmark.fingerprint
        original_artifact = native_execution._ArtifactRecorder.run_artifact
        original_step = benchmark.train_step
        memory_values = ((1000, 2000), (100, 200), (300, 600))
        memory_count = 0
        def measure(**kwargs):
            nonlocal active
            active = True
            events.append("measurement_start")
            try:
                return original_measure(**kwargs)
            finally:
                active = False
                events.append("measurement_end")
        def finite(*args, **kwargs):
            self.assertFalse(active, "Finite scan ran inside the measured window")
            events.append("finite")
            return original_finite(*args, **kwargs)
        def fingerprint(*args, **kwargs):
            self.assertFalse(active, "Tensor hashing ran inside the measured window")
            events.append("hash")
            return original_fingerprint(*args, **kwargs)
        def artifact(owner, *args, **kwargs):
            self.assertFalse(active, "Artifact observer ran inside the measured window")
            events.append("artifact")
            return original_artifact(owner, *args, **kwargs)
        def step(*args, **kwargs):
            self.assertIsNotNone(args[0]._compiled_call_impl)
            steps.append(active)
            if active:
                self.assertIsNone(current_native_boundary_observer())
                self.assertFalse(args[1]._optimizer_step_pre_hooks)
                self.assertIsNone(kwargs.get("progress"))
            return original_step(*args, **kwargs)
        def memory(*args, **kwargs):
            nonlocal memory_count
            allocated, reserved = memory_values[min(memory_count, 2)]
            memory_count += 1
            events.append("memory_" + str(memory_count))
            return {"scope": "unit_simulated_allocator_metadata_only", "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved}
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _receipt(directory, tiny=True)
            args = _worker_args(receipt, Path(directory) / "timed.json", "native", cpu=True)
            with patch.object(native_execution, "_benchmark_updates", side_effect=measure), \
                    patch.object(native_execution, "_finite", side_effect=finite), \
                    patch.object(native_execution, "_memory", side_effect=memory), \
                    patch.object(native_execution._ArtifactRecorder, "run_artifact", new=artifact), \
                    patch.object(benchmark, "fingerprint", side_effect=fingerprint), \
                    patch.object(benchmark, "train_step", side_effect=step), \
                    patch("examples.vae.qualification.paired_trajectory", side_effect=AssertionError("paired oracle")), \
                    patch("examples.vae.qualification.TensorOracle.__init__", side_effect=AssertionError("tensor oracle")), \
                    redirect_stdout(io.StringIO()):
                result = native_execution.run_native_execution(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(result, 0, report.get("error"))
            self.assertTrue(report["measurement"]["valid"], report["measurement"])
            self.assertEqual(steps, [False, True, True])
            measured_events = events[events.index("measurement_start") + 1:events.index("measurement_end")]
            self.assertFalse(set(measured_events) & {"finite", "hash", "artifact"})
            self.assertIn("memory_2", measured_events)
            self.assertEqual(report["measurement"]["memory"]["peak_allocated_bytes"], 100)
            self.assertEqual(report["memory_segments"]["startup"]["peak_allocated_bytes"], 1000)
            self.assertEqual(report["memory_segments"]["postchecks"]["peak_allocated_bytes"], 300)
            self.assertEqual(report["memory_segments"]["total"]["peak_allocated_bytes"], 1000)
            self.assertEqual(report["measurement"]["native_graph_count_before"],
                             report["measurement"]["native_graph_count_after"])
            self.assertEqual(report["tensor_files_written"], 0)
            self.assertFalse(report["correctness_assessed"])


@contextmanager
def _integration_directory():
    destination = os.environ.get("DEEPOPT_VAE_BENCHMARK_TEST_EVIDENCE")
    if destination:
        path = Path(destination).resolve()
        path.mkdir(parents=True, exist_ok=False)
        yield path
    else:
        with tempfile.TemporaryDirectory() as directory:
            yield Path(directory)


class VAEExecutionBenchmarkSupervisorTests(unittest.TestCase):
    def setUp(self):
        from importlib.metadata import version

        # Child workers re-import the module, so the default expected release
        # travels through the environment; arm manifests still pin exactly.
        previous = os.environ.get("DEEPOPT_VAE_EXPECTED_VERSION")
        os.environ["DEEPOPT_VAE_EXPECTED_VERSION"] = version("deep-optimization")
        self.addCleanup(lambda: os.environ.pop("DEEPOPT_VAE_EXPECTED_VERSION", None) if previous is None
                        else os.environ.__setitem__("DEEPOPT_VAE_EXPECTED_VERSION", previous))

    def arguments(self, directory, *, cpu=False, trials=1):
        receipt, _ = _receipt(directory, tiny=cpu)
        arguments = ["--from-result", str(receipt), "--output", str(Path(directory) / "comparison.json"),
                     "--warmup", "1", "--updates", "2", "--trials", str(trials)]
        if cpu:
            arguments += ["--device", "cpu", "--batch-size", "1", "--synthetic", "--optimizer", "sgd"]
        return benchmark_execution.parser().parse_args(arguments)

    def test_failed_routes_are_retained_and_other_modes_still_launch_in_alternating_order(self):
        from benchmarks.compiler.installed_runtime import installed_identity
        from tests.benchmarks.test_vae_memory_comparison import audited_row

        modes = []
        identity = installed_identity(native_execution.ROOT, full=True)
        def execute(command, **kwargs):
            options = native_execution.parser().parse_args(command[command.index("--from-result"):])
            mode = options.execution_mode
            modes.append(mode)
            trial = (len(modes) + 1) // 2
            row = _successful_worker(native_execution.execution_plan(options), mode, trial=trial, identity=identity)
            row["memory_audit"] = audited_row(arm=mode, pid=100 + len(modes))["memory_audit"]
            if len(modes) in (1, 3):
                oom = len(modes) == 1
                row.update(status="failed", execution_completed=False,
                    capacity_status="cuda_oom" if oom else "not_established",
                    error="OutOfMemoryError: original batch" if oom else "RuntimeError: native failure")
                row["measurement"].update(valid=False, invalid_reasons=["worker_failed"])
            options.output.write_text(json.dumps(row))
            return SimpleNamespace(returncode=0 if row["status"] == "completed" else 1)
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory, trials=2)
            with patch.object(benchmark_execution.subprocess, "run", side_effect=execute), \
                    redirect_stdout(io.StringIO()):
                result = benchmark_execution.run_benchmark(args)
            report = json.loads(args.output.read_text())
        self.assertEqual(result, 1)
        self.assertEqual(modes, ["eager", "native", "native", "eager"])
        self.assertEqual([row["status"] for row in report["results"]],
                         ["failed", "completed", "failed", "completed"])
        self.assertEqual(report["results"][0]["capacity_status"], "cuda_oom")
        self.assertIn("OutOfMemoryError", report["results"][0]["error"])
        self.assertIn("native failure", report["results"][2]["error"])
        self.assertFalse(report["summary"]["valid"])
        self.assertFalse(report["summary"]["ratios"])
        for mode in ("eager", "native"):
            self.assertEqual(report["summary"]["modes"][mode]["successful_trials"], 1)
            self.assertEqual(report["summary"]["modes"][mode]["total_trials"], 2)

    def test_malformed_nested_receipts_preserve_raw_files_and_continue_independent_peers(self):
        from benchmarks.compiler.installed_runtime import installed_identity
        from tests.benchmarks.test_vae_memory_comparison import audited_row

        identity = installed_identity(native_execution.ROOT, full=True)
        for field in ("measurement", "memory", "origin", "model", "invalid_reasons", "nonfinite"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                args = self.arguments(directory, trials=2)
                modes = []
                def execute(command, **kwargs):
                    options = native_execution.parser().parse_args(command[command.index("--from-result"):])
                    modes.append(options.execution_mode)
                    row = _successful_worker(native_execution.execution_plan(options), options.execution_mode,
                        trial=(len(modes) + 1) // 2, identity=identity)
                    row["memory_audit"] = audited_row(arm=options.execution_mode, pid=100 + len(modes))["memory_audit"]
                    if len(modes) == 1:
                        if field == "invalid_reasons":
                            row["measurement"].update(valid=False, invalid_reasons=[{"bad": "reason"}])
                        elif field == "nonfinite":
                            row["measurement"]["wall_ms_per_update"] = float("nan")
                        else:
                            row[field] = None
                    options.output.write_text(json.dumps(row))
                    return SimpleNamespace(returncode=0)
                with patch.object(benchmark_execution.subprocess, "run", side_effect=execute), \
                        redirect_stdout(io.StringIO()):
                    code = benchmark_execution.run_benchmark(args)
                report = json.loads(args.output.read_text())
                self.assertEqual(code, 1)
                self.assertEqual(modes, ["eager", "native", "native", "eager"])
                self.assertEqual([row["status"] for row in report["results"]],
                                 ["failed", "completed", "completed", "completed"])
                self.assertIn("malformed", report["results"][0]["worker_result_error"])
                self.assertEqual(report["summary"]["modes"]["native"]["successful_trials"], 2)
                self.assertEqual(report["summary"]["modes"]["native"]["metrics"]["wall_ms_per_update"]["median"], 10.)
                self.assertFalse(report["summary"]["valid"])
                self.assertFalse(report["summary"]["ratios"])
                for receipt in report["results"]:
                    path = args.output.parent / receipt["result"]["path"]
                    self.assertTrue(path.is_file())
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), receipt["result"]["sha256"])

    def test_current_supervisor_requires_audits_even_when_all_worker_fields_are_otherwise_valid(self):
        from benchmarks.compiler.installed_runtime import installed_identity

        identity = installed_identity(native_execution.ROOT, full=True)
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            def execute(command, **kwargs):
                options = native_execution.parser().parse_args(command[command.index("--from-result"):])
                row = _successful_worker(native_execution.execution_plan(options), options.execution_mode, identity=identity)
                options.output.write_text(json.dumps(row))
                return SimpleNamespace(returncode=0)
            with patch.object(benchmark_execution.subprocess, "run", side_effect=execute), \
                    redirect_stdout(io.StringIO()):
                code = benchmark_execution.run_benchmark(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(len(report["results"]), 2)
            self.assertTrue(all(row["status"] == "completed" for row in report["results"]))
            self.assertFalse(report["summary"]["valid"])
            self.assertFalse(report["summary"]["ratios"])
            self.assertIn("memory_audit_missing_or_disabled", " ".join(report["summary"]["invalid_reasons"]))

    def test_aggregation_failure_keeps_both_completed_peer_receipts_and_printable_summary(self):
        from benchmarks.compiler.installed_runtime import installed_identity
        from tests.benchmarks.test_vae_memory_comparison import audited_row

        identity = installed_identity(native_execution.ROOT, full=True)
        original = benchmark_execution.aggregate_trials
        def aggregate(rows, **kwargs):
            if len(rows) == 2:
                raise RuntimeError("injected aggregate failure")
            return original(rows, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            launches = []
            def execute(command, **kwargs):
                options = native_execution.parser().parse_args(command[command.index("--from-result"):])
                launches.append(options.execution_mode)
                row = _successful_worker(native_execution.execution_plan(options), options.execution_mode, identity=identity)
                row["memory_audit"] = audited_row(arm=options.execution_mode, pid=100 + len(launches))["memory_audit"]
                options.output.write_text(json.dumps(row))
                return SimpleNamespace(returncode=0)
            with patch.object(benchmark_execution.subprocess, "run", side_effect=execute), \
                    patch.object(benchmark_execution, "aggregate_trials", side_effect=aggregate), \
                    redirect_stdout(io.StringIO()):
                code = benchmark_execution.run_benchmark(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(launches, ["eager", "native"])
            self.assertEqual(len(report["results"]), 2)
            self.assertTrue(all(row["status"] == "completed" for row in report["results"]))
            self.assertIn("injected aggregate failure", report["summary"]["aggregation_error"]["error"])
            self.assertFalse(report["summary"]["valid"])
            self.assertFalse(report["summary"]["ratios"])
            self.assertEqual(report["summary"]["modes"]["native"]["successful_trials"], 1)
            self.assertEqual(report["summary"]["modes"]["native"]["metrics"]["wall_ms_per_update"]["median"], 10.)

    def test_real_installed_cpu_children_match_identity_without_oracle_or_tensor_files(self):
        # The subprocess bootstrap adds test guards; both workers still run
        # their real main(), model math, optimizers and installed identity gate.
        bootstrap = """import os, sys
import torch
torch.set_num_threads(1)
from examples.vae import qualification
def forbidden(*args, **kwargs):
    raise AssertionError('Benchmark invoked a numerical oracle')
qualification.paired_trajectory = forbidden
qualification.TensorOracle.__init__ = forbidden
print('TEST_CHILD_PID=' + str(os.getpid()), flush=True)
from examples.vae.native_execution import main
raise SystemExit(main(sys.argv[1:]))
"""
        real_run = subprocess.run
        launched = []
        def execute(command, **kwargs):
            launched.append(command[command.index("--execution-mode") + 1])
            guarded = [command[0], "-B", "-c", bootstrap, *command[command.index("--from-result"):]]
            return real_run(guarded, **kwargs)
        with _integration_directory() as directory:
            args = self.arguments(directory, cpu=True)
            with patch.object(benchmark_execution.subprocess, "run", side_effect=execute), \
                    redirect_stdout(io.StringIO()):
                result = benchmark_execution.run_benchmark(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(result, 0, report.get("summary"))
            self.assertEqual(launched, ["eager", "native"])
            self.assertTrue(report["summary"]["valid"], report["summary"]["invalid_reasons"])
            self.assertFalse(report["summary"]["target_cuda_comparison"])
            children, pids = {}, []
            for row in report["results"]:
                for reference in (row["result"], row["log"]):
                    data = (directory / reference["path"]).read_bytes()
                    self.assertEqual(len(data), reference["bytes"])
                    self.assertEqual(hashlib.sha256(data).hexdigest(), reference["sha256"])
                child = json.loads((directory / row["result"]["path"]).read_text())
                children[row["execution_mode"]] = child
                text = (directory / row["log"]["path"]).read_text()
                pid, = [line.split("=", 1)[1] for line in text.splitlines() if line.startswith("TEST_CHILD_PID=")]
                pids.append(pid)
                self.assertEqual(child["status"], "completed")
                self.assertTrue(child["measurement"]["valid"], child["measurement"])
                self.assertEqual(child["measurement"]["measured_updates"], 2)
                self.assertEqual(child["input"]["shape"], [1, 3, 16, 16])
                self.assertEqual(child["tensor_files_written"], 0)
                self.assertFalse(child["numerical_comparisons_performed"])
                self.assertFalse(child["correctness_assessed"])
            self.assertEqual(len(set(pids)), 2)
            eager, native = children["eager"], children["native"]
            self.assertEqual(benchmark_execution.comparable_identity(eager),
                             benchmark_execution.comparable_identity(native))
            self.assertTrue(eager["eager_ownership_verified"])
            self.assertFalse(eager["native_ownership_verified"])
            self.assertFalse(eager["settings"]["runtime"]["compile"]["enabled"])
            self.assertTrue(native["native_ownership_verified"])
            self.assertTrue(native["settings"]["runtime"]["compile"]["enabled"])
            self.assertEqual(native["measurement"]["native_graph_count_before"],
                             native["measurement"]["native_graph_count_after"])
            forbidden = {".pt", ".pth", ".pkl", ".pickle", ".npy", ".npz", ".safetensors", ".ckpt"}
            self.assertFalse([path for path in directory.rglob("*") if path.suffix in forbidden])


if __name__ == "__main__":
    unittest.main()
