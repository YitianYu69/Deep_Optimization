"""Optional profiles and manifest-pinned worker inputs preserve clean timing."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.vae import execution_profile, native_execution
from examples.vae import benchmark_vae as benchmark
from tests.benchmarks.test_vae_native_execution import _receipt


def _options(directory, values):
    path = Path(directory) / "native-options.json"
    path.write_text(json.dumps(values))
    return path


def _arm(directory, identity=None, options=None):
    identity = identity or {"version": native_execution.EXPECTED_VERSION,
        "source_sha256": "1" * 64, "files_hashed": 457}
    value = {"schema_version": 1, "arm": "B", "expected_installed": {
        key: identity[key] for key in ("version", "source_sha256", "files_hashed")},
        "wheel_sha256": "2" * 64,
        "native_options_sha256": hashlib.sha256(options.read_bytes()).hexdigest() if options else None,
        "metadata": {"scope": "test_fixture"}}
    path = Path(directory) / "arm.json"
    path.write_text(json.dumps(value))
    return path


def _args(directory, *, tiny=False, extra=()):
    receipt, raw = _receipt(directory, tiny=tiny)
    args = native_execution.parser().parse_args(["--from-result", str(receipt),
        "--output", str(Path(directory) / "execution.json"), "--benchmark", "--warmup", "1", "--updates", "2", *extra])
    return args, raw


class ExecutionProfilePlanningTests(unittest.TestCase):
    def test_pinned_override_dry_run_is_stdlib_only_and_preserves_raw_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            options = _options(directory, {"max_pointwise_fusion_length": 32})
            arm = _arm(directory, options=options)
            args, raw = _args(directory, extra=("--native-options-json", str(options),
                "--arm-manifest", str(arm), "--expected-version", native_execution.EXPECTED_VERSION,
                "--profile-after-measurement", "--profile-memory"))
            before = args.from_result.read_bytes()
            result = subprocess.run([sys.executable, "-S", str(Path(native_execution.__file__)),
                "--from-result", str(args.from_result), "--benchmark", "--dry-run",
                "--native-options-json", str(options), "--arm-manifest", str(arm),
                "--profile-after-measurement"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["target_settings"], raw)
            self.assertIsNotNone(raw["runtime"]["compile"]["mode"])
            self.assertTrue(raw["runtime"]["compile"]["options"]["use_native_autodiff"])
            self.assertEqual(args.from_result.read_bytes(), before)
            self.assertIsNone(plan["settings"]["runtime"]["compile"]["mode"])
            self.assertEqual(plan["settings"]["runtime"]["compile"]["options"]["max_pointwise_fusion_length"], 32)
            self.assertEqual(plan["native_options_override"]["sha256"], hashlib.sha256(options.read_bytes()).hexdigest())
            self.assertEqual(plan["experiment_arm"]["manifest"]["expected_installed"]["files_hashed"], 457)
            self.assertTrue(plan["profiling"]["enabled"])
            self.assertFalse(plan["profiling"]["record_shapes"])

    def test_profile_counts_and_phase_require_an_explicit_benchmark(self):
        for extra in (("--profile-updates", "0"), ("--profile-updates", "4"),
                      ("--profile-memory",), ("--profile-record-shapes",)):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                args, _ = _args(directory, extra=extra)
                with self.assertRaises(ValueError):
                    native_execution.execution_plan(args)
        with tempfile.TemporaryDirectory() as directory:
            args, _ = _args(directory, extra=("--profile-after-measurement",))
            args.benchmark = False
            with self.assertRaisesRegex(ValueError, "requires --benchmark"):
                native_execution.execution_plan(args)

    def test_expected_version_requires_a_matching_hash_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = _args(directory, extra=("--expected-version", "candidate"))
            with self.assertRaisesRegex(ValueError, "pinned --arm-manifest"):
                native_execution.execution_plan(args)
            args.arm_manifest = _arm(directory)
            with self.assertRaisesRegex(ValueError, "differs from the arm manifest"):
                native_execution.execution_plan(args)
            args.expected_version = native_execution.EXPECTED_VERSION
            manifest = json.loads(args.arm_manifest.read_text())
            manifest["expected_installed"]["files_hashed"] = True
            args.arm_manifest.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "positive integer"):
                native_execution.execution_plan(args)

    def test_native_override_cannot_change_storage_or_derivative_owner(self):
        cases = ({"bf16_policy": "torch_amp"}, {"activation_memory": {}},
                 {"use_native_autodiff": False}, {"require_full_aot": False},
                 {"enable_aot_autodiff": 1}, {"capture_deepopt": False}, {"mutation_backend": "inductor"})
        for values in cases:
            with self.subTest(values=values), tempfile.TemporaryDirectory() as directory:
                path = _options(directory, values)
                args, _ = _args(directory, extra=("--native-options-json", str(path)))
                with self.assertRaises(ValueError):
                    native_execution.execution_plan(args)
        with tempfile.TemporaryDirectory() as directory:
            path = _options(directory, {})
            args, _ = _args(directory, extra=("--native-options-json", str(path), "--execution-mode", "eager"))
            with self.assertRaisesRegex(ValueError, "only for native"):
                native_execution.execution_plan(args)

    def test_duplicate_nonfinite_and_mutated_option_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _options(directory, {})
            args, _ = _args(directory, extra=("--native-options-json", str(path)))
            for text in ('{"debug":false,"debug":true}', '{"block_size":NaN}', '[]'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    native_execution.execution_plan(args)
            path.write_text('{"debug":false}')
            plan = native_execution.execution_plan(args)
            path.write_text('{"debug":true}')
            with self.assertRaisesRegex(RuntimeError, "changed after planning"):
                native_execution._verify_experiment_inputs(plan)

    def test_arm_pin_requires_the_exact_options_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            options = _options(directory, {"debug": False})
            arm = _arm(directory, options=options)
            options.write_text('{ "debug": false }')
            args, _ = _args(directory, extra=("--native-options-json", str(options), "--arm-manifest", str(arm)))
            with self.assertRaisesRegex(ValueError, "options bytes"):
                native_execution.execution_plan(args)


class ExecutionProfileCPUTests(unittest.TestCase):
    def setUp(self):
        import torch
        from deepopt import reset_config

        # In-process execution installs process defaults; restore them here.
        reset_config()
        self.addCleanup(reset_config)
        self.torch = torch
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def arguments(self, directory, *, profile=True, options=None, pinned=True):
        from benchmarks.compiler.installed_runtime import installed_identity

        extra = ["--device", "cpu", "--batch-size", "1", "--synthetic", "--optimizer", "sgd"]
        if profile:
            extra.append("--profile-after-measurement")
        if options is not None:
            extra += ["--native-options-json", str(options)]
        if pinned:
            arm = _arm(directory, identity=installed_identity(native_execution.ROOT, full=True), options=options)
            extra += ["--arm-manifest", str(arm)]
        return _args(directory, tiny=True, extra=extra)

    def check_artifacts(self, report):
        def visit(value):
            if isinstance(value, dict):
                if {"path", "bytes", "sha256"} <= value.keys():
                    payload = Path(value["path"]).read_bytes()
                    self.assertEqual(len(payload), value["bytes"])
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), value["sha256"])
                for part in value.values():
                    visit(part)
            elif isinstance(value, list):
                for part in value:
                    visit(part)
        visit(report)

    def test_real_installed_cpu_profile_is_after_the_frozen_measurement(self):
        active, measured, steps = False, False, []
        original_measure = native_execution._benchmark_updates
        original_profile = execution_profile.run_execution_profile
        original_step = benchmark.train_step
        original_finite = native_execution._finite

        def measure(**kwargs):
            nonlocal measured
            measured = True
            try:
                return original_measure(**kwargs)
            finally:
                measured = False

        def profile(**kwargs):
            nonlocal active
            self.assertFalse(measured)
            frozen_path = kwargs["directory"].parent / "measurement-frozen.json"
            frozen = json.loads(frozen_path.read_text())
            self.assertEqual(frozen["updates_completed"], 2)
            self.assertTrue(frozen["measurement"]["valid"])
            self.assertTrue(frozen["native_ownership_verified"])
            self.assertTrue(frozen["compiler_report_artifact"])
            active = True
            try:
                return original_profile(**kwargs)
            finally:
                active = False

        def step(*args, **kwargs):
            from deepopt.compiler.runtime import current_native_boundary_observer

            steps.append((measured, active, kwargs.get("progress") is not None))
            if measured or active:
                self.assertIsNone(current_native_boundary_observer())
                self.assertFalse(args[1]._optimizer_step_pre_hooks)
            if measured:
                self.assertIsNone(kwargs.get("progress"))
            return original_step(*args, **kwargs)

        def finite(*args, **kwargs):
            self.assertFalse(measured)
            self.assertFalse(self.torch.autograd._profiler_enabled(), "Finite scan contaminated profiler execution")
            return original_finite(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            options = _options(directory, {"max_pointwise_fusion_length": 32})
            args, raw = self.arguments(directory, options=options)
            with patch.object(native_execution, "_benchmark_updates", side_effect=measure), \
                    patch.object(execution_profile, "run_execution_profile", side_effect=profile), \
                    patch.object(benchmark, "train_step", side_effect=step), \
                    patch.object(native_execution, "_finite", side_effect=finite), \
                    patch("examples.vae.qualification.TensorOracle.__init__", side_effect=AssertionError("oracle")), \
                    redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 0, report.get("error"))
            self.assertEqual(report["status"], "completed")
            self.assertTrue(report["measurement"]["valid"])
            self.assertTrue(report["profile"]["complete"], report["profile"])
            self.assertEqual(report["profile"]["evidence_scope"], "CPU_diagnostic_profile_only")
            self.assertFalse(report["profile"]["cuda_execution_evidence"])
            self.assertTrue(report["profile"]["same_measured_program"])
            self.assertEqual(report["profile"]["first_update"], 4)
            self.assertEqual((report["warmup_completed"], report["updates_completed"]), (1, 2))
            self.assertEqual(len(report["steps"]), 3)
            self.assertEqual(steps, [(False, False, True), (True, False, False),
                                     (True, False, False), (False, True, True)])
            self.assertEqual(report["target_settings"], raw)
            self.assertEqual(report["settings"]["runtime"]["compile"]["options"]["max_pointwise_fusion_length"], 32)
            frozen = json.loads(Path(report["measurement_frozen_artifact"]["path"]).read_text())
            self.assertEqual(report["measurement"], frozen["measurement"])
            self.assertEqual(report["memory_segments"], frozen["memory_segments"])
            self.assertEqual(report["cuda_allocator_diagnostics"], frozen["cuda_allocator_diagnostics"])
            self.assertEqual(report["compiler_report_artifact"], frozen["compiler_report_artifact"])
            trace = json.loads(next(Path(value["path"]) for value in report["profile"]["files"]
                                    if Path(value["path"]).name == "trace.json").read_text())
            names = {event["name"] for event in trace["traceEvents"] if "name" in event}
            for phase in ("forward_and_loss", "backward", "optimizer"):
                self.assertIn("vae.profile." + phase, names)
            self.assertFalse([path for path in Path(directory).rglob("*")
                              if path.suffix in {".pt", ".pth", ".pkl", ".pickle", ".npy", ".safetensors"}])
            self.check_artifacts(report)

    def test_profiler_failure_preserves_measurement_and_canonical_compiler_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.arguments(directory)
            counters = [{"allocator_stats": {"num_alloc_retries": value, "num_ooms": value}} for value in (0, 10, 20)]
            with patch.object(self.torch.profiler, "profile", side_effect=RuntimeError("optional profiler unavailable")), \
                    patch.object(benchmark, "cuda_failure_diagnostics", side_effect=counters), \
                    redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 0)
            self.assertTrue(report["measurement"]["valid"])
            self.assertEqual(report["profile"]["status"], "incomplete")
            self.assertIn("optional profiler unavailable", report["profile"]["error"])
            self.assertIn("traceback", report["profile"])
            frozen = json.loads(Path(report["measurement_frozen_artifact"]["path"]).read_text())
            self.assertEqual(report["measurement"], frozen["measurement"])
            self.assertEqual(report["cuda_allocator_diagnostics"], frozen["cuda_allocator_diagnostics"])
            self.assertEqual(report["cuda_allocator_diagnostics"]["allocator_stats"]["num_alloc_retries"], 0)
            self.assertEqual(report["profile"]["allocator_after"]["allocator_stats"]["num_alloc_retries"], 20)
            self.assertEqual(report["compiler_report_artifact"], frozen["compiler_report_artifact"])
            self.check_artifacts(report)

    def test_profile_graph_growth_invalidates_only_its_attribution(self):
        growth = False
        original_evidence = benchmark.native_compilation_evidence
        original_summary = execution_profile._operator_summary
        def summary(profile):
            nonlocal growth
            value = original_summary(profile)
            growth = True
            return value
        def evidence(model):
            value = original_evidence(model)
            if growth:
                value["captured_graphs"] += 1  # Explicit metadata-only graph-growth fault injection.
            return value
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.arguments(directory)
            with patch.object(execution_profile, "_operator_summary", side_effect=summary), \
                    patch.object(benchmark, "native_compilation_evidence", side_effect=evidence), \
                    redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 0)
            self.assertTrue(report["measurement"]["valid"])
            self.assertFalse(report["profile"]["complete"])
            self.assertFalse(report["profile"]["same_measured_program"])
            self.assertEqual(report["profile"]["ownership_after"]["captured_graphs"],
                             report["measurement"]["native_graph_count_after"] + 1)
            compiler = json.loads(Path(report["compiler_report_artifact"]["path"]).read_text())
            self.assertEqual(compiler["captured_graphs"], report["measurement"]["native_graph_count_after"])

    def test_unknown_installed_compiler_option_fails_before_model_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            options = _options(directory, {"not_an_installed_native_pass": True})
            args, _ = self.arguments(directory, profile=False, options=options)
            with patch.object(benchmark, "make_optimizer", side_effect=AssertionError("model allocated")) as allocation, \
                    redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            allocation.assert_not_called()
            self.assertIn("not_an_installed_native_pass", report["error"])
            self.assertFalse(report["execution_completed"])

    def test_installed_package_hash_pin_is_checked_before_model_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.arguments(directory, profile=False)
            manifest = json.loads(args.arm_manifest.read_text())
            manifest["expected_installed"]["source_sha256"] = "f" * 64
            args.arm_manifest.write_text(json.dumps(manifest))
            with patch.object(benchmark, "make_optimizer", side_effect=AssertionError("allocated")) as allocation, \
                    redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            allocation.assert_not_called()
            self.assertIn("differs from the experiment arm pin", report["error"])

    def test_actual_cpu_trace_byte_limit_reports_explicit_omission(self):
        with tempfile.TemporaryDirectory() as directory:
            def update(number, progress):
                progress({"phase": "forward_and_loss"})
                self.torch.ones(3).sin()
            with patch.object(execution_profile, "MAX_TRACE_BYTES", 1):
                result = execution_profile.run_execution_profile(torch=self.torch,
                    device=self.torch.device("cpu"), directory=Path(directory) / "profile",
                    update=update, first_update=16, updates=1)
            self.assertFalse(result["complete"])
            self.assertEqual(result["updates_completed"], 1)
            self.assertEqual(result["omissions"][0]["reason"], "trace_byte_limit")
            self.assertFalse((Path(directory) / "profile/trace.json").exists())
            self.assertFalse(result["cuda_execution_evidence"])
            self.check_artifacts(result)


class ExecutionProfileBoundTests(unittest.TestCase):
    def test_allocator_metadata_is_bounded_and_contains_no_payload(self):
        snapshot = [{"device": 0, "address": 1, "total_size": 100,
                     "blocks": [{"address": index, "size": 1, "state": "active_allocated", "tensor": object()}
                                for index in range(10)]}]
        fake = SimpleNamespace(cuda=SimpleNamespace(memory_snapshot=lambda: snapshot, current_device=lambda: 0))
        with patch.object(execution_profile, "MAX_ALLOCATOR_RECORDS", 3):
            result = execution_profile._allocator_snapshot(fake, SimpleNamespace(type="cuda", index=None))
        self.assertEqual(result["records"], 3)
        self.assertEqual(result["omitted_records"], 8)
        self.assertNotIn('"tensor":', json.dumps(result))


if __name__ == "__main__":
    unittest.main()
