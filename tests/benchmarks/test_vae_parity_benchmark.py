"""Matched model-only compiler scope, real CPU AOT backward, and failed peers."""

from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
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

from examples.vae import benchmark_parity, parity_execution
from tests.benchmarks.test_vae_memory_comparison import audited_row
from tests.benchmarks.test_vae_native_execution import _receipt


def _arguments(directory, *, cohort=False, extra=()):
    receipt, raw = _receipt(directory)
    owner = benchmark_parity if cohort else parity_execution
    args = owner.parser().parse_args(["--from-result", str(receipt), "--output", str(Path(directory) / "result.json"),
        "--device", "cpu", "--cpu-tiny", "--batch-size", "1", "--synthetic", "--optimizer", "sgd",
        "--warmup", "1", "--updates", "1", *extra])
    return args, raw


@contextmanager
def _evidence_directory():
    destination = os.environ.get("DEEPOPT_VAE_PARITY_TEST_EVIDENCE")
    if destination:
        path = Path(destination).resolve()
        path.mkdir(parents=True, exist_ok=False)
        yield str(path)
    else:
        with tempfile.TemporaryDirectory() as directory:
            yield directory


def _worker(plan, identity=None):
    """Scalar-only acceptance fixture; no compiler/CUDA execution is implied."""
    compiled = plan["route"] in ("native_torch_inductor", "deepopt_native")
    row = {**deepcopy(plan), "status": "completed", "execution_completed": True,
        "warmup_completed": plan["warmup"], "updates_completed": plan["updates"],
        "installed": identity or {"version": parity_execution.EXPECTED_VERSION, "source_sha256": "1" * 64},
        "installed_stable": True, "helper_source_stable": True, "source_stable": True, "route_ownership_verified": True,
        "native_ownership_verified": plan["route"] == "deepopt_native", "target_CUDA_execution": False,
        "input": {"sha256": "2" * 64, "shape": plan["requested_input_shape"], "requires_grad": False},
        "model": {"initial_state_sha256": "3" * 64, "parameters": 100, "trainable_parameter_tensors": 20},
        "optimizer": {"initial_state_sha256": "4" * 64, "name": "sgd"}, "provenance": {"optimizer_source_sha256": "5" * 64},
        "software": {"torch": "CPU_metadata_fixture"}, "helper_sha256": {"fixture": "6" * 64},
        "settings": deepcopy(plan["common_settings"]), "ownership": {"captured_graphs": int(compiled), "runtime_fallbacks": 0,
            "owner": "deepopt_native_autodiff" if plan["route"] == "deepopt_native" else "pytorch_aotautograd" if compiled else "pytorch_autograd"},
        "rng_before_warmup_sha256": "7" * 64, "rng_after_warmup_sha256": "8" * 64, "rng_after_measurement_sha256": "9" * 64,
        "measurement": {"valid": True, "wall_ms_per_update": 2 if compiled else 4,
            "measured_updates": plan["updates"], "native_graph_count_before": int(compiled), "native_graph_count_after": int(compiled),
            "postchecks": {"finite_losses": True, "finite_parameters": True}, "memory": {"peak_allocated_bytes": None, "peak_reserved_bytes": None}},
        "memory": {"startup": {}, "total": {}}, "inductor_validation": {"passed": True}}
    row["memory_audit"] = audited_row(device="cpu")["memory_audit"]
    controls = {"backward_pass_autocast": "off", "fallback_random": True, "emulate_precision_casts": plan["inductor_preset"] == "cast-preserving"}
    row["numerical_controls"] = {"preset": plan["inductor_preset"], "requested": controls, "effective": dict(controls),
        "applied": dict(controls), "tf32_matmul": False, "tf32_cudnn": True, "bf16_reduced_precision_reduction": True,
        "backward_execution": "outside autocast", "random_policy": "fallback_random=True (explicit eager RNG diagnostic)"}
    return row


def _pin_installed(args, directory):
    from benchmarks.compiler.installed_runtime import installed_identity

    identity = installed_identity(parity_execution.ROOT, full=True)
    pin = Path(directory) / "release.arm.json"
    pin.write_text(json.dumps({"schema_version": 1, "arm": "parity_release",
        "expected_installed": {name: identity[name] for name in ("version", "source_sha256", "files_hashed")},
        "native_options_sha256": None}))
    args.arm_manifest = pin


class VAEParityPlanningTests(unittest.TestCase):
    def test_four_routes_and_two_inductor_presets_are_stdlib_only_and_preserve_original(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, raw = _receipt(directory)
            original = receipt.read_bytes()
            command = [sys.executable, "-S", str(Path(benchmark_parity.__file__)), "--from-result", str(receipt), "--dry-run"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(receipt.read_bytes(), original)
        self.assertEqual(len(plan["workers"]), 5)
        recipes = set()
        for worker in plan["workers"].values():
            self.assertEqual(worker["target_settings"], raw)
            self.assertEqual(worker["requested_input_shape"], [16, 3, 224, 224])
            self.assertEqual(worker["common_settings"]["application"]["model"], raw["application"]["model"])
            self.assertEqual(worker["common_settings"]["training"]["optimizer"], "custom")
            self.assertEqual(worker["common_settings"]["runtime"]["bf16_policy"], "torch_amp")
            self.assertFalse(worker["common_settings"]["runtime"]["activation_compression"]["enabled"])
            self.assertEqual(worker["common_settings"]["runtime"]["activation_memory"]["multiply_backward"], "materialized")
            self.assertEqual(worker["common_settings"]["runtime"]["activation_compression"]["group_size"], 1024)
            self.assertFalse(worker["external_input_requires_grad"])
            self.assertEqual(worker["compiled_scope"]["loss"], "eager_FP32_L1_KL")
            self.assertEqual(worker["compiled_scope"]["optimizer"], "eager")
            self.assertFalse(worker["diagnostic_overrides"])
            recipes.add(worker["common_recipe_sha256"])
        self.assertEqual(len(recipes), 1)

    def test_cpu_and_reduced_model_require_explicit_diagnostic_control(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = _arguments(directory)
            args.cpu_tiny = False
            with self.assertRaisesRegex(ValueError, "explicit --cpu-tiny"):
                parity_execution.execution_plan(args)
            args.cpu_tiny, args.device = True, "cuda:0"
            with self.assertRaises(ValueError):
                parity_execution.execution_plan(args)

    def test_compression_and_owner_overrides_are_rejected(self):
        for options in ({"fuse_packed_multiply": True}, {"share_compatible_packs": True},
                        {"bf16_policy": "bf16_storage"}, {"use_native_autodiff": False}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "options.json"
                path.write_text(json.dumps(options))
                args, _ = _arguments(directory, extra=("--native-options-json", str(path)))
                with self.assertRaises(ValueError):
                    parity_execution.execution_plan(args)


class VAEParityAggregationTests(unittest.TestCase):
    def setUp(self):
        from importlib.metadata import version

        # Aggregation fixtures exercise accounting, not one historical release;
        # the arm-manifest tests keep the exact wheel pin.
        expected = patch.object(parity_execution, "EXPECTED_VERSION", version("deep-optimization"))
        expected.start()
        self.addCleanup(expected.stop)

    def test_profile_only_OOM_counters_do_not_replace_the_clean_window(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = _arguments(directory)
            row = _worker(parity_execution.execution_plan(args))
        row["device"] = "cuda:0"
        row["memory_audit"] = audited_row(device="cuda:0")["memory_audit"]
        row["measurement"].update(cuda_event_ms_per_update=1, memory={"peak_allocated_bytes": 100, "peak_reserved_bytes": 200})
        snapshots = row["memory_audit"]["snapshots"]
        snapshots["profile_end"] = deepcopy(snapshots["after_postchecks"])
        snapshots["profile_end"]["ooms"] = snapshots["cleanup"]["ooms"] = 1
        self.assertNotIn("CUDA_OOM_or_allocator_retry", benchmark_parity.worker_errors(row))
        snapshots["measurement_end"]["ooms"] = 1
        self.assertIn("CUDA_OOM_or_allocator_retry", benchmark_parity.worker_errors(row))

    def test_matched_metadata_has_separate_presets_and_does_not_declare_CPU_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = _arguments(directory, cohort=True)
            plan = benchmark_parity.benchmark_plan(args)
            rows = [_worker(value) | {"job": key, "block": 1} for key, value in plan["workers"].items()]
            result = benchmark_parity.aggregate(rows, plan["workers"], 1)
        self.assertTrue(result["valid"], result["invalid_reasons"])
        self.assertEqual(set(result["ratios"]), {"default", "cast-preserving"})
        self.assertFalse(result["target_CUDA_parity_established"])
        self.assertFalse(result["performance_policy_promoted"])

    def test_state_rng_scope_and_software_drift_suppress_ratios(self):
        for field in ("common_recipe_sha256", "rng_after_measurement_sha256", "model", "software", "compiled_scope", "tf32"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                args, _ = _arguments(directory, cohort=True)
                plan = benchmark_parity.benchmark_plan(args)
                rows = [_worker(value) | {"job": key, "block": 1} for key, value in plan["workers"].items()]
                if field == "tf32":
                    rows[-1]["numerical_controls"]["tf32_matmul"] = True
                else:
                    rows[-1][field] = "changed"
                result = benchmark_parity.aggregate(rows, plan["workers"], 1)
                self.assertFalse(result["valid"])
                self.assertFalse(result["ratios"])

    def test_failed_and_malformed_peers_do_not_stop_remaining_workers(self):
        from benchmarks.compiler.installed_runtime import installed_identity

        identity = installed_identity(parity_execution.ROOT, full=True)
        for failure in ("OOM", "malformed"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                args, _ = _arguments(directory, cohort=True)
                launched = []
                def execute(command, **kwargs):
                    options = parity_execution.parser().parse_args(command[3:])
                    launched.append(options.route)
                    row = _worker(parity_execution.execution_plan(options), identity)
                    if len(launched) == 1:
                        if failure == "OOM":
                            row.update(status="failed", execution_completed=False, error="original CUDA OOM")
                        else:
                            row["measurement"] = None
                    options.output.write_text(json.dumps(row))
                    return SimpleNamespace(returncode=1 if failure == "OOM" and len(launched) == 1 else 0)
                with patch.object(benchmark_parity.subprocess, "run", side_effect=execute), \
                        patch.object(benchmark_parity, "_verify_route_artifacts", return_value={"passed": True, "scope": "metadata_fixture_only"}), \
                        redirect_stdout(io.StringIO()):
                    code = benchmark_parity.run_benchmark(args)
                report = json.loads(args.output.read_text())
                self.assertEqual(code, 1)
                self.assertEqual(len(launched), 5)
                self.assertEqual(len(report["results"]), 5)
                self.assertEqual(report["summary"]["modes"]["deepopt_native:cast-preserving"]["successful_trials"], 1)
                self.assertFalse(report["summary"]["ratios"])
                for row in report["results"]:
                    path = args.output.parent / row["result"]["path"]
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["result"]["sha256"])


class VAEParityInstalledCPUTests(unittest.TestCase):
    def setUp(self):
        from deepopt import reset_config

        # In-process workers install process defaults; restore them here.
        reset_config()
        self.addCleanup(reset_config)

    def test_blocked_CUDA_attempt_is_not_target_execution_evidence(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = _receipt(directory)
            args = parity_execution.parser().parse_args(["--from-result", str(receipt), "--route", "native_torch_inductor",
                "--output", str(Path(directory) / "blocked.json")])
            _pin_installed(args, directory)
            with patch.object(torch.cuda, "is_available", return_value=False), redirect_stdout(io.StringIO()):
                code = parity_execution.run_worker(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            self.assertIn("CUDA unavailable", report["error"])
            self.assertFalse(report["execution_completed"])
            self.assertFalse(report["target_CUDA_execution"])
            self.assertFalse(report["measurement"]["valid"])

    def test_actual_all_routes_include_model_only_inductor_forward_and_lazy_AOT_backward(self):
        # Every child imports the actual pinned installed wheel. The VAE is
        # reduced only by the explicit --cpu-tiny flag, with no fake backend.
        with _evidence_directory() as directory:
            args, _ = _arguments(directory, cohort=True)
            from benchmarks.compiler.installed_runtime import installed_identity
            identity = installed_identity(parity_execution.ROOT, full=True)
            pin = Path(directory) / "release.arm.json"
            pin.write_text(json.dumps({"schema_version": 1, "arm": "parity_release",
                "expected_installed": {name: identity[name] for name in ("version", "source_sha256", "files_hashed")},
                "native_options_sha256": None}))
            options = Path(directory) / "native-options.json"
            options.write_text(json.dumps({"max_pointwise_fusion_length": 32}))
            args.arm_manifest, args.native_options_json = pin, options
            with redirect_stdout(io.StringIO()):
                code = benchmark_parity.run_benchmark(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 0, report.get("error") or report.get("summary"))
            self.assertTrue(report["summary"]["valid"])
            self.assertFalse(report["summary"]["target_CUDA_parity_established"])
            rows = []
            for receipt in report["results"]:
                row = json.loads((args.output.parent / receipt["result"]["path"]).read_text())
                rows.append(row)
                self.assertEqual(row["status"], "completed", row.get("error"))
                self.assertTrue(row["memory_audit"]["complete"])
                self.assertTrue(row["memory_audit"]["cleanup"]["completed"])
                self.assertFalse(row["memory_audit"]["cuda_execution_evidence"])
                self.assertTrue(row["route_ownership_verified"])
                self.assertTrue(row["measurement"]["postchecks"]["finite_losses"])
                self.assertTrue(row["measurement"]["postchecks"]["finite_parameters"])
                self.assertEqual(row["measurement"]["native_graph_count_before"], row["measurement"]["native_graph_count_after"])
                self.assertTrue(row["compiler_cache_protocol"]["fresh_per_worker"])
                self.assertTrue(all(step["fp32_gradients"] and step["all_trainable_gradients_present"] for step in row["steps"] if step["stage"] == "warmup"))
                if row["route"] == "native_torch_inductor":
                    self.assertTrue(row["inductor_validation"]["passed"], row["inductor_validation"])
                    self.assertFalse(row["inductor_validation"]["cuda_execution_observed"])
                    self.assertTrue(row["inductor_artifacts"]["compiled_phase_artifacts"]["forward"])
                    self.assertTrue(row["inductor_artifacts"]["compiled_phase_artifacts"]["backward"])
                    self.assertEqual(row["numerical_controls"]["effective"]["backward_pass_autocast"], "off")
                    self.assertTrue(row["numerical_controls"]["effective"]["fallback_random"])
                    self.assertEqual(row["numerical_controls"]["effective"]["emulate_precision_casts"], row["inductor_preset"] == "cast-preserving")
                    for capture in row["native_torch_captures"]:
                        self.assertTrue(capture["provenance"]["native_ops_only"])
                        self.assertFalse(capture["provenance"]["opaque_custom_boundaries"])
                        self.assertNotIn("l1_loss", Path(capture["artifact"]["path"]).read_text())
            self.assertEqual(len({row["model"]["initial_state_sha256"] for row in rows}), 1)
            self.assertEqual(len({row["optimizer"]["initial_state_sha256"] for row in rows}), 1)
            self.assertEqual(len({row["input"]["sha256"] for row in rows}), 1)
            self.assertFalse([path for path in Path(directory).rglob("*") if path.suffix in {".pt", ".pth", ".pkl", ".pickle", ".npy", ".safetensors"}])

    def test_optional_CPU_profile_failure_preserves_the_clean_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = _arguments(directory, extra=("--route", "native_torch_eager", "--profile-after-measurement"))
            _pin_installed(args, directory)
            with patch("examples.vae.execution_profile.run_execution_profile", side_effect=RuntimeError("optional profiler failed")), redirect_stdout(io.StringIO()):
                code = parity_execution.run_worker(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 0, report.get("error"))
            self.assertTrue(report["measurement"]["valid"])
            self.assertFalse(report["profile"]["complete"])
            self.assertIn("optional profiler failed", report["profile"]["error"])
            frozen = json.loads(Path(report["measurement_frozen_artifact"]["path"]).read_text())
            self.assertEqual(report["measurement"], frozen["measurement"])
            self.assertEqual(report["memory"], frozen["memory"])


if __name__ == "__main__":
    unittest.main()
