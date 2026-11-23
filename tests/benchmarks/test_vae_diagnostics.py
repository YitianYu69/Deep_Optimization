"""Diagnostic planning and failure orchestration; CUDA execution is separate."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from examples.vae import benchmark_vae as benchmark
from examples.vae import diagnostics


class VAEDiagnosticPlanningTests(unittest.TestCase):
    def arguments(self, directory, extra=()):
        config = Path(benchmark.__file__).with_name("h200_bf16_storage_diagnostics.toml")
        raw = benchmark.settings(benchmark.parser().parse_args(
            ["--config", str(config), "--bf16-policy", "bf16_storage"]))
        raw["training"]["batch_size"] = 128
        raw["runtime"]["activation_memory"] = dict(multiply_backward="packed_streaming", pack_reuse="inherit",
            reduction_scratch_bytes=8388608, recompute_modules=[])
        receipt = Path(directory) / "previous.json"
        receipt.write_text(json.dumps({"settings": raw, "optimizer_factory": "project:OriginalOptimizer",
            "model_factory": "mbconv_vae:MBConvVAE_V2", "workload_sha256": "target"}))
        args = benchmark.parser().parse_args(["--diagnose", "--diagnostic-from-result", str(receipt),
            "--output", str(Path(directory) / "diagnostic.json"), *extra])
        return args, raw

    def test_complete_matrix_preserves_target_and_separates_small_diagnostic_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            args, raw = self.arguments(directory)
            plan = diagnostics.diagnostic_plan(args)
        self.assertEqual(len(plan["cases"]), 13)
        self.assertEqual(plan["cases"][0]["name"], "streamed_eager_native_untraced")
        self.assertFalse(plan["cases"][0]["observe_boundaries"])
        self.assertEqual(plan["target_settings"], raw)
        self.assertEqual(plan["settings"]["training"]["batch_size"], 1)
        self.assertEqual(plan["settings"]["application"], raw["application"])
        self.assertEqual(plan["optimizer_factory"], "project:OriginalOptimizer")
        self.assertFalse(plan["qualification_eligible"])
        self.assertFalse(plan["timing_eligible"])
        self.assertEqual(plan["diagnostic_overrides"]["input"], "first_dataset_batch")
        self.assertEqual(raw["training"]["batch_size"], 128)

    def test_route_variants_freeze_explicit_fusion_and_keep_optimizer_and_group_size(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.arguments(directory)
            raw = diagnostics.diagnostic_plan(args)["settings"]
        for profile, (compressed, algorithm) in diagnostics.PROFILES.items():
            with self.subTest(profile=profile):
                selected = diagnostics._route_settings(raw, {"execution": "native", "profile": profile})
                runtime = selected["runtime"]
                self.assertEqual(runtime["activation_compression"]["enabled"], compressed)
                self.assertEqual(runtime["activation_compression"]["group_size"], 1024)
                self.assertEqual(runtime["activation_memory"]["multiply_backward"], algorithm)
                self.assertEqual(runtime["compile"]["options"]["fuse_packed_multiply"], algorithm == "packed_streaming")
                self.assertEqual(selected["training"]["optimizer"], "custom")
        self.assertEqual(raw["runtime"]["activation_memory"]["multiply_backward"], "packed_streaming")

    def test_recorded_native_options_and_inductor_mode_reach_real_forward_backward(self):
        import torch
        from deepopt.config import ConfigError, config_from_dict

        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.arguments(directory, ("--diagnostic-device", "cpu", "--diagnostic-synthetic",
                "--optimizer", "sgd", "--diagnostic-cases", "dense_eager_repeat"))
            receipt = json.loads(args.diagnostic_from_result.read_text())
            receipt["settings"]["application"]["model"] = dict(
                base_channels=2, max_channels=8, latent_channels=1, blocks_per_stage=1)
            receipt["settings"]["application"]["data"].update(image_size=16, resize_size=20)
            args.diagnostic_from_result.write_text(json.dumps(receipt))
            plan = diagnostics.diagnostic_plan(args)
            frozen = deepcopy(plan["settings"])
            self.assertEqual(frozen["runtime"]["compile"]["mode"], "max-autotune-no-cudagraphs")
            self.assertTrue(frozen["runtime"]["compile"]["options"]["use_native_autodiff"])
            with self.assertRaisesRegex(ConfigError, "Choose runtime.compile.mode"):
                config_from_dict(frozen)
            task = {**plan, "case": plan["cases"][0], "directory": directory}
            result = {}
            torch.manual_seed(42)
            with redirect_stdout(io.StringIO()):
                diagnostics._run_trajectory(task, result, torch.device("cpu"), torch, benchmark)
        self.assertEqual(plan["settings"], frozen)
        self.assertEqual(result["beta"], 0.0)
        self.assertTrue(result["trajectory"]["numerical_comparisons_passed"])
        self.assertFalse(result["trajectory"]["qualification_eligible"])
        self.assertTrue(result["trajectory"]["updates"][0]["native"]["backward_completed"])

    def test_dry_run_requires_no_torch_and_never_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.arguments(directory)
            command = [sys.executable, "-S", str(Path(benchmark.__file__).resolve()), "--diagnose",
                "--diagnostic-from-result", str(args.diagnostic_from_result), "--diagnostic-dry-run"]
            completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertFalse(report["qualification_eligible"])
        self.assertEqual(report["diagnostic_overrides"]["batch_size"], 1)

    def test_cpu_oracle_budget_is_declared_and_explicit_lower_limit_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.arguments(directory)
            self.assertEqual(diagnostics.diagnostic_plan(args)["max_bytes"], 1024 * 2**20)
            args.diagnostic_max_bytes = 512 * 2**20
            self.assertEqual(diagnostics.diagnostic_plan(args)["max_bytes"], 512 * 2**20)

    def test_explicit_determinism_control_preserves_original_receipt_and_selects_small_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            args, raw = self.arguments(directory, ("--diagnostic-deterministic", "--diagnostic-cases",
                "backward_repeatability", "dense_eager_repeat", "dense_native_repeat", "dense_eager_native"))
            original = args.diagnostic_from_result.read_bytes()
            plan = diagnostics.diagnostic_plan(args)
            self.assertEqual(args.diagnostic_from_result.read_bytes(), original)
        self.assertEqual(plan["target_settings"], raw)
        self.assertFalse(raw["runtime"]["deterministic"])
        self.assertTrue(plan["settings"]["runtime"]["deterministic"])
        self.assertEqual(plan["cases"][0]["name"], "backward_repeatability")
        self.assertEqual(len(plan["cases"]), 4)
        self.assertEqual(plan["diagnostic_overrides"]["deterministic_algorithms"]["source"], "explicit_diagnostic_control")
        self.assertFalse(plan["qualification_eligible"])

    def test_legacy_deterministic_enables_control_and_explicit_disable_is_respected(self):
        for options, expected in [(("--deterministic",), True),
                                  (("--deterministic", "--no-diagnostic-deterministic"), False)]:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                args, _ = self.arguments(directory, options)
                self.assertEqual(diagnostics.diagnostic_plan(args)["settings"]["runtime"]["deterministic"], expected)

    def test_invalid_values_and_duplicate_cases_reject_before_launch(self):
        for extra in (("--diagnostic-batch-size", "0"), ("--diagnostic-batch-size", "129"),
                      ("--diagnostic-max-bytes", "0"),
                      ("--diagnostic-cases", "dense_eager_repeat", "dense_eager_repeat"),
                      ("--diagnostic-cases", "unknown")):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                args, _ = self.arguments(directory, extra)
                with self.assertRaises(ValueError):
                    diagnostics.diagnostic_plan(args)

    def test_parent_retains_failed_case_and_runs_remaining_fresh_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.arguments(directory, ("--diagnostic-deterministic", "--diagnostic-cases",
                "dense_eager_repeat", "streamed_eager_native"))
            identity = {"version": "candidate", "install_root": directory}
            commands = []
            def execute(command, **kwargs):
                commands.append(command)
                self.assertEqual(kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
                task = json.loads(Path(command[command.index("--_diagnostic-task") + 1]).read_text())
                target = Path(command[command.index("--output") + 1])
                failed = len(commands) == 1
                target.write_text(json.dumps({"case": task["case"], "settings_sha256": task["settings_sha256"],
                    "qualification_eligible": False, "timing_eligible": False, "installed": identity,
                    "status": "failed" if failed else "passed",
                    "error": "ConfigError: invalid diagnostic settings" if failed else None,
                    "qualification_evidence": {"oracle": {"comparisons": [{"name": "gradient/input", "passed": False}]}} if failed else {}}))
                return SimpleNamespace(returncode=1 if failed else 0)
            console = io.StringIO()
            with patch("benchmarks.compiler.installed_runtime.installed_identity", return_value=identity), \
                    patch("benchmarks.compiler.installed_runtime.child_environment", return_value={"PYTHONPATH": directory}), \
                    patch.object(diagnostics.subprocess, "run", side_effect=execute), redirect_stdout(console):
                self.assertEqual(diagnostics.run_diagnostics(args), 1)
            report = json.loads(args.output.read_text())
            self.assertEqual(len(commands), 2)
            self.assertEqual([row["status"] for row in report["runs"]], ["failed", "passed"])
            canonical = json.loads(Path(report["runs"][0]["result_path"]).read_text())
            self.assertEqual(canonical["qualification_evidence"]["oracle"]["comparisons"][0]["name"], "gradient/input")
            self.assertIn("result_file", report["runs"][0])
            self.assertFalse(report["all_cases_passed"])
            self.assertFalse(report["qualification_eligible"])
            self.assertNotEqual(commands[0], commands[1])
            self.assertIn("ConfigError: invalid diagnostic settings", console.getvalue())

    def test_exception_evidence_is_metadata_only_and_original_cause_is_retained(self):
        error = AssertionError("first numerical failure")
        error.qualification_evidence = {"oracle": {"comparisons": [{"name": "gradient/input", "passed": False}]}}
        try:
            raise RuntimeError("transfer rejected") from error
        except RuntimeError as wrapped:
            details = benchmark.failure_error_details(wrapped)
        self.assertEqual(details["qualification_evidence"], error.qualification_evidence)
        self.assertIn("first numerical failure", details["traceback"])
        error.qualification_evidence = {"payload": object()}
        details = benchmark.failure_error_details(error)
        self.assertNotIn("qualification_evidence", details)
        self.assertEqual(details["qualification_evidence_collection_error"], "non_serializable_metadata")

    def test_elementwise_and_relative_gradient_failure_reasons_are_preserved_independently(self):
        import torch
        from examples.vae.qualification import TensorOracle

        pairs = ((torch.ones(1), torch.zeros(1), False, False),
                 (torch.tensor([1e-8]), torch.tensor([2e-8]), True, False))
        reference = torch.ones(16384)
        candidate = reference.clone()
        candidate[-1] += 1e-4
        pairs += ((reference, candidate, False, True),)
        for expected, actual, pointwise, relative in pairs:
            with self.subTest(pointwise=pointwise, relative=relative):
                oracle = TensorOracle(diagnostic=True)
                oracle.record("gradient/input", expected)
                oracle.compare("gradient/input", actual)
                row = oracle.snapshot()["comparisons"][0]
                self.assertFalse(row["passed"])
                self.assertEqual(row["elementwise_passed"], pointwise)
                self.assertEqual(row["relative_l2_passed"], relative)
                self.assertEqual(len(row["failure_reasons"]), int(not pointwise) + int(not relative))

    def test_external_model_factory_is_preserved_and_synthetic_input_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.arguments(directory, ("--diagnostic-synthetic",))
            old = json.loads(args.diagnostic_from_result.read_text())
            old["model_factory"] = "external:OriginalVAE"
            args.diagnostic_from_result.write_text(json.dumps(old))
            plan = diagnostics.diagnostic_plan(args)
        self.assertEqual(plan["model_factory"], "external:OriginalVAE")
        self.assertEqual(plan["diagnostic_overrides"]["input"], "seeded_synthetic")

    def test_operator_worker_preserves_completed_probes_and_scratch_cap_after_failure(self):
        import torch

        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.arguments(directory, ("--diagnostic-device", "cpu", "--diagnostic-synthetic",))
            plan = diagnostics.diagnostic_plan(args)
            plan["settings"]["runtime"]["activation_memory"]["reduction_scratch_bytes"] = 4096
            identity = {"version": "candidate"}
            task = {**plan, "case": {"name": "operator_replays", "kind": "operators"},
                    "installed": identity, "helper_sha256": diagnostics._helper_identity(), "directory": directory}
            args._diagnostic_task = Path(directory) / "task.json"
            args._diagnostic_task.write_text(json.dumps(task))
            with patch("benchmarks.compiler.installed_runtime.installed_identity", return_value=identity), \
                    patch("examples.vae.diagnostic_operators.run_packed_diagnostics", return_value={"passed": True}) as packed, \
                    patch("examples.vae.diagnostic_localization.nearest_replay_diagnostics", side_effect=ValueError("nearest probe failure")), \
                    patch("examples.vae.diagnostic_localization.chain_replay_diagnostics", return_value={"passed": True}) as chain:
                self.assertEqual(diagnostics.diagnostic_worker(args), 1)
            result = json.loads(args.output.read_text())
        self.assertEqual(packed.call_args.kwargs["scratch_bytes"], 4096)
        self.assertTrue(result["operators"]["packed"]["passed"])
        self.assertFalse(result["operators"]["nearest"]["passed"])
        self.assertIn("nearest probe failure", result["operators"]["nearest"]["traceback"])
        self.assertTrue(result["operators"]["chain"]["passed"])
        chain.assert_called_once()
        self.assertFalse(result["qualification_eligible"])

    def test_failed_secondary_cuda_collection_preserves_original_worker_error(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.arguments(directory)
            plan = diagnostics.diagnostic_plan(args)
            task = {**plan, "case": plan["cases"][0], "installed": {"version": "expected"},
                    "directory": directory, "helper_sha256": diagnostics._helper_identity()}
            args._diagnostic_task = Path(directory) / "task.json"
            args._diagnostic_task.write_text(json.dumps(task))
            with patch("benchmarks.compiler.installed_runtime.installed_identity", side_effect=RuntimeError("original identity failure")), \
                    patch.object(benchmark, "cuda_failure_diagnostics", side_effect=RuntimeError("secondary failure")):
                self.assertEqual(diagnostics.diagnostic_worker(args), 1)
            result = json.loads(args.output.read_text())
        self.assertIn("original identity failure", result["error"])
        self.assertEqual(result["cuda_diagnostics_collection_error"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
