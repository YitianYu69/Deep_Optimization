"""Real installed schema handoff before any VAE input/model/CUDA allocation."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.vae import benchmark_packed_offload, benchmark_vae, native_execution
from tests.benchmarks.test_vae_native_execution import _receipt


RECORDED_DDP_OPTIONS = {"find_unused_parameters": False, "broadcast_buffers": True,
    "gradient_as_bucket_view": True, "bucket_cap_mb": 25.0, "static_graph": False,
    "validation_sync_buffers": True}


def _arguments(directory, *, mode="resident", device="cuda:0", identity=None, preflight=True,
               invalid_ddp=False):
    source, raw = _receipt(directory, tiny=device == "cpu")
    if invalid_ddp:
        raw["runtime"]["ddp"]["bucket_cap_mb"] = "invalid"
        source.write_text(json.dumps({"settings": raw, "optimizer_factory": benchmark_vae.CUSTOM_OPTIMIZER,
                                      "model_factory": None}))
    policy = Path(directory) / (mode + ".policy.json")
    policy.write_text(json.dumps({**benchmark_packed_offload.POLICY_DEFAULTS, "enabled": mode != "resident",
        "policy": "demand" if mode == "demand_offload" else "memory_first_overlap",
        "host_pinned_bytes": 16 << 20, "device_staging_bytes": 16 << 20, "d2h_inflight_bytes": 16 << 20}))
    command = ["--from-result", str(source), "--device", device, "--benchmark", "--warmup", "1", "--updates", "1",
        "--packed-offload-json", str(policy), "--output", str(Path(directory) / (mode + ".json"))]
    if device == "cpu":
        command += ["--batch-size", "1", "--synthetic", "--optimizer", "sgd"]
    if preflight:
        command.append("--configuration-preflight")
    if identity is not None:
        pin = Path(directory) / (mode + ".arm.json")
        pin.write_text(json.dumps({"schema_version": 1, "arm": mode,
            "expected_installed": {key: identity[key] for key in ("version", "source_sha256", "files_hashed")},
            "native_options_sha256": None, "packed_offload_sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}))
        command += ["--arm-manifest", str(pin)]
    return native_execution.parser().parse_args(command), raw


class WorkerConfigurationSchemaTests(unittest.TestCase):
    def setUp(self):
        from deepopt.config import reset_config

        reset_config()
        self.addCleanup(reset_config)

    def test_all_three_actual_worker_plans_validate_the_recorded_envelope_and_keep_DDP_knobs(self):
        import deepopt
        from deepopt.compiler import BackendConfig
        from deepopt.config import config_from_dict

        for mode in benchmark_packed_offload.ARMS:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args, raw = _arguments(directory, mode=mode)
                original = args.from_result.read_bytes()
                plan = native_execution.execution_plan(args)
                self.assertIsNotNone(raw["runtime"]["compile"]["mode"])
                self.assertTrue(raw["runtime"]["compile"]["options"]["use_native_autodiff"])
                self.assertEqual(plan["target_settings"], raw)
                self.assertEqual(plan["settings"]["runtime"]["ddp"], RECORDED_DDP_OPTIONS)
                config = deepopt.load_config(deepcopy(plan["settings"]))
                self.assertEqual(config_from_dict(asdict(config)), config)
                self.assertEqual(config.runtime.ddp, RECORDED_DDP_OPTIONS)
                self.assertEqual(config.runtime.packed_activation_offload.enabled, mode != "resident")
                BackendConfig(**plan["settings"]["runtime"]["compile"]["options"])
                self.assertIsNone(config.runtime.compile.mode)
                self.assertTrue(config.runtime.compile.fullgraph)
                self.assertTrue(config.runtime.compile.options["require_full_aot"])
                self.assertEqual(args.from_result.read_bytes(), original)


class WorkerConfigurationTopologyTests(unittest.TestCase):
    """Guard-boundary metadata units; no group initialization or CUDA evidence."""

    def test_explicit_multi_rank_and_malformed_world_size_fail_without_initialization(self):
        import torch

        for value in ("2", "invalid", "0", "-1", "1.5", ""):
            with self.subTest(value=value), patch.dict(os.environ, {"WORLD_SIZE": value}), \
                    patch.object(torch.distributed, "init_process_group", side_effect=AssertionError("group initialized")), \
                    patch.object(torch.cuda, "set_device", side_effect=AssertionError("CUDA initialized")):
                with self.assertRaisesRegex(ValueError, "WORLD_SIZE"):
                    native_execution._configuration_topology()
                self.assertEqual(os.environ["WORLD_SIZE"], value)

    def test_initialized_multi_rank_group_is_rejected_and_single_process_defaults_are_preserved(self):
        import torch

        with patch.dict(os.environ, {"WORLD_SIZE": "1"}), \
                patch.object(torch.distributed, "is_available", return_value=True), \
                patch.object(torch.distributed, "is_initialized", return_value=True), \
                patch.object(torch.distributed, "get_world_size", return_value=2), \
                patch.object(torch.distributed, "init_process_group", side_effect=AssertionError("group initialized")):
            with self.assertRaisesRegex(ValueError, "initialized multi-rank"):
                native_execution._configuration_topology()
        environment = {key: value for key, value in os.environ.items() if key != "WORLD_SIZE"}
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(torch.distributed, "is_available", return_value=True), \
                patch.object(torch.distributed, "is_initialized", return_value=False), \
                patch.object(torch.distributed, "init_process_group", side_effect=AssertionError("group initialized")), \
                patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")):
            result = native_execution._configuration_topology()
            self.assertEqual(result["requested_world_size"], 1)
            self.assertEqual(result["process_group_world_size"], 1)
            self.assertFalse(result["process_group_initialized"])
            self.assertNotIn("WORLD_SIZE", os.environ)


class WorkerConfigurationPreflightInstalledTests(unittest.TestCase):
    """Run against a real installed candidate; no identity/parser substitution."""

    def setUp(self):
        from deepopt.config import reset_config
        from benchmarks.compiler.installed_runtime import installed_identity

        reset_config()
        self.addCleanup(reset_config)
        self.identity = installed_identity(native_execution.ROOT, full=True)

    def test_all_three_pinned_configuration_preflights_use_the_real_schema_without_allocations(self):
        import torch

        for mode in benchmark_packed_offload.ARMS:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args, raw = _arguments(directory, mode=mode, identity=self.identity)
                with patch.object(benchmark_vae, "make_optimizer", side_effect=AssertionError("optimizer allocated")), \
                        patch.object(benchmark_vae, "build_loader", side_effect=AssertionError("input allocated")), \
                        patch.object(native_execution, "_benchmark_updates", side_effect=AssertionError("measurement executed")), \
                        patch.object(torch.cuda, "set_device", side_effect=AssertionError("CUDA device selected")), \
                        patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA synchronized")), \
                        patch.object(torch.cuda, "Stream", side_effect=AssertionError("CUDA stream allocated")), \
                        redirect_stdout(io.StringIO()):
                    code = native_execution.run_native_execution(args)
                report = json.loads(args.output.read_text())
                self.assertEqual(code, 0, report.get("traceback"))
                self.assertEqual(report["scope"], "configuration_preflight_only")
                self.assertTrue(report["configuration_preflight"]["valid"])
                self.assertTrue(all(report["configuration_preflight"]["checks"].values()))
                self.assertTrue(report["configuration_preflight"]["no_model_allocated"])
                self.assertTrue(report["configuration_preflight"]["no_input_allocated"])
                self.assertFalse(report["configuration_preflight"]["cuda_execution"])
                self.assertFalse(report["execution_completed"])
                self.assertNotIn("measurement", report)
                self.assertNotIn("memory_audit", report)
                self.assertEqual(report["installed"], self.identity)
                self.assertTrue(report["installed_stable"])
                self.assertTrue(report["helper_source_stable"])
                self.assertEqual(report["target_settings"], raw)
                self.assertEqual(report["validated_config"]["runtime"]["ddp"], RECORDED_DDP_OPTIONS)
                self.assertEqual(report["settings"]["runtime"]["ddp"], RECORDED_DDP_OPTIONS)
                self.assertFalse(args.output.parent.joinpath(args.output.stem + "-artifacts").exists())

    def test_invalid_DDP_value_reports_config_validation_before_runtime_or_input(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = _arguments(directory, identity=self.identity, invalid_ddp=True)
            with patch.object(benchmark_vae, "make_optimizer", side_effect=AssertionError("optimizer allocated")), \
                    redirect_stdout(io.StringIO()):
                code = native_execution.run_native_execution(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(code, 1)
            self.assertEqual(report["failure_context"]["phase"], "config_validation")
            self.assertIn("runtime.ddp.bucket_cap_mb", report["error"])
            self.assertFalse(report["configuration_preflight"]["valid"])
            self.assertFalse(report["configuration_preflight"]["checks"]["typed_config"])
            self.assertNotIn("model", report)
            self.assertNotIn("input", report)

    def test_real_CPU_resident_worker_keeps_rich_DDP_settings_and_enabled_CPU_stays_rejected(self):
        import torch

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        for mode in ("resident", "demand_offload"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args, raw = _arguments(directory, mode=mode, device="cpu", identity=self.identity, preflight=False)
                with redirect_stdout(io.StringIO()):
                    code = native_execution.run_native_execution(args)
                report = json.loads(args.output.read_text())
                self.assertEqual(report["target_settings"], raw)
                self.assertEqual(report["settings"]["runtime"]["ddp"], RECORDED_DDP_OPTIONS)
                if mode == "resident":
                    self.assertEqual(code, 0, report.get("traceback"))
                    self.assertTrue(report["native_ownership_verified"])
                    self.assertTrue(report["measurement"]["valid"])
                    self.assertEqual(report["runtime_config"]["runtime"]["ddp"], RECORDED_DDP_OPTIONS)
                    self.assertEqual(report["updates_completed"], 1)
                else:
                    self.assertEqual(code, 1)
                    self.assertEqual(report["failure_context"]["phase"], "packed_offload_device_validation")
                    self.assertIn("requires native compiled CUDA execution", report["error"])
                    self.assertNotIn("model", report)


if __name__ == "__main__":
    unittest.main()
