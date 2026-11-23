"""Cold policy/configuration checks; these do not execute CUDA transport."""

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt._config_resolution import ResolvedConfig, resolve_config
from deepopt._packed_offload import PackedActivationOffloadPolicy, resolve_packed_offload, resolve_owned_packed_offload
from deepopt.compiler import BackendConfig, compile_fx
from deepopt.compiler.registry import KernelCodegenContext
from deepopt.config import ConfigError, RuntimeConfig, config_from_dict
from deepopt.engine.execution.configuration import prepare_execution_model, runtime_signature


def _policy(**changes):
    return {"enabled": True, "host_pinned_bytes": 4096, "device_staging_bytes": 4096,
            "d2h_inflight_bytes": 4096, **changes}


def _config(**changes):
    runtime = {"device": "cuda:0", "precision": "bf16",
               "compile": {"enabled": True, "backend": "deepopt", "options": {}},
               "packed_activation_offload": _policy(), **changes}
    return {"name": "offload-policy", "seed": 17,
            "training": {"epochs": 1, "batch_size": 1, "learning_rate": .01}, "runtime": runtime}


# Exactly the inactive option table retained by the original H200 receipt.
RECORDED_DDP_OPTIONS = {"find_unused_parameters": False, "broadcast_buffers": True,
    "gradient_as_bucket_view": True, "bucket_cap_mb": 25.0, "static_graph": False,
    "validation_sync_buffers": True}


class PackedOffloadPolicyTests(unittest.TestCase):
    def setUp(self):
        from deepopt.config import reset_config

        reset_config()
        self.addCleanup(reset_config)

    def test_torch_free_frozen_owned_json_descriptor_and_disabled_defaults(self):
        environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
        result = subprocess.run([sys.executable, "-c", """
import sys
from deepopt._packed_offload import resolve_packed_offload
from deepopt.config import RuntimeConfig
assert not RuntimeConfig().packed_activation_offload.enabled
assert resolve_packed_offload().as_dict()['version'] == 1
assert 'torch' not in sys.modules
assert 'triton' not in sys.modules
"""], env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        raw = _policy()
        policy = resolve_packed_offload(raw)
        raw["host_pinned_bytes"] = 1
        self.assertEqual(policy.host_pinned_bytes, 4096)
        with self.assertRaises(FrozenInstanceError):
            policy.enabled = False
        self.assertEqual(resolve_packed_offload(json.loads(json.dumps(policy.as_dict()))), policy)
        self.assertEqual(policy.mode, "predictive")
        self.assertEqual(resolve_packed_offload(_policy(policy="demand")).mode, "demand")
        self.assertFalse(BackendConfig().packed_activation_offload.enabled)
        self.assertIsNone(KernelCodegenContext().packed_activation_offload)

    def test_strict_types_unknown_versions_and_explicit_enabled_budgets(self):
        invalid = ({"enabled": 1}, {"enabled": True}, {"host_pinned_bytes": True},
                   {"copy_chunk_bytes": 0}, {"max_live_forwards": False}, {"max_backward_tasks": -1},
                   {"policy": "predictive"}, {"admission": "ignore"}, {"peak_budget_bytes": False},
                   {"host_pageable_bytes": -1}, {"latency_target_percent": float("nan")},
                   {"latency_target_percent": True}, {"latency_target_percent": 10**1000},
                   {"version": True}, {"version": 3}, {"surprise": 1}, {1: "bad"})
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                resolve_packed_offload(raw)
        pageable = resolve_packed_offload(_policy(host_pageable_bytes=32768))
        self.assertEqual(pageable.canonical_host_bytes, 32768)
        self.assertEqual(resolve_packed_offload(_policy()).canonical_host_bytes, 4096)
        self.assertIs(type(resolve_packed_offload(_policy(latency_target_percent=1)).latency_target_percent), float)

    def test_explicit_v2_preserves_physical_semantics_and_v1_wire_format(self):
        legacy = resolve_packed_offload(_policy())
        expected = {"enabled": True, "policy": "memory_first_overlap", "host_pinned_bytes": 4096,
            "device_staging_bytes": 4096, "d2h_inflight_bytes": 4096, "copy_chunk_bytes": 16777216,
            "max_live_forwards": 1, "max_backward_tasks": 1, "peak_budget_bytes": None,
            "latency_target_percent": 1.0, "admission": "resident_fallback", "host_pageable_bytes": 0}
        self.assertEqual(legacy.as_config(), expected)
        self.assertEqual(legacy.as_dict(), {"version": 1, **expected})
        self.assertFalse(legacy.physical_host_budget)
        self.assertNotIn("version", resolve_config(_config(), base=None)["runtime"]["packed_activation_offload"])
        physical = resolve_packed_offload({"version": 2, **expected})
        self.assertTrue(physical.physical_host_budget)
        self.assertNotEqual(legacy, physical)
        self.assertEqual(physical.as_config(), {**expected, "version": 2})
        self.assertEqual(resolve_packed_offload(json.loads(json.dumps(physical.as_dict()))), physical)
        raw = _config(packed_activation_offload=physical.as_config())
        raw["runtime"]["compile"]["options"]["packed_activation_offload"] = physical.as_config()
        parsed = config_from_dict(raw)
        self.assertEqual(resolve_config(raw, base=None)["runtime"]["packed_activation_offload"]["version"], 2)
        self.assertEqual(parsed.runtime.packed_activation_offload, physical)
        self.assertEqual(config_from_dict(asdict(parsed)), parsed)
        self.assertEqual(BackendConfig.from_options(parsed.runtime.compile.options).packed_activation_offload, physical)
        raw["runtime"]["compile"]["options"]["packed_activation_offload"] = legacy.as_config()
        with self.assertRaisesRegex(ConfigError, "policies conflict"):
            config_from_dict(raw)

    def test_runtime_compiler_policy_equivalence_and_conflicts(self):
        raw = _config()
        raw["runtime"]["compile"]["options"]["packed_activation_offload"] = _policy()
        config = config_from_dict(raw)
        self.assertTrue(config.runtime.packed_activation_offload.enabled)
        self.assertEqual(BackendConfig.from_options(config.runtime.compile.options).packed_activation_offload,
                         config.runtime.packed_activation_offload)
        raw["runtime"]["compile"]["options"]["packed_activation_offload"]["policy"] = "demand"
        with self.assertRaisesRegex(ConfigError, "policies conflict"):
            config_from_dict(raw)
        compiler_only = _config()
        compiler_only["runtime"].pop("packed_activation_offload")
        compiler_only["runtime"]["compile"]["options"]["packed_activation_offload"] = _policy()
        normalized = resolve_config(compiler_only, base=None)
        self.assertTrue(normalized["runtime"]["packed_activation_offload"]["enabled"])
        absent = resolve_config({"runtime": {"checks": False}}, base=None)
        self.assertNotIn("packed_activation_offload", absent["runtime"])
        with self.assertRaisesRegex(ConfigError, "configuration table"):
            config_from_dict(_config(packed_activation_offload=None))

    def test_unsupported_execution_owners_reject_before_lowering(self):
        for changes in ({"device": "cpu", "precision": "fp32"},
                        {"compile": {"enabled": False}}, {"compile": {"enabled": True, "backend": "inductor"}},
                        {"cuda_graph": {"warmup": 1}}):
            with self.subTest(changes=changes), self.assertRaises(ConfigError):
                config_from_dict(_config(**changes))
        for key in ("enable_aot_autodiff", "use_native_autodiff"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "native AutoDiff"):
                BackendConfig(packed_activation_offload=_policy(), **{key: False})
        graph = torch.fx.symbolic_trace(nn.ReLU())
        with patch("deepopt.compiler.backend.GraphLowering.run", side_effect=AssertionError("lowering ran")):
            with self.assertRaisesRegex(ValueError, "CUDA execution"):
                compile_fx(graph, [torch.randn(3)], config=BackendConfig(packed_activation_offload=_policy()))

    def test_disabled_cuda_graph_options_allow_packed_offload_configuration(self):
        raw = _config(cuda_graph={"enabled": False, "warmup": 3, "autograd": True})
        parsed = config_from_dict(raw)
        self.assertTrue(parsed.runtime.packed_activation_offload.enabled)
        self.assertFalse(parsed.runtime.cuda_graph["enabled"])
        self.assertFalse(resolve_config(raw, base=None)["runtime"]["cuda_graph"]["enabled"])
        raw["runtime"]["cuda_graph"]["enabled"] = True
        with self.assertRaisesRegex(ConfigError, "CUDA Graph"):
            config_from_dict(raw)

    def test_saved_DDP_options_are_preserved_by_schema_load_and_normalized_route(self):
        import deepopt
        from examples.vae import benchmark_vae as benchmark

        path = Path(benchmark.__file__).with_name("h200_bf16_storage_diagnostics.toml")
        recorded = benchmark.settings(benchmark.parser().parse_args(["--config", str(path),
            "--bf16-policy", "bf16_storage"]))
        original = deepcopy(recorded)
        self.assertEqual(recorded["runtime"]["ddp"], RECORDED_DDP_OPTIONS)
        self.assertIsNotNone(recorded["runtime"]["compile"]["mode"])
        self.assertTrue(recorded["runtime"]["compile"]["options"]["use_native_autodiff"])
        with self.assertRaisesRegex(ConfigError, "Choose runtime.compile.mode"):
            config_from_dict(recorded)
        for mode in (None, "demand", "memory_first_overlap"):
            with self.subTest(mode=mode):
                raw = benchmark.mode_settings(recorded, "deepopt-bf16-native-compressed")
                policy = {**_policy(), "enabled": mode is not None, "policy": mode or "memory_first_overlap"}
                raw["runtime"]["packed_activation_offload"] = deepcopy(policy)
                raw["runtime"]["compile"]["options"]["packed_activation_offload"] = deepcopy(policy)
                parsed = config_from_dict(deepcopy(raw))
                loaded = deepopt.load_config(deepcopy(raw))
                roundtrip = config_from_dict(asdict(loaded))
                self.assertEqual(parsed, loaded)
                self.assertEqual(roundtrip, loaded)
                self.assertEqual(loaded.runtime.ddp, RECORDED_DDP_OPTIONS)
                self.assertEqual(loaded.runtime.packed_activation_offload.enabled, mode is not None)
                self.assertIsNone(loaded.runtime.compile.mode)
                self.assertTrue(loaded.runtime.compile.fullgraph)
                self.assertTrue(loaded.runtime.compile.options["use_native_autodiff"])
                self.assertTrue(loaded.runtime.compile.options["require_full_aot"])
        self.assertEqual(recorded, original)

    def test_DDP_knob_schema_remains_strict_for_enabled_and_resident_policies(self):
        invalid = ({"enabled": False}, {"unknown_knob": True}, {"static_graph": 1},
            {"broadcast_buffers": "true"}, {"bucket_cap_mb": 0}, {"bucket_cap_mb": float("nan")})
        for enabled in (False, True):
            for changes in invalid:
                with self.subTest(enabled=enabled, changes=changes), self.assertRaises(ConfigError):
                    config_from_dict(_config(ddp={**RECORDED_DDP_OPTIONS, **changes},
                        packed_activation_offload=_policy(enabled=enabled)))

    def test_multi_rank_cuda_policy_is_not_rejected_as_unsupported(self):
        from deepopt.engine.execution import runtime as runtime_module

        # Metadata-only check of removal of the single-rank guard. Numerical
        # DDP, startup admission and CUDA transport have separate real workers.
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        snapshot = ResolvedConfig(asdict(config_from_dict(_config(ddp=RECORDED_DDP_OPTIONS))))
        with patch.object(runtime_module.distributed_ops, "collective_errors", side_effect=lambda *args: nullcontext()), \
                patch.object(runtime_module.distributed_ops, "initialize_distributed", return_value=SimpleNamespace(device=torch.device("cuda:0"))), \
                patch.object(runtime_module.distributed_ops, "world_size", return_value=2), \
                patch.object(runtime_module, "PrecisionPolicy", side_effect=AssertionError("precision reached")), \
                patch.object(runtime_module, "TrainingCore", side_effect=AssertionError("single core constructed")), \
                patch.object(runtime_module, "DDPCore", side_effect=AssertionError("distributed core constructed")):
            with self.assertRaisesRegex(AssertionError, "precision reached"):
                runtime_module.TrainingRuntime(model, optimizer, config=snapshot)

    def test_owned_policy_cannot_change_before_preparation_and_signature_records_it(self):
        model = nn.Linear(2, 2)
        model._deepopt_packed_activation_offload_policy = resolve_packed_offload(_policy())
        self.assertEqual(resolve_owned_packed_offload(model._deepopt_packed_activation_offload_policy),
                         model._deepopt_packed_activation_offload_policy)
        with patch("deepopt.engine.execution.configuration.prepare_fp8_model", side_effect=AssertionError("prepared")):
            with self.assertRaisesRegex(ValueError, "owned compiled-model"):
                prepare_execution_model(model, {"runtime": {"packed_activation_offload": {"enabled": False}}})
        enabled = ResolvedConfig(asdict(config_from_dict(_config())))
        disabled = deepcopy(enabled)
        disabled["runtime"]["packed_activation_offload"] = asdict(PackedActivationOffloadPolicy())
        self.assertNotEqual(runtime_signature(enabled), runtime_signature(disabled))
        self.assertFalse(RuntimeConfig().packed_activation_offload.enabled)


if __name__ == "__main__":
    unittest.main()
