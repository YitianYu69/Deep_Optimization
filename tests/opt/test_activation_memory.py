"""Owned activation-memory plans, without claiming CUDA kernel qualification."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import ActivationMemoryConfig, Trainer
from deepopt._activation_memory import (
    ActivationMemoryPlan, activation_memory_compiler_options, activation_memory_scope,
    active_activation_memory, current_activation_memory, resolve_activation_memory,
)
from deepopt._config_resolution import ResolvedConfig, resolve_config
from deepopt.config import ConfigError, activation_memory_config_from_dict, config_from_dict, reset_config
from deepopt.compiler import BackendConfig, compile_fx
from deepopt.compiler.operations import numerical_scope
from deepopt.engine.checkpointing.checkpoint import _checkpoint_payload, restore_checkpoint
from deepopt.engine.checkpointing.identity import compatibility_config, model_activation_memory
from deepopt.engine.execution.configuration import prepare_execution_model
from deepopt.opt._autocast import autocast
from deepopt.opt.policy import prepare_fp8_model


class ActivationMemoryTests(unittest.TestCase):
    def setUp(self):
        reset_config()
        self.addCleanup(reset_config)

    def test_configuration_and_resolver_are_torch_free(self):
        environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
        result = subprocess.run([sys.executable, "-c", """
import sys
from deepopt import ActivationMemoryConfig
from deepopt._config_resolution import resolve_config
from deepopt._activation_memory import resolve_activation_memory
assert ActivationMemoryConfig().reduction_scratch_bytes == 8388608
assert resolve_activation_memory().version == 1
assert resolve_config({'runtime': {'activation_memory': {'pack_reuse': 'off'}}}, base=None)['runtime']['activation_memory']['pack_reuse'] == 'off'
assert 'torch' not in sys.modules
assert 'triton' not in sys.modules
"""], env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_descriptor_is_owned_frozen_versioned_and_canonical(self):
        raw = {"multiply_backward": "packed_streaming", "recompute_modules": ["decoder.10", "encoder.2"]}
        plan = resolve_activation_memory(raw)
        raw["recompute_modules"].append("encoder.4")
        self.assertEqual(plan.recompute_modules, ("decoder.10", "encoder.2"))
        self.assertEqual(resolve_activation_memory(plan.as_dict()), plan)
        with self.assertRaises(FrozenInstanceError):
            plan.reduction_scratch_bytes = 4
        descriptor = plan.as_dict()
        descriptor["recompute_modules"].clear()
        self.assertEqual(len(plan.recompute_modules), 2)
        for version in (True, 0, 2):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "version"):
                resolve_activation_memory({**plan.as_dict(), "version": version})

    def test_parser_defaults_round_trip_and_partial_provenance(self):
        raw = {"name": "memory", "seed": 3,
               "training": {"epochs": 1, "batch_size": 2, "learning_rate": .1},
               "runtime": {"activation_memory": {"multiply_backward": "packed_streaming", "pack_reuse": "off"}}}
        config = config_from_dict(raw)
        self.assertEqual(config_from_dict(asdict(config)), config)
        self.assertEqual(config.runtime.activation_memory,
                         ActivationMemoryConfig(multiply_backward="packed_streaming", pack_reuse="off"))
        self.assertNotIn("activation_memory", resolve_config({"runtime": {}}, base=None)["runtime"])
        self.assertEqual(resolve_activation_memory().multiply_backward, "inherit")

    def test_invalid_values_paths_overlaps_and_options_reject(self):
        cases = (None, True, {"multiply_backward": "streaming"}, {"pack_reuse": False},
                 {"reduction_scratch_bytes": True}, {"reduction_scratch_bytes": 0},
                 {"recompute_modules": "encoder.2"}, {"recompute_modules": [""]},
                 {"recompute_modules": ["encoder..2"]}, {"recompute_modules": ["encoder.*"]},
                 {"recompute_modules": ["encoder.2", "encoder.2"]},
                 {"recompute_modules": ["encoder.2", "encoder.2.conv"]},
                 {"version": 1}, {"unknown": True})
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                activation_memory_config_from_dict(raw)
        for choice, legacy in (("packed_streaming", False), ("materialized", True)):
            with self.subTest(choice=choice), self.assertRaisesRegex(ConfigError, "conflicts"):
                resolve_config({"runtime": {"activation_memory": {"multiply_backward": choice},
                               "compile": {"backend": "deepopt", "options": {"fuse_packed_multiply": legacy}}}}, base=None)

    def test_native_and_eager_use_one_plan_preserving_legacy_options(self):
        plan = ActivationMemoryPlan(multiply_backward="packed_streaming", pack_reuse="off",
                                    reduction_scratch_bytes=65536)
        native = BackendConfig(activation_memory=plan)
        self.assertTrue(native.fuse_packed_multiply)
        self.assertFalse(native.share_compatible_packs)
        self.assertEqual(native.activation_memory, plan)
        self.assertEqual(native.reduction_scratch_bytes, 65536)
        self.assertTrue(BackendConfig.from_options({"fuse_packed_multiply": True}).fuse_packed_multiply)
        self.assertTrue(BackendConfig.from_options({"share_compatible_packs": True}).share_compatible_packs)
        self.assertEqual(activation_memory_compiler_options(None, {"fuse_packed_multiply": True}),
                         {"fuse_packed_multiply": True})
        for changes in ({"fuse_packed_multiply": False}, {"kernel_backend": "tilelang"},
                        {"fusion_phases": "forward"}, {"enable_aot_autodiff": False}, {"use_native_autodiff": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                BackendConfig(activation_memory=plan, **changes)
        with self.assertRaisesRegex(ValueError, "compiler.backward"):
            autocast("cpu", enabled=False, config={"runtime": {
                "activation_memory": plan.as_config(), "kernels": {
                    "backend": "mixed", "roles": {"compiler.backward": "tilelang"}}}})

    def test_scope_snapshot_inheritance_restoration_and_capture_identity(self):
        raw = {"runtime": {"activation_memory": {"multiply_backward": "materialized", "reduction_scratch_bytes": 8192}}}
        scope = autocast("cpu", enabled=False, config=raw)
        raw["runtime"]["activation_memory"]["reduction_scratch_bytes"] = 4
        original = numerical_scope()
        with scope:
            self.assertEqual(current_activation_memory("cpu").reduction_scratch_bytes, 8192)
            self.assertNotEqual(numerical_scope(), original)
            with autocast("cpu", enabled=False):
                self.assertEqual(current_activation_memory("cpu").reduction_scratch_bytes, 8192)
            with self.assertRaisesRegex(RuntimeError, "probe"):
                with activation_memory_scope({"multiply_backward": "packed_streaming"}, "cpu"):
                    raise RuntimeError("probe")
            self.assertEqual(current_activation_memory("cpu").multiply_backward, "materialized")
            self.assertTrue(current_activation_memory("cuda").is_default)
        self.assertIsNone(active_activation_memory("cpu"))

    def test_explicit_native_option_conflicts_with_scoped_plan_before_lowering(self):
        graph = torch.fx.symbolic_trace(nn.ReLU())
        with activation_memory_scope({"multiply_backward": "packed_streaming"}, "cpu"):
            for options in ({"fuse_packed_multiply": False}, {"fusion_phases": "forward"},
                            {"use_native_autodiff": False},
                            {"activation_memory": {"multiply_backward": "materialized"}}):
                with self.subTest(options=options), patch("deepopt.compiler.backend.GraphLowering.run") as lower:
                    with self.assertRaises(ValueError):
                        compile_fx(graph, [torch.ones(2)], config=BackendConfig(**options))
                    lower.assert_not_called()

    def test_direct_backend_option_merge_preserves_implicit_fusion_defaults(self):
        from deepopt.compiler.backend import do_backend

        graph = torch.fx.symbolic_trace(nn.ReLU())
        plan = ActivationMemoryPlan(multiply_backward="packed_streaming")
        config = BackendConfig()
        value = torch.ones(2, requires_grad=True)
        with activation_memory_scope(plan, "cpu"):
            compiled = do_backend(graph, [value], config=config,
                                  options={"max_pointwise_fusion_length": 64})
            output = compiled(value)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                do_backend(graph, [value], config=BackendConfig(fuse_packed_multiply=False),
                           options={"max_pointwise_fusion_length": 64})
        output.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value))
        self.assertEqual(config._activation_memory_explicit_options, ())
        self.assertEqual(compiled.compilation_report.effective_compiler_options["activation_memory_plans"]["cpu"], plan.as_dict())
        explicit = do_backend(graph, [value], config=config,
                              options={"activation_memory": plan.as_dict()})
        self.assertEqual(explicit.compilation_report.effective_compiler_options["activation_memory_plans"]["cpu"], plan.as_dict())

    def test_model_owned_plan_survives_empty_ambient_and_runtime_snapshot(self):
        class Probe(nn.Module):
            def forward(self, value):
                return current_activation_memory(value.device.type), value.square()

        raw = {"runtime": {"activation_memory": {"multiply_backward": "materialized", "reduction_scratch_bytes": 4096}}}
        model, _ = prepare_fp8_model(Probe(), config=raw)
        plan = model._deepopt_activation_memory_plan
        raw["runtime"]["activation_memory"]["multiply_backward"] = "packed_streaming"
        value = torch.ones(2, requires_grad=True)
        with autocast("cpu", enabled=False, config={}):
            selected, output = model(value)
        output.sum().backward()
        self.assertEqual(selected, plan)
        torch.testing.assert_close(value.grad, torch.full_like(value, 2))
        copied = deepcopy(model)
        self.assertEqual(copied(value)[0], plan)
        snapshot = ResolvedConfig()
        same, _, _ = prepare_execution_model(model, snapshot)
        self.assertIs(same, model)
        self.assertEqual(snapshot["runtime"]["activation_memory"], plan.as_config())
        with activation_memory_scope({"multiply_backward": "packed_streaming"}, "cpu"):
            with self.assertRaisesRegex(ValueError, "differs"):
                model(value)
        with self.assertRaisesRegex(ValueError, "differs"):
            prepare_fp8_model(model, config=raw)

    def test_prepared_operations_and_fp8_builder_retain_memory_descriptor(self):
        from deepopt.opt.policy import fp8_plan

        config = {"runtime": {"activation_memory": {"multiply_backward": "materialized"},
                              "activation_compression": {"enabled": True, "group_size": 1024}}}
        model, _ = prepare_fp8_model(nn.Sequential(nn.Linear(2, 2), nn.GELU()), config=config)
        expected = model._deepopt_activation_memory_plan.as_dict()
        self.assertEqual(model[0].plan["activation_memory"], expected)
        self.assertEqual(model[1].plan["activation_memory"], expected)
        self.assertEqual(fp8_plan({"enabled": True, "backend": "reference"}, config=config)["activation_memory"], expected)

    def test_native_model_records_owned_plan_and_backwards_after_scope_exit(self):
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model

        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        config = {"runtime": {"bf16_policy": "bf16_storage", "activation_compression": {
            "enabled": True, "group_size": 1024}, "activation_memory": {
                "multiply_backward": "materialized", "pack_reuse": "off", "reduction_scratch_bytes": 4096}}}
        model, _ = prepare_fp8_model(nn.Sequential(nn.Linear(4, 4), nn.GELU()), config=config)
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
            options={"require_full_aot": True}), config=config)
        value = torch.randn(2, 4, requires_grad=True)
        with autocast("cpu", dtype=torch.bfloat16, config={}):
            output = model(value)
        with activation_memory_scope({"multiply_backward": "packed_streaming"}, "cpu"):
            output.float().sum().backward()
        report = compilation_report(model).as_dict()
        self.assertEqual(report["runtime_fallbacks"], 0)
        regions = [region for graph in report["graphs"] for region in graph["native_autodiff_graphs"]]
        self.assertTrue(regions)
        self.assertTrue(all(region["activation_memory"] == model._deepopt_activation_memory_plan.as_dict()
                            for region in regions))
        self.assertEqual(value.grad.dtype, torch.float32)

    def test_unimplemented_features_and_missing_modules_fail_before_effects(self):
        for settings, exception in (({"pack_reuse": "forward"}, NotImplementedError),
                                    ({"recompute_modules": ["0"]}, NotImplementedError),
                                    ({"recompute_modules": ["missing"]}, ValueError)):
            model = nn.Sequential(nn.Linear(2, 2))
            parameters = tuple(model.parameters())
            rng = torch.get_rng_state()
            with self.subTest(settings=settings), self.assertRaises(exception):
                prepare_fp8_model(model, config={"runtime": {"activation_memory": settings}})
            self.assertEqual(tuple(map(id, model.parameters())), tuple(map(id, parameters)))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(hasattr(model, "_deepopt_activation_memory_plan"))
        with self.assertRaises(NotImplementedError):
            autocast("cpu", enabled=False, activation_memory={"pack_reuse": "forward"}, config={})

    def test_direct_native_and_fp8_operations_reject_owned_plan_conflicts_before_effects(self):
        from deepopt.opt.nn import functional as do
        from deepopt.opt.policy import linear_plan, fp8_plan

        config = {"runtime": {"activation_memory": {"multiply_backward": "materialized"}}}
        plans = (linear_plan(config=config), fp8_plan({"enabled": True, "backend": "reference"}, config=config))
        value = torch.randn(2, 4, 4, 4)
        linear_input, linear_weight = torch.randn(2, 4), nn.Parameter(torch.randn(3, 4))
        weight = nn.Parameter(torch.randn(4, 4, 1, 1))
        running_mean, running_var = torch.zeros(4), torch.ones(4)
        query = torch.randn(2, 2, 4, 8)
        before = value.clone()
        for plan in plans:
            calls = (
                lambda: do.do_linear(linear_input, linear_weight, plan=plan),
                lambda: do.do_conv2d(value, weight, plan=plan),
                lambda: do.do_conv_transpose2d(value, weight, plan=plan),
                lambda: do.do_group_norm(value, 2, plan=plan),
                lambda: do.do_layer_norm(value, (4, 4), plan=plan),
                lambda: do.do_batch_norm(value, running_mean, running_var, training=True, plan=plan),
                lambda: do.do_relu(value, inplace=True, plan=plan),
                lambda: do.do_max_pool2d(value, 2, plan=plan),
                lambda: do.do_fractional_max_pool2d(value, 2, output_size=2, plan=plan),
                lambda: do.do_scaled_dot_product_attention(query, query, query, dropout_p=.5, plan=plan),
            )
            for index, call in enumerate(calls):
                rng = torch.get_rng_state()
                with self.subTest(fp8="fp8_training" in plan, operation=index), \
                        activation_memory_scope({"multiply_backward": "packed_streaming"}, "cpu"):
                    with self.assertRaisesRegex(ValueError, "differs"):
                        call()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                torch.testing.assert_close(value, before, rtol=0, atol=0)
                torch.testing.assert_close(running_mean, torch.zeros_like(running_mean), rtol=0, atol=0)
                torch.testing.assert_close(running_var, torch.ones_like(running_var), rtol=0, atol=0)
                self.assertIsNone(weight.grad)
                self.assertIsNone(linear_weight.grad)

    def test_checkpoint_plan_version_and_scratch_are_validated_before_load(self):
        def trainer():
            model = nn.Linear(2, 1)
            return Trainer(model, torch.optim.SGD(model.parameters(), lr=.01), device="cpu",
                           loss_fn=nn.MSELoss(), config={"runtime": {"activation_memory": {
                               "multiply_backward": "materialized", "reduction_scratch_bytes": 4096}}})

        source, target = trainer(), trainer()
        payload = deepcopy(_checkpoint_payload(source, source.config, generators={}, history=[]))
        restore_checkpoint(payload, target, {}, generators={})
        self.assertEqual(payload["trainer_settings"]["activation_memory"],
                         model_activation_memory(target.model, {}).as_dict())
        for changes in ({"version": 0}, {"reduction_scratch_bytes": 8192}, {"multiply_backward": "packed_streaming"}):
            changed = deepcopy(payload)
            changed["trainer_settings"]["activation_memory"].update(changes)
            rng = torch.get_rng_state()
            with self.subTest(changes=changes), patch.object(target.model, "load_state_dict") as load:
                with self.assertRaises(ValueError):
                    restore_checkpoint(changed, target, {}, generators={})
                load.assert_not_called()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(compatibility_config({}), compatibility_config({"runtime": {"activation_memory": {}}}))
        self.assertNotEqual(compatibility_config({}), compatibility_config({"runtime": {"activation_memory": {
            "multiply_backward": "materialized"}}}))


if __name__ == "__main__":
    unittest.main()
