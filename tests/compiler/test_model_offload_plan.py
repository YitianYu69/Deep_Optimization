"""Whole-model selection and mixed-residency native runtime contracts."""

from dataclasses import replace
import gc
import json
import unittest
from unittest.mock import patch
from copy import deepcopy

import torch

from deepopt.compiler.model_offload_plan import build_model_offload_plan, scope_identity
from deepopt.compiler.offload_plan import build_packed_offload_plan
from deepopt.compiler.runtime.manager import CompiledGraphRuntimeManager, compiled_forward_scope
from deepopt.compiler.runtime.packed_offload import TestingTransportBackend, OffloadAdmissionError
from tests.compiler.test_packed_offload_plan import _fixture, _plan
from tests.compiler.test_packed_offload_lifecycle import fixture, graph


def candidates(*, budget=4096, shape=(4, 1024)):
    forward_graph, trace = _fixture(shape)
    original, forward, backward = _plan(forward_graph, trace)
    policy = replace(original.policy, version=2, host_pinned_bytes=budget)
    return build_packed_offload_plan(forward_graph, trace.graph, forward, backward, trace.saved_value_specs,
        policy=policy, saved_tensors=trace.saved_tensors, backward_argument_sources=trace.backward_argument_sources,
        profile_identity={"gpu": "explicit_metadata_fixture"}, defer_host_budget=True)


class ModelOffloadPlanTests(unittest.TestCase):
    def test_two_regions_share_physical_placement_instead_of_regional_fallback(self):
        plans = candidates(), candidates()
        selected = build_model_offload_plan(plans)
        self.assertEqual(tuple(len(p.bundles) for p in plans), (2, 2))
        self.assertEqual(sum(len(p.bundles) for p in selected.regions), 3)
        self.assertTrue(all(p.bundles for p in selected.regions))
        self.assertLessEqual(selected.physical_host_bytes, 4096)
        self.assertEqual(selected.logical_host_bytes, 3 * 1056)
        self.assertEqual(selected.profile_source, "cold_logical_estimate")
        self.assertFalse(selected.as_dict()["performance_qualified"])
        self.assertEqual(selected, build_model_offload_plan(plans))
        json.dumps(selected.as_dict(), allow_nan=False)
        for plan in selected.regions:
            self.assertEqual(plan.model_scope_signature, selected.signature)
            self.assertTrue(set(plan.selected_component_names).issubset(plans[0].selected_component_names))

    def test_target_window_requires_full_scope_identity_and_cuda_receipt(self):
        plans = candidates(), candidates()
        profile = dict(schema_version=1, scope_identity=scope_identity(plans), cuda_execution_verified=True,
            source_receipt_sha256="a" * 64,
            windows=[dict(region=1, phase="forward", boundary=100, baseline_bytes=20000)])
        selected = build_model_offload_plan(plans, peak_profile=profile)
        self.assertEqual(selected.profile_source, "target_storage_ledger")
        self.assertEqual(selected.estimated_peak_after_selection, 20000 - 3 * 1056)
        for changed in (dict(scope_identity="stale"), dict(cuda_execution_verified=False), dict(extra=True)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                build_model_offload_plan(plans, peak_profile={**profile, **changed})
        demand = tuple(replace(p, policy=replace(p.policy, policy="demand")) for p in plans)
        self.assertEqual(scope_identity(plans), scope_identity(demand))
        self.assertEqual(tuple(p.selected_component_names for p in selected.regions),
                         tuple(p.selected_component_names for p in build_model_offload_plan(demand,
                               peak_profile=profile).regions))

    def test_selection_ablation_holds_physical_pool_fixed(self):
        plans = candidates(), candidates(shape=(8, 1024))
        profile = dict(schema_version=1, scope_identity=scope_identity(plans), cuda_execution_verified=True,
            source_receipt_sha256="c" * 64,
            windows=[dict(region=1, phase="forward", boundary=100, baseline_bytes=20000)])
        peak = build_model_offload_plan(plans, peak_profile=profile)
        ordered = build_model_offload_plan(plans, peak_profile=profile, selection_mode="regional_order")
        self.assertEqual(ordered.regions[0].bundles, plans[0].bundles)
        self.assertEqual(len(ordered.regions[1].bundles), 0)
        self.assertGreater(peak.logical_host_bytes, ordered.logical_host_bytes)
        self.assertLessEqual(peak.physical_host_bytes, 4096)
        self.assertLessEqual(ordered.physical_host_bytes, 4096)
        self.assertEqual(peak.eligible_inventory_sha256, ordered.eligible_inventory_sha256)
        self.assertEqual(peak.identity, ordered.identity)

    def test_outer_scope_refuses_mismatched_region_selection_controls(self):
        from types import SimpleNamespace
        from deepopt.compiler.model_offload_plan import finalize_model_offload_regions

        regions = tuple(SimpleNamespace(_offload_candidates=candidates(), context=SimpleNamespace(
            packed_offload_selection_mode=mode, packed_offload_peak_profile=None)) for mode in ("peak", "regional_order"))
        with self.assertRaisesRegex(ValueError, "same selection mode"):
            finalize_model_offload_regions(regions)

    def test_resident_native_artifacts_record_recipe_and_schedule_without_storage(self):
        from deepopt import autocast
        from deepopt.compiler import BackendConfig, compile_fx
        from torch.fx.passes.shape_prop import ShapeProp
        from tests.compiler.test_native_swiglu_fusion import _ProjectionSwiGLU, _PreparedTracer

        model = _ProjectionSwiGLU("torch_amp")
        traced = torch.fx.GraphModule(model, _PreparedTracer().trace(model))
        x = torch.randn(2, 4, 13, 17)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, bf16_policy="torch_amp",
                activation_compression=dict(enabled=True, bits=2, group_size=1024)):
            ShapeProp(traced).propagate(x)
            program = compile_fx(traced, [x], config=BackendConfig(require_full_aot=True))
            output = program(x)
        output.sum().backward()
        regions = program.backend_artifact.joint_autograd_regions
        self.assertTrue(regions)
        for index, region in enumerate(regions):
            for artifact in (region.forward_artifact, region.backward_artifact):
                self.assertEqual(artifact.native_region_index, index)
                self.assertIsNone(artifact.native_scope_identity)
                for key in ("native_forward_signature", "native_backward_signature", "native_recipe_identity_sha256"):
                    self.assertEqual(len(getattr(artifact, key)), 64)

    def test_custom_resident_policy_protocol_does_not_require_offload_metadata_schema(self):
        from deepopt.compiler.autodiff.joint import JointProgram
        from deepopt.compiler.ir import IRGraph, TensorMeta
        from deepopt.compiler.registry import BackendRegistry, KernelCodegenContext
        from deepopt.compiler.trace.builder import GraphBuilder

        class Policy:
            enabled = False
            recipe = (("bits", 2), ("group_size", 1024))

            def select(self, value, use):
                return None, "consumer_requires_exact"

        graph = IRGraph()
        builder = GraphBuilder(graph)
        x = builder.input(0, TensorMeta((3,), torch.float64, "cpu", (1,), True))
        y = x * x
        builder.output((y,))
        program = JointProgram(graph, (x.value,), (y.value,), [], BackendRegistry(),
            KernelCodegenContext(saved_activation_policy=Policy()))
        value = torch.tensor([1., 2., 3.], dtype=torch.float64, requires_grad=True)
        output = program(value)
        output.sum().backward()
        torch.testing.assert_close(output, torch.tensor([1., 4., 9.], dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(value.grad, torch.tensor([2., 4., 6.], dtype=torch.float64), rtol=0, atol=0)
        for artifact in (program.forward_artifact, program.backward_artifact):
            self.assertIsNone(artifact.native_recipe_identity_sha256)
            self.assertIn("metadata only", artifact.native_profile_unavailable_reason)

    def test_profile_is_frozen_and_recipe_identity_includes_geometry_and_options(self):
        from types import SimpleNamespace
        from deepopt.compiler import BackendConfig
        from deepopt.compiler.registry import KernelCodegenContext
        from deepopt.compiler.autodiff.joint import _packed_execution_identity
        from deepopt.opt.nn.convolution.geometry import ConvGeometry

        plans = candidates(), candidates()
        profile = dict(schema_version=1, scope_identity=scope_identity(plans), cuda_execution_verified=True,
            source_receipt_sha256="b" * 64,
            windows=[dict(region=1, phase="forward", boundary=9, baseline_bytes=20000)])
        config = BackendConfig(packed_offload_peak_profile=profile)
        profile["windows"][0]["baseline_bytes"] = 1
        self.assertEqual(config.packed_offload_peak_profile.windows[0].baseline_bytes, 20000)
        geometries = [ConvGeometry((4, 2, 7, 7), (2, 2, 3, 3), (1, 1), (padding, padding),
                                   (dilation, dilation), 1) for padding, dilation in ((1, 1), (2, 2))]
        self.assertEqual(geometries[0].output_shape, geometries[1].output_shape)
        left, right = [_packed_execution_identity([SimpleNamespace(geometry=g)], KernelCodegenContext(), "cpu")
                       for g in geometries]
        self.assertNotEqual(left["numerical_recipe_sha256"], right["numerical_recipe_sha256"])
        changed = _packed_execution_identity([SimpleNamespace(geometry=geometries[0])],
            KernelCodegenContext(native_conv_dx_batch_size=2), "cpu")
        self.assertNotEqual(left["compiler_options_sha256"], changed["compiler_options_sha256"])
        with self.assertRaisesRegex(ValueError, "version 2"):
            BackendConfig(packed_activation_offload=candidates().policy.as_config() | {"version": 1},
                          packed_offload_selection_mode="all_eligible")

    def test_original_geometry_capacity_uses_64bit_metadata_without_allocating(self):
        plans = candidates(budget=4 << 30, shape=(128, 768, 224, 224)), candidates(
            budget=4 << 30, shape=(128, 768, 224, 224))
        result = build_model_offload_plan(plans)
        self.assertGreater(result.logical_host_bytes, 2**31)
        self.assertLessEqual(result.physical_host_bytes, 4 << 30)
        self.assertLess(len(result.regions[0].bundles) + len(result.regions[1].bundles), 4)

    def test_strict_scope_admission_rejects_capacity_instead_of_dropping_groups(self):
        plan = candidates()
        plan = replace(plan, policy=replace(plan.policy, admission="error", device_staging_bytes=0, d2h_inflight_bytes=0))
        plans = (plan, plan)
        with self.assertRaisesRegex(ValueError, "Whole-model physical.*8192 bytes, rank budget 4096 bytes"):
            build_model_offload_plan(plans)

    def test_maximum_offload_point_requires_complete_inventory_and_reports_ceiling(self):
        small = (candidates(), candidates())
        with self.assertRaisesRegex(ValueError, "All-eligible.*partial selection forbidden"):
            build_model_offload_plan(small, selection_mode="all_eligible")
        enough = (candidates(budget=8192), candidates(budget=8192))
        maximum = build_model_offload_plan(enough, selection_mode="all_eligible")
        inventory = maximum.inventory()
        self.assertEqual(inventory["eligible_bundle_count"], 4)
        self.assertEqual(inventory["eligible_history_bytes"], 4 * 1056)
        self.assertTrue(inventory["all_eligible_selected"])
        self.assertEqual(inventory["selected_history_bytes"], inventory["eligible_history_bytes"])
        self.assertEqual(inventory["all_eligible_host_physical_capacity_bytes"], 8192)
        self.assertEqual(len(inventory["eligible_inventory_sha256"]), 64)
        self.assertTrue(all(p.selection_mode == "all_eligible" for p in maximum.regions))

    def test_physical_pageable_stage_is_proven_before_any_producer_runs(self):
        plan = fixture(version=2, pageable=True, groups=20, host_pinned_bytes=7000)
        backend = TestingTransportBackend()
        manager = CompiledGraphRuntimeManager(plan.policy, backend=backend, start_worker=False)
        try:
            with self.assertRaisesRegex(OffloadAdmissionError, "physical pinned staging capacity before forward"):
                manager.begin_forward(plan)
            self.assertEqual(len(backend.events), 0)
            self.assertEqual(len(backend.allocations), 0)
        finally:
            manager.close()

    def test_runtime_can_mix_resident_and_cpu_saved_groups_inside_one_region(self):
        plan = fixture(bundles=4, groups=4, version=2, host_pinned_bytes=4096,
                       device_staging_bytes=4096, d2h_inflight_bytes=4096, admission="resident_fallback")
        manager = CompiledGraphRuntimeManager(plan.policy, backend=TestingTransportBackend(), start_worker=False)
        try:
            with compiled_forward_scope((plan,), manager):
                frame, value, output = graph(manager, plan)
            self.assertFalse(frame.resident_fallback)
            self.assertEqual(len(frame.records), 3)
            self.assertEqual(len(manager.snapshot()["admission_history"][-1]["selected_bundle_ids"]), 3)
            output.backward(retain_graph=True)
            output.backward()
            self.assertEqual(value.grad.item(), 4)
            del output
            gc.collect()
            manager.drain()
            self.assertEqual(manager.snapshot()["host"]["pinned_active_bytes"], 0)
        finally:
            manager.close()

    def test_shared_physical_allocator_domain_is_counted_once(self):
        from deepopt.compiler.runtime import manager as managers

        plan = fixture(version=2, host_pinned_bytes=4096, admission="resident_fallback")
        backend = TestingTransportBackend()
        left = CompiledGraphRuntimeManager(plan.policy, backend=backend, start_worker=False)
        right = CompiledGraphRuntimeManager(plan.policy, backend=backend, start_worker=False)
        try:
            left._initialize(plan)
            right._initialize(plan)
            lease = left.pinned.acquire(plan.bundles[0].components)
            expected = left.snapshot()["host"]
            with patch.object(managers, "_managers", {left, right}):
                total = managers.packed_offload_runtime_snapshot()["totals"]["host"]
            self.assertEqual(total["pinned_active_bytes"], expected["pinned_active_bytes"])
            lease.release()
        finally:
            left.close()
            right.close()


@unittest.skipUnless(torch.cuda.is_available(), "Actual CUDA native execution is required")
class ModelOffloadCudaTests(unittest.TestCase):
    def test_compiler_finalized_v2_native_training_retained_backward(self):
        from torch.fx.passes.shape_prop import ShapeProp
        from deepopt import autocast
        from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
        from tests.compiler.test_native_swiglu_fusion import _PreparedTracer, _ProjectionSwiGLU
        from tests.compiler.test_native_padding import _strict_algorithms

        with _strict_algorithms():
            torch.manual_seed(971)
            original = _ProjectionSwiGLU("bf16_storage").cuda()
            sample = torch.randn(2, 4, 13, 17, device="cuda")
            reference = None
            for mode in (None, "demand", "memory_first_overlap"):
                model = deepcopy(original)
                gm = torch.fx.GraphModule(model, _PreparedTracer().trace(model))
                report = CompilationReport()
                policy = None if mode is None else dict(version=2, enabled=True, policy=mode,
                    host_pinned_bytes=16 << 20, admission="error")
                optimizer = torch.optim.AdamW(gm.parameters(), lr=.001)
                snapshots = []
                with autocast("cuda", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                        activation_compression=dict(enabled=True, bits=2, group_size=1024)):
                    ShapeProp(gm).propagate(sample)
                    compiled = compile_fx(gm, [sample], config=BackendConfig(require_full_aot=True,
                        packed_activation_offload=policy, fuse_swiglu_residuals=True,
                        swiglu_fusion_stage="forward_backward"), report=report)
                    for _ in range(2):
                        output = compiled(sample)
                        seed = torch.full_like(output, .137)
                        first = torch.autograd.grad(output, tuple(gm.parameters()), seed, retain_graph=True)
                        second = torch.autograd.grad(output, tuple(gm.parameters()), seed)
                        for left, right in zip(first, second):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                        snapshots.append((output.detach().cpu(), tuple(g.cpu() for g in first)))
                        for parameter, gradient in zip(gm.parameters(), first):
                            self.assertEqual(gradient.dtype, torch.float32)
                            parameter.grad = gradient
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                self.assertIsNone(sample.grad)
                evidence = report.as_dict()
                self.assertEqual(evidence["runtime_fallbacks"], 0)
                self.assertGreater(evidence["native_autodiff_operations"], 0)
                for region in compiled.backend_artifact.joint_autograd_regions:
                    if mode is not None and region.offload_runtime is not None:
                        inventory = region.model_offload_plan["inventory"]
                        self.assertTrue(inventory["all_eligible_selected"])
                        self.assertLessEqual(inventory["selected_physical_capacity_bytes"], 16 << 20)
                        region.offload_runtime.drain()
                        snapshot = region.offload_runtime.snapshot()
                        self.assertEqual(snapshot["transport_limits"], dict(
                            device_staging_bytes=inventory["required_staging_bytes"],
                            d2h_inflight_bytes=inventory["required_d2h_inflight_bytes"]))
                        self.assertLessEqual(snapshot["device"]["staging_peak_bytes"], inventory["required_staging_bytes"])
                        self.assertLessEqual(snapshot["device"]["peak_d2h_inflight_bytes"], inventory["required_d2h_inflight_bytes"])
                        counters = snapshot["counters"]
                        self.assertGreater(counters["d2h_bytes"], 0)
                        self.assertGreater(counters["h2d_bytes"], 0)
                if reference is None:
                    reference = snapshots
                else:
                    for (output, gradients), (expected, expected_gradients) in zip(snapshots, reference):
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)
                        for actual, expected_gradient in zip(gradients, expected_gradients):
                            torch.testing.assert_close(actual, expected_gradient, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
