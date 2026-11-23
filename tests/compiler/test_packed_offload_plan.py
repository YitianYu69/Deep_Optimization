"""Final native-AD plan metadata, without CUDA allocation or execution claims."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
import json
import operator
import unittest

import torch

from deepopt._packed_offload import resolve_packed_offload
from deepopt.compiler.autodiff.activation_saving import SavedActivationPolicy
from deepopt.compiler.autodiff.aot_auto_diff import AOTAutoDiff
from deepopt.compiler.autodiff.optimization import rebuild
from deepopt.compiler.ir import CallNode, IRGraph, InputNode, OutputNode, PointwiseNode, TensorMeta, TensorValue
from deepopt.compiler.offload_plan import build_packed_offload_plan
from deepopt.compiler.scheduler import Scheduler
from deepopt.compiler.trace.metadata_utils import contiguous_stride


def _fixture(shape=(4, 1024)):
    graph = IRGraph()
    metadata = TensorMeta(shape, torch.bfloat16, "cuda:0", contiguous_stride(shape), True)
    left, right, output = (TensorValue(name, metadata) for name in ("left", "right", "product"))
    graph.add(InputNode(0, left))
    graph.add(InputNode(1, right))
    graph.add(PointwiseNode("mul", (left, right), {}, output, operator.mul))
    graph.add(OutputNode(output))
    trace = AOTAutoDiff(graph, saved_activation_policy=SavedActivationPolicy(True, 2, 1024)).diff()
    graph.nodes[-1].result = (output, trace.saved_tensors)
    rebuild(graph)
    return graph, trace


def _plan(graph, trace, **changes):
    forward, backward = Scheduler(graph).run(), Scheduler(trace.graph, backward=True).run()
    policy = resolve_packed_offload({"enabled": True, "host_pinned_bytes": 16 * 2**30,
        "device_staging_bytes": 4 * 2**30, "d2h_inflight_bytes": 2 * 2**30, **changes})
    plan = build_packed_offload_plan(graph, trace.graph, forward, backward, trace.saved_value_specs,
        policy=policy, saved_tensors=trace.saved_tensors, backward_argument_sources=trace.backward_argument_sources,
        profile_identity={"gpu": "metadata-only", "torch": torch.__version__, "recipe": "2bit_group1024"})
    return plan, forward, backward


class PackedOffloadPlanTests(unittest.TestCase):
    def test_real_native_derivation_exports_early_atomic_lossless_triplets(self):
        graph, trace = _fixture()
        saved_before, result_before = trace.saved_tensors, trace.graph.nodes[-1].result
        plan, forward, backward = _plan(graph, trace)
        self.assertEqual(len(plan.bundles), 2)
        self.assertFalse(plan.rejections)
        self.assertEqual(plan.selected_bytes, 2 * 4 * (256 + 8))
        self.assertEqual(trace.saved_tensors, saved_before)
        self.assertIs(trace.graph.nodes[-1].result, result_before)
        for bundle in plan.bundles:
            self.assertLess(bundle.last_forward_use, len(forward.items) - 1)
            self.assertEqual(bundle.producer_boundary, bundle.last_forward_use)
            self.assertEqual(tuple(component.role for component in bundle.components), ("payload", "codec_step", "codec_minimum"))
            self.assertEqual(tuple(component.dtype for component in bundle.components), ("torch.int32", "torch.float32", "torch.float32"))
            for component in bundle.components:
                self.assertEqual(trace.saved_tensors[component.saved_index].name, component.name)
                self.assertIsInstance(component.stride, tuple)
        self.assertTrue(all(consumer.bytes <= plan.policy.device_staging_bytes for consumer in plan.consumers))
        self.assertEqual(len(plan.prefetches), sum(len(consumer.bundle_ids) for consumer in plan.consumers))
        self.assertTrue(all(not request.calibrated for request in plan.prefetches))
        self.assertFalse(plan.as_dict()["physical_storage_observed"])
        plan.validate_current(forward, backward, profile_identity={"gpu": "metadata-only", "torch": torch.__version__, "recipe": "2bit_group1024"})

    def test_pointer_free_frozen_plan_and_large_geometry_byte_arithmetic(self):
        graph, trace = _fixture((128, 768, 224, 224))
        plan, _, _ = _plan(graph, trace)
        self.assertEqual(plan.selected_bytes, 2 * 1271660544)
        self.assertGreater(plan.selected_bytes, 2**31)
        json.dumps(plan.as_dict(), allow_nan=False)
        with self.assertRaises(FrozenInstanceError):
            plan.bundles = ()

        def check(value):
            self.assertNotIsInstance(value, (torch.Tensor, TensorValue, IRGraph))
            if is_dataclass(value):
                for item in fields(value):
                    check(getattr(value, item.name))
            elif isinstance(value, tuple):
                for item in value:
                    check(item)
        check(plan)

    def test_automatic_transport_accepts_large_bundles_and_explicit_caps_still_reject(self):
        graph, trace = _fixture((256, 1024, 224, 224))
        plan, _, _ = _plan(graph, trace, device_staging_bytes=0, d2h_inflight_bytes=0, admission="error")
        bundle_bytes = 3_391_094_784
        self.assertEqual(plan.selected_bytes, 2 * bundle_bytes)
        self.assertEqual(plan.required_d2h_inflight_bytes, bundle_bytes)
        self.assertEqual(plan.required_staging_bytes, max(consumer.bytes for consumer in plan.consumers))
        with self.assertRaisesRegex(ValueError, "whole_bundle_exceeds_pinned_or_D2H_budget"):
            _plan(graph, trace, device_staging_bytes=0, d2h_inflight_bytes=3_000_000_000, admission="error")
        with self.assertRaisesRegex(ValueError, "atomic_consumer_staging_budget_exceeded"):
            _plan(graph, trace, device_staging_bytes=bundle_bytes - 1, d2h_inflight_bytes=0, admission="error")

    def test_host_and_atomic_consumer_bounds_strict_reject_and_recorded_fallback(self):
        graph, trace = _fixture()
        bundle_bytes = 4 * 264
        plan, _, _ = _plan(graph, trace, host_pinned_bytes=bundle_bytes, d2h_inflight_bytes=bundle_bytes)
        self.assertEqual(len(plan.bundles), 1)
        self.assertTrue(any("canonical_history" in row.reason for row in plan.rejections))
        with self.assertRaisesRegex(ValueError, "admission rejected"):
            _plan(graph, trace, admission="error", host_pinned_bytes=bundle_bytes)
        plan, _, _ = _plan(graph, trace, device_staging_bytes=bundle_bytes - 1)
        self.assertFalse(plan.bundles)
        self.assertTrue(any("atomic_consumer" in row.reason for row in plan.rejections))
        pageable, _, _ = _plan(graph, trace, host_pinned_bytes=bundle_bytes, host_pageable_bytes=4 * bundle_bytes,
                               d2h_inflight_bytes=bundle_bytes)
        self.assertEqual(len(pageable.bundles), 2)

    def test_conflicting_memberships_unknown_producers_and_output_alias_are_rejected(self):
        for kind in ("membership", "constructor", "output", "layout", "optional"):
            graph, trace = _fixture()
            target = trace.saved_tensors[0]
            if kind == "membership":
                record = deepcopy(trace.saved_value_specs[0])
                record["physical_source"] = "different_owner"
                trace = replace(trace, saved_value_specs=(*trace.saved_value_specs, record))
            elif kind == "constructor":
                target.producer.target = lambda x: x
            elif kind == "output":
                graph.nodes[-1].result = ((graph.nodes[-1].result[0], target), trace.saved_tensors)
                rebuild(graph)
            elif kind == "layout":
                target.meta = replace(target.meta, stride=(0, 1))
            else:
                target.provenance = "parameter"
                target.storage_source = "module.weight"
            with self.subTest(kind=kind):
                plan, _, _ = _plan(graph, trace)
                self.assertTrue(plan.rejections)
                self.assertNotIn(target.name, plan.selected_component_names)

    def test_stale_schedule_metadata_profile_and_changed_backward_slots_fail(self):
        graph, trace = _fixture()
        plan, forward, backward = _plan(graph, trace)
        value = trace.saved_tensors[0]
        value.meta = replace(value.meta)
        with self.assertRaisesRegex(ValueError, "stale"):
            plan.validate_current(forward, backward)
        plan, forward, backward = _plan(graph, trace)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            plan.validate_current(forward, backward, profile_identity={"gpu": "different"})
        graph, trace = _fixture()
        wrong = tuple(reversed(trace.saved_tensors))
        with self.assertRaisesRegex(ValueError, "sources changed"):
            build_packed_offload_plan(graph, trace.graph, Scheduler(graph).run(), Scheduler(trace.graph, backward=True).run(),
                trace.saved_value_specs, policy=resolve_packed_offload({"enabled": True,
                    "host_pinned_bytes": 4096, "device_staging_bytes": 4096, "d2h_inflight_bytes": 4096}),
                saved_tensors=wrong, backward_argument_sources=trace.backward_argument_sources)

    def test_disabled_plan_keeps_no_transport_or_prefetch_actions(self):
        graph, trace = _fixture()
        plan = build_packed_offload_plan(graph, trace.graph, Scheduler(graph).run(), Scheduler(trace.graph, backward=True).run())
        self.assertFalse(plan.bundles)
        self.assertFalse(plan.prefetches)
        self.assertFalse(plan.boundary_ids)
        self.assertFalse(plan.policy.enabled)

    def test_backward_views_escapes_mutation_and_custom_streams_cannot_release_staging_early(self):
        for kind in ("view", "escape", "mutation", "stream"):
            graph, trace = _fixture()
            packed = next(node.value for node in trace.graph.nodes if isinstance(node, InputNode)
                          and node.index == len(trace.grad_output_source))
            if kind == "view":
                alias = TensorValue("packed_alias", packed.meta)
                view = CallNode("view", lambda x: x.view_as(x), (packed,), {}, (alias,))
                trace.graph.nodes.insert(-1, view)
                rebuild(trace.graph)
            elif kind == "escape":
                trace.graph.nodes[-1].result = (trace.graph.nodes[-1].result, packed)
                rebuild(trace.graph)
            else:
                restore = next(node for node in trace.graph.nodes if packed in node.inputs())
                if kind == "mutation":
                    restore.scheduling_constraints["mutation"] = "unknown_write"
                else:
                    restore.stream_id = "other-stream"
            with self.subTest(kind=kind):
                plan, _, _ = _plan(graph, trace)
                self.assertNotIn(trace.saved_tensors[0].name, plan.selected_component_names)
                self.assertTrue(any("backward" in row.reason for row in plan.rejections))

    def test_native_packed_storage_exp_uses_exact_fresh_bound_executor_contract(self):
        from deepopt._bf16_policy import BF16Policy
        from deepopt.compiler.autodiff.primitives import register_rules
        from deepopt.compiler.autodiff.semantic import packed_metas
        from deepopt.compiler.registry import BackendRegistry
        from deepopt.opt.nn.primitive_state import StorageExpState

        graph, registry = IRGraph(), BackendRegistry()
        register_rules(registry)
        shape = (4, 1024)
        metadata = TensorMeta(shape, torch.bfloat16, "cuda:0", contiguous_stride(shape), True)
        x, output = TensorValue("source", metadata), TensorValue("exp_output", metadata)
        spec = StorageExpState(shape, BF16Policy("bf16_storage"), 2, 1024, True)
        components = tuple(TensorValue(f"exp_save_{index}", meta) for index, meta in enumerate(
            packed_metas((1, 4096), {"bits": 2, "group_size": 1024}, "cuda:0")))
        graph.add(InputNode(0, x))
        graph.add(CallNode("autodiff_storage_exp", spec.forward, (x,), {}, (output, *components), spec))
        graph.add(OutputNode(output))
        trace = AOTAutoDiff(graph, registry=registry).diff()
        graph.nodes[-1].result = (output, trace.saved_tensors)
        rebuild(graph)
        plan, _, _ = _plan(graph, trace)
        self.assertEqual(len(plan.bundles), 1)
        self.assertFalse(plan.rejections)

    def test_forward_alias_escape_is_rejected_and_proven_decode_extends_last_read(self):
        graph, trace = _fixture()
        leaves = trace.saved_tensors[:3]
        alias = TensorValue("public_packed_view", leaves[0].meta)
        graph.nodes.insert(-1, CallNode("view", lambda x: x.view_as(x), (leaves[0],), {}, (alias,)))
        graph.nodes[-1].result = ((graph.nodes[-1].result[0], alias), trace.saved_tensors)
        rebuild(graph)
        plan, _, _ = _plan(graph, trace)
        self.assertNotIn(leaves[0].name, plan.selected_component_names)
        self.assertTrue(any("forward_consumer" in row.reason for row in plan.rejections))
        graph, trace = _fixture()
        leaves = trace.saved_tensors[:3]
        restore = next(node for node in trace.graph.nodes if getattr(node, "op", None) == "codec.restore_activation")
        decoded = TensorValue("forward_decoded_copy", restore.values[0].meta)
        graph.nodes.insert(-1, CallNode("codec.restore_activation", restore.target, tuple(leaves), {}, (decoded,), restore.specification))
        graph.nodes[-1].result = ((graph.nodes[-1].result[0], decoded), trace.saved_tensors)
        rebuild(graph)
        plan, forward, _ = _plan(graph, trace)
        bundle = next(bundle for bundle in plan.bundles if bundle.components[0].name == leaves[0].name)
        self.assertEqual(bundle.last_forward_use, len(forward.items) - 2)
        self.assertGreater(bundle.last_forward_use, bundle.producer_boundary)


if __name__ == "__main__":
    unittest.main()
