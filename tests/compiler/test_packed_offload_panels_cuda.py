"""Named packed-window consumer checks; CPU diagnostics are not CUDA evidence."""

from copy import deepcopy
from dataclasses import replace
from math import prod
import os
import unittest
from unittest.mock import patch

import torch

from deepopt.compiler.autodiff import swiglu_fusion
from deepopt.compiler.offload_plan import PackedOffloadPanel
from deepopt.opt.quantization.activation import quantize_pack_activation
from tests.compiler.test_packed_offload_plan import _fixture, _plan


def _decode(triplet, count):
    data, step, minimum = triplet
    index = torch.arange(count, dtype=torch.int64, device=data.device)
    group, offset = index // 1024, index % 1024
    code = ((data.flatten()[group * 64 + offset // 16].to(torch.int64) >> (offset % 16 * 2)) & 3).float()
    return code * step[group] + minimum[group]


def _reference_panel(gradient, outputs, triplets, start, count):
    dy = gradient.contiguous().flatten()[start:start + count].float()
    coefficient, gate, up = (_decode(triplet, count) for triplet in triplets)
    product_adjoint = (dy * up.to(torch.bfloat16).float()).to(torch.bfloat16).float()
    outputs[0].flatten()[start:start + count] = (product_adjoint * coefficient).to(torch.bfloat16)
    outputs[1].flatten()[start:start + count] = (dy * gate.to(torch.bfloat16).float()).to(torch.bfloat16)


class PackedOffloadPanelContractTests(unittest.TestCase):
    def test_CPU_group_panels_preserve_three_triplets_global_offsets_and_BF16_boundaries(self):
        torch.manual_seed(511)
        shape = (2, 3, 5, 333)  # Includes a partial final group.
        sources = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(3)]
        packs = [quantize_pack_activation(source.reshape(1, -1), 1024, bits=2) for source in sources]
        saved = tuple((pack.data, pack.step, pack.minimum) for pack in packs)
        gradient = torch.randn((*shape[:-1], shape[-1] * 2), dtype=torch.bfloat16)[..., ::2]
        full = tuple(torch.empty(shape, dtype=torch.bfloat16) for _ in range(2))
        panels = tuple(torch.full(shape, float("nan"), dtype=torch.bfloat16) for _ in range(2))
        _reference_panel(gradient, full, saved, 0, prod(shape))
        max_staged = 0
        for start in range(0, prod(shape), 3 * 1024):
            count = min(3 * 1024, prod(shape) - start)
            group, groups = start // 1024, (count + 1023) // 1024
            staged = tuple(tuple(value[group:group + groups].clone() for value in triplet) for triplet in saved)
            max_staged = max(max_staged, sum(value.numel() * value.element_size() for triplet in staged for value in triplet))
            _reference_panel(gradient, panels, staged, start, count)
        self.assertLessEqual(max_staged, 3 * 3 * 264)
        self.assertLess(max_staged, sum(value.numel() * value.element_size() for triplet in saved for value in triplet))
        for actual, expected in zip(panels, full):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Python metadata arithmetic above signed int32; no giant CPU payload.
        start = 2**31 + 1024
        self.assertGreater(start + 1023, 2**31)
        self.assertEqual((start // 1024) * 1024, start)

    def test_only_trusted_complete_named_consumer_can_use_small_staging(self):
        graph, trace = _fixture((8, 1024))
        # A generic multiply remains a whole-bundle consumer, even if a user
        # forges a suggestive attribute on its Python target.
        for node in trace.graph.nodes:
            if hasattr(node, "target"):
                try:
                    node.target.packed_offload_panel = object()
                except AttributeError:
                    pass
        plan, _, _ = _plan(graph, trace, device_staging_bytes=1584)
        self.assertFalse(plan.bundles)
        self.assertTrue(any("atomic_consumer" in row.reason for row in plan.rejections))
        for invalid in (0, True, -1):
            with self.subTest(groups=invalid), self.assertRaises(ValueError):
                PackedOffloadPanel("swiglu_backward_v1", (2, 16, 16, 16), tuple(str(i) for i in range(9)),
                    invalid, 2, invalid * 2 * 3 * 264)

    def test_native_callable_adapter_exists_only_on_proven_fused_backward(self):
        # Exercise the existing actual CPU native VJP and then its static CUDA
        # fusion proof. The resulting adapter/plan still does not execute CUDA.
        from deepopt.compiler.autodiff.joint import JointProgram
        from deepopt.compiler.autodiff.optimization import rebuild
        from tests.compiler.test_native_training_fusion_combined import NativeTrainingFusionCombinedTests

        programs = []
        original = JointProgram.__post_init__

        def capture(program):
            original(program)
            programs.append(program)

        with patch.object(JointProgram, "__post_init__", new=capture):
            NativeTrainingFusionCombinedTests("test_actual_CPU_graph_admits_both_passes_under_static_CUDA_metadata_only").run(
                result := unittest.TestResult())
        self.assertFalse(result.errors, result.errors)
        self.assertFalse(result.failures, result.failures)
        program = programs[-1]
        graph, trace = program.graph, program.trace
        graph.nodes[-1].result = (graph.nodes[-1].result, trace.saved_tensors)
        rebuild(graph)
        records = deepcopy(trace.saved_value_specs)
        for record in records:
            if record.get("logical_device") == "cpu":
                record["logical_device"] = "cuda:0"
            for component in record.get("physical_components", ()):
                if component.get("device") == "cpu":
                    component["device"] = "cuda:0"
        trace = replace(trace, saved_value_specs=records)
        plan, _, _ = _plan(graph, trace, device_staging_bytes=1584, copy_chunk_bytes=792)
        consumers = [consumer for consumer in plan.consumers if consumer.panel is not None]
        self.assertEqual(len(consumers), 1)
        consumer = consumers[0]
        self.assertEqual(consumer.panel.staging_bytes, 1584)
        self.assertEqual(consumer.panel.buffer_count, 2)
        self.assertEqual(consumer.panel.groups_per_panel, 1)
        self.assertGreater(consumer.bytes, consumer.panel.staging_bytes)
        bundles = {bundle.bundle_id: bundle for bundle in plan.bundles}
        self.assertEqual(tuple(component.name for identifier in consumer.bundle_ids for component in bundles[identifier].components),
                         consumer.panel.component_names)
        self.assertFalse(any(request.consumer_id == consumer.consumer_id for request in plan.prefetches))
        node = next(node for node in trace.graph.nodes if getattr(node, "op", None) == "swiglu.backward")
        self.assertIs(type(node.target.packed_offload_panel), swiglu_fusion.SwiGLUPackedOffloadPanel)
        whole, _, _ = _plan(graph, trace)
        self.assertFalse(any(consumer.panel is not None for consumer in whole.consumers))
        pageable, _, _ = _plan(graph, trace, device_staging_bytes=90000, host_pinned_bytes=33792,
            host_pageable_bytes=2**20, d2h_inflight_bytes=33792, copy_chunk_bytes=2**30)
        pageable_panel = next(consumer.panel for consumer in pageable.consumers if consumer.panel is not None)
        self.assertLessEqual(pageable_panel.staging_bytes // pageable_panel.buffer_count, pageable.policy.host_pinned_bytes)
        self.assertEqual(pageable_panel.groups_per_panel, 33792 // (3 * 264))
        node.target.packed_offload_panel = object()
        rejected, _, _ = _plan(graph, trace, device_staging_bytes=1584)
        self.assertFalse(any(consumer.panel is not None for consumer in rejected.consumers))


def _cuda_compare(test, shape, *, strided=False, groups_per_panel=3):
    from deepopt.opt.backends.triton.activation import swiglu

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(761)
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        up = torch.randn_like(x)
        saved = swiglu.forward(x, up, shape=shape, product=True)[1:]
        gradient = (torch.randn((*shape[:-1], shape[-1] * 2), dtype=torch.bfloat16, device="cuda")[..., ::2]
                    if strided else torch.tensor(.173, dtype=torch.bfloat16, device="cuda").expand(shape))
        expected = swiglu.backward(gradient, *saved, shape=shape)
        actual = swiglu.allocate_panel_outputs(gradient, shape=shape)
        maximum = 0
        with patch.object(swiglu, "backward", side_effect=AssertionError("whole VJP called from panel path")):
            for start in range(0, prod(shape), groups_per_panel * 1024):
                count = min(groups_per_panel * 1024, prod(shape) - start)
                first, groups = start // 1024, (count + 1023) // 1024
                triplets = tuple(tuple(value[first:first + groups].clone() for value in saved[index:index + 3])
                                 for index in (0, 3, 6))
                staged = sum(value.numel() * value.element_size() for triplet in triplets for value in triplet)
                maximum = max(maximum, staged)
                test.assertLessEqual(staged, groups_per_panel * 3 * 264)
                swiglu.backward_panel(gradient, actual, triplets, start, count, shape=shape)
        for result, baseline in zip(actual, expected):
            torch.testing.assert_close(result, baseline, rtol=0, atol=0)
        test.assertLess(maximum, sum(value.numel() * value.element_size() for value in saved))


@unittest.skipUnless(torch.cuda.is_available(), "requires actual packed SwiGLU panel CUDA execution")
class PackedOffloadPanelCudaTests(unittest.TestCase):
    def test_expanded_and_strided_cotangents_tail_are_exact(self):
        for shape in ((2, 16, 16, 16), (1, 1, 7, 777)):
            for strided in (False, True):
                with self.subTest(shape=shape, strided=strided):
                    _cuda_compare(self, shape, strided=strided)

    def test_panel_alignment_and_out_of_range_reject_before_kernel(self):
        from deepopt.opt.backends.triton.activation import swiglu

        gradient = torch.empty((2, 16, 16, 16), dtype=torch.bfloat16, device="cuda")
        for start, count in ((1, 1024), (-1024, 1024), (0, 0), (8192, 1024), (0, 1025)):
            with self.subTest(start=start, count=count), self.assertRaisesRegex(ValueError, "aligned"):
                swiglu.backward_panel(gradient, (), (), start, count, shape=tuple(gradient.shape))


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("DEEPOPT_TEST_LARGE_CUDA_PACKED_OFFLOAD_PANELS") == "1",
                     "requires explicit original B128 packed-panel CUDA capacity run")
class PackedOffloadPanelOriginalShapeCudaTests(unittest.TestCase):
    def test_original_B128_global_offsets_are_exact(self):
        _cuda_compare(self, (128, 384, 224, 224), groups_per_panel=32768)


if __name__ == "__main__":
    unittest.main()
