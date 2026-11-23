"""Actual CUDA transport gates; CPU skips provide no launch evidence."""

from dataclasses import replace
from math import prod
import gc
import os
import unittest
import weakref

import torch

from deepopt.compiler.runtime.manager import CompiledGraphRuntimeManager
from deepopt.compiler.runtime.packed_offload import OffloadProtocolError
from deepopt.compiler.offload_plan import PackedOffloadConsumer, PackedOffloadPanel
from tests.compiler.test_packed_offload_lifecycle import bitwise, components, fixture, graph, publish


@unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA packed transport")
class PackedOffloadCudaTests(unittest.TestCase):
    def test_deterministic_allocation_fill_orders_before_h2d_on_nondefault_stream(self):
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        fill = torch.utils.deterministic.fill_uninitialized_memory
        plan = fixture(groups=65_536, device="cuda:0")
        manager = CompiledGraphRuntimeManager(plan.policy)
        try:
            torch.use_deterministic_algorithms(True)
            torch.utils.deterministic.fill_uninitialized_memory = True
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                _, x, y = graph(manager, plan)
                y.backward(retain_graph=True)
                y.backward()
            manager.drain()
            self.assertEqual(x.grad.item(), 4)
            self.assertEqual(manager.staging_bytes, 0)
        finally:
            try:
                manager.close()
            finally:
                torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
                torch.utils.deterministic.fill_uninitialized_memory = fill

    def test_bitwise_tail_triplet_repeated_retained_and_pageable(self):
        for pageable in (False, True):
            with self.subTest(pageable=pageable):
                plan = fixture(groups=7, device="cuda:0", pageable=pageable)
                manager = CompiledGraphRuntimeManager(plan.policy)
                frame, x, y = graph(manager, plan)
                y.backward(retain_graph=True)
                manager.drain()
                y.backward()
                manager.drain()
                self.assertEqual(x.grad.item(), 4)
                self.assertEqual(manager.pinned.active, 0)
                self.assertEqual(manager.staging_bytes, 0)
                self.assertEqual(manager.counters["h2d_bytes"], 2 * plan.selected_bytes)
                self.assertTrue(all(record.closed for record in frame.records.values()))
                manager.close()

    def test_delayed_custom_stream_producer_and_independent_gpu_restorations(self):
        plan = fixture(groups=17, device="cuda:0", predictive=True)
        manager = CompiledGraphRuntimeManager(plan.policy)
        stream = torch.cuda.Stream()
        frame = manager.begin_forward(plan)
        with torch.cuda.stream(stream):
            source = components(plan.bundles[0])
            torch.cuda._sleep(2_000_000)
            source[0].add_(7)
            expected = tuple(value.clone() for value in source)
            refs = [weakref.ref(value) for value in source]
            exports = publish(frame, [source])
            del source
        gc.collect()
        self.assertTrue(all(reference() is None for reference in refs))
        with frame.backward_scope():
            task = frame.begin_backward(exports)
            task.advance(plan.consumers[0].boundary)
            restored = task.take("b0", "c0")
            self.assertEqual(len({value.untyped_storage()._cdata for value in restored}), 3)
            self.assertTrue(bitwise(restored, expected))
            task.release_after("b0", "c0")
            del restored
            task.finish()
        frame.abort()
        manager.drain()
        manager.close()

    def test_interleaved_graphs_and_concurrent_stream_backward_have_separate_staging(self):
        plan = fixture(groups=11, device="cuda:0", max_backward_tasks=2)
        manager = CompiledGraphRuntimeManager(plan.policy)
        first, x, y = graph(manager, plan)
        second, z, w = graph(manager, plan)
        self.assertNotEqual(first.invocation, second.invocation)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            y.backward(retain_graph=True)
        w.backward()
        with torch.cuda.stream(stream):
            y.backward()
        manager.drain()
        self.assertEqual(x.grad.item(), 4)
        self.assertEqual(z.grad.item(), 2)
        self.assertEqual(manager.pinned.active, 0)
        manager.close()

    def test_source_layout_alias_and_cuda_graph_capture_fail_before_transfer(self):
        plan = fixture(device="cuda:0")
        manager = CompiledGraphRuntimeManager(plan.policy)
        frame = manager.begin_forward(plan)
        source = components(plan.bundles[0])
        with self.assertRaises(OffloadProtocolError):
            frame.pack_ready("b0", (source[0], source[1], source[1]))
        frame.abort()
        frame = manager.begin_forward(plan)
        with self.assertRaisesRegex(OffloadProtocolError, "ABI"):
            frame.pack_ready("b0", (source[0].t().contiguous().t(), *source[1:]))
        frame.abort()
        capture_stream = torch.cuda.Stream()
        captured = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(OffloadProtocolError, "CUDA Graph capture"):
            with torch.cuda.graph(captured, stream=capture_stream):
                manager.begin_forward(plan)
        manager.close()

    def test_true_swiglu_panel_transport_never_restores_whole_saved_operands(self):
        from deepopt.opt.backends.triton.activation import swiglu
        from tests.compiler.test_native_swiglu_fusion import fixture as native_fixture, select

        shape = (2, 3, 5, 333)
        graph_ir, trace, _, _ = native_fixture(shape, device="cuda:0")
        select(graph_ir, trace, "forward_backward")
        native = next(node.target for node in trace.graph.nodes if getattr(node, "op", None) == "swiglu.backward")
        groups = (prod(shape) + 1023) // 1024
        plan = fixture(bundles=3, groups=groups, device="cuda:0", device_staging_bytes=1584)
        names = tuple(component.name for bundle in plan.bundles for component in bundle.components)
        panel = PackedOffloadPanel("swiglu_backward_v1", shape, names, 1, 2, 1584)
        consumer = PackedOffloadConsumer("panel", 3, tuple(b.bundle_id for b in plan.bundles), plan.selected_bytes, panel)
        plan = replace(plan, consumers=(consumer,))
        manager = CompiledGraphRuntimeManager(plan.policy)
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        up = torch.randn_like(x)
        saved = swiglu.forward(x, up, shape=shape, product=True)[1:]
        gradient = torch.tensor(.173, dtype=torch.bfloat16, device="cuda").expand(shape)
        expected = swiglu.backward(gradient, *saved, shape=shape)
        frame = manager.begin_forward(plan)
        exports = publish(frame, [saved[i:i + 3] for i in (0, 3, 6)])
        with frame.backward_scope():
            task = frame.begin_backward(exports)
            task.advance(3)
            actual = task.run_panel_consumer("panel", native, (gradient,))
        frame.abort()
        manager.drain()
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        self.assertEqual(manager.counters["h2d_bytes"], plan.selected_bytes)
        self.assertLessEqual(manager.staging_peak, 1584)
        self.assertGreater(manager.counters["panels_consumed"], 1)
        manager.close()


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("DEEPOPT_TEST_LARGE_CUDA_PACKED_OFFLOAD") == "1",
                     "requires opt-in actual original-shape CUDA host/device capacity")
class PackedOffloadOriginalShapeCudaTests(unittest.TestCase):
    def test_original_128_768_224_224_encoded_bundle(self):
        # Original 4,932,501,504-element activation, copied in encoded form.
        plan = fixture(groups=4_816_896, device="cuda:0")
        self.assertEqual(plan.selected_bytes, 1_271_660_544)
        manager = CompiledGraphRuntimeManager(plan.policy)
        _, x, y = graph(manager, plan)
        y.backward()
        manager.drain()
        self.assertEqual(x.grad.item(), 2)
        self.assertEqual(manager.counters["d2h_bytes"], plan.selected_bytes)
        self.assertEqual(manager.counters["h2d_bytes"], plan.selected_bytes)
        self.assertEqual(manager.pinned.active, 0)
        manager.close()


if __name__ == "__main__":
    unittest.main()
