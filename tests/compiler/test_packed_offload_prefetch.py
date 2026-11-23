"""Prefetch usefulness and retirement ordering, without CUDA performance claims."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from deepopt.compiler.offload_plan import PackedOffloadConsumer
from deepopt.compiler.runtime.manager import CompiledGraphRuntimeManager
from deepopt.compiler.runtime.packed_offload import TestingTransportBackend, _CpuEvent, _Staging
from tests.compiler.test_packed_offload_lifecycle import fixture, components, publish, bitwise


class _Exercise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frame, value, consume):
        ctx.frame, ctx.consume = frame, consume
        saves = publish(frame, [components(bundle) for bundle in frame.plan.bundles])
        ctx.save_for_backward(*saves)
        return value * 2

    @staticmethod
    def backward(ctx, gradient):
        with ctx.frame.backward_scope():
            task = ctx.frame.begin_backward(ctx.saved_tensors)
            ctx.maybe_clear_saved_tensors()
            try:
                ctx.consume(task)
            finally:
                task.finish()
        return None, gradient * 2, None


class PackedOffloadUsefulPrefetchTests(unittest.TestCase):
    def exercise(self, plan, consume, backend=None):
        backend = backend or TestingTransportBackend()
        manager = CompiledGraphRuntimeManager(plan.policy, backend=backend, start_worker=False)
        self.addCleanup(manager.close)
        frame = manager.begin_forward(plan)
        value = torch.tensor(1., requires_grad=True)
        with frame.saved_token_scope():
            output = _Exercise.apply(frame, value, consume)
        output.backward()
        self.assertEqual(value.grad.item(), 2.)
        manager.drain()
        return manager

    def test_future_prefetch_is_not_evicted_when_current_operands_fit(self):
        plan = fixture(bundles=3, predictive=True)

        def consume(task):
            manager = task.manager
            manager._restore(task, "b2", prefetch=True)
            for consumer in plan.consumers:
                task.advance(consumer.boundary)
                for bid in consumer.bundle_ids:
                    self.assertTrue(bitwise(task.take(bid), components(manager._bundle(plan, bid))))
                    task.release_after(bid)

        manager = self.exercise(plan, consume)
        self.assertEqual(manager.counters["prefetch_evictions"], 0)
        self.assertEqual(manager.counters["h2d_bytes"], plan.selected_bytes)
        self.assertGreaterEqual(manager.counters["prefetch_consumed"], 1)
        self.assertEqual(manager.counters["prefetch_consumed"], manager.counters["prefetches"])

    def test_short_shared_reuse_does_not_transfer_the_bundle_twice(self):
        plan = fixture(bundles=2, predictive=True)
        first = replace(plan.bundles[0], consumers=("c0", "again"), last_backward_use=9)
        last = PackedOffloadConsumer("again", 9, ("b0",), first.bytes)
        plan = replace(plan, bundles=(first, plan.bundles[1]), consumers=(*plan.consumers, last), signature="shared")

        def consume(task):
            for consumer in plan.consumers:
                task.advance(consumer.boundary)
                for bid in consumer.bundle_ids:
                    task.take(bid)
                    task.release_after(bid)

        manager = self.exercise(plan, consume)
        self.assertEqual(manager.counters["h2d_bytes"], plan.selected_bytes)
        self.assertEqual(manager.counters["shared_restore_reuses"], 1)
        self.assertEqual(manager.counters["short_reuse_retained"], 1)
        self.assertEqual(manager.staging_bytes, 0)

    def test_pressure_evicts_only_the_farthest_unused_bundle_needed_for_credits(self):
        plan = fixture(bundles=4, predictive=True, device_staging_bytes=2 * 528)

        def consume(task):
            task.manager._restore(task, "b2", prefetch=True)
            task.manager._restore(task, "b3", prefetch=True)
            for consumer in plan.consumers:
                task.advance(consumer.boundary)
                for bid in consumer.bundle_ids:
                    task.take(bid)
                    task.release_after(bid)

        manager = self.exercise(plan, consume)
        self.assertEqual(manager.counters["prefetch_evictions"], 1)
        self.assertEqual(manager.counters["prefetch_evicted_before_use_bytes"], 528)
        self.assertEqual(manager.counters["h2d_bytes"], plan.selected_bytes + 528)

    def test_three_component_bindings_insert_one_consumer_wait(self):
        class CountingBackend(TestingTransportBackend):
            waits = 0

            def wait(self, event):
                self.waits += 1

        backend = CountingBackend()
        plan = fixture()

        def consume(task):
            consumer = plan.consumers[0]
            task.advance(consumer.boundary)
            for component in plan.bundles[0].components:
                task.resolve(component.saved_index)
            task.release_after("b0")

        manager = self.exercise(plan, consume, backend)
        self.assertEqual(backend.waits, 1)
        self.assertEqual(manager.counters["consumer_bundle_reads"], 1)

    def test_retained_copy_retires_after_last_consumer_not_original_dma(self):
        plan = fixture()
        manager = CompiledGraphRuntimeManager(plan.policy, backend=TestingTransportBackend(), start_worker=False)
        self.addCleanup(manager.close)
        dma, last_read = _CpuEvent(True), _CpuEvent(False)
        record = SimpleNamespace(generation=1)
        stage = _Staging(record, (torch.empty(2),), dma, bytes=8, consumed=True, dma_done=True,
                         last_consumer_event=last_read)
        manager.staging_bytes = 8
        manager._retire(stage)
        self.assertIs(stage.completion, last_read)
        try:
            manager.poll()
            self.assertEqual(manager.staging_bytes, 8)
        finally:
            last_read.complete()
        manager.poll()
        self.assertEqual(manager.staging_bytes, 0)


if __name__ == "__main__":
    unittest.main()
