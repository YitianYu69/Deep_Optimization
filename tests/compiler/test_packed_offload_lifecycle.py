"""CPU state-machine/autograd tests, explicitly not CUDA transport evidence."""

from dataclasses import replace
import gc
import json
import threading
import unittest
from unittest.mock import patch
import weakref

import torch

from deepopt._packed_offload import PackedActivationOffloadPolicy
from deepopt.compiler.offload_plan import (
    PackedOffloadBundle, PackedOffloadComponent, PackedOffloadConsumer, PackedOffloadPanel, PackedOffloadPlan,
)
from deepopt.compiler.runtime.manager import (
    CompiledGraphRuntimeManager, compiled_forward_scope, packed_offload_runtime_snapshot,
    compiled_runtime_scope,
)
from deepopt.compiler.runtime.packed_offload import (
    OffloadAdmissionError, OffloadProtocolError, TestingTransportBackend,
)


def fixture(*, bundles=1, groups=2, device="cpu", pageable=False, predictive=False, **overrides):
    size = groups * 264
    options = dict(enabled=True, policy="memory_first_overlap" if predictive else "demand",
        host_pinned_bytes=size if pageable else size * bundles * 2,
        host_pageable_bytes=size * bundles * 2 if pageable else 0,
        device_staging_bytes=size * bundles, d2h_inflight_bytes=size * bundles,
        max_live_forwards=2, max_backward_tasks=2, admission="error")
    options.update(overrides)
    policy = PackedActivationOffloadPolicy(**options)
    saves, consumers = [], []
    for index in range(bundles):
        components = tuple(PackedOffloadComponent(f"b{index}.{role}", index * 3 + offset, shape, dtype, stride,
                device, prod_bytes, role)
            for offset, (role, shape, dtype, stride, prod_bytes) in enumerate((
                ("payload", (groups, 64), "torch.int32", (64, 1), groups * 256),
                ("codec_step", (groups,), "torch.float32", (1,), groups * 4),
                ("codec_minimum", (groups,), "torch.float32", (1,), groups * 4))))
        boundary = 3 + index * 3
        saves.append(PackedOffloadBundle(f"b{index}", components, index, index + 1, (f"c{index}",), boundary, boundary))
        consumers.append(PackedOffloadConsumer(f"c{index}", boundary, (f"b{index}",), size))
    return PackedOffloadPlan(policy, tuple(saves), tuple(consumers), "test-plan", "forward", "backward")


def components(bundle, *, offset=0):
    groups = bundle.components[0].shape[0]
    device = bundle.components[0].device
    payload = (torch.arange(groups * 64, dtype=torch.int32, device=device) * 173 - 127 + offset).reshape(groups, 64)
    # Exercise signed zero, subnormals, NaNs, and full FP32 bit preservation.
    bits = torch.tensor([0x80000000, 0x7FC12345, 0x00000001, 0x3EAAAAAB], dtype=torch.uint32)
    indices = torch.arange(groups) % 4
    step = bits.view(torch.int32)[indices].clone().to(device).view(torch.float32)
    minimum = (bits.view(torch.int32)[(indices + 1) % 4].clone().to(device).view(torch.float32))
    return payload, step, minimum


def bitwise(left, right):
    return all(torch.equal(a.view(torch.int32), b.view(torch.int32)) for a, b in zip(left, right))


def publish(frame, values):
    exports = [None] * (len(frame.plan.bundles) * 3)
    for bundle, saved in zip(frame.plan.bundles, values):
        frame.pack_ready(bundle.bundle_id, saved)
        frame.last_forward_use(bundle.bundle_id)
        for component in bundle.components:
            exports[component.saved_index] = frame.export_slot(component.saved_index)
    frame.finish_forward()
    return exports


class _SavedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frame, x, resident):
        ctx.frame = frame
        values = [components(bundle) for bundle in frame.plan.bundles]
        exports = publish(frame, values)
        ctx.save_for_backward(*exports, resident)
        return x * 2

    @staticmethod
    def backward(ctx, gradient):
        frame = ctx.frame
        with frame.backward_scope():
            saved = ctx.saved_tensors
            task = frame.begin_backward(saved)
            ctx.maybe_clear_saved_tensors()
            for consumer in frame.plan.consumers:
                task.advance(consumer.boundary)
                for bundle_id in consumer.bundle_ids:
                    bundle = next(b for b in frame.plan.bundles if b.bundle_id == bundle_id)
                    restored = task.take(bundle_id, consumer.consumer_id)
                    if not bitwise(restored, components(bundle)):
                        raise AssertionError("Encoded byte round trip differs")
                    task.release_after(bundle_id, consumer.consumer_id)
            task.finish()
        return None, gradient * 2, None


def graph(manager, plan, resident=None):
    frame = manager.begin_forward(plan)
    x = torch.tensor(1.5, device=plan.bundles[0].components[0].device, requires_grad=True)
    resident = torch.tensor(3., device=x.device) if resident is None else resident
    with frame.saved_token_scope():
        output = _SavedFunction.apply(frame, x, resident)
    return frame, x, output


class PackedOffloadLifecycleTests(unittest.TestCase):
    def manager(self, plan, **kwargs):
        result = CompiledGraphRuntimeManager(plan.policy, backend=TestingTransportBackend(), start_worker=False, **kwargs)
        self.addCleanup(result.close)
        return result

    def test_retained_repeated_backward_uses_independent_staging_and_releases_graph_tokens(self):
        plan = fixture()
        manager = self.manager(plan)
        frame, x, y = graph(manager, plan)
        self.assertEqual(manager.snapshot()["lifecycle"]["graph_leases"], 3)
        y.backward(retain_graph=True)
        manager.drain()
        self.assertEqual(x.grad.item(), 2)
        self.assertEqual(manager.snapshot()["lifecycle"]["graph_leases"], 3)
        first = manager.counters["h2d_bytes"]
        y.backward()
        manager.drain()
        self.assertEqual(x.grad.item(), 4)
        self.assertEqual(manager.counters["h2d_bytes"], 2 * first)
        self.assertEqual(manager.snapshot()["lifecycle"]["graph_leases"], 0)
        self.assertEqual(manager.pinned.active, 0)
        self.assertTrue(all(record.closed for record in frame.records.values()))

    def test_inspection_is_completed_nonpooled_snapshot_and_does_not_trigger_h2d(self):
        plan = fixture()
        manager = self.manager(plan)
        frame, _, y = graph(manager, plan)
        inspected = y.grad_fn.saved_tensors[:3]
        expected = tuple(value.clone() for value in inspected)
        self.assertTrue(bitwise(inspected, components(plan.bundles[0])))
        for value, pooled in zip(inspected, frame.records["b0"].host.tensors):
            self.assertNotEqual(value.untyped_storage()._cdata, pooled.untyped_storage()._cdata)
        self.assertEqual(manager.counters["h2d_bytes"], 0)
        del y
        gc.collect()
        manager.drain()
        second, _, y = graph(manager, plan)
        self.assertTrue(bitwise(inspected, expected))
        self.assertNotEqual(frame.invocation, second.invocation)
        y.backward()

    def test_abandoned_graph_keeps_dma_lease_until_completion_then_recycles(self):
        plan = fixture()
        backend = TestingTransportBackend(auto_complete=False)
        manager = CompiledGraphRuntimeManager(plan.policy, backend=backend, start_worker=False)
        frame, _, y = graph(manager, plan)
        del y
        gc.collect()
        self.assertEqual(manager.snapshot()["lifecycle"]["graph_leases"], 0)
        self.assertGreater(manager.pinned.active, 0)
        self.assertEqual(manager.snapshot()["lifecycle"]["dma_leases"], 1)
        for event in backend.events:
            event.complete()
        manager.drain()
        self.assertEqual(manager.pinned.active, 0)
        self.assertTrue(frame.records["b0"].closed)
        manager.close()

    def test_external_hooks_rejected_before_admission_and_restored(self):
        plan = fixture()
        manager = self.manager(plan)
        packed = []
        with torch.autograd.graph.saved_tensors_hooks(lambda value: packed.append(value) or value, lambda value: value):
            with self.assertRaisesRegex(RuntimeError, "cannot nest external"):
                manager.begin_forward(plan)
            x = torch.tensor(2., requires_grad=True)
            (x * x).backward()
        self.assertEqual(len(packed), 2)
        self.assertEqual(len(manager._plans), 0)

    def test_resident_passthrough_version_resize_and_storage_rebind_guards(self):
        for mutation in (lambda t: t.add_(1), lambda t: t.resize_(2), lambda t: t.set_(torch.ones(3))):
            with self.subTest(mutation=mutation):
                plan = fixture()
                manager = self.manager(plan)
                resident = torch.tensor(3.)
                _, _, y = graph(manager, plan, resident)
                mutation(resident)
                with self.assertRaisesRegex(RuntimeError, "modified by an inplace"):
                    y.backward()
                del y
                gc.collect()
                manager.drain()

    def test_resident_data_rebinding_matches_stock_saved_variable(self):
        class UseSaved(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, saved):
                ctx.save_for_backward(saved)
                return x * 2

            @staticmethod
            def backward(ctx, gradient):
                return gradient * ctx.saved_tensors[0].sum(), None

        for leaf in (False, True):
            with self.subTest(leaf=leaf):
                expected = torch.tensor([3., 4.], requires_grad=leaf)
                x = torch.tensor(1., requires_grad=True)
                reference = UseSaved.apply(x, expected)
                expected.data = torch.tensor([8., 9., 10.])
                reference.backward()
                plan = fixture()
                manager = self.manager(plan)
                frame = manager.begin_forward(plan)
                saved = torch.tensor([3., 4.], requires_grad=leaf)
                candidate_x = torch.tensor(1., requires_grad=True)
                with frame.saved_token_scope():
                    candidate = UseSaved.apply(candidate_x, saved)
                saved.data = torch.tensor([8., 9., 10.])
                candidate.backward()
                self.assertEqual(candidate_x.grad.item(), x.grad.item())

    def test_full_model_scope_admits_all_regions_before_producers_and_shares_limit(self):
        plan = fixture(host_pinned_bytes=1056, max_live_forwards=1)
        manager = self.manager(plan)
        with compiled_forward_scope((plan, plan), manager):
            self.assertEqual(manager.pinned.active, 1056)
            first = manager.begin_forward(plan)
            second = manager.begin_forward(plan)
            self.assertEqual(first.execution_scope, second.execution_scope)
            self.assertEqual(manager.snapshot()["lifecycle"]["live_forwards"], 1)
            first.abort()
            second.abort()
        self.assertEqual(manager.pinned.active, 0)
        with self.assertRaisesRegex(OffloadAdmissionError, "scope admission"):
            with compiled_forward_scope((plan, plan, plan), manager):
                self.fail("No producer may run on failed whole-model admission")
        self.assertEqual(manager.pinned.active, 0)

    def test_automatic_limits_cover_shared_regions_and_complete_operand_sets(self):
        options = dict(version=2, host_pinned_bytes=8192, device_staging_bytes=0, d2h_inflight_bytes=0)
        small = fixture(**options)
        wide = fixture(bundles=2, groups=3, **options)
        consumer = PackedOffloadConsumer("c0", 3, ("b0", "b1"), wide.selected_bytes)
        wide = replace(wide, signature="wide", consumers=(consumer,), bundles=tuple(
            replace(bundle, consumers=(consumer.consumer_id,), first_backward_use=3, last_backward_use=3)
            for bundle in wide.bundles))
        manager = self.manager(small)
        with compiled_forward_scope((small, wide), manager):
            limits = dict(device_staging_bytes=wide.selected_bytes * 2, d2h_inflight_bytes=792)
            self.assertEqual(manager.snapshot()["transport_limits"], limits)
            first = manager.begin_forward(small)
            frame = manager.begin_forward(wide)
            first.abort()
            values = [components(bundle) for bundle in wide.bundles]
            saved = publish(frame, values)
        with frame.backward_scope():
            task = frame.begin_backward(saved)
            task.advance(consumer.boundary)
            for bundle, expected in zip(wide.bundles, values):
                self.assertTrue(bitwise(task.take(bundle.bundle_id), expected))
            for bid in consumer.bundle_ids:
                task.release_after(bid)
        frame.abort()
        del saved  # CPU export bindings still hold their arena spans.
        manager.drain()
        snapshot = manager.snapshot()
        self.assertEqual(snapshot["device"]["staging_peak_bytes"], wide.selected_bytes)
        self.assertEqual(snapshot["device"]["peak_d2h_inflight_bytes"], 792)
        self.assertEqual(snapshot["counters"]["d2h_bytes"], wide.selected_bytes)
        self.assertEqual(snapshot["host"]["pinned_active_bytes"], 0)

    def test_peak_target_uses_actual_allocated_headroom_and_freeze_labels_cpu_cold(self):
        plan = fixture(peak_budget_bytes=600)
        backend = TestingTransportBackend()
        backend.external_allocated_bytes = 200
        manager = CompiledGraphRuntimeManager(plan.policy, backend=backend, start_worker=False)
        frame = manager.begin_forward(plan)
        saved = publish(frame, [components(plan.bundles[0])])
        with frame.backward_scope():
            task = frame.begin_backward(saved)
            task.advance(3)
            with self.assertRaisesRegex(OffloadAdmissionError, "peak target"):
                task.take("b0", "c0")
        frame.abort()
        manager.freeze_profiles()
        self.assertTrue(manager.snapshot()["profile_frozen"])
        self.assertFalse(manager.snapshot()["calibrated"])
        self.assertTrue(manager.snapshot()["cold"])
        manager.close()

    def test_named_panel_runtime_bounds_all_nine_leaves_and_preserves_tail_offsets(self):
        for pageable in (False, True):
            with self.subTest(pageable=pageable):
                plan = fixture(bundles=3, groups=9, pageable=pageable, predictive=True, device_staging_bytes=3168)
                names = tuple(component.name for bundle in plan.bundles for component in bundle.components)
                shape = (1, 1, 1, 9 * 1024 - 13)
                panel = PackedOffloadPanel("swiglu_backward_v1", shape, names, 2, 2, 3168)
                consumer = PackedOffloadConsumer("panel", 3, tuple(b.bundle_id for b in plan.bundles), plan.selected_bytes, panel)
                plan = replace(plan, consumers=(consumer,))
                backend = TestingTransportBackend()
                manager = CompiledGraphRuntimeManager(plan.policy, backend=backend, start_worker=False)
                frame = manager.begin_forward(plan)
                originals = [components(bundle) for bundle in plan.bundles]
                saved = publish(frame, originals)
                gradient = torch.linspace(-.7, .3, shape[-1], dtype=torch.bfloat16).reshape(shape)
                observed = []

                class Adapter:
                    kind, group_size, bits = "swiglu_backward_v1", 1024, 2

                    def __init__(self):
                        self.shape = shape

                    def allocate_outputs(self, gradient):
                        return torch.empty_like(gradient), torch.empty_like(gradient)

                    def run(self, dy, outputs, triplets, start, count):
                        self_outer.assertEqual(len({v.untyped_storage()._cdata for t in triplets for v in t}), 9)
                        first, groups = start // 1024, (count + 1023) // 1024
                        for triplet, original in zip(triplets, originals):
                            self_outer.assertTrue(bitwise(triplet, tuple(v[first:first + groups] for v in original)))
                        self_outer.assertLessEqual(backend.allocated_bytes(), plan.policy.device_staging_bytes)
                        outputs[0].flatten()[start:start + count] = dy.flatten()[start:start + count] * 2
                        outputs[1].flatten()[start:start + count] = dy.flatten()[start:start + count] * 3
                        observed.append((start, count))

                self_outer = self

                def native(*args):
                    raise AssertionError("Whole packed restore is forbidden for named panel path")

                native.packed_offload_panel = Adapter()
                with frame.backward_scope():
                    task = frame.begin_backward(saved)
                    task.advance(2)
                    manager.poll()
                    self.assertEqual(manager.counters["h2d_bytes"], 0)
                    task.advance(3)
                    with self.assertRaisesRegex(OffloadProtocolError, "whole-bundle"):
                        task.take("b0", "panel")
                    results = task.run_panel_consumer("panel", native, (gradient,))
                    torch.testing.assert_close(results[0], gradient * 2, rtol=0, atol=0)
                    torch.testing.assert_close(results[1], gradient * 3, rtol=0, atol=0)
                frame.abort()
                manager.drain()
                self.assertEqual(observed[-1], (8192, 1011))
                self.assertEqual(manager.counters["h2d_bytes"], plan.selected_bytes)
                self.assertLessEqual(manager.staging_peak, 3168)
                self.assertEqual(manager.staging_bytes, 0)
                manager.close()

    def test_retirement_error_preserves_primary_and_discards_task_after_owned_drain(self):
        class FailingEvents(TestingTransportBackend):
            fail = False
            drains = 0

            def event(self, stream=None, *, timing=False):
                if self.fail:
                    raise RuntimeError("retirement-secondary")
                return super().event(stream, timing=timing)

            def synchronize_consumer(self):
                self.drains += 1

        plan = fixture()
        backend = FailingEvents()
        manager = CompiledGraphRuntimeManager(plan.policy, backend=backend, start_worker=False)
        frame = manager.begin_forward(plan)
        saved = publish(frame, [components(plan.bundles[0])])
        with self.assertRaisesRegex(ValueError, "kernel-primary"):
            with frame.backward_scope():
                task = frame.begin_backward(saved)
                task.advance(3)
                task.resolve(0)
                backend.fail = True
                raise ValueError("kernel-primary")
        self.assertEqual(len(manager._active_tasks), 0)
        self.assertGreater(backend.drains, 0)
        frame.abort()
        manager.drain()
        self.assertEqual(manager.staging_bytes, 0)
        self.assertEqual(manager.pinned.active, 0)
        self.assertTrue(any(row["exception_class"] == "RuntimeError" for row in manager.snapshot()["cleanup_errors"]))
        manager.close()

    def test_panel_kernel_error_survives_failed_retirement_event_and_releases_task(self):
        class FailingEvents(TestingTransportBackend):
            fail = False

            def event(self, stream=None, *, timing=False):
                if self.fail:
                    raise RuntimeError("panel-retirement-secondary")
                return super().event(stream, timing=timing)

        plan = fixture(bundles=3, groups=3, device_staging_bytes=1584)
        shape = (1, 1, 1, 3072)
        names = tuple(component.name for bundle in plan.bundles for component in bundle.components)
        panel = PackedOffloadPanel("swiglu_backward_v1", shape, names, 1, 2, 1584)
        consumer = PackedOffloadConsumer("panel", 3, tuple(b.bundle_id for b in plan.bundles), plan.selected_bytes, panel)
        plan = replace(plan, consumers=(consumer,))
        backend = FailingEvents()
        manager = CompiledGraphRuntimeManager(plan.policy, backend=backend, start_worker=False)
        frame = manager.begin_forward(plan)
        saved = publish(frame, [components(bundle) for bundle in plan.bundles])

        class Adapter:
            kind, group_size, bits = "swiglu_backward_v1", 1024, 2

            def __init__(self):
                self.shape = shape

            def allocate_outputs(self, gradient):
                return torch.empty_like(gradient), torch.empty_like(gradient)

            def run(self, *args):
                backend.fail = True
                raise ValueError("panel-kernel-primary")

        def native(*args):
            raise AssertionError("Whole saved operands must not be restored")

        native.packed_offload_panel = Adapter()
        with self.assertRaisesRegex(ValueError, "panel-kernel-primary"):
            with frame.backward_scope():
                task = frame.begin_backward(saved)
                task.advance(3)
                task.run_panel_consumer("panel", native, (torch.ones(shape, dtype=torch.bfloat16),))
        self.assertFalse(task._panel_stages)
        self.assertFalse(manager._active_tasks)
        self.assertTrue(any(row["role"] == "consumer_retirement_event"
                            and row["message"] == "panel-retirement-secondary"
                            for row in manager.snapshot()["cleanup_errors"]))
        frame.abort()
        manager.drain()
        self.assertEqual(manager.staging_bytes, 0)
        self.assertEqual(manager.pinned.active, 0)
        manager.close()

    def test_dma_only_previous_history_backpressures_next_forward_instead_of_fallback(self):
        plan = fixture(host_pinned_bytes=528, max_live_forwards=1)
        backend = TestingTransportBackend()
        manager = CompiledGraphRuntimeManager(plan.policy, backend=backend, start_worker=False)
        _, _, y = graph(manager, plan)
        backend.auto_complete = False
        y.backward()
        entered, finished = threading.Event(), threading.Event()
        results, errors = [], []

        def next_forward():
            entered.set()
            try:
                results.append(manager.begin_forward(plan))
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        thread = threading.Thread(target=next_forward)
        thread.start()
        self.assertTrue(entered.wait(timeout=5))
        self.assertFalse(finished.wait(timeout=.03))
        for event in tuple(backend.events):
            event.complete()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        self.assertFalse(results[0].resident_fallback)
        self.assertGreater(manager.counters["history_backpressure_waits"], 0)
        results[0].abort()
        manager.close()

    def test_forward_source_registration_during_worker_poll(self):
        plan = fixture(bundles=2)
        manager = self.manager(plan)
        frame = manager.begin_forward(plan)
        first, second = plan.bundles
        first_sources, second_sources = components(first), components(second)
        with frame.saved_token_scope():
            frame.pack_ready(first.bundle_id, first_sources)
            frame.last_forward_use(first.bundle_id)
            source_ref = next(iter(manager._source_ids.values()))[0]
            expired = type(source_ref).expired
            polling, resume = threading.Event(), threading.Event()
            runtime_lock = manager._lock
            worker = threading.Thread(target=manager._pump, daemon=True)

            def pause_expiry(reference):
                if reference is source_ref and threading.current_thread() is worker:
                    polling.set()
                    if not resume.wait(timeout=5):
                        raise TimeoutError("Forward did not reach the runtime lock")
                return expired(reference)

            def enter_lock():
                if threading.current_thread() is not worker:
                    resume.set()
                return runtime_lock.__enter__()

            # Resume the worker once the forward tries to take its lock.
            with (
                patch.object(type(source_ref), "expired", pause_expiry),
                patch.object(manager, "_lock") as observed_lock,
            ):
                observed_lock.__enter__.side_effect = enter_lock
                observed_lock.__exit__.side_effect = runtime_lock.__exit__
                worker.start()
                try:
                    self.assertTrue(polling.wait(timeout=5), "Worker did not enter storage expiry")
                    frame.pack_ready(second.bundle_id, second_sources)
                finally:
                    resume.set()
                    worker.join(timeout=5)

            self.assertFalse(worker.is_alive())
            manager.drain()
            self.assertEqual(len(manager._source_ids), 6)
            frame.last_forward_use(second.bundle_id)
            frame.finish_forward()

    def test_rejected_source_overlap_leaves_new_storages_available(self):
        plan = fixture(bundles=2)
        manager = self.manager(plan)
        frame = manager.begin_forward(plan)
        first, second = plan.bundles
        first_sources, second_sources = components(first), components(second)
        with frame.saved_token_scope():
            frame.pack_ready(first.bundle_id, first_sources)
            frame.last_forward_use(first.bundle_id)
            with self.assertRaisesRegex(OffloadProtocolError, "overlap live source storage"):
                frame.pack_ready(second.bundle_id, (second_sources[0], first_sources[1], second_sources[2]))
            frame.pack_ready(second.bundle_id, second_sources)
            frame.last_forward_use(second.bundle_id)
            frame.finish_forward()

    def test_concurrent_backward_credit_waits_for_other_enqueued_task_without_aliasing(self):
        plan = fixture(device_staging_bytes=528)
        backend = TestingTransportBackend()
        manager = CompiledGraphRuntimeManager(plan.policy, backend=backend, start_worker=False)
        frame = manager.begin_forward(plan)
        saved = publish(frame, [components(plan.bundles[0])])
        first_ready, release_first, second_entered, second_done = (threading.Event() for _ in range(4))
        errors = []

        def backward(first):
            try:
                with frame.backward_scope():
                    task = frame.begin_backward(saved)
                    task.advance(3)
                    if not first:
                        second_entered.set()
                    values = task.take("b0", "c0")
                    self.assertTrue(bitwise(values, components(plan.bundles[0])))
                    if first:
                        first_ready.set()
                        if not release_first.wait(timeout=5):
                            raise RuntimeError("test concurrent release timed out")
                    del values
                    task.release_after("b0", "c0")
            except BaseException as error:
                errors.append(error)
            finally:
                if not first:
                    second_done.set()

        a, b = threading.Thread(target=backward, args=(True,)), threading.Thread(target=backward, args=(False,))
        a.start()
        self.assertTrue(first_ready.wait(timeout=5))
        b.start()
        self.assertTrue(second_entered.wait(timeout=5))
        self.assertFalse(second_done.wait(timeout=.03))
        release_first.set()
        for thread in (a, b):
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        frame.abort()
        manager.close()

    def test_owner_gc_does_not_abort_live_loss_graph_and_closes_after_last_lease(self):
        class Owner:
            pass

        plan = fixture()
        manager = CompiledGraphRuntimeManager(plan.policy, backend=TestingTransportBackend(), start_worker=False)
        owner = Owner()
        reference = weakref.ref(owner)
        with patch("deepopt.compiler.runtime.manager.CompiledGraphRuntimeManager", return_value=manager):
            with compiled_runtime_scope(plan.policy, owner):
                with compiled_forward_scope((plan,), manager):
                    frame, x, y = graph(manager, plan)
        del owner
        gc.collect()
        self.assertIsNone(reference())
        self.assertTrue(manager._owner_gone)
        self.assertFalse(frame.aborted)
        y.backward()
        manager.drain()
        self.assertEqual(x.grad.item(), 2)
        self.assertTrue(manager.snapshot()["lifecycle"]["closed"])
        self.assertEqual(manager.pinned.cached, 0)

    def test_copy_failures_release_host_and_task_leases(self):
        for direction in ("d2h", "h2d"):
            with self.subTest(direction=direction):
                class FailingCopy(TestingTransportBackend):
                    def copy(self, destinations, sources, **kwargs):
                        if kwargs["direction"] == direction:
                            raise RuntimeError("copy-failure")
                        return super().copy(destinations, sources, **kwargs)

                plan = fixture(pageable=True)
                manager = CompiledGraphRuntimeManager(plan.policy, backend=FailingCopy(), start_worker=False)
                frame = manager.begin_forward(plan)
                with self.assertRaisesRegex(RuntimeError, "copy-failure"):
                    saved = publish(frame, [components(plan.bundles[0])])
                    with frame.backward_scope():
                        task = frame.begin_backward(saved)
                        task.advance(3)
                        task.take("b0", "c0")
                frame.abort()
                manager.drain()
                self.assertEqual(len(manager._active_tasks), 0)
                self.assertEqual(manager.pinned.active, 0)
                self.assertEqual(manager.pageable.active, 0)
                manager.close()

    def test_atomic_history_admission_with_retained_graphs_and_explicit_fallback(self):
        plan = fixture(host_pinned_bytes=528, max_live_forwards=3)
        manager = self.manager(plan)
        _, _, first = graph(manager, plan)
        with self.assertRaisesRegex(OffloadAdmissionError, "Whole-forward"):
            manager.begin_forward(plan)
        self.assertEqual(manager.pinned.active, 528)
        del first
        gc.collect()
        manager.drain()
        frame = manager.begin_forward(plan)
        frame.abort()
        fallback_plan = replace(plan, policy=replace(plan.policy, admission="resident_fallback"))
        fallback = self.manager(fallback_plan)
        _, _, y = graph(fallback, fallback_plan)
        second, _, z = graph(fallback, fallback_plan)
        self.assertTrue(second.resident_fallback)
        z.backward()
        y.backward()

    def test_predictive_destination_allocation_requires_actual_progress_event(self):
        plan = fixture(bundles=2, predictive=True)
        backend = TestingTransportBackend()
        manager = CompiledGraphRuntimeManager(plan.policy, backend=backend, start_worker=False)
        frame = manager.begin_forward(plan)
        exports = publish(frame, [components(b) for b in plan.bundles])
        manager.poll()
        backend.auto_complete = False
        with frame.backward_scope():
            task = frame.begin_backward(exports)
            task.advance(2)
            manager.poll()
            self.assertEqual(manager.staging_bytes, 0)
            gate = backend.events[-1]
            gate.complete()
            manager.poll()
            self.assertEqual(manager.staging_bytes, plan.bundles[0].bytes)
            for event in backend.events:
                event.complete()
            task.finish()
            for event in backend.events:
                event.complete()
        frame.abort()
        manager.close()

    def test_enabled_actual_cpu_execution_rejected_without_testing_backend(self):
        plan = fixture()
        manager = CompiledGraphRuntimeManager(plan.policy)
        with self.assertRaisesRegex(OffloadProtocolError, "CUDA"):
            manager.begin_forward(plan)
        self.assertIsNone(manager.pinned)

    def test_metadata_only_snapshot_and_invalid_source_fail_closed(self):
        plan = fixture()
        manager = self.manager(plan)
        frame = manager.begin_forward(plan)
        value = components(plan.bundles[0])
        with self.assertRaisesRegex(OffloadProtocolError, "independent"):
            frame.pack_ready("b0", (value[0], value[1], value[1]))
        frame.abort()
        report = packed_offload_runtime_snapshot()
        json.dumps(report, allow_nan=False)
        self.assertTrue(report["metadata_only"])


if __name__ == "__main__":
    unittest.main()
