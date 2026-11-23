"""V2 physical-arena contracts; CPU state tests are not CUDA evidence."""

import gc
import os
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import torch

from deepopt.compiler.runtime.manager import CompiledGraphRuntimeManager
from deepopt.compiler.runtime.packed_offload import TestingTransportBackend, _CudaBackend, _HostLease
from deepopt.compiler.runtime.pinned_arena import (
    PinnedArenaAllocation, PinnedArenaPool, plan_pinned_placement,
)
from tests.compiler.test_packed_offload_lifecycle import fixture, graph


def leaf(nbytes, name="payload", dtype="torch.int32"):
    return SimpleNamespace(name=name, shape=(nbytes // 4,), dtype=dtype, stride=(1,), nbytes=nbytes, role=name)


class PinnedArenaPlacementTests(unittest.TestCase):
    def test_original_204_leaf_inventory_fits_one_16GiB_physical_arena(self):
        # Immutable run061727 encoder inventory, represented as metadata only.
        histogram = {741801984: 7, 1271660544: 7, 25344: 6, 46464: 6, 4224: 5,
            635830272: 4, 1584: 4, 2904: 4, 14784: 3, 35904: 3, 57024: 3, 67584: 3,
            264: 2, 1056: 2, 2376: 2, 3696: 2, 4967424: 1, 370900992: 1, 145711104: 1,
            13246464: 1, 103488: 1}
        components = tuple(value for size, repetitions in histogram.items() for _ in range(repetitions)
                           for value in fixture(groups=size//264).bundles[0].components)
        self.assertEqual(len(components), 204)
        plan = plan_pinned_placement(components, budget_bytes=2**34)
        self.assertEqual(plan.logical_bytes, 17173498848)
        self.assertEqual(plan.physical_bytes, 2**34)
        self.assertEqual(sum(1 << (x.nbytes-1).bit_length() for x in components), 28540299216)

    def test_whole_large_components_have_deterministic_metadata_only_placement(self):
        values = (leaf(1_233_125_376), leaf(19_267_584, "step", "torch.float32"),
                  leaf(19_267_584, "minimum", "torch.float32"))
        plan = plan_pinned_placement(values, budget_bytes=2**31)
        self.assertEqual(plan, plan_pinned_placement(values, budget_bytes=2**31))
        self.assertEqual(plan.physical_bytes, 2**31)
        self.assertEqual(plan.logical_bytes, 1_271_660_544)
        self.assertEqual([x.component_index for x in plan.spans], [0, 1, 2])
        for span in plan.spans:
            self.assertEqual(span.byte_offset % 64, 0)
            self.assertLessEqual(span.byte_offset + span.nbytes, plan.arena_capacities[span.arena_index])
        self.assertEqual(len(plan.signature), 64)
        self.assertIsNone(plan_pinned_placement(values, budget_bytes=2**30))

    def test_non_power_of_two_budget_and_existing_cached_capacity_remain_charged(self):
        plan = plan_pinned_placement((leaf(7680), leaf(4000)), budget_bytes=12288)
        self.assertEqual(plan.arena_capacities, (8192, 4096))
        self.assertEqual(plan.physical_bytes, 12288)
        self.assertIsNone(plan_pinned_placement((leaf(4),), budget_bytes=4096,
                                             arena_capacities=(8192,)))
        self.assertIsNone(plan_pinned_placement((leaf(4096),), budget_bytes=8192,
            arena_capacities=(8192,), free_ranges=(((0, 2048), (4096, 6144)),)))
        for kwargs in ({"alignment": 3}, {"budget_bytes": True},
                       {"arena_capacities": (5000,)}, {"free_ranges": (((0, 2),),)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plan_pinned_placement((leaf(4),), **({"budget_bytes": 8192} | kwargs))


class PinnedArenaPoolTests(unittest.TestCase):
    def test_disjoint_writes_have_independent_versions_and_split_lease_compatibility(self):
        backend = TestingTransportBackend()
        pool = PinnedArenaPool(4096, backend)
        lease = pool.acquire(fixture(groups=2).bundles[0].components)
        first, second, third = lease.tensors
        self.assertEqual(len({x.untyped_storage()._cdata for x in lease.tensors}), 1)
        before = first._version
        second.fill_(3)
        third.fill_(-0.0)
        self.assertEqual(first._version, before)
        first.fill_(17)
        self.assertNotEqual(first._version, before)
        self.assertTrue(all(x._base is None and x.is_contiguous() for x in lease.tensors))
        parts = [_HostLease(pool, lease.tensors[i:i+1], lease.keys[i:i+1],
                           lease.tensors[i].numel()*4) for i in range(3)]
        lease.released, lease.tensors = True, ()
        for part in parts:
            part.release()
        self.assertGreater(pool.active, 0)  # Local bindings still own their spans.
        del first, second, third
        gc.collect()
        self.assertEqual(pool.active, 0)
        self.assertEqual(pool.snapshot()["physical_capacity_bytes"], 4096)
        pool.trim()
        self.assertEqual(pool.snapshot()["physical_cached_bytes"], 4096)
        self.assertFalse(pool.snapshot()["physical_return_observed"])

    def test_binding_owners_and_generations_prevent_early_reuse(self):
        pool = PinnedArenaPool(4096, TestingTransportBackend())
        lease = pool.acquire((leaf(4000),))
        old = lease.tensors[0]
        key = lease.keys[0]
        lease.release()
        self.assertFalse(pool.can_acquire((leaf(4000),)))
        self.assertEqual(pool.snapshot()["physical_pending_retirement_bytes"], 4096)
        del old
        gc.collect()
        newer = pool.acquire((leaf(4000),))
        self.assertEqual(newer.keys[0].byte_offset, key.byte_offset)
        self.assertNotEqual(newer.keys[0].generation, key.generation)
        self.assertEqual(pool.allocations, 1)
        newer.release()

    def test_abandoned_reservation_and_graph_release_spans_but_keep_capacity_charged(self):
        pool = PinnedArenaPool(4096, TestingTransportBackend())
        lease = pool.acquire((leaf(4000),))
        del lease
        gc.collect()
        self.assertEqual(pool.active, 0)
        self.assertEqual(pool.snapshot()["physical_cached_bytes"], 4096)
        plan = fixture(version=2, host_pinned_bytes=16384)
        manager = CompiledGraphRuntimeManager(plan.policy, backend=TestingTransportBackend(), start_worker=False)
        try:
            _, _, output = graph(manager, plan)
            del output
            gc.collect()
            manager.drain()
            self.assertEqual(manager.pinned.active, 0)
            self.assertEqual(manager.pinned.snapshot()["physical_capacity_bytes"], 4096)
        finally:
            manager.close()

    def test_resize_and_unversioned_data_rebind_do_not_escape_binding_validation(self):
        for change in (lambda value: value.resize_(0),
                       lambda value: setattr(value, "data", torch.empty_like(value))):
            with self.subTest(change=change):
                pool = PinnedArenaPool(4096, TestingTransportBackend())
                lease = pool.acquire((leaf(512),))
                pool.validate_lease(lease)
                change(lease.tensors[0])
                with self.assertRaisesRegex(RuntimeError, "storage or metadata"):
                    pool.validate_lease(lease)
                lease.release()

    def test_retired_owner_capacity_is_reused_or_rejects_smaller_budget(self):
        backend = TestingTransportBackend()
        old = PinnedArenaPool(8192, backend)
        lease = old.acquire((leaf(6000),))
        address = lease.tensors[0].untyped_storage().data_ptr()
        lease.release()
        old.trim()
        replacement = PinnedArenaPool(8192, backend)
        newer = replacement.acquire((leaf(6000),))
        self.assertEqual(newer.tensors[0].untyped_storage().data_ptr(), address)
        self.assertEqual(replacement.allocations, 1)
        newer.release()
        shrinking = PinnedArenaPool(4096, backend)
        self.assertIsNone(shrinking.acquire((leaf(4),)))
        self.assertFalse(shrinking.snapshot()["accounting_verified"])
        self.assertEqual(shrinking.snapshot()["physical_capacity_bytes"], 8192)

    def test_old_new_capacity_overlap_is_charged_without_compacting_live_or_cached_arenas(self):
        pool = PinnedArenaPool(12288, TestingTransportBackend())
        first = pool.acquire((leaf(4000),))
        first.release()
        second = pool.acquire((leaf(6000),))
        self.assertEqual(pool.snapshot()["physical_capacity_bytes"], 12288)
        self.assertEqual(pool.snapshot()["physical_peak_bytes"], 12288)
        self.assertEqual(pool.allocations, 2)
        second.release()
        pool.trim()
        # Total free bytes suffice, but a whole 9KB leaf cannot straddle arenas.
        self.assertFalse(pool.can_acquire((leaf(9000),)))
        self.assertIsNone(pool.acquire((leaf(9000),)))
        self.assertEqual(pool.snapshot()["physical_capacity_bytes"], 12288)

    def test_partial_allocation_failure_keeps_the_first_arena_charged(self):
        class Failing(TestingTransportBackend):
            attempts = 0

            def host_arena(self, capacity, *, pinned):
                self.attempts += 1
                if self.attempts == 2:
                    raise MemoryError("second arena failed")
                return super().host_arena(capacity, pinned=pinned)

        backend = Failing()
        pool = PinnedArenaPool(12288, backend)
        with self.assertRaisesRegex(MemoryError, "second arena failed"):
            pool.acquire((leaf(7680), leaf(4000)))
        self.assertEqual(pool.active, 0)
        self.assertEqual(pool.snapshot()["physical_capacity_bytes"], 8192)
        self.assertIsNotNone(pool.acquire((leaf(4000),)))
        self.assertEqual(backend.attempts, 2)

    def test_partial_binding_failure_retains_traceback_owned_span_until_release(self):
        pool = PinnedArenaPool(4096, TestingTransportBackend())
        original = torch.Tensor.set_
        attempts = 0

        def fail_second(tensor, *args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise MemoryError("binding failure")
            return original(tensor, *args, **kwargs)

        traceback = None
        with patch.object(torch.Tensor, "set_", fail_second):
            try:
                pool.acquire(fixture(groups=2).bundles[0].components)
            except MemoryError as error:
                traceback = error.__traceback__
        self.assertIsNotNone(traceback)
        self.assertEqual(pool.snapshot()["physical_capacity_bytes"], 4096)
        self.assertGreater(pool.active, 0)
        self.assertFalse(pool.can_acquire((leaf(4000),)))
        del traceback
        gc.collect()
        self.assertEqual(pool.active, 0)
        self.assertTrue(pool.can_acquire((leaf(4000),)))

    def test_unverified_allocator_capacity_is_quarantined_not_forgotten(self):
        class Unknown(TestingTransportBackend):
            def host_arena(self, capacity, *, pinned):
                return PinnedArenaAllocation(super().host_arena(capacity, pinned=pinned),
                                             capacity, accounting_verified=False)

        pool = PinnedArenaPool(4096, Unknown())
        with self.assertRaisesRegex(RuntimeError, "cannot be reconciled"):
            pool.acquire((leaf(4),))
        self.assertEqual(pool.snapshot()["physical_capacity_bytes"], 4096)
        self.assertFalse(pool.snapshot()["accounting_verified"])
        self.assertIsNone(pool.acquire((leaf(4),)))

    def test_failed_attempt_retired_capacity_is_charged_and_original_error_survives(self):
        primary = MemoryError("primary host allocation failure")
        primary._deepopt_pinned_unowned_bytes = 4096
        primary._deepopt_pinned_failure_verified = True

        class Failing(TestingTransportBackend):
            def host_arena(self, capacity, *, pinned):
                raise primary

        pool = PinnedArenaPool(8192, Failing())
        with self.assertRaises(MemoryError) as raised:
            pool.acquire((leaf(4),))
        self.assertIs(raised.exception, primary)
        self.assertEqual(pool.snapshot()["physical_unowned_retired_bytes"], 4096)
        self.assertIsNone(pool.acquire((leaf(4),)))

    def test_inference_mode_is_rejected_before_allocating_unversioned_bindings(self):
        pool = PinnedArenaPool(4096, TestingTransportBackend())
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "ordinary versioned"):
            pool.acquire((leaf(4),))
        self.assertEqual(pool.allocations, 0)

    def test_concurrent_reservations_cannot_replace_a_live_span(self):
        backend = TestingTransportBackend()
        pools = [PinnedArenaPool(4096, backend), PinnedArenaPool(4096, backend)]
        barrier = threading.Barrier(3)
        results = []

        def acquire(pool):
            barrier.wait()
            results.append(pool.acquire((leaf(4000),)))

        threads = [threading.Thread(target=acquire, args=(pool,)) for pool in pools]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(sum(x is not None for x in results), 1)
        next(x for x in results if x is not None).release()
        self.assertEqual(pools[0].snapshot()["physical_capacity_bytes"], 4096)

    def test_real_autograd_retained_roundtrip_inspection_and_pageable_stage(self):
        for pageable in (False, True):
            with self.subTest(pageable=pageable):
                plan = fixture(bundles=2, groups=7, pageable=pageable, version=2,
                               host_pinned_bytes=16384)
                manager = CompiledGraphRuntimeManager(plan.policy, backend=TestingTransportBackend(),
                                                     start_worker=False)
                frame, x, y = graph(manager, plan)
                manager.poll()
                self.assertIsInstance(manager.pinned, PinnedArenaPool)
                for record in frame.records.values():
                    record.validate()
                inspection = y.grad_fn.saved_tensors[0]
                inspection.fill_(0)  # External unpack owns a separate CPU clone.
                y.backward(retain_graph=True)
                y.backward()
                manager.drain()
                self.assertEqual(x.grad.item(), 4)
                self.assertEqual(manager.pinned.active, 0)
                self.assertLessEqual(manager.pinned.snapshot()["physical_peak_bytes"], 16384)
                manager.close()

    def test_many_real_autograd_updates_have_a_stable_arena_allocation_plateau(self):
        plan = fixture(bundles=2, groups=7, version=2, host_pinned_bytes=16384)
        manager = CompiledGraphRuntimeManager(plan.policy, backend=TestingTransportBackend(), start_worker=False)
        try:
            for _ in range(20):
                _, x, y = graph(manager, plan)
                y.backward()
                manager.drain()
                self.assertEqual(x.grad.item(), 2)
                self.assertEqual(manager.pinned.active, 0)
            self.assertEqual(manager.pinned.allocations, 1)
            self.assertEqual(manager.pinned.snapshot()["physical_peak_bytes"], 4096)
        finally:
            manager.close()


@unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA pinned arena transport")
class PinnedArenaCudaTests(unittest.TestCase):
    def test_actual_pinning_independent_versions_and_rounding_counters(self):
        # Run physical-budget families in fresh processes: retired arenas stay
        # charged rather than being silently reclassified as outside cache.
        backend = _CudaBackend("cuda:0")
        before = torch.cuda.memory.host_memory_stats()
        pool = PinnedArenaPool(2**22, backend)
        lease = pool.acquire(fixture(groups=4096).bundles[0].components)
        self.assertTrue(all(value.is_pinned() for value in lease.tensors))
        version = lease.tensors[0]._version
        lease.tensors[1].fill_(2)
        self.assertEqual(lease.tensors[0]._version, version)
        after = torch.cuda.memory.host_memory_stats()
        charged = pool.snapshot()["physical_capacity_bytes"]
        self.assertTrue(pool.snapshot()["accounting_verified"])
        self.assertLessEqual(after["allocated_bytes.current"]-before["allocated_bytes.current"], charged)
        lease.release()

    def test_actual_transport_retained_backward_and_independent_restored_GPU_storages(self):
        plan = fixture(bundles=2, groups=65, device="cuda:0", version=2,
                       host_pinned_bytes=2**22)
        manager = CompiledGraphRuntimeManager(plan.policy)
        try:
            _, x, y = graph(manager, plan)
            y.backward(retain_graph=True)
            y.backward()
            manager.drain()
            self.assertEqual(x.grad.item(), 4)
            self.assertTrue(manager.pinned.snapshot()["accounting_verified"])
            self.assertLessEqual(manager.pinned.snapshot()["physical_peak_bytes"], 2**22)
        finally:
            manager.close()


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("DEEPOPT_TEST_LARGE_CUDA_PINNED_ARENA") == "1",
                     "requires opt-in original CUDA pinned-arena geometry in a fresh process")
class PinnedArenaOriginalShapeCudaTests(unittest.TestCase):
    def test_original_g1024_leaf_larger_than_one_GiB_is_whole_contiguous_and_pinned(self):
        backend = _CudaBackend("cuda:0")
        pool = PinnedArenaPool(2**32, backend)
        values = fixture(groups=4_816_896, device="cuda:0").bundles[0].components
        lease = pool.acquire(values)
        self.assertTrue(all(x.is_pinned() and x.is_contiguous() for x in lease.tensors))
        self.assertEqual([x.numel()*4 for x in lease.tensors], [x.nbytes for x in values])
        self.assertGreaterEqual(pool.snapshot()["physical_capacity_bytes"], 2**31)
        self.assertLessEqual(pool.snapshot()["physical_capacity_bytes"], 2**32)
        sources = tuple(torch.full(x.shape, 17 if i == 0 else 3.0,
            dtype=torch.int32 if i == 0 else torch.float32, device="cuda:0") for i, x in enumerate(values))
        ready = backend.copy(lease.tensors, sources, direction="d2h", dependency=backend.event(),
                             chunk_bytes=16*1024*1024)
        ready.synchronize()
        destinations = tuple(backend.allocate(x) for x in values)
        self.assertEqual(len({x.untyped_storage()._cdata for x in destinations}), 3)
        self.assertTrue(all(x.storage_offset() == 0 for x in destinations))
        back = backend.copy(destinations, lease.tensors, direction="h2d", dependency=ready,
                            chunk_bytes=16*1024*1024)
        back.synchronize()
        self.assertTrue(all(torch.equal(x.view(torch.int32), y.view(torch.int32))
                            for x, y in zip(sources, destinations)))
        lease.release()


if __name__ == "__main__":
    unittest.main()
