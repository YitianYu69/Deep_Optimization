"""Bounded host-pool state tests using the explicit CPU testing backend."""

from dataclasses import replace
import threading
import unittest

from deepopt.compiler.runtime.packed_offload import PinnedOffloadPool, TestingTransportBackend
from tests.compiler.test_packed_offload_lifecycle import fixture


class PinnedOffloadPoolTests(unittest.TestCase):
    def test_history_acquisition_is_atomic_and_active_plus_cached_stays_bounded(self):
        bundle = fixture().bundles[0]
        pool = PinnedOffloadPool(bundle.bytes, TestingTransportBackend())
        first = pool.acquire(bundle.components)
        self.assertIsNotNone(first)
        self.assertIsNone(pool.acquire(bundle.components))
        addresses = tuple(t.untyped_storage()._cdata for t in first.tensors)
        self.assertEqual(len(set(addresses)), 3)
        first.release()
        self.assertEqual(pool.cached, bundle.bytes)
        second = pool.acquire(bundle.components)
        self.assertEqual(addresses, tuple(t.untyped_storage()._cdata for t in second.tensors))
        self.assertEqual(pool.allocations, 3)
        second.release()
        second.release()
        self.assertEqual(pool.active, 0)
        self.assertLessEqual(pool.active + pool.cached, pool.budget)

    def test_incompatible_cached_capacity_evicted_before_replacement_allocation(self):
        bundle = fixture(groups=4).bundles[0]
        backend = TestingTransportBackend()
        pool = PinnedOffloadPool(bundle.bytes, backend)
        first = pool.acquire(bundle.components)
        first.release()
        smaller = fixture(groups=3).bundles[0]
        second = pool.acquire(smaller.components)
        self.assertEqual(pool.active, smaller.bytes)
        self.assertLessEqual(pool.active + pool.cached, pool.budget)
        second.release()
        pool.trim()
        self.assertEqual(pool.cached, 0)

    def test_allocation_error_rolls_back_reservations_and_keeps_cached_accounting(self):
        class Failing(TestingTransportBackend):
            def host(self, component, *, pinned):
                if component.role == "codec_step":
                    raise MemoryError("injected pinned allocation error")
                return super().host(component, pinned=pinned)

        bundle = fixture().bundles[0]
        pool = PinnedOffloadPool(bundle.bytes, Failing())
        with self.assertRaisesRegex(MemoryError, "injected pinned"):
            pool.acquire(bundle.components)
        self.assertEqual(pool.active, 0)
        self.assertEqual(pool.cached, 0)

    def test_concurrent_reservation_has_single_winner_before_any_producer(self):
        bundle = fixture().bundles[0]
        pool = PinnedOffloadPool(bundle.bytes, TestingTransportBackend())
        barrier = threading.Barrier(3)
        results = []

        def reserve():
            barrier.wait()
            results.append(pool.acquire(bundle.components))

        workers = [threading.Thread(target=reserve) for _ in range(2)]
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
        self.assertEqual(sum(value is not None for value in results), 1)
        self.assertEqual(pool.active, bundle.bytes)
        next(value for value in results if value is not None).release()

    def test_large_arithmetic_rejects_without_allocating_truncated_capacity(self):
        component = fixture().bundles[0].components[0]
        huge = replace(component, shape=(2**32, 64), nbytes=2**40)
        backend = TestingTransportBackend()
        pool = PinnedOffloadPool(2**31, backend)
        self.assertIsNone(pool.acquire((huge,)))
        self.assertEqual(pool.allocations, 0)


if __name__ == "__main__":
    unittest.main()
