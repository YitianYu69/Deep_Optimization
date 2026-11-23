"""Pageable canonical history and bounded pinned stage lifecycle tests."""

import gc
import unittest

from deepopt.compiler.runtime.manager import CompiledGraphRuntimeManager
from deepopt.compiler.runtime.packed_offload import OffloadAdmissionError, TestingTransportBackend
from tests.compiler.test_packed_offload_lifecycle import fixture, graph


class OffloadHostTierTests(unittest.TestCase):
    def test_pageable_history_round_trip_and_repeated_backward_use_pinned_stage(self):
        plan = fixture(bundles=3, pageable=True)
        manager = CompiledGraphRuntimeManager(plan.policy, backend=TestingTransportBackend(), start_worker=False)
        frame, x, y = graph(manager, plan)
        manager.poll()
        self.assertEqual(manager.pageable.active, plan.selected_bytes)
        self.assertEqual(manager.pinned.active, 0)
        self.assertLessEqual(manager.pinned.peak, plan.policy.host_pinned_bytes)
        y.backward(retain_graph=True)
        manager.drain()
        self.assertEqual(manager.pageable.active, plan.selected_bytes)
        self.assertEqual(x.grad.item(), 2)
        y.backward()
        manager.drain()
        self.assertEqual(x.grad.item(), 4)
        self.assertEqual(manager.pageable.active, 0)
        self.assertEqual(manager.pinned.active, 0)
        self.assertEqual(manager.counters["cpu_copy_bytes"], plan.selected_bytes * 3)
        self.assertTrue(all(record.closed for record in frame.records.values()))
        manager.close()

    def test_pageable_budget_is_full_history_not_a_small_pinned_ring(self):
        plan = fixture(bundles=2, pageable=True, host_pageable_bytes=1056, max_live_forwards=3)
        manager = CompiledGraphRuntimeManager(plan.policy, backend=TestingTransportBackend(), start_worker=False)
        _, _, y = graph(manager, plan)
        with self.assertRaisesRegex(OffloadAdmissionError, "Whole-forward"):
            manager.begin_forward(plan)
        self.assertEqual(manager.pageable.active, plan.selected_bytes)
        del y
        gc.collect()
        manager.drain()
        self.assertEqual(manager.pageable.active, 0)
        manager.close()

    def test_pageable_snapshots_not_recycled_while_retained_graph_exists(self):
        plan = fixture(pageable=True)
        manager = CompiledGraphRuntimeManager(plan.policy, backend=TestingTransportBackend(), start_worker=False)
        _, _, y = graph(manager, plan)
        y.backward(retain_graph=True)
        manager.drain()
        _, _, z = graph(manager, plan)
        self.assertEqual(manager.pageable.active, 2 * plan.selected_bytes)
        z.backward()
        y.backward()
        manager.drain()
        self.assertEqual(manager.pageable.active, 0)
        self.assertLessEqual(manager.pinned.peak, plan.policy.host_pinned_bytes)
        manager.close()


if __name__ == "__main__":
    unittest.main()
