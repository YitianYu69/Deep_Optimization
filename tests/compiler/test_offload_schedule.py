"""Measured-cost and bounded deadline hints; no CUDA performance claims."""

from dataclasses import replace
import unittest

from deepopt.compiler.runtime.offload_cost import OffloadCostProfile, TransferCostModel, plan_prefetches
from tests.compiler.test_packed_offload_plan import _fixture, _plan


class OffloadScheduleTests(unittest.TestCase):
    def test_no_invented_bandwidth_and_contention_domains_are_separate(self):
        model = TransferCostModel(safety_margin_ms=.25)
        self.assertIsNone(model.estimate_ms("H2D", 4096))
        model.observe("H2D", 4096, 2, enqueue_ms=.1, queue_ms=.3, load="idle")
        self.assertIsNone(model.estimate_ms("H2D", 4096, load="compute"))
        model.observe("H2D", 8192, 8, enqueue_ms=.2, queue_ms=.4)
        # One size cannot separate fixed latency from bandwidth; do not scale
        # its duration down and claim a calibrated smaller-copy rate.
        self.assertEqual(model.estimate_ms("H2D", 4096), 8 + .6 + .25)
        self.assertIsNone(model.estimate_ms("D2H", 4096))
        profile = model.freeze(profile_identity={"gpu": "test", "schedule": "a"})
        profile.validate_identity({"schedule": "a", "gpu": "test"})
        with self.assertRaisesRegex(ValueError, "Stale"):
            profile.validate_identity({"gpu": "other"})
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            model.observe("H2D", 1, 1)
        for value in (0, True, float("nan"), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TransferCostModel().observe("H2D", 1, value)

    def test_demand_cold_and_calibrated_hints_never_grant_allocation_authority(self):
        graph, trace = _fixture()
        plan, _, backward = _plan(graph, trace)
        cold = plan_prefetches(plan)
        self.assertTrue(all(not request.calibrated and request.requires_device_progress for request in cold))
        demand = replace(plan, policy=replace(plan.policy, policy="demand"))
        self.assertTrue(all(request.submit_boundary == request.consumer_boundary for request in plan_prefetches(demand)))
        costs = TransferCostModel(safety_margin_ms=.1)
        for _ in range(5):
            for bundle in plan.bundles:
                costs.observe("H2D", bundle.bytes, .5)
        profile = costs.freeze(profile_identity=plan.profile_identity)
        durations = [2.] * len(backward.items)
        requests = plan_prefetches(plan, durations, profile)
        self.assertTrue(all(request.calibrated for request in requests))
        self.assertTrue(all(request.submit_boundary <= request.consumer_boundary for request in requests))
        enriched = plan.with_prefetches(requests)
        self.assertNotEqual(enriched.signature, plan.signature)
        self.assertTrue(all(request.submit_boundary in enriched.boundary_ids for request in requests))
        with self.assertRaisesRegex(ValueError, "contradict"):
            plan.with_prefetches((replace(requests[0], bundle_id="not-selected"),))
        with self.assertRaisesRegex(ValueError, "Stale"):
            plan_prefetches(plan, durations, OffloadCostProfile(identity=(("gpu", "other"),)))

    def test_missing_compute_cost_does_not_claim_calibrated_prediction(self):
        graph, trace = _fixture()
        plan, _, _ = _plan(graph, trace)
        model = TransferCostModel(max_samples=3)
        for _ in range(10):
            model.observe("H2D", 1024, 1)
        profile = model.freeze(profile_identity=plan.profile_identity)
        self.assertLessEqual(len(profile.samples), 3)
        self.assertTrue(all(not request.calibrated for request in plan_prefetches(plan, {}, profile)))

    def test_coarse_marker_profile_can_start_after_input_binding_indices(self):
        graph, trace = _fixture()
        plan, _, _ = _plan(graph, trace)
        origin = min(plan.boundary_ids)
        self.assertGreater(origin, 0)
        costs = TransferCostModel()
        for _ in range(5):
            for bundle in plan.bundles:
                costs.observe("H2D", bundle.bytes, .1)
        profile = costs.freeze(profile_identity=plan.profile_identity)
        last = max(consumer.schedule_index for consumer in plan.consumers)
        durations = {boundary: 2. if boundary in plan.boundary_ids else 0.
                     for boundary in range(origin, last + 1)}
        requests = plan_prefetches(plan, durations, profile)
        self.assertTrue(all(request.calibrated for request in requests))

    def test_tiny_latency_does_not_become_gigabyte_bandwidth(self):
        model = TransferCostModel(max_samples=64)
        big = 1271660544
        for repeat in range(8):
            model.observe("H2D", 264, .0945919976, enqueue_ms=.158)
            model.observe("H2D", 741818368, 13.55 + repeat * .01)
            model.observe("H2D", big, 23.15 + repeat * .02)
        profile = model.freeze(profile_identity={"gpu": "recorded_H200_fixture"})
        prediction = profile.prediction("H2D", big)
        self.assertTrue(prediction["qualified"])
        self.assertGreater(prediction["held_out"], 0)
        self.assertLess(prediction["estimated_ms"], 30)
        self.assertGreater(prediction["estimated_ms"], 20)
        self.assertIsNone(profile.estimate_ms("H2D", big, load="idle"))
        self.assertFalse(profile.prediction("H2D", big * 2)["qualified"])

    def test_unsupported_predictions_use_demand_and_validation_is_separate(self):
        graph, trace = _fixture()
        plan, _, backward = _plan(graph, trace)
        model = TransferCostModel()
        model.observe("H2D", plan.bundles[0].bytes, 1)
        profile = model.freeze(profile_identity=plan.profile_identity)
        self.assertIsNotNone(profile.estimate_ms("H2D", plan.bundles[0].bytes))
        self.assertFalse(profile.prediction("H2D", plan.bundles[0].bytes)["qualified"])
        requests = plan_prefetches(plan, [2.] * len(backward.items), profile)
        self.assertTrue(all(not request.calibrated and request.submit_boundary == request.consumer_boundary
                            for request in requests))

    def test_bounded_collector_keeps_size_domains_instead_of_worst_rate(self):
        model = TransferCostModel(max_samples=12)
        for _ in range(100):
            model.observe("H2D", 264, .1)
            model.observe("H2D", 1 << 30, 20.)
        profile = model.freeze()
        self.assertEqual(len(profile.samples), 12)
        self.assertEqual(sum(sample.nbytes == 264 for sample in profile.samples), 6)
        self.assertTrue(profile.prediction("H2D", 1 << 30)["qualified"])


if __name__ == "__main__":
    unittest.main()
