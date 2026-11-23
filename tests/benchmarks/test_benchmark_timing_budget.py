"""Slow GPU steps must not be blindly repeated hundreds of times."""

from types import SimpleNamespace
import unittest

from deepopt._benchmarking import plan_timing, share_timing_counts, timing_plan_from_cost


class BenchmarkTimingBudgetTests(unittest.TestCase):
    def args(self, **changes):
        return SimpleNamespace(**({"warmup": 10, "iterations": 50, "repeats": 5, "time_budget": 2.0} | changes))

    def test_exhaustive_counts_need_no_pilot_or_extra_step(self):
        def forbidden():
            raise AssertionError("Disabled budget added an untimed operation")
        plan = plan_timing(forbidden, self.args(time_budget=0))
        self.assertEqual((plan.warmup, plan.iterations, plan.repeats), (10, 50, 5))
        self.assertFalse(plan.sampling(1)["budget_exhausted"])

    def test_cold_compilation_does_not_determine_warm_sample_counts(self):
        clock = [0.]
        calls = []
        def operation():
            clock[0] += 100. if not calls else .001
            calls.append(1)
        sync = []
        plan = plan_timing(operation, self.args(), synchronize=lambda: sync.append(1), clock=lambda: clock[0])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(sync), 3)
        self.assertEqual(plan.cold_first_call_seconds, 100.)
        self.assertEqual((plan.warmup, plan.iterations, plan.repeats), (10, 50, 5))

    def test_slow_step_reduces_work_and_reports_requested_and_actual_counts(self):
        plan = timing_plan_from_cost(self.args(), .1)
        self.assertLessEqual((plan.warmup + plan.iterations * plan.repeats) * .1, 2.)
        self.assertLess(plan.iterations * plan.repeats, 50 * 5)
        sampling = plan.sampling(2.)
        self.assertEqual(sampling["requested_iterations"], 50)
        self.assertEqual(sampling["actual_iterations_per_repeat"], [plan.iterations] * plan.repeats)
        self.assertTrue(sampling["budget_exhausted"])

    def test_single_uninterruptible_step_retains_one_sample_instead_of_250(self):
        plan = timing_plan_from_cost(self.args(), 30.)
        self.assertEqual((plan.warmup, plan.iterations, plan.repeats), (0, 1, 1))
        self.assertEqual(plan.sampling(30.)["actual_repeats"], 1)

    def test_invalid_budget_fails_before_pilot(self):
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "time-budget"):
                plan_timing(lambda: self.fail("executed invalid plan"), self.args(time_budget=value))

    def test_compiler_can_share_one_count_plan_for_equal_training_updates(self):
        args = SimpleNamespace(warmup=10, steps=50, time_budget=2.)
        plan = timing_plan_from_cost(args, .1, iterations=args.steps, repeats=1)
        self.assertEqual(plan.requested_iterations, 50)
        self.assertEqual(plan.repeats, 1)
        self.assertLess(plan.iterations, args.steps)

    def test_stateful_variants_share_update_counts_and_keep_their_own_calibration(self):
        first = timing_plan_from_cost(self.args(), .1, cold_seconds=10., calibration_calls=2)
        second = timing_plan_from_cost(self.args(), .001, cold_seconds=1., calibration_calls=2)
        shared = share_timing_counts(second, first)
        self.assertEqual((shared.warmup, shared.iterations, shared.repeats),
                         (first.warmup, first.iterations, first.repeats))
        self.assertEqual(shared.pilot_seconds, .001)
        self.assertEqual(shared.cold_first_call_seconds, 1.)
        self.assertEqual(shared.calibration_calls, first.calibration_calls)


if __name__ == "__main__":
    unittest.main()
