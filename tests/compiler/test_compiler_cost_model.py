"""Static admission of pointwise partitions; predictions, never measurements."""

import unittest

import torch

from deepopt.compiler.backend import BackendConfig
from deepopt.compiler.heuristic import (
    RegionCost,
    apply,
    decline_partitions,
    estimate_region,
    numel_of,
)
from deepopt.compiler.ir import (
    IRGraph,
    InputNode,
    OutputNode,
    PointwiseNode,
    TensorMeta,
    TensorValue,
)
from deepopt.compiler.reporting import CompilationReport, GraphReport
from deepopt.compiler.scheduler import FusionGroup, Schedule, SinglePointwiseOp

BANDWIDTH = 3.3e12


def fused_schedule(numel=1 << 22, ops=("neg", "silu", "relu"), device="cuda"):
    """Build a schedule whose metadata claims CUDA; nothing here executes it."""
    meta = TensorMeta((numel,), torch.bfloat16, device, (1,))
    values = [TensorValue(f"v{index}", meta) for index in range(len(ops) + 1)]
    graph = IRGraph()
    graph.add(InputNode(0, values[0]))
    nodes = []
    for index, (op, target) in enumerate(zip(ops, (torch.neg, torch.nn.functional.silu, torch.relu))):
        node = PointwiseNode(op, (values[index],), {}, values[index + 1], target)
        graph.add(node)
        nodes.append(node)
    graph.add(OutputNode(values[-1]))
    return graph, nodes


class EstimateRuleTests(unittest.TestCase):
    def test_single_operation_region_is_declined_without_any_bandwidth(self):
        cost = estimate_region(numel=1 << 20, inputs=1, operations=1, dtype=torch.bfloat16)
        self.assertEqual(cost.decision, "declined")
        self.assertEqual(cost.reason, "single_operation_region_has_no_fusion_gain")

    def test_fused_region_is_admitted_when_bandwidth_is_known(self):
        cost = estimate_region(numel=1 << 20, inputs=2, operations=3,
                               dtype=torch.bfloat16, bandwidth_bytes_per_s=BANDWIDTH)
        self.assertEqual(cost.decision, "admitted")
        self.assertGreater(cost.predicted_speedup, 1.0)
        self.assertGreater(cost.eager_bytes, cost.fused_bytes)
        self.assertEqual(cost.as_dict()["basis"], "static_estimate_not_measurement")

    def test_unknown_geometry_is_unestimated_not_declined(self):
        cost = estimate_region(numel=None, inputs=2, operations=3, dtype=torch.bfloat16)
        self.assertEqual(cost.decision, "unestimated")
        self.assertEqual(cost.reason, "unknown_geometry")

    def test_advisory_rows_are_never_marked_applied(self):
        graph, nodes = fused_schedule()
        schedule = Schedule([graph.nodes[0], *(SinglePointwiseOp(node) for node in nodes),
                             graph.nodes[-1]])
        _, estimates = apply(schedule, mode="advisory", bandwidth_bytes_per_s=BANDWIDTH)
        self.assertFalse(any(estimate["applied"] for estimate in estimates))
        self.assertTrue(all(estimate["schedule_position"] == "pre_admission"
                            for estimate in estimates))
        _, enforced = apply(schedule, mode="enforce", bandwidth_bytes_per_s=BANDWIDTH)
        self.assertTrue(all(estimate["applied"] for estimate in enforced))

    def test_unknown_bandwidth_never_declines_work(self):
        cost = estimate_region(numel=1 << 20, inputs=2, operations=3, dtype=torch.bfloat16)
        self.assertEqual(cost.decision, "unestimated")
        self.assertEqual(cost.reason, "no_bandwidth_for_device")
        self.assertIsNone(cost.predicted_speedup)

    def test_small_region_and_empty_region_are_declined(self):
        small = estimate_region(numel=512, inputs=2, operations=3,
                                dtype=torch.bfloat16, bandwidth_bytes_per_s=BANDWIDTH)
        self.assertEqual(small.reason, "below_minimum_region_size")
        empty = estimate_region(numel=0, inputs=1, operations=2, dtype=torch.bfloat16)
        self.assertEqual(empty.reason, "empty_region")

    def test_unknown_dtype_is_unestimated(self):
        cost = estimate_region(numel=1 << 20, inputs=1, operations=2, dtype=torch.complex64)
        self.assertEqual(cost.reason, "unknown_dtype")

    def test_symbolic_geometry_is_unestimated(self):
        self.assertIsNone(numel_of(None))
        self.assertIsNone(numel_of((-1, 4)))
        self.assertEqual(numel_of((4, 8)), 32)


class ScheduleAdmissionTests(unittest.TestCase):
    def test_advisory_records_without_changing_the_schedule(self):
        graph, nodes = fused_schedule()
        schedule = Schedule([graph.nodes[0], FusionGroup(nodes), graph.nodes[-1]])
        before = list(schedule.items)
        admitted, estimates = apply(schedule, mode="advisory", bandwidth_bytes_per_s=BANDWIDTH)
        self.assertIs(admitted, schedule)
        self.assertEqual(list(schedule.items), before)
        self.assertEqual(len(estimates), 1)
        self.assertEqual(estimates[0]["decision"], "admitted")
        self.assertEqual(estimates[0]["operations"], ("neg", "silu", "relu"))

    def test_enforce_returns_declined_partitions_to_plain_nodes(self):
        graph, nodes = fused_schedule()
        schedule = Schedule([graph.nodes[0], *(SinglePointwiseOp(node) for node in nodes),
                             graph.nodes[-1]])
        enforced, estimates = apply(schedule, mode="enforce", bandwidth_bytes_per_s=BANDWIDTH)
        self.assertEqual([estimate["decision"] for estimate in estimates],
                         ["declined", "declined", "declined"])
        self.assertEqual([type(item).__name__ for item in enforced.items],
                         ["InputNode", "PointwiseNode", "PointwiseNode", "PointwiseNode", "OutputNode"])
        enforced.validate()
        self.assertEqual(len(enforced.nodes()), 5)

    def test_enforce_keeps_admitted_partitions(self):
        graph, nodes = fused_schedule()
        schedule = Schedule([graph.nodes[0], FusionGroup(nodes), graph.nodes[-1]])
        enforced, _ = apply(schedule, mode="enforce", bandwidth_bytes_per_s=BANDWIDTH)
        self.assertEqual([type(item).__name__ for item in enforced.items],
                         ["InputNode", "FusionGroup", "OutputNode"])

    def test_declining_preserves_effect_epochs_and_barriers(self):
        graph, nodes = fused_schedule()
        schedule = Schedule([graph.nodes[0], *(SinglePointwiseOp(node) for node in nodes),
                             graph.nodes[-1]])
        rebuilt = decline_partitions(schedule, frozenset(id(item) for item in schedule.items[1:-1]))
        self.assertEqual(rebuilt.effect_epochs, schedule.effect_epochs)
        self.assertEqual(rebuilt.effect_barriers, schedule.effect_barriers)
        rebuilt.validate()

    def test_off_mode_is_a_no_op(self):
        graph, nodes = fused_schedule()
        schedule = Schedule([graph.nodes[0], FusionGroup(nodes), graph.nodes[-1]])
        returned, estimates = apply(schedule, mode="off")
        self.assertIs(returned, schedule)
        self.assertEqual(estimates, ())

    def test_unknown_mode_is_rejected(self):
        graph, nodes = fused_schedule()
        schedule = Schedule([graph.nodes[0], FusionGroup(nodes), graph.nodes[-1]])
        with self.assertRaisesRegex(ValueError, "cost_model"):
            apply(schedule, mode="sometimes")


class ConfigValidationTests(unittest.TestCase):
    def test_mode_bandwidth_and_margin_are_validated(self):
        for kwargs in ({"cost_model": "bogus"},
                       {"cost_model_bandwidth_bytes_per_s": 0},
                       {"cost_model_bandwidth_bytes_per_s": "fast"},
                       {"cost_model_admit_margin": 1.0},
                       {"cost_model_admit_margin": -0.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BackendConfig(**kwargs)

    def test_strict_ownership_rejects_returning_work_to_eager(self):
        with self.assertRaisesRegex(ValueError, "require_full_aot"):
            BackendConfig(cost_model="enforce", require_full_aot=True)
        # Advisory is the documented way to inspect predictions under strictness.
        config = BackendConfig(cost_model="advisory", require_full_aot=True)
        self.assertEqual(config.cost_model, "advisory")

    def test_defaults_keep_execution_unchanged(self):
        self.assertEqual(BackendConfig().cost_model, "advisory")
        self.assertIsNone(BackendConfig().cost_model_bandwidth_bytes_per_s)


class DeclineExecutionTests(unittest.TestCase):
    """A declined partition must run on its original eager callable."""

    def test_declined_partition_executes_eagerly_and_matches_torch(self):
        from torch import nn

        from deepopt.compiler.codegen.wrapper import WrapperCodegen

        meta = TensorMeta((1 << 18,), torch.float32, "cpu", (1,))
        values = [TensorValue(f"v{index}", meta) for index in range(3)]
        graph = IRGraph()
        graph.add(InputNode(0, values[0]))
        first = PointwiseNode("neg", (values[0],), {}, values[1], torch.neg)
        second = PointwiseNode("relu", (values[1],), {}, values[2], torch.relu)
        graph.add(first)
        graph.add(second)
        graph.add(OutputNode(values[2]))
        schedule = Schedule([graph.nodes[0], SinglePointwiseOp(first),
                             SinglePointwiseOp(second), graph.nodes[-1]])

        enforced, estimates = apply(schedule, mode="enforce", bandwidth_bytes_per_s=BANDWIDTH)
        self.assertEqual([estimate["decision"] for estimate in estimates],
                         ["declined", "declined"])
        artifact = WrapperCodegen(nn.Module(), enforced, enable_autodiff=False).run()
        self.assertNotIn("_launch_kernel", artifact.source)
        self.assertIn("_native_ops", artifact.source)
        sample = torch.randn(1 << 18)
        torch.testing.assert_close(artifact.fn(sample), torch.relu(-sample))


class ReportingTests(unittest.TestCase):
    def test_graph_report_carries_estimates_and_aggregates_them(self):
        report = CompilationReport()
        report.graphs.append(GraphReport(
            operations=3,
            pointwise_cost_estimates=({"index": 1, "decision": "declined",
                                       "reason": "single_operation_region_has_no_fusion_gain"},),
            static_regions={"declined_pointwise_regions": 1},
        ))
        payload = report.as_dict()
        self.assertEqual(len(payload["pointwise_cost_estimates"]), 1)
        self.assertEqual(payload["pointwise_cost_estimates"][0]["graph_index"], 0)
        self.assertEqual(payload["static_regions"]["declined_pointwise_regions"], 1)

    def test_region_cost_dict_is_serializable(self):
        cost = RegionCost(1, 1, 1, 2, 4, 4, 1, 1)
        import json

        self.assertIn("predicted_speedup", json.loads(json.dumps(cost.as_dict())))


if __name__ == "__main__":
    unittest.main()
