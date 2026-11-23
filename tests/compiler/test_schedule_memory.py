"""Final emitted order and logical reference cleanup contracts."""

import gc
import json
import unittest
import weakref

import torch
from torch import nn

from deepopt.compiler.codegen.wrapper import WrapperCodegen
from deepopt.compiler.ir import AuxiliaryTensorMeta, CallNode, IRGraph, InputNode, OutputNode, TensorMeta, TensorValue
from deepopt.compiler.runtime.memory import MemoryManager
from deepopt.compiler.scheduler import Scheduler


class ScheduleMemoryTests(unittest.TestCase):
    def test_scratch_contracts_use_final_order_and_invalidate_stale_codegen(self):
        graph, fused = self.constrained_graph([])
        fused.scheduling_constraints["scratch_requirements"] = (
            {"name": "decoded_panel", "bytes": 1024 * 7 * 2, "lifetime": "call",
             "owner": "invocation", "capture_safe_after_warmup": True},)
        schedule = Scheduler(graph, backward=True).run()
        codegen = WrapperCodegen(nn.Module(), schedule, enable_autodiff=False)
        artifact = codegen.run()
        row, = artifact.scratch_lifetimes
        self.assertEqual(row["birth"], row["last_use"])
        self.assertEqual(row["local_release_after"], 2)
        self.assertIsNone(row["physical_bytes_observed"])
        self.assertEqual(artifact.fn._deepopt_scratch_lifetimes, artifact.scratch_lifetimes)
        fused.scheduling_constraints["scratch_requirements"][0]["bytes"] += 4
        with self.assertRaisesRegex(ValueError, "Schedule changed"):
            codegen.runtime_memory_manager.scratch_plan()

    def test_scratch_cannot_escape_or_claim_cross_invocation_ownership(self):
        for invalid in ({"name": "x", "bytes": -1},
                        {"name": "x", "bytes": 4, "owner": "global"},
                        {"name": "x", "bytes": 4, "reuse": "global_pool"},
                        {"name": "x", "bytes": 4, "capture_safe_after_warmup": torch.ones(2)},
                        {"name": "x", "bytes": 4, "lifetime": "graph"}):
            graph, fused = self.constrained_graph([])
            fused.scheduling_constraints["scratch_requirements"] = [invalid]
            with self.assertRaisesRegex(ValueError, "Scratch|scratch"):
                codegen = WrapperCodegen(nn.Module(), Scheduler(graph).run(), enable_autodiff=False)
                codegen.run()

    def test_compiler_report_contains_final_order_and_lifted_model_ownership(self):
        from torch.fx.passes.shape_prop import ShapeProp
        from deepopt.compiler import BackendConfig, CompilationReport, compile_fx

        model = torch.fx.symbolic_trace(nn.Linear(3, 2).bfloat16())
        sample = torch.ones(2, 3, dtype=torch.bfloat16, requires_grad=True)
        ShapeProp(model).propagate(sample)
        report = CompilationReport()
        compiled = compile_fx(model, [sample], config=BackendConfig(require_full_aot=True), report=report)
        compiled(sample).sum().backward()
        evidence = report.as_dict()
        json.dumps(evidence)
        region = evidence["graphs"][0]["native_autodiff_graphs"][0]
        self.assertEqual(region["backward_orders"][0]["requested_order"], "dx_first")
        self.assertEqual(region["backward_orders"][0]["final_order"], ("dx", "dw"))
        self.assertEqual(tuple(op for row in region["final_backward_schedule"] for op in row["operations"]),
                         region["backward_operations"])
        model_inputs = [row for row in region["forward_logical_lifetimes"]
                        if row["provenance"] in ("parameter", "persistent_buffer")]
        self.assertTrue(model_inputs)
        self.assertTrue(all(row["owner"] == "model" for row in model_inputs))
        saved_parameters = [row for row in region["backward_logical_lifetimes"]
                            if row["provenance"] == "parameter"]
        self.assertTrue(saved_parameters)
        self.assertTrue(all(row["owner"] == "model" and row["source_name"] for row in saved_parameters))

    @staticmethod
    def constrained_graph(events, *, effect=False, dependency=False):
        meta = TensorMeta((2,), torch.float32, "cpu", (1,))
        x, saved, dw, dx = (TensorValue(name, meta) for name in ("x", "saved", "dw", "dx"))
        graph = IRGraph()
        graph.add(InputNode(0, x))

        def weight(value):
            events.append("dw")
            return value * 2

        graph.add(CallNode("bf16.gemm_dw", weight, (x,), {}, (dw,)))
        if effect:
            def observe():
                events.append("effect")
                return torch.ones(2)
            graph.add(CallNode("external_effect", observe, (), {}, (TensorValue("effect_token", meta),)))
        graph.add(InputNode(1, saved))

        def input_gradient(value, coefficient):
            events.append("dx")
            return value + coefficient

        source = dw if dependency else x
        fused = CallNode("bf16.gemm_dx_native_derivative", input_gradient, (source, saved), {}, (dx,))
        fused.scheduling_constraints = {"requested_order": "dx_first", "prefer_before": ("dw",),
                                        "dependencies": (source.name, saved.name)}
        graph.add(fused)
        graph.add(OutputNode((dx, dw)))
        return graph, fused

    def test_fused_dx_placement_moves_late_saved_binding_and_rebuilds_lifetimes(self):
        events = []
        graph, fused = self.constrained_graph(events)
        schedule = Scheduler(graph, backward=True).run()
        artifact = WrapperCodegen(nn.Module(), schedule, enable_autodiff=False).run()
        actual = artifact.fn(torch.ones(2), torch.full((2,), 3.))
        self.assertEqual(events, ["dx", "dw"])
        self.assertEqual(fused.scheduling_constraints["placement"], "earliest_legal")
        torch.testing.assert_close(actual, (torch.full((2,), 4.), torch.full((2,), 2.)))
        operations = [op for row in artifact.final_schedule for op in row["operations"]]
        self.assertLess(operations.index("bf16.gemm_dx_native_derivative"), operations.index("bf16.gemm_dw"))
        records = {row["name"]: row for row in artifact.logical_lifetimes}
        self.assertEqual(records["saved"]["consumers"], (2,))
        self.assertEqual(records["x"]["consumers"], (2, 3))

    def test_fused_dx_preserves_effect_and_data_dependencies(self):
        for effect, dependency in ((True, False), (False, True)):
            with self.subTest(effect=effect, dependency=dependency):
                events = []
                graph, fused = self.constrained_graph(events, effect=effect, dependency=dependency)
                artifact = WrapperCodegen(nn.Module(), Scheduler(graph, backward=True).run(), enable_autodiff=False).run()
                artifact.fn(torch.ones(2), torch.ones(2))
                self.assertEqual(events, ["dw", "effect", "dx"] if effect else ["dw", "dx"])
                if effect:
                    self.assertEqual(fused.scheduling_constraints["effect_barrier"], "external_effect")

    def test_unused_auxiliary_result_released_before_next_call(self):
        graph = IRGraph()
        meta = TensorMeta((4,), torch.float32, "cpu", (1,))
        x, y, unused, z = (TensorValue(name, meta) for name in ("x", "y", "unused", "z"))
        references = []

        def first(value):
            temporary = torch.ones(1024)
            references.append(weakref.ref(temporary))
            return value + 1, temporary

        unused.meta = AuxiliaryTensorMeta()

        def second(value):
            gc.collect()
            self.assertIsNone(references[0]())
            return value * 2

        graph.add(InputNode(0, x))
        graph.add(CallNode("first", first, (x,), {}, (y, unused)))
        graph.add(CallNode("second", second, (y,), {}, (z,)))
        graph.add(OutputNode(z))
        artifact = WrapperCodegen(nn.Module(), Scheduler(graph).run(), enable_autodiff=False).run()
        torch.testing.assert_close(artifact.fn(torch.zeros(4)), torch.full((4,), 2.))
        records = {entry["name"]: entry for entry in artifact.logical_lifetimes}
        self.assertEqual(records["unused"]["birth"], records["unused"]["local_release_after"])
        self.assertIsNone(records["unused"]["logical_bytes"])
        self.assertEqual(records["y"]["consumers"], (2,))
        self.assertEqual(records["z"]["owner"], "invocation")
        self.assertTrue(records["z"]["escaped"])
        self.assertIsNone(records["z"]["local_release_after"])
        self.assertTrue(all(entry["physical_bytes"] is None for entry in records.values()))

    def test_rewrite_invalidates_liveness_before_codegen(self):
        meta = TensorMeta((2,), torch.float32, "cpu", (1,))
        x, y = TensorValue("x", meta), TensorValue("y", meta)
        graph = IRGraph()
        graph.add(InputNode(0, x))
        graph.add(CallNode("neg", torch.neg, (x,), {}, (y,)))
        graph.add(OutputNode(y))
        schedule = Scheduler(graph).run()
        codegen = WrapperCodegen(nn.Module(), schedule, enable_autodiff=False)
        schedule.items[-1].result = x
        with self.assertRaisesRegex(ValueError, "Schedule changed"):
            codegen.run()

    def test_repeated_operands_count_one_consumer_and_outputs_keep_aliases(self):
        meta = TensorMeta((2,), torch.float32, "cpu", (1,))
        x, y = TensorValue("x", meta), TensorValue("y", meta)
        graph = IRGraph()
        graph.add(InputNode(0, x))
        graph.add(CallNode("square", torch.mul, (x, x), {}, (y,)))
        graph.add(OutputNode((x, y, y)))
        schedule = Scheduler(graph).run()
        memory = MemoryManager(schedule)
        memory.collect_schedule_level_remaining_users()
        self.assertEqual(memory.remaining_users[x], 2)
        self.assertEqual(memory.remaining_users[y], 1)
        records = {entry["name"]: entry for entry in memory.lifetime_plan()}
        self.assertEqual(records["x"]["owner"], "caller")
        self.assertEqual(records["x"]["consumers"], (1, 2))
        self.assertEqual(records["x"]["logical_bytes"], 8)
        artifact = WrapperCodegen(nn.Module(), schedule, enable_autodiff=False).run()
        original = torch.tensor([2., 3.])
        a, b, c = artifact.fn(original)
        self.assertIs(a, original)
        self.assertIs(b, c)
        torch.testing.assert_close(b, original.square())

    def test_logical_fp8_wrapper_bytes_do_not_claim_physical_component_size(self):
        metadata = TensorMeta((2, 4), torch.bfloat16, "cpu", (4, 1),
                              representation="fp8", storage_dtype=torch.float8_e4m3fn)
        value = TensorValue("fp8_wrapper", metadata)
        graph = IRGraph()
        graph.add(InputNode(0, value))
        graph.add(OutputNode(value))
        memory = MemoryManager(Scheduler(graph).run())
        memory.collect_schedule_level_remaining_users()
        row, = memory.lifetime_plan()
        self.assertEqual(row["logical_bytes"], 16)
        self.assertEqual(row["storage_dtype"], "torch.float8_e4m3fn")
        self.assertIsNone(row["physical_bytes"])

    def test_non_dominating_schedule_rejected(self):
        meta = TensorMeta((2,), torch.float32, "cpu", (1,))
        x, y = TensorValue("x", meta), TensorValue("y", meta)
        graph = IRGraph()
        graph.add(InputNode(0, x))
        graph.add(CallNode("neg", torch.neg, (x,), {}, (y,)))
        graph.add(OutputNode(y))
        schedule = Scheduler(graph).run()
        schedule.items[0], schedule.items[1] = schedule.items[1], schedule.items[0]
        with self.assertRaisesRegex(ValueError, "topologically ordered"):
            MemoryManager(schedule).collect_schedule_level_remaining_users()

    def test_effect_reordering_rejected_before_liveness_collection(self):
        meta = TensorMeta((2,), torch.float32, "cpu", (1,))
        x, a, b = (TensorValue(name, meta) for name in ("x", "a", "b"))
        graph = IRGraph()
        graph.add(InputNode(0, x))
        graph.add(CallNode("external_write_a", torch.neg, (x,), {}, (a,)))
        graph.add(CallNode("external_write_b", torch.neg, (x,), {}, (b,)))
        graph.add(OutputNode((a, b)))
        schedule = Scheduler(graph).run()
        schedule.items[1], schedule.items[2] = schedule.items[2], schedule.items[1]
        with self.assertRaisesRegex(ValueError, "effect/alias barrier"):
            MemoryManager(schedule).collect_schedule_level_remaining_users()

    def test_unsafe_alias_values_form_scheduling_barriers(self):
        graph, fused = self.constrained_graph([])
        dw = next(value for node in graph.nodes for value in node.outputs() if value.name == "dw")
        schedule = Scheduler(graph, unsafe_values=frozenset({dw}), backward=True).run()
        operations = [getattr(node, "op", "") for node in schedule.nodes()]
        self.assertLess(operations.index("bf16.gemm_dw"), operations.index("bf16.gemm_dx_native_derivative"))
        self.assertEqual(fused.scheduling_constraints["effect_barrier"], "bf16.gemm_dw")


if __name__ == "__main__":
    unittest.main()
