"""Explicit pack CSE preserves every saved binding and consumer lifetime."""

from dataclasses import replace
import json
from math import prod
from types import MappingProxyType
import unittest

import torch

from deepopt.compiler.autodiff.aot_auto_diff import BackwardTraceResult
from deepopt.compiler.autodiff.pack_fusion import share_packs
from deepopt.compiler.ir import CallNode, InputNode, IRGraph, OutputNode, PointwiseNode, TensorMeta, TensorValue
from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation, quantize_pack_activation


def evaluate(graph, inputs):
    values = {}

    def tree(value):
        if isinstance(value, TensorValue):
            return values[value]
        if isinstance(value, tuple):
            return tuple(tree(child) for child in value)
        if isinstance(value, dict):
            return {key: tree(child) for key, child in value.items()}
        return value

    for node in graph.nodes:
        if isinstance(node, InputNode):
            values[node.value] = inputs[node.index]
        elif isinstance(node, OutputNode):
            return tree(node.result), values
        else:
            result = node.target(*tree(node.args), **tree(node.kwargs))
            outputs = node.outputs()
            values.update(zip(outputs, (result,) if len(outputs) == 1 else result))


class PackFusionTests(unittest.TestCase):
    def graph(self, *, duplicate_options=None, separate_source=False, barrier=False, packs=2):
        metadata = TensorMeta((4, 17), torch.bfloat16, torch.device("cpu"), (17, 1), True)
        graph = IRGraph()
        source = TensorValue("source", metadata)
        graph.add(InputNode(0, source))
        other = source
        if separate_source:
            other = TensorValue("alias", metadata, source_name="source")
            graph.add(InputNode(1, other))
        components, options_by_pack, calls = [], [], []
        for index in range(packs):
            if index and barrier:
                graph.add(CallNode("opaque_side_effect", lambda x: x, (source,), {},
                                   (TensorValue(f"effect{index}", metadata),)))
            actual_source = other if index else source
            options = {"bits": 2, "group_size": 16, "logical_shape": metadata.shape,
                       "logical_dtype": metadata.dtype, "logical_stride": metadata.stride,
                       "restored_stride": metadata.stride, "traversal": "logical_row_major",
                       "rounding_domain": "operand_dtype", "source_cast": None}
            if index:
                options.update(duplicate_options or {})
            groups = (prod(metadata.shape) + options["group_size"] - 1) // options["group_size"]
            words = (options["group_size"] + 32 // options["bits"] - 1) // (32 // options["bits"])
            outputs = tuple(TensorValue(f"pack{index}_{name}", TensorMeta(shape, dtype, metadata.device,
                representation="packed" if name == "payload" else "dense")) for name, shape, dtype in (
                    ("payload", (groups, words), torch.int32), ("step", (groups,), torch.float32),
                    ("minimum", (groups,), torch.float32)))

            def pack(x, options=options):
                packed = quantize_pack_activation(x, options["group_size"], bits=options["bits"])
                calls.append(tuple(options.items()))
                return packed.data, packed.step, packed.minimum

            graph.add(CallNode("codec.save_activation", pack, (actual_source,), {}, outputs, MappingProxyType(options)))
            components.append(outputs)
            options_by_pack.append(options)
        output = OutputNode((source,))
        output.saved_tensors_for_bwd = components[-1]
        graph.add(output)
        backward = IRGraph()
        seed = TensorValue("seed", replace(metadata, requires_grad=False))
        backward.add(InputNode(0, seed))
        results, records = [], []
        for index, (stored, options) in enumerate(zip(components, options_by_pack)):
            aliases = tuple(TensorValue("saved_" + value.name, value.meta) for value in stored)
            for component, alias in enumerate(aliases):
                backward.add(InputNode(1 + index * 3 + component, alias))

            def restore(data, step, minimum, options=options):
                packed = PackedActivation(data, step, minimum, metadata.shape, options["group_size"], bits=options["bits"])
                return dequantize_unpack_activation(packed, metadata.dtype, row_alignment=1)

            restored = TensorValue(f"restored{index}", replace(metadata, requires_grad=False))
            backward.add(CallNode("codec.restore_activation", restore, aliases, {}, (restored,), MappingProxyType(options)))
            product = TensorValue(f"product{index}", replace(metadata, requires_grad=False))
            backward.add(PointwiseNode("mul", (seed, restored), {}, product, torch.mul))
            results.append(product)
            records.append({"source": "source", "components": tuple(v.name for v in stored),
                            "representation": "packed_saved_operand", "consumers": [{"role": f"reader{index}"}]})
        backward.add(OutputNode(tuple(results)))
        trace = BackwardTraceResult(backward, (source,), tuple(v for group in components for v in group), (), tuple(records))
        graph.validate()
        backward.validate()
        return graph, trace, calls

    def test_equivalent_packs_are_produced_once_with_all_saved_readers_remapped(self):
        graph, trace, calls = self.graph(packs=3)
        source = torch.linspace(-3, 4, 68).reshape(4, 17).bfloat16()
        gradient = torch.randn_like(source)
        old_trace = trace
        trace, decisions = share_packs(graph, trace)
        self.assertEqual(len([node for node in graph.nodes if getattr(node, "op", None) == "codec.save_activation"]), 1)
        self.assertEqual(len(trace.saved_tensors), 3)
        self.assertEqual(len([node for node in trace.graph.nodes if isinstance(node, InputNode)]), 4)
        self.assertEqual([record["selected"] for record in decisions], [False, True, True])
        self.assertTrue(all(not record["kernel_fusion"] for record in decisions))
        self.assertEqual(decisions[0]["status"], "shared_producer")
        self.assertEqual(decisions[1]["produced_components"], decisions[0]["original_components"])
        self.assertGreater(decisions[1]["estimated_removed_component_bytes"], 0)
        self.assertEqual(graph.nodes[-1].saved_tensors_for_bwd, trace.saved_tensors)
        for record in trace.saved_value_specs:
            self.assertEqual(record["components"], tuple(v.name for v in trace.saved_tensors))
        self.assertEqual(len(old_trace.saved_tensors), 9)
        _output, forward_values = evaluate(graph, [source])
        self.assertEqual(len(calls), 1)
        actual, _ = evaluate(trace.graph, [gradient, *(forward_values[value] for value in trace.saved_tensors)])
        packed = quantize_pack_activation(source, 16, bits=2)
        expected = gradient * dequantize_unpack_activation(packed, source.dtype, row_alignment=1)
        for result in actual:
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        for component in trace.saved_tensors:
            # Every physical component has one surviving backward binding;
            # that binding feeds all three restore readers until their last use.
            inputs = [node for node in trace.graph.nodes if isinstance(node, InputNode)
                      and node.index == trace.saved_tensors.index(component) + 1]
            self.assertEqual(len(inputs[0].value.users), 3)
        json.dumps(decisions, allow_nan=False)

    def test_all_representation_cast_layout_and_correction_boundaries_are_part_of_key(self):
        variants = ({"bits": 4}, {"group_size": 32}, {"restored_stride": (1, 4)},
                    {"traversal": "channel_major"}, {"source_cast": torch.bfloat16},
                    {"representation_role": "activation_derivative"},
                    {"correction_recipe": {"recenter": True}}, {"rounding_domain": "accumulator_fp32"})
        for options in variants:
            with self.subTest(options=options):
                graph, trace, _ = self.graph(duplicate_options=options)
                trace, decisions = share_packs(graph, trace)
                self.assertFalse(any(record["selected"] for record in decisions))
                self.assertEqual(len(trace.saved_tensors), 6)
                self.assertEqual(decisions[-1]["reason"], "incompatible_representation_or_effect_interval")

    def test_equal_shape_and_source_ancestry_do_not_establish_source_identity(self):
        graph, trace, _ = self.graph(separate_source=True)
        trace, decisions = share_packs(graph, trace)
        self.assertFalse(any(record["selected"] for record in decisions))
        self.assertEqual(len(trace.saved_tensors), 6)

    def test_unknown_operations_are_effect_barriers(self):
        graph, trace, _ = self.graph(barrier=True)
        trace, decisions = share_packs(graph, trace)
        self.assertFalse(any(record["selected"] for record in decisions))
        self.assertEqual(len(trace.saved_tensors), 6)
        self.assertNotEqual(decisions[0]["compatibility"]["source_version"], decisions[1]["compatibility"]["source_version"])

    def test_disabled_control_discovers_duplicates_without_rewriting(self):
        graph, trace, calls = self.graph()
        original_nodes = tuple(graph.nodes)
        updated, decisions = share_packs(graph, trace, enabled=False)
        self.assertIs(updated, trace)
        self.assertEqual(tuple(graph.nodes), original_nodes)
        self.assertEqual(decisions[1]["would_share_with"], decisions[0]["encoding_id"])
        self.assertTrue(all(record["status"] == "disabled" for record in decisions))
        evaluate(graph, [torch.randn(4, 17).bfloat16()])
        self.assertEqual(len(calls), 2)

    def test_backward_alias_is_kept_at_the_earliest_existing_binding(self):
        graph, trace, _ = self.graph()
        nodes = trace.graph.nodes
        # Both groups are independent: bind/read the duplicate first.
        nodes[:] = [nodes[0], *nodes[6:11], *nodes[1:6], nodes[-1]]
        trace.graph.validate()
        trace, _ = share_packs(graph, trace)
        trace.graph.validate()
        self.assertTrue(all(node.value.name.startswith("saved_pack1_") for node in trace.graph.nodes
                            if isinstance(node, InputNode) and node.index > 0))
        self.assertEqual(sorted(node.index for node in trace.graph.nodes if isinstance(node, InputNode)), [0, 1, 2, 3])

    def test_missing_or_inconsistent_codec_contract_remains_explicit(self):
        for mutation, reason in ((lambda options: options.pop("source_cast"), "incomplete_codec_contract"),
                                 (lambda options: options.update(logical_stride=(1, 4)), "source_contract_mismatch")):
            with self.subTest(reason=reason):
                graph, trace, _ = self.graph()
                pack = next(node for node in graph.nodes if getattr(node, "op", None) == "codec.save_activation")
                options = dict(pack.specification)
                mutation(options)
                pack.specification = MappingProxyType(options)
                updated, decisions = share_packs(graph, trace)
                self.assertEqual(decisions[0]["reason"], reason)
                self.assertFalse(any(record["selected"] for record in decisions))
                self.assertEqual(len(updated.saved_tensors), 6)


if __name__ == "__main__":
    unittest.main()
