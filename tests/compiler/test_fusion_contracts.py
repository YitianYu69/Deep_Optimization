"""Fusion identities distinguish semantics while implementation controls vary."""

from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

import torch

from deepopt.compiler import BackendConfig
from deepopt.compiler.autodiff.fusion_contracts import candidate, select
from deepopt.compiler.ir import CallNode, IRGraph, InputNode, OutputNode, TensorMeta, TensorValue


class FusionContractTests(unittest.TestCase):
    def graph(self, approximate="none", tile=64):
        graph = IRGraph()
        metadata = TensorMeta((2, 17), torch.bfloat16, torch.device("cpu"), (17, 1))
        value = TensorValue("x", metadata)
        output = TensorValue("y", metadata)
        graph.add(InputNode(0, value))
        spec = SimpleNamespace(node_name="projection", module_path="linear", module_training=True,
                               tile=tile, profile_ranges=False)
        def implementation(x, **kwargs):
            return torch.nn.functional.gelu(x, **kwargs)
        node = CallNode("gelu", implementation, (value,), {"approximate": approximate}, (output,), spec)
        graph.add(node)
        graph.add(OutputNode(output))
        return graph, node, spec

    def test_semantic_approximation_affects_candidate_identity(self):
        records = [candidate(graph, node, spec, "forward", "gemm_gelu")
                   for graph, node, spec in (self.graph("none"), self.graph("tanh"), self.graph("none"))]
        self.assertNotEqual(records[0]["candidate_id"], records[1]["candidate_id"])
        self.assertEqual(records[0]["candidate_id"], records[2]["candidate_id"])

    def test_implementation_control_changes_fingerprint_without_changing_candidate(self):
        records = []
        for tile in (32, 64):
            graph, node, spec = self.graph(tile=tile)
            record = candidate(graph, node, spec, "forward", "gemm_gelu")
            select(node, record, roles={"epilogue": ["_bf16_gemm"]}, consumed=("gemm", "gelu"))
            records.append(record)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                node.target(torch.randn(2, 17), **node.kwargs)
            self.assertIn(record["profile_range"], {event.key for event in profile.key_averages()})
            json.dumps(record, allow_nan=False)
        self.assertEqual(records[0]["candidate_id"], records[1]["candidate_id"])
        self.assertNotEqual(records[0]["implementation_fingerprint"], records[1]["implementation_fingerprint"])

    def test_independent_phase_and_pack_controls_validate(self):
        self.assertEqual(BackendConfig(fusion_phases="both").fusion_phases, ("forward", "backward"))
        self.assertEqual(BackendConfig(fusion_phases="none").fusion_phases, ())
        self.assertEqual(replace(BackendConfig(), fusion_phases="backward").fusion_phases, ("backward",))
        for phases in ("invalid", ("forward", "forward"), ("invalid",)):
            with self.assertRaises(ValueError):
                BackendConfig(fusion_phases=phases)
        for option in ("share_compatible_packs", "fuse_packed_multiply"):
            with self.assertRaises(ValueError):
                BackendConfig(**{option: 1})


if __name__ == "__main__":
    unittest.main()
