"""Physical qualification of the implemented explicit saved-pack CSE pass.

This is an IR contract fixture. Public shared Linear projections currently
hide their packs inside numerical calls and do not exercise this pass.
"""

from dataclasses import replace
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from deepopt.compiler.autodiff.pack_fusion import share_packs
from deepopt.compiler.ir import InputNode
from tests.compiler import test_pack_fusion as pack_fixture


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton"),
                     "Explicit pack CSE qualification requires CUDA and Triton")
class ExplicitPackCSECudaTests(unittest.TestCase):
    def test_two_explicit_encodes_become_one_with_all_backward_readers_preserved(self):
        from benchmarks.analysis.measurement import compiler_launch_evidence, fusion_candidate_evidence

        graph, trace, calls = pack_fixture.PackFusionTests().graph(packs=2)
        evaluate = pack_fixture.evaluate
        for compiled in (graph, trace.graph):
            for node in compiled.nodes:
                for value in node.outputs():
                    value.meta = replace(value.meta, device=torch.device("cuda:0"))
        source = torch.linspace(-3, 4, 68, device="cuda").reshape(4, 17).bfloat16()
        gradient = torch.randn_like(source)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

        # Warm both the encoder and backward numerical readers before profiling.
        with torch.no_grad():
            _, control_values = evaluate(graph, [source])
            expected, _ = evaluate(trace.graph, [gradient, *(control_values[value] for value in trace.saved_tensors)])
            torch.cuda.synchronize()
            calls.clear()
            with torch.profiler.profile(activities=activities) as control_profile:
                evaluate(graph, [source])
                torch.cuda.synchronize()
            self.assertEqual(len(calls), 2)

            trace, decisions = share_packs(graph, trace)
            self.assertEqual(len(trace.saved_tensors), 3)
            aliases = [node for node in trace.graph.nodes if isinstance(node, InputNode) and node.index > 0]
            self.assertEqual(len(aliases), 3)
            self.assertTrue(all(len(node.value.users) == 2 for node in aliases))
            evaluate(graph, [source])
            torch.cuda.synchronize()
            calls.clear()
            with torch.profiler.profile(activities=activities) as shared_profile:
                _, actual_values = evaluate(graph, [source])
                torch.cuda.synchronize()
            self.assertEqual(len(calls), 1)
            actual, _ = evaluate(trace.graph, [gradient, *(actual_values[value] for value in trace.saved_tensors)])
            for value, control in zip(actual, expected):
                torch.testing.assert_close(value, control, rtol=0, atol=0)

        def encoder_launches(profile):
            return [event for event in profile.events() if str(event.device_type).endswith("CUDA")
                    and "_quantize_pack" in event.name]

        self.assertEqual(len(encoder_launches(control_profile)), 2)
        self.assertEqual(len(encoder_launches(shared_profile)), 1)
        selected = [row for row in decisions if row["selected"]]
        self.assertEqual(len(selected), 1)
        with TemporaryDirectory(prefix="deepopt-pack-cse-profile-") as directory:
            trace_path = Path(directory) / "trace.json"
            shared_profile.export_chrome_trace(str(trace_path))
            evidence = fusion_candidate_evidence({"fusion_candidates": selected}, {
                "compiler_regions": compiler_launch_evidence(shared_profile.events(), 1, trace=trace_path)})
        self.assertTrue(evidence["all_selected_observed"], evidence)
        self.assertFalse(any(row["kernel_fusion"] for row in decisions))


if __name__ == "__main__":
    unittest.main()
