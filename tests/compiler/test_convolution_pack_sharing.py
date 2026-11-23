"""Public shared convolution projections reuse compatible saved source packs."""

from copy import deepcopy
import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx


class ConvolutionPackSharingTests(unittest.TestCase):
    def test_public_projections_share_source_and_preserve_every_gradient(self):
        class Projections(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Conv2d(4, 5, 1, bias=False)
                self.up = nn.Conv2d(4, 5, 1, bias=False)

            def forward(self, x):
                return nn.functional.silu(self.gate(x)), self.up(x)

        torch.manual_seed(901)
        source = Projections()
        outputs, gradients, decisions = [], [], []
        for enabled in (False, True):
            module = deepcopy(source)
            graph = torch.fx.symbolic_trace(module)
            x = torch.randn(2, 4, 16, 16, requires_grad=True)
            if not enabled:
                sample = x.detach().clone()
            else:
                x = sample.clone().requires_grad_()
            report = CompilationReport()
            with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                          activation_compression=dict(enabled=True, bits=2, group_size=1024,
                                                      norm_xhat_recenter=True), config={}):
                ShapeProp(graph).propagate(x)
                compiled = compile_fx(graph, [x], config=BackendConfig(
                    require_full_aot=True, share_compatible_packs=enabled), report=report)
                result = compiled(x)
            sum(value.float().square().mean() for value in result).backward()
            outputs.append(tuple(value.detach() for value in result))
            gradients.append((x.grad, *(parameter.grad for parameter in graph.parameters())))
            decisions.append(report.graphs[0].pack_sharing_decisions)
            from examples.vae.qualification import saved_state_evidence
            from deepopt.compiler.reporting import classify_saved_state

            # This unprepared functional SiLU saves its dense input exactly.
            # Sharing valid convolution packs must not hide that other save.
            with self.assertRaisesRegex(AssertionError, "large dense.*gate"):
                saved_state_evidence(report.as_dict(), True)
            classified = [row for region in report.graphs[0].native_autodiff_graphs
                          for row in classify_saved_state(region, False)]
            self.assertEqual(sum(row["category"] == "codec_metadata" for row in classified), 2 if enabled else 4)
        for value, control in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(value, control, rtol=0, atol=0)
        for value, control in zip(gradients[0], gradients[1]):
            torch.testing.assert_close(value, control, rtol=0, atol=0)
        self.assertFalse(any(item["selected"] for item in decisions[0]))
        selected = [item for item in decisions[1] if item["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertGreater(selected[0]["estimated_removed_component_bytes"], 0)
        self.assertEqual(selected[0]["compatibility"]["representation_role"], "convolution_source")


if __name__ == "__main__":
    unittest.main()
