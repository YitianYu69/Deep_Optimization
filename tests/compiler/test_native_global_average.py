"""Global adaptive pooling uses the public mean derivative's BF16 rounding."""

import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt.compiler import BackendConfig, CompilationReport, compile_fx


class NativeGlobalAverageTests(unittest.TestCase):
    def test_public_global_pooling_exact_adjoint_without_source_values(self):
        for rank in (1, 2, 3):
            for dtype in (torch.bfloat16, torch.float32, torch.float64):
                for spatial in ((3,) * rank, (4,) * rank):
                    with self.subTest(rank=rank, dtype=dtype, spatial=spatial):
                        model = nn.Sequential(getattr(nn, f"AdaptiveAvgPool{rank}d")(1))
                        graph = torch.fx.symbolic_trace(model)
                        x = torch.randn((2, 3, *spatial), dtype=dtype, requires_grad=True)
                        reference = x.detach().clone().requires_grad_()
                        ShapeProp(graph).propagate(x)
                        report = CompilationReport()
                        compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True), report=report)
                        actual, expected = compiled(x), graph(reference)
                        gradient = torch.randn_like(actual)
                        actual.backward(gradient)
                        expected.backward(gradient)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                        region, = compiled.backend_artifact.joint_autograd_regions
                        self.assertEqual(region.trace.saved_tensors, ())
                        self.assertEqual(region.trace.saved_value_specs[0]["representation"], "metadata_only")
                        self.assertFalse(report.graphs[0].eager_autograd_operations)
                        self.assertFalse(report.graphs[0].legacy_backward_operations)


if __name__ == "__main__":
    unittest.main()
