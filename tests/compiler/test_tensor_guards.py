"""Input guards distinguish addressed layouts from irrelevant singleton strides."""

import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt.compiler import BackendConfig, compile_fx


class TensorGuardTests(unittest.TestCase):
    def test_singleton_stride_changes_preserve_addresses_and_gradients(self):
        class Double(nn.Module):
            def forward(self, x):
                return x * 2

        graph = torch.fx.symbolic_trace(Double())
        sample = torch.randn(1, 2, 2, 2, requires_grad=True)
        ShapeProp(graph).propagate(sample)
        compiled = compile_fx(graph, [sample], config=BackendConfig(require_full_aot=True))
        x = sample.detach().as_strided(sample.shape, (16, 4, 2, 1)).requires_grad_()
        y = compiled(x)
        torch.testing.assert_close(y, x * 2, rtol=0, atol=0)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 2), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "compiled tensor metadata"):
            compiled(x.transpose(-1, -2))


if __name__ == "__main__":
    unittest.main()
