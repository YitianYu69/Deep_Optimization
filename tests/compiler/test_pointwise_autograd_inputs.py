"""Pointwise backward setup must map saved values in the cloned local graph."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt.compiler.autodiff_runtime import build_autograd_region
from deepopt.compiler.ir import PointwiseNode
from deepopt.compiler.lowering import GraphLowering
from deepopt.compiler.registry import BackendRegistry
from deepopt.compiler.scheduler import FusionGroup, SinglePointwiseOp


class PointwiseAutogradInputTests(unittest.TestCase):
    def check_region(
        self, module, *, two_inputs=False, frozen_second=False, dtype=torch.float32
    ):
        torch.manual_seed(42)
        samples = [torch.randn(2, 7, 32, dtype=dtype, requires_grad=True)]
        if two_inputs:
            samples.append(
                torch.randn_like(samples[0], requires_grad=not frozen_second)
            )
        graph = torch.fx.symbolic_trace(module)
        ShapeProp(graph).propagate(*samples)
        ir = GraphLowering(graph, samples).run()
        nodes = [node for node in ir.nodes if isinstance(node, PointwiseNode)]
        partition = (
            SinglePointwiseOp(nodes[0]) if len(nodes) == 1 else FusionGroup(nodes)
        )
        region = build_autograd_region(
            partition,
            None,
            graph,
            registry=BackendRegistry(),
            cache_dir=None,
            block_size=1024,
        )
        self.assertEqual(region.input_count, len(samples))
        self.assertTrue(region.saved_input_indices)
        self.assertTrue(
            all(0 <= index < len(samples) for index in region.saved_input_indices)
        )
        reference_inputs = [
            x.detach().clone().requires_grad_(x.requires_grad) for x in samples
        ]
        # Replace only the CUDA forward launch. The real region builder, saved
        # tensor mapping, autograd boundary and generated backward execute on
        # CPU; this test does not establish Triton/CUDA kernel correctness.
        with patch.object(region, "_forward", side_effect=module):
            actual = region(*samples)
        expected = module(*reference_inputs)
        actual.float().sum().backward()
        expected.float().sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for sample, reference in zip(samples, reference_inputs):
            torch.testing.assert_close(sample.grad, reference.grad)
        return region

    def test_single_activation_saves_local_input_in_fp32_and_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            for activation in (nn.GELU(), nn.Sigmoid(), nn.Tanh()):
                with self.subTest(dtype=dtype, activation=type(activation).__name__):
                    self.check_region(activation, dtype=dtype)

    def test_fused_region_rematerializes_intermediates_from_both_local_inputs(self):
        class Fused(nn.Module):
            def forward(self, x, y):
                return torch.nn.functional.gelu(x) * torch.sigmoid(y)

        self.check_region(Fused(), two_inputs=True)

    def test_frozen_second_input_is_saved_in_its_original_argument_position(self):
        class Multiply(nn.Module):
            def forward(self, x, y):
                return x * y

        region = self.check_region(Multiply(), two_inputs=True, frozen_second=True)
        self.assertEqual(region.saved_input_indices, (1,))
        self.assertEqual(region.gradient_target_input_indices, (0,))


if __name__ == "__main__":
    unittest.main()
