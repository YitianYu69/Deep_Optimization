"""CPU regression checks for FX aliases, mutations, and Inductor routing."""

import unittest
import copy

import torch
from torch.fx.passes.shape_prop import ShapeProp

from deepopt.compiler import BackendConfig, compile_fx
from deepopt.compiler.effects import analyze_fx_effects


class ViewOnly(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ViewMutation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        y = x.clone()
        view = y[:, :2]
        view.add_(1.0)
        return self.relu(y)


class PositionalInplace(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        return torch.nn.functional.relu(y, True)


class EffectsTest(unittest.TestCase):
    def test_residual_block_with_inplace_relu_keeps_safe_aot_regions(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
                self.bn = torch.nn.BatchNorm2d(3)
                self.relu = torch.nn.ReLU(inplace=True)
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(3, 2)

            def forward(self, x):
                y = self.relu(self.bn(self.conv(x)))
                return self.fc(torch.flatten(self.pool(y + x), 1))

        gm = torch.fx.symbolic_trace(Block())
        sample = torch.randn(2, 3, 8, 8, requires_grad=True)
        ShapeProp(gm).propagate(sample)
        reference = copy.deepcopy(gm)
        compiled = compile_fx(gm, (sample,))
        self.assertTrue(hasattr(compiled, "backend_artifact"))
        self.assertGreater(len(compiled.backend_artifact.native_autograd_regions)
                           + len(compiled.backend_artifact.joint_autograd_regions), 0)
        actual_input = sample.detach().clone().requires_grad_()
        expected_input = sample.detach().clone().requires_grad_()
        actual = compiled(actual_input)
        expected = reference(expected_input)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual_input.grad, expected_input.grad)
        for (name, parameter), (_, expected_parameter) in zip(
            gm.named_parameters(), reference.named_parameters()
        ):
            torch.testing.assert_close(
                parameter.grad, expected_parameter.grad, msg=name
            )

    def test_view_write_keeps_mutated_producer_native(self):
        class ViewBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                y = self.conv(x)
                y[:, :1].add_(1.0)
                return self.pool(y)

        gm = torch.fx.symbolic_trace(ViewBlock())
        sample = torch.randn(2, 3, 8, 8, requires_grad=True)
        ShapeProp(gm).propagate(sample)
        reference = copy.deepcopy(gm)
        report = analyze_fx_effects(gm)
        self.assertIn("conv", report.unsafe_nodes)
        compiled = compile_fx(gm, (sample,))
        self.assertTrue(hasattr(compiled, "backend_artifact"))
        self.assertEqual(len(compiled.backend_artifact.native_autograd_regions), 1)
        x = sample.detach().clone().requires_grad_()
        x_ref = sample.detach().clone().requires_grad_()
        result = compiled(x)
        expected = reference(x_ref)
        torch.testing.assert_close(result, expected)
        result.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(x.grad, x_ref.grad)
        torch.testing.assert_close(gm.conv.weight.grad, reference.conv.weight.grad)

    def test_pure_view_preserves_alias(self):
        gm = torch.fx.symbolic_trace(ViewOnly())
        report = analyze_fx_effects(gm)
        self.assertTrue(report.alias_nodes)
        self.assertFalse(report.mutation_nodes)
        x = torch.randn(2, 3)
        compiled = compile_fx(gm, (x,))
        result = compiled(x)
        self.assertEqual(
            result.untyped_storage().data_ptr(), x.untyped_storage().data_ptr()
        )

    def test_view_write_and_inplace_relu_match_eager_gradients(self):
        gm = torch.fx.symbolic_trace(ViewMutation())
        report = analyze_fx_effects(gm)
        self.assertTrue(report.mutation_nodes)
        x_eager = torch.randn(2, 4, requires_grad=True)
        x_compiled = x_eager.detach().clone().requires_grad_()
        expected = gm(x_eager)
        expected.sum().backward()
        compiled = compile_fx(gm, (x_compiled,), config=BackendConfig())
        actual = compiled(x_compiled)
        actual.sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x_compiled.grad, x_eager.grad)

    def test_positional_inplace_runs_in_native_mode(self):
        gm = torch.fx.symbolic_trace(PositionalInplace())
        report = analyze_fx_effects(gm)
        self.assertTrue(report.mutation_nodes)
        x = torch.randn(2, 4, requires_grad=True)
        expected = gm(x)
        compiled = compile_fx(gm, (x,))
        torch.testing.assert_close(compiled(x), expected)
        torch.testing.assert_close(gm(x), expected)

    def test_named_backend_registration(self):
        from deepopt.compiler import register_torch_backend
        from torch._dynamo.backends.registry import lookup_backend

        register_torch_backend("deepopt_effects_test")
        self.assertIs(
            lookup_backend("deepopt_effects_test"), compile_fx.__globals__["do_backend"]
        )
        model = ViewMutation()
        compiled = torch.compile(model, backend="deepopt_effects_test", fullgraph=True)
        x = torch.randn(2, 4, requires_grad=True)
        expected = model(x)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected)

    def test_functional_batch_norm_running_stats_are_writes(self):
        class FunctionalBN(torch.nn.Module):
            def forward(self, x, mean, var):
                return torch.nn.functional.batch_norm(x, mean, var, training=True)

        gm = torch.fx.symbolic_trace(FunctionalBN())
        report = analyze_fx_effects(gm)
        self.assertTrue(report.mutation_nodes)
        self.assertIn("mean", report.unsafe_nodes)
        self.assertIn("var", report.unsafe_nodes)
        self.assertNotIn("x", report.unsafe_nodes)


if __name__ == "__main__":
    unittest.main()
