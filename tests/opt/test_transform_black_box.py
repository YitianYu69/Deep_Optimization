"""Public model transformation, training, aliases, and checkpoint behavior."""

from copy import deepcopy
import io
import unittest

import torch
from torch import fx, nn
from torch.nn import functional as F

from deepopt.opt import TransformationTraceError, transform_model
from deepopt.opt.do_nn.linear_module import DOLinear


def plan():
    return {
        "fp8_training": {"fp8_dtypes": {
            "activation": torch.float8_e4m3fn,
            "weight": torch.float8_e4m3fn,
            "grad": torch.float8_e5m2,
        }},
        "block_shapes": {role: None for role in ("activation", "weight", "grad")},
    }


class MixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8)
        self.weight = nn.Parameter(torch.randn(4, 8) * .1)
        self.bias = nn.Parameter(torch.zeros(4))

    def forward(self, x):
        return F.linear(input=torch.tanh(self.projection(x)),
                        weight=self.weight, bias=self.bias)


class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = MixedModel()

    def forward(self, x):
        if x.sum() > 0:
            return self.block(x)
        return -self.block(x)

    def custom_method(self):
        return "root preserved"


class SharedModel(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.Linear(8, 8)
        self.left = nn.Sequential(layer, nn.Tanh())
        self.right = nn.Sequential(layer, nn.Sigmoid())
        self.alias = layer
        self.tied = nn.Linear(8, 8)
        self.tied.weight = layer.weight

    def forward(self, x):
        return self.left(x) + self.right(x) + self.alias(x) + self.tied(x)


class TransformationBlackBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)

    def assert_relative_close(self, actual, expected, tolerance=.12):
        error = (actual.float() - expected.float()).norm()
        scale = expected.float().norm().clamp_min(1e-6)
        self.assertLess((error / scale).item(), tolerance)

    def test_full_forward_backward_and_optimizer_step(self):
        model = MixedModel()
        baseline = deepcopy(model)
        original_parameters = dict(model.named_parameters())
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        transformed, report = transform_model(model, plan=plan())
        x = torch.randn(5, 8, requires_grad=True)
        baseline_x = x.detach().clone().requires_grad_()
        output = transformed(x)
        expected = baseline(baseline_x)
        self.assert_relative_close(output, expected)
        output.square().mean().backward()
        expected.square().mean().backward()
        self.assert_relative_close(x.grad, baseline_x.grad, .2)
        for name, parameter in transformed.named_parameters():
            self.assertIs(parameter, original_parameters[name])
            self.assert_relative_close(parameter.grad, dict(baseline.named_parameters())[name].grad, .2)
        before = transformed.projection.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, transformed.projection.weight))
        self.assertEqual(report.trace_mode, "full")
        self.assertEqual(report.converted_modules, ("projection",))
        self.assertEqual(len(report.function_calls), 1)
        self.assertEqual(report.untraced_modules, ())

    def test_modules_preserves_root_methods_and_dynamic_branches(self):
        model = DynamicModel()
        baseline = deepcopy(model)
        transformed, report = transform_model(model, plan=plan(), trace_mode="modules")
        self.assertIs(transformed, model)
        self.assertEqual(transformed.custom_method(), "root preserved")
        self.assertIsInstance(transformed.block, fx.GraphModule)
        self.assertIn("<root>", report.untraced_modules)
        for sign in (-1, 1):
            x = torch.ones(3, 8) * sign
            self.assert_relative_close(transformed(x), baseline(x))

    def test_auto_records_fallback_and_full_success_separately(self):
        dynamic = DynamicModel()
        transformed, report = transform_model(dynamic, plan=plan(), trace_mode="auto")
        self.assertIs(transformed, dynamic)
        self.assertEqual(report.trace_mode, "modules")
        self.assertTrue(report.fallback_reason)
        transformed, report = transform_model(MixedModel(), plan=plan(), trace_mode="auto")
        self.assertIsInstance(transformed, fx.GraphModule)
        self.assertEqual(report.trace_mode, "full")
        self.assertIsNone(report.fallback_reason)

    def test_full_trace_error_propagates(self):
        with self.assertRaises(TransformationTraceError) as raised:
            transform_model(DynamicModel(), plan=plan(), trace_mode="full")
        self.assertIsNotNone(raised.exception.__cause__)

    def test_deepcopy_isolates_source_and_preserves_ties(self):
        for mode in ("full", "modules"):
            with self.subTest(mode=mode):
                model = SharedModel()
                transformed, _ = transform_model(model, plan=plan(), trace_mode=mode, inplace=False)
                self.assertIsInstance(model.alias, nn.Linear)
                self.assertIsInstance(transformed.alias, DOLinear)
                self.assertIs(transformed.alias, transformed.left[0] if mode == "full"
                              else transformed.left.get_submodule("0"))
                self.assertIs(transformed.alias.weight, transformed.tied.weight)
                self.assertIsNot(transformed.alias.weight, model.alias.weight)

    def test_shared_modules_and_parameters_across_boundaries(self):
        for mode in ("full", "modules"):
            with self.subTest(mode=mode):
                source = SharedModel()
                weight, bias = source.alias.weight, source.alias.bias
                transformed, _ = transform_model(source, plan=plan(), trace_mode=mode)
                self.assertIs(transformed.left.get_submodule("0"), transformed.alias)
                self.assertIs(transformed.right.get_submodule("0"), transformed.alias)
                self.assertIs(transformed.tied.weight, weight)
                self.assertIs(transformed.alias.bias, bias)
                transformed(torch.randn(3, 8)).sum().backward()
                self.assertIsNotNone(weight.grad)
                self.assertTrue(torch.isfinite(weight.grad).all())

    def test_state_layout_keeps_unused_state_and_nonpersistent_buffers(self):
        class Stateful(MixedModel):
            def __init__(self):
                super().__init__()
                self.unused = nn.Linear(8, 2)
                self.register_parameter("unused_parameter", nn.Parameter(torch.ones(2)))
                self.register_buffer("saved", torch.ones(2))
                self.register_buffer("temporary", torch.ones(3), persistent=False)
                self.projection.eval()

        for mode in ("full", "modules"):
            with self.subTest(mode=mode):
                model = Stateful()
                before = model.state_dict()
                transformed, _ = transform_model(model, plan=plan(), trace_mode=mode)
                self.assertEqual(set(before), set(transformed.state_dict()))
                self.assertNotIn("temporary", transformed.state_dict())
                self.assertTrue(transformed.training)
                self.assertFalse(transformed.projection.training)
                fresh = Stateful()
                fresh.load_state_dict(transformed.state_dict(), strict=True)
                transformed.load_state_dict(fresh.state_dict(), strict=True)

    def test_model_and_optimizer_checkpoint_resume_matches_next_update(self):
        model, _ = transform_model(MixedModel(), plan=plan())
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        batches = [torch.randn(3, 8), torch.randn(3, 8)]

        def step(module, opt, x):
            opt.zero_grad(set_to_none=True)
            loss = module(x).square().mean()
            loss.backward()
            opt.step()
            return loss.detach()

        step(model, optimizer, batches[0])
        checkpoint = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=True)
        resumed, _ = transform_model(MixedModel(), plan=plan())
        resumed.load_state_dict(saved["model"], strict=True)
        resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=.01)
        resumed_optimizer.load_state_dict(saved["optimizer"])
        torch.testing.assert_close(step(model, optimizer, batches[1]),
                                   step(resumed, resumed_optimizer, batches[1]), rtol=0, atol=0)
        for actual, expected in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_parameter_order_preserves_pretransformation_optimizer_checkpoint(self):
        class ReverseExecution(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Linear(8, 8)
                self.second = nn.Linear(8, 8)

            def forward(self, x):
                # FX encounters these in the opposite order to registration.
                return self.first(self.second(x))

        source = ReverseExecution()
        names_before = tuple(dict(source.named_parameters()))
        optimizer = torch.optim.Adam(source.parameters(), lr=.01)
        transformed, _ = transform_model(source, plan=plan())
        self.assertEqual(tuple(dict(transformed.named_parameters())), names_before)
        x = torch.randn(3, 8)
        transformed(x).square().mean().backward()
        optimizer.step()
        checkpoint = deepcopy({"model": transformed.state_dict(), "optimizer": optimizer.state_dict()})
        resumed, _ = transform_model(ReverseExecution(), plan=plan())
        resumed.load_state_dict(checkpoint["model"])
        resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=.01)
        resumed_optimizer.load_state_dict(checkpoint["optimizer"])
        for module, opt in ((transformed, optimizer), (resumed, resumed_optimizer)):
            opt.zero_grad(set_to_none=True)
            module(x).square().mean().backward()
            opt.step()
        for name, parameter in transformed.named_parameters():
            torch.testing.assert_close(parameter, dict(resumed.named_parameters())[name], rtol=0, atol=0)

    def test_existing_graph_and_repeated_transformation(self):
        model, _ = transform_model(MixedModel(), plan=plan())
        original_projection = model.projection
        output = model(torch.ones(2, 8))
        transformed, report = transform_model(model, plan=plan())
        self.assertIs(transformed, model)
        self.assertIs(transformed.projection, original_projection)
        self.assertEqual(report.converted_modules, ())
        self.assertEqual(report.function_calls, ())
        torch.testing.assert_close(output, transformed(torch.ones(2, 8)), rtol=0, atol=0)

    def test_functional_root_linear(self):
        source = nn.Linear(8, 4)
        transformed, report = transform_model(source, plan=plan())
        self.assertEqual(report.converted_modules, ())
        self.assertEqual(len(report.function_calls), 1)
        self.assertIs(transformed.weight, source.weight)
        transformed(torch.randn(3, 8)).sum().backward()
        self.assertIsNotNone(source.weight.grad)

    def test_concrete_args_specializes_full_graph(self):
        class OptionalBranch(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(8, 4)

            def forward(self, x, enabled=True):
                return self.layer(x) if enabled else x

        transformed, _ = transform_model(OptionalBranch(), plan=plan(), concrete_args={"enabled": True})
        self.assertEqual(transformed(torch.ones(2, 8)).shape, (2, 4))
        with self.assertRaises(ValueError):
            transform_model(OptionalBranch(), plan=plan(), trace_mode="modules", concrete_args={"enabled": True})
        with self.assertRaises(TransformationTraceError):
            transform_model(DynamicModel(), plan=plan(), trace_mode="auto", concrete_args={})

    def test_traceable_child_still_converts_unused_registered_descendants(self):
        class PartialChild(nn.Module):
            def __init__(self):
                super().__init__()
                self.used = nn.Linear(8, 8)
                self.unused = nn.Linear(8, 8)

            def forward(self, x):
                return self.used(x)

        model = nn.Sequential(PartialChild())
        transformed, report = transform_model(model, plan=plan(), trace_mode="modules")
        self.assertIsInstance(transformed[0].used, DOLinear)
        self.assertIsInstance(transformed[0].unused, DOLinear)
        self.assertIn("0.unused", report.converted_modules)

    def test_failed_child_trace_keeps_control_flow_and_converts_descendants(self):
        model = nn.Sequential(DynamicModel())
        transformed, report = transform_model(model, plan=plan(), trace_mode="modules")
        self.assertIs(transformed, model)
        self.assertIn("0", report.untraced_modules)
        self.assertIsInstance(transformed[0].block.projection, DOLinear)
        self.assertEqual(transformed(torch.ones(2, 8)).shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
