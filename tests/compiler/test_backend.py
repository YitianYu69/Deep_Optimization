"""Public compiler configuration, ordinary autograd and structured FX contracts."""

from copy import deepcopy
import operator
import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt.compiler import (
    BackendConfig,
    BackendRegistry,
    compilation_report,
    compile_fx,
    register_torch_backend,
)
from deepopt.compiler.lowering import GraphLowering
from deepopt.compiler.autodiff.native_layer_backward_rules import (
    is_registered_native_layer_node,
)
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model


class BackendTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_registration_is_idempotent_and_native_model_preserves_ownership(self):
        register_torch_backend()
        register_torch_backend()
        torch.manual_seed(3)
        model = nn.Sequential(nn.Linear(3, 7), nn.Sigmoid(), nn.Linear(7, 2))
        reference = deepcopy(model)
        identities = [id(parameter) for parameter in model.parameters()]
        compiled = compile_model(
            model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True)
        )
        self.assertIs(compiled, model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
        for size in (4, 2):
            x = torch.randn(size, 3, requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            actual, expected = model(x), reference(xr)
            torch.testing.assert_close(actual, expected)
            actual.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(x.grad, xr.grad)
            optimizer.step()
            reference_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            for parameter, baseline in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, baseline)
        self.assertEqual(
            identities, [id(parameter) for parameter in model.parameters()]
        )
        self.assertEqual(tuple(model.state_dict()), tuple(reference.state_dict()))
        self.assertEqual(compilation_report(model).captured_graphs, 2)
        self.assertEqual(compilation_report(model).runtime_fallbacks, 0)
        self.assertIsNone(deepcopy(model)._compiled_call_impl)

    def test_options_reject_invalid_or_unsupported_requests_before_compilation(self):
        invalid = (
            {"block_size": True},
            {"block_size": 3},
            {"max_pointwise_fusion_length": 1.5},
            {"capture_deepopt": "yes"},
            {"debug": 1},
            {"unknown": 1},
            {"require_full_aot": True, "enable_aot_autodiff": False},
            {"cache_dir": ""},
        )
        for options in invalid:
            with (
                self.subTest(options=options),
                self.assertRaises((ValueError, TypeError)),
            ):
                compile_model(
                    nn.Linear(2, 2),
                    CompileConfig(enabled=True, backend="deepopt", options=options),
                )
        with self.assertRaisesRegex(ValueError, "dynamic=False"):
            compile_model(
                nn.Linear(2, 2),
                CompileConfig(enabled=True, backend="deepopt", dynamic=True),
            )
        with self.assertRaisesRegex(ValueError, "mode"):
            compile_model(
                nn.Linear(2, 2),
                CompileConfig(enabled=True, backend="deepopt", mode="reduce-overhead"),
            )

    def test_nested_outputs_broadcast_and_noncontiguous_gradients(self):
        class Model(nn.Module):
            def forward(self, x, bias):
                y = x.transpose(0, 1) + bias
                values, indices = torch.topk(y, 2, dim=-1)
                return {"result": [values, (indices, y)], "size": y.shape[0]}

        model = Model()
        compile_model(
            model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True)
        )
        x = torch.randn(3, 5, requires_grad=True)
        bias = torch.randn(3, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        br = bias.detach().clone().requires_grad_()
        actual, expected = model(x, bias), Model()(xr, br)
        torch.testing.assert_close(actual, expected)
        actual["result"][0].sum().backward()
        expected["result"][0].sum().backward()
        torch.testing.assert_close(x.grad, xr.grad)
        torch.testing.assert_close(bias.grad, br.grad)

    def test_native_amp_retains_casts_and_backward_outside_its_scope(self):
        model = nn.Sequential(nn.Linear(8, 8), nn.Sigmoid(), nn.Linear(8, 3))
        baseline = deepcopy(model)
        compile_model(
            model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True)
        )
        x = torch.randn(4, 8, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual, expected = model(x), baseline(xr)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
        for parameter, reference in zip(model.parameters(), baseline.parameters()):
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)

    def test_complex_operations_retain_pytorch_conjugate_gradient_rules(self):
        model = compile_model(
            nn.Linear(3, 2, dtype=torch.complex64),
            CompileConfig(enabled=True, backend="deepopt", fullgraph=True),
        )
        baseline = deepcopy(model)
        x = torch.randn(4, 3, dtype=torch.complex64, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        actual, expected = model(x), baseline(xr)
        actual.real.sum().backward()
        expected.real.sum().backward()
        torch.testing.assert_close(x.grad, xr.grad)
        torch.testing.assert_close(model.weight.grad, baseline.weight.grad)
        graph = torch.fx.symbolic_trace(lambda x: x * (2 + 3j))
        sample = torch.randn(3, dtype=torch.complex64, requires_grad=True)
        ShapeProp(graph).propagate(sample)
        compile_fx(graph, [sample])(sample).real.sum().backward()
        torch.testing.assert_close(sample.grad, torch.full_like(sample, 2 - 3j))

    def test_explicit_fx_selects_native_execution_when_numerical_scope_changes(self):
        from deepopt.compiler import CompilationReport

        gm = torch.fx.symbolic_trace(nn.Linear(8, 3))
        sample = torch.randn(4, 8)
        ShapeProp(gm).propagate(sample)
        report = CompilationReport()
        compiled = compile_fx(gm, [sample], report=report)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual, expected = compiled(sample), gm(sample)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(report.fallback_reasons, {"numerical scope changed": 1})

    def test_native_activation_options_and_module_configuration_changes(self):
        from deepopt.compiler import CompilationReport

        for activation in (
            nn.GELU(approximate="tanh"),
            nn.LeakyReLU(0.3),
            nn.Hardtanh(-0.2, 0.4),
            nn.Softplus(beta=2, threshold=5),
        ):
            with self.subTest(activation=type(activation).__name__):
                graph = torch.fx.symbolic_trace(nn.Sequential(activation))
                x = torch.randn(4, 5, requires_grad=True)
                xr = x.detach().clone().requires_grad_()
                ShapeProp(graph).propagate(x)
                compiled = compile_fx(graph, [x])
                actual, expected = compiled(x), graph(xr)
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(x.grad, xr.grad)
        graph = torch.fx.symbolic_trace(nn.Sequential(nn.BatchNorm2d(3)))
        sample = torch.randn(2, 3, 4, 4, requires_grad=True)
        ShapeProp(graph).propagate(sample)
        report = CompilationReport()
        compiled = compile_fx(graph, [sample], report=report)
        graph.eval()
        actual, expected = compiled(sample), graph(sample)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(report.fallback_reasons, {"module configuration changed": 1})
        graph = torch.fx.symbolic_trace(nn.Sequential(nn.Linear(5, 3)))
        sample = torch.randn(2, 5)
        ShapeProp(graph).propagate(sample)
        compiled = compile_fx(graph, [sample])
        graph._modules["0"] = nn.Linear(5, 3)
        torch.testing.assert_close(compiled(sample), graph(sample))

    def test_private_scaled_inplace_add_preserves_alpha_and_gradient(self):
        class Model(nn.Module):
            def forward(self, x, y):
                return x.clone().add_(y, alpha=3)

        graph = torch.fx.symbolic_trace(Model())
        x, y = torch.randn(4, requires_grad=True), torch.randn(4, requires_grad=True)
        ShapeProp(graph).propagate(x, y)
        compile_fx(graph, [x, y])(x, y).sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        torch.testing.assert_close(y.grad, torch.full_like(y, 3))

    def test_custom_linear_named_callable_keeps_its_original_backward(self):
        def custom_linear(x):
            return x.sin() * 3

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.call_function(custom_linear, (x,))
        graph.output(y)
        gm = torch.fx.GraphModule({}, graph)
        sample = torch.randn(4, requires_grad=True)
        ShapeProp(gm).propagate(sample)
        ir = GraphLowering(gm, [sample]).run()
        node = next(
            node for node in ir.nodes if getattr(node, "target", None) is custom_linear
        )
        self.assertFalse(is_registered_native_layer_node(node))
        result = compile_fx(gm, [sample])(sample)
        result.sum().backward()
        torch.testing.assert_close(sample.grad, 3 * sample.detach().cos())

    def test_backward_extensions_are_scoped_to_the_registry(self):
        registry = BackendRegistry()
        derivative = lambda *args: ()
        registry.register_backward_rule("custom_square", derivative)
        copied = registry.copy()
        self.assertIs(copied.backward_rule("custom_square"), derivative)
        with self.assertRaises(NotImplementedError):
            BackendRegistry().backward_rule("custom_square")
        with self.assertRaises(KeyError):
            registry.register_backward_rule("mul", derivative)

    def test_strict_aot_rejects_uncovered_operator(self):
        graph = torch.fx.symbolic_trace(lambda x: torch.erf(x))
        sample = torch.randn(4, requires_grad=True)
        ShapeProp(graph).propagate(sample)
        with self.assertRaisesRegex(NotImplementedError, "autograd"):
            compile_fx(graph, [sample], config=BackendConfig(require_full_aot=True))
        structured = torch.fx.symbolic_trace(lambda x: torch.topk(x, 2))
        ShapeProp(structured).propagate(sample)
        with self.assertRaisesRegex(NotImplementedError, "autograd"):
            compile_fx(
                structured, [sample], config=BackendConfig(require_full_aot=True)
            )

    def test_cache_source_is_atomic_deterministic_and_external(self):
        from deepopt.compiler.codegen.triton.pointwise import write_module_source

        with tempfile.TemporaryDirectory() as directory:
            first = write_module_source("# source\n", directory)
            self.assertEqual(first, write_module_source("# source\n", directory))
            self.assertNotEqual(first, write_module_source("# other\n", directory))
            self.assertEqual(first.parent, Path(directory))

    def test_input_guards_prevent_using_static_metadata_for_another_layout(self):
        graph = torch.fx.symbolic_trace(lambda x: operator.mul(x, 2))
        sample = torch.randn(3, 3)
        ShapeProp(graph).propagate(sample)
        compiled = compile_fx(graph, [sample])
        with self.assertRaisesRegex(ValueError, "metadata"):
            compiled(sample.t())
        with self.assertRaisesRegex(ValueError, "metadata"):
            compiled(sample.requires_grad_())
        from deepopt.opt.quantization.activation_tensor import to_fp8_activation

        sample.requires_grad_(False)
        with self.assertRaisesRegex(TypeError, "metadata"):
            compiled(to_fp8_activation(sample, layout="last"))
        model = torch.fx.symbolic_trace(nn.Linear(3, 3))
        ShapeProp(model).propagate(sample)
        frozen = compile_fx(model, [sample])
        model.double()
        with self.assertRaisesRegex(ValueError, "Attribute"):
            frozen(sample)

    def test_registered_extension_derivative_runs_in_its_own_aot_region(self):
        from dataclasses import dataclass
        from deepopt.compiler import OperatorExtension
        from deepopt.compiler.ir import PointwiseNode
        from deepopt.compiler.autodiff.backward_rules import GradWrapper

        @dataclass
        class Square(PointwiseNode):
            pass

        def lower(context, node):
            lowered = Square(
                "custom_square",
                context.map_arg(node.args),
                context.map_arg(node.kwargs),
                context.tensor_value(node),
                node.target,
            )
            context.add_node(node, lowered)
            return lowered

        def backward(context, node, gradient):
            x = node.args[0]
            return (GradWrapper(x, gradient * 2 * context.saved(x)),)

        registry = BackendRegistry()
        registry.register(
            OperatorExtension(
                "custom_square",
                Square,
                lambda node, gm: node.target is torch.square,
                lower,
            )
        )
        registry.register_backward_rule("custom_square", backward)
        graph = torch.fx.symbolic_trace(lambda x: torch.square(x))
        x = torch.randn(4, requires_grad=True)
        ShapeProp(graph).propagate(x)
        compiled = compile_fx(graph, [x], registry=registry)
        compiled(x).sum().backward()
        torch.testing.assert_close(x.grad, 2 * x.detach())

    def test_standalone_extension_cannot_silently_drop_training_gradients(self):
        from dataclasses import dataclass
        from deepopt.compiler import OperatorExtension
        from deepopt.compiler.ir import IRNode, find_tensor_values

        @dataclass
        class Square(IRNode):
            op: str
            args: tuple
            kwargs: dict
            value: object
            target: object

            def inputs(self):
                return list(find_tensor_values((self.args, self.kwargs)))

            def outputs(self):
                return (self.value,)

        def lower(context, node):
            result = Square(
                "square_kernel",
                context.map_arg(node.args),
                context.map_arg(node.kwargs),
                context.tensor_value(node),
                node.target,
            )
            context.add_node(node, result)
            return result

        graph = torch.fx.symbolic_trace(lambda x: torch.square(x))
        sample = torch.randn(3, requires_grad=True)
        ShapeProp(graph).propagate(sample)
        for fallback in (
            None,
            lambda node, args, kwargs: torch.square(*args, **kwargs),
        ):
            registry = BackendRegistry()
            registry.register(
                OperatorExtension(
                    "square_kernel",
                    Square,
                    lambda node, gm: node.target is torch.square,
                    lower,
                    kernel_codegen=lambda node, context: torch.square,
                    native_fallback=fallback,
                )
            )
            if fallback is None:
                with self.assertRaisesRegex(NotImplementedError, "native_fallback"):
                    compile_fx(graph, [sample], registry=registry)
            else:
                compile_fx(graph, [sample], registry=registry)(sample).sum().backward()
                torch.testing.assert_close(sample.grad, 2 * sample.detach())


if __name__ == "__main__":
    unittest.main()
