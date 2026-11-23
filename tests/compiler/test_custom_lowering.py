"""Executed custom training programs, fusion legality and normalization work."""

from copy import deepcopy
from dataclasses import dataclass, replace
import operator
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import FP8Config, autocast
from deepopt.compiler import BackendConfig, BackendRegistry, CompilationReport, compile_fx, compilation_report
from deepopt.compiler.ir import IRNode, TensorValue, find_tensor_values
from deepopt.compiler.linear import DeepOptLinearNode, LinearProgram, fuse_activations
from deepopt.compiler.lowering import GraphLowering
from deepopt.compiler.registry import OperatorExtension
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model
from deepopt.opt.nn.autograd.linear import _DOLinearFunction
from deepopt.opt.policy import prepare_fp8_model


class CustomLoweringTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        torch.manual_seed(79)

    def test_lowered_training_does_not_invoke_original_forward_or_backward(self):
        for compressed in (False, True):
            for input_grad, weight_grad, bias in ((True, True, True), (False, True, True), (True, False, False)):
                with self.subTest(compressed=compressed, gradients=(input_grad, weight_grad), bias=bias):
                    torch.compiler.reset()
                    model = prepare_fp8_model(
                        nn.Linear(17, 11, bias=bias),
                        FP8Config(enabled=True, out_dtype="bf16"),
                        activation_compression={"enabled": compressed, "bits": 2, "group_size": 1024},
                    )[0]
                    model.weight.requires_grad_(weight_grad)
                    reference = deepcopy(model)
                    x = torch.randn(65, 17, requires_grad=input_grad)
                    xr = x.detach().clone().requires_grad_(input_grad)
                    expected = reference(xr)
                    expected.float().square().sum().backward()
                    compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True))
                    with (
                        patch.object(_DOLinearFunction, "forward", side_effect=AssertionError("imperative forward")),
                        patch.object(_DOLinearFunction, "_backward", side_effect=AssertionError("imperative backward")),
                        patch("deepopt.compiler.capture.deepcopy", side_effect=AssertionError("whole model copy")),
                    ):
                        actual = model(x)
                        actual.float().square().sum().backward()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
                    for a, b in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                    report = compilation_report(model)
                    self.assertEqual(len(report.graphs[0].lowered_deepopt_operations), 1)
                    self.assertEqual(report.graphs[0].compiled_deepopt_backward_regions, 1)
                    self.assertFalse(report.graphs[0].unsupported_deepopt_operations)

    def test_scope_override_and_native_delayed_recipe_have_explicit_owners(self):
        for settings, enabled in ((FP8Config(enabled=True), False), (FP8Config(enabled=True, scaling_recipe="delayed"), True)):
            torch.compiler.reset()
            model = prepare_fp8_model(nn.Sequential(nn.Linear(8, 8)), settings)[0]
            reference = deepcopy(model)
            compile_model(model, CompileConfig(enabled=True, backend="deepopt"))
            with autocast("cpu", enabled=False, fp8=settings if enabled else False):
                x = torch.randn(2, 8)
                actual, expected = model(x), reference(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            report = compilation_report(model)
            self.assertEqual(bool(report.graphs[0].lowered_deepopt_operations), enabled)
            self.assertEqual(bool(report.graphs[0].unsupported_deepopt_operations), not enabled)
            if enabled:
                self.assertGreater(report.graphs[0].native_autodiff_operations, 0)

    def test_replaced_module_recipe_is_guarded_before_generated_work(self):
        from deepopt.opt.nn.functional import _prepare_fp8_plan
        from deepopt.opt.policy import fp8_plan

        gm, sample = self.linear_graph()
        report = CompilationReport()
        compiled = compile_fx(gm, [sample], report=report)
        gm.projection.plan = _prepare_fp8_plan(fp8_plan(FP8Config(enabled=True, out_dtype="bf16")))
        actual, expected = compiled(sample), gm(sample)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(report.fallback_reasons, {"module configuration changed": 1})

    def test_native_layernorm_has_no_extra_forward_during_backward(self):
        graph = torch.fx.symbolic_trace(nn.Sequential(nn.LayerNorm(32)))
        reference = deepcopy(graph)
        sample = torch.randn(2, 3, 32, requires_grad=True)
        ShapeProp(graph).propagate(sample)
        compiled = compile_fx(graph, [sample])
        x = sample.detach().clone().requires_grad_()
        xr = sample.detach().clone().requires_grad_()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            actual = compiled(x)
            actual.square().sum().backward()
        expected = reference(xr)
        expected.square().sum().backward()
        counts = {event.key: event.count for event in profile.key_averages()}
        self.assertEqual(counts["aten::native_layer_norm"], 1)
        self.assertEqual(counts["aten::native_layer_norm_backward"], 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)

    def linear_graph(self, compressed=False, activation=None, kind="gelu"):
        model = prepare_fp8_model(
            nn.Sequential(nn.Linear(17, 17)), FP8Config(enabled=True),
            activation_compression={"enabled": compressed, "bits": 2, "group_size": 1024},
        )[0]
        root = nn.Module()
        root.projection = model[0]
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        value = graph.call_function(
            getattr(nn.functional, kind), (x,),
            {"approximate": activation or "none"} if kind == "gelu" else {},
        )
        value = graph.call_function(nn.functional.dropout, (value,), {"p": 0.0, "training": True})
        value = graph.call_module("projection", (value,))
        graph.output(value)
        gm = torch.fx.GraphModule(root, graph)
        sample = torch.randn(257, 17, requires_grad=True)
        ShapeProp(gm).propagate(sample)
        return gm, sample

    def test_fusion_crosses_identity_dropout_but_not_effects_or_extra_users(self):
        gm, sample = self.linear_graph()
        ir = GraphLowering(gm, [sample]).run()
        for node in ir.nodes:
            for value in node.outputs():
                if value.meta is not None:
                    value.meta = replace(value.meta, device=torch.device("cuda:0"))
        patterns = fuse_activations(ir, enabled=True, unsafe_values=frozenset())
        self.assertEqual(len(patterns), 1)
        linear = next(node for node in ir.nodes if isinstance(node, DeepOptLinearNode))
        self.assertEqual(linear.activation, ("gelu", "none"))
        self.assertEqual(linear.args[0].name, "x")
        ir.validate()
        gm, sample = self.linear_graph()
        ir = GraphLowering(gm, [sample]).run()
        self.assertFalse(fuse_activations(ir, enabled=True, unsafe_values=frozenset()))
        for case in ("extra_user", "unsafe", "effect", "detached"):
            with self.subTest(case=case):
                from deepopt.compiler.ir import ExternKernelNode, PointwiseNode, register_uses

                gm, sample = self.linear_graph()
                ir = GraphLowering(gm, [sample]).run()
                for node in ir.nodes:
                    for value in node.outputs():
                        if value.meta is not None:
                            value.meta = replace(value.meta, device=torch.device("cuda:0"))
                producer = next(node for node in ir.nodes if isinstance(node, PointwiseNode))
                unsafe = frozenset()
                if case == "extra_user":
                    output = ir.nodes[-1]
                    output.result = output.result, producer.value
                    register_uses(output)
                elif case == "unsafe":
                    unsafe = frozenset((producer.value,))
                elif case == "effect":
                    effect = ExternKernelNode(
                        op="rand_like", target=torch.rand_like,
                        args=producer.args, kwargs={},
                        value=TensorValue("random", producer.value.meta),
                    )
                    ir.add(effect)
                    ir.nodes.remove(effect)
                    ir.nodes.insert(ir.nodes.index(producer) + 1, effect)
                else:
                    producer.value.meta = replace(producer.value.meta, requires_grad=False)
                self.assertFalse(fuse_activations(ir, enabled=True, unsafe_values=unsafe))

    def test_fused_training_program_matches_reference_with_and_without_packing(self):
        for compressed in (False, True):
            for kind, approximate in (("gelu", "none"), ("gelu", "tanh"), ("silu", "none")):
                with self.subTest(compressed=compressed, activation=(kind, approximate)):
                    gm, sample = self.linear_graph(compressed, approximate, kind)
                    reference = deepcopy(gm)
                    ir = GraphLowering(gm, [sample]).run()
                    linear = next(node for node in ir.nodes if isinstance(node, DeepOptLinearNode))
                    producer = linear.args[0].producer
                    linear.args = (producer.args[0], *linear.args[1:])
                    linear.activation = (kind, approximate)
                    program = LinearProgram(linear)
                    x = sample.detach().clone().requires_grad_()
                    xr = sample.detach().clone().requires_grad_()
                    actual = program(x, gm.projection.weight, gm.projection.bias)
                    expected = reference(xr)
                    actual.sum().backward()
                    expected.sum().backward()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
                    for a, b in zip(gm.parameters(), reference.parameters()):
                        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                    self.assertIn("activation_input", program.saved_components)
                    self.assertIn("prepare_activation", program.source)
                    self.assertIn(f"{kind}_backward", program.source)

    def test_multiple_results_training_codegen_and_structured_outputs(self):
        def pair(x):
            return x.square(), x * 3

        class PairTraining(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return pair(x)

            @staticmethod
            def backward(ctx, first, second):
                x, = ctx.saved_tensors
                return 2 * x * first + 3 * second

        @dataclass(eq=False)
        class PairNode(IRNode):
            args: tuple
            kwargs: dict
            values: tuple
            target: object = pair

            def inputs(self):
                return list(find_tensor_values(self.args))

            def outputs(self):
                return self.values

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        value = graph.call_function(pair, (x,))
        left = graph.call_function(operator.getitem, (value, 0))
        right = graph.call_function(operator.getitem, (value, 1))
        graph.output({"left": left, "right": right})
        gm = torch.fx.GraphModule(nn.Module(), graph)
        sample = torch.randn(5, requires_grad=True)
        ShapeProp(gm).propagate(sample)
        registry = BackendRegistry()

        def lower(context, fx_node):
            outputs = context.tensor_values(fx_node)
            node = PairNode(context.map_arg(fx_node.args), {}, outputs)
            context.add_node(fx_node, node)
            return node

        registry.register(OperatorExtension(
            name="pair", node_type=PairNode,
            matcher=lambda node, gm: node.target is pair,
            lowering=lower, kernel_codegen=lambda node, context: pair,
            training_codegen=lambda node, context: PairTraining.apply, pure=False,
        ))
        # A custom training callback alone no longer proves native AutoDiff
        # ownership. Register the inspectable multi-result VJP explicitly.
        with self.assertRaisesRegex(NotImplementedError, "coverage incomplete"):
            compile_fx(gm, [sample], registry=registry, config=BackendConfig(require_full_aot=True))

        def derivative(ctx, node, gradients):
            from deepopt.compiler.autodiff.backward_rules import GradWrapper

            first, second = gradients
            return (GradWrapper(node.args[0], 2 * ctx.saved(node.args[0]) * first + 3 * second),)

        registry.register_backward_rule("pair", derivative, multi_output=True)
        compiled = compile_fx(gm, [sample], registry=registry, config=BackendConfig(require_full_aot=True))
        output = compiled(sample)
        (output["left"].sum() + output["right"].sum()).backward()
        torch.testing.assert_close(sample.grad, 2 * sample.detach() + 3)
        self.assertEqual(len(compiled.backend_artifact.custom_kernels), 1)


if __name__ == "__main__":
    unittest.main()
