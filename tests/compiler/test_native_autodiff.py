"""Native derivative provenance, state exports and graph-level execution."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.compiler.autodiff import AOTAutoDiff
from deepopt.compiler.autodiff.backward_rules import GradWrapper
from deepopt.compiler.ir import CallNode, TensorValue
from deepopt.compiler.registry import BackendRegistry
from deepopt.compiler.trace.builder import GraphBuilder
from deepopt.compiler.ir import IRGraph, TensorMeta


class NativeAutoDiffTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
    def test_multi_output_vjp_unused_state_and_repeated_outputs(self):
        graph = IRGraph()
        builder = GraphBuilder(graph)
        meta = TensorMeta((3,), torch.float64, "cpu", (1,), True)
        x = builder.input(0, meta)
        a, b = TensorValue("a", meta), TensorValue("b", meta)
        graph.add(CallNode("two_results", lambda t: (t * 2, t * 3), (x.value,), {}, (a, b)))
        builder.output((a, a, b))
        registry = BackendRegistry()

        def vjp(ctx, node, cotangents):
            ga, gb = cotangents
            return (GradWrapper(node.args[0], ga * 2 + gb * 3),)

        registry.register_backward_rule("two_results", vjp, multi_output=True)
        trace = AOTAutoDiff(graph, registry=registry).diff()
        self.assertEqual(len(trace.grad_output_source), 3)
        from deepopt.compiler.scheduler import Scheduler
        from deepopt.compiler.codegen.wrapper import WrapperCodegen

        executable = WrapperCodegen(nn.Module(), Scheduler(trace.graph, registry=registry).run(),
                                    registry=registry, enable_autodiff=False).run().fn
        result, = executable(torch.ones(3, dtype=torch.float64),
                             torch.ones(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64))
        torch.testing.assert_close(result, torch.full((3,), 7.0, dtype=torch.float64))

    def test_ir_calls_use_identity_when_inserting_saved_components(self):
        meta = TensorMeta((2,), torch.float32, "cpu", (1,), False)
        first = CallNode("first", torch.neg, (), {}, (TensorValue("first", meta),))
        second = CallNode("second", torch.neg, (), {}, (TensorValue("second", meta),))
        self.assertIsNot(first, second)
        self.assertNotEqual(first, second)
        values = [first, second]
        values.remove(second)
        self.assertEqual(values, [first])

    def test_bf16_graph_uses_native_diff_without_legacy_programs(self):
        torch.manual_seed(121)
        graph = torch.fx.symbolic_trace(nn.Sequential(nn.Linear(7, 11), nn.GELU(), nn.Linear(11, 3)).bfloat16())
        reference = deepcopy(graph)
        x = torch.randn(2, 4, 7, dtype=torch.bfloat16, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        ShapeProp(graph).propagate(x)
        report = CompilationReport()
        with patch("deepopt.compiler.native_linear._BF16Function.forward", side_effect=AssertionError("legacy")), \
             patch("deepopt.compiler.native_linear.NativeLinearProgram.backward", side_effect=AssertionError("legacy")):
            compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True), report=report)
            actual = compiled(x)
            expected = reference(xr)
            gradient = torch.randn_like(actual)
            actual.backward(gradient)
            expected.backward(gradient)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
        for a, b in zip(graph.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        self.assertGreater(report.graphs[0].native_autodiff_operations, 0)
        self.assertEqual(report.graphs[0].legacy_backward_operations, ())
        self.assertEqual(len(compiled.backend_artifact.joint_autograd_regions), 1)

    def test_compressed_amp_stateful_graph_avoids_legacy_backward_helpers(self):
        from deepopt import autocast
        from deepopt.config import CompileConfig
        from deepopt.compiler import compilation_report
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model

        for bits in (1, 2, 4, 8):
            torch.compiler.reset()
            torch.manual_seed(132)
            compression = {"enabled": True, "bits": bits, "group_size": 1024}
            model = prepare_fp8_model(nn.Sequential(nn.Linear(7, 1024), nn.SiLU(), nn.LayerNorm(1024), nn.Linear(1024, 3)),
                                      False, activation_compression=compression, config={})[0]
            reference = deepcopy(model)
            x = torch.randn(2, 4, 7, requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=compression, config={}):
                expected = reference(xr)
            gradient = torch.randn_like(expected)
            expected.backward(gradient)
            compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                                               options={"require_full_aot": True}))
            with patch("deepopt.compiler.native_linear.NativeLinearProgram.backward", side_effect=AssertionError("legacy")), \
                 patch("deepopt.opt.nn.autograd.activation.elementwise.activation_backward", side_effect=AssertionError("legacy")), \
                 patch("deepopt.opt.nn.autograd.normalization.layer.layer_norm_backward", side_effect=AssertionError("legacy")):
                with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=compression, config={}):
                    actual = model(x)
                actual.backward(gradient)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            evidence = compilation_report(model).graphs[0]
            self.assertEqual(evidence.native_autodiff_regions, 1)
            self.assertFalse(evidence.eager_autograd_operations)
            self.assertFalse(evidence.legacy_backward_operations)

    def test_fp8_linear_uses_native_vjp_and_compressed_state(self):
        from deepopt import FP8Config
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model

        for compressed in (False, True):
            torch.compiler.reset()
            model = prepare_fp8_model(nn.Linear(13, 7), FP8Config(enabled=True),
                activation_compression={"enabled": compressed, "group_size": 1024})[0]
            reference = deepcopy(model)
            x = torch.randn(80, 13, requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            expected = reference(xr)
            expected.square().sum().backward()
            compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                options={"require_full_aot": True}))
            with patch("deepopt.compiler.linear._CompiledLinearFunction.forward", side_effect=AssertionError("legacy")), \
                 patch("deepopt.opt.nn.autograd.linear._DOLinearFunction.backward", side_effect=AssertionError("legacy")):
                actual = model(x)
                actual.square().sum().backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            self.assertFalse(compilation_report(model).graphs[0].legacy_backward_operations)

    def test_native_geometry_matrix_rules_and_unused_multi_results(self):
        class Model(nn.Module):
            def forward(self, x, w):
                a, b, c = (x @ w).transpose(0, 1).chunk(3, dim=0)
                return torch.cat((a.sin(), c.cos()), dim=0).mean(dim=-1)

        graph = torch.fx.symbolic_trace(Model())
        x, w = torch.randn(4, 5, dtype=torch.float64, requires_grad=True), torch.randn(5, 9, dtype=torch.float64, requires_grad=True)
        xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
        ShapeProp(graph).propagate(x, w)
        compiled = compile_fx(graph, [x, w], config=BackendConfig(require_full_aot=True))
        actual, expected = compiled(x, w), graph(xr, wr)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, xr.grad)
        torch.testing.assert_close(w.grad, wr.grad)

    def test_shared_parameters_interleaved_forwards_and_retained_graph(self):
        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 5).bfloat16()

            def forward(self, x):
                return self.linear(x).sigmoid() + self.linear(x * 2)

        graph = torch.fx.symbolic_trace(Shared())
        reference = deepcopy(graph)
        x = torch.randn(3, 5, dtype=torch.bfloat16, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        ShapeProp(graph).propagate(x)
        compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True))
        a, b = compiled(x), compiled(x * 3)
        ar, br = reference(xr), reference(xr * 3)
        a.sum().backward(retain_graph=True)
        ar.sum().backward(retain_graph=True)
        (a.sum() + b.sum()).backward()
        (ar.sum() + br.sum()).backward()
        # Region aggregation changes the legal accumulation order at shared
        # BF16 leaves. Check within the dtype's established numerical budget.
        torch.testing.assert_close(x.grad, xr.grad)
        for left, right in zip(graph.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad)

    def test_attention_native_vjp_preserves_backend_and_qkv_compression(self):
        from deepopt import autocast
        from deepopt.opt.nn.functional import do_scaled_dot_product_attention

        for compressed in (False, True):
            for math_backend in (False, True):
                with self.subTest(compressed=compressed, math=math_backend):
                    torch.manual_seed(311)
                    compression = {"enabled": compressed, "bits": 2, "group_size": 1024}
                    graph = torch.fx.Graph()
                    args = tuple(graph.placeholder(name) for name in ("q", "k", "v"))
                    graph.output(graph.call_function(do_scaled_dot_product_attention, args,
                        {"plan": {"fp8": False, "activation_compression": compression}}))
                    gm = torch.fx.GraphModule(nn.Module(), graph)
                    inputs = tuple(torch.randn(2, 3, length, 8, dtype=torch.bfloat16, requires_grad=True)
                                   for length in (23, 25, 25))
                    refs = tuple(value.detach().clone().requires_grad_() for value in inputs)
                    from torch.nn.attention import sdpa_kernel, SDPBackend
                    backends = [SDPBackend.MATH] if math_backend else [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]
                    with sdpa_kernel(backends):
                        with autocast("cpu", fp8=False, activation_compression=compression, config={}):
                            ShapeProp(gm).propagate(*inputs)
                            expected = gm(*refs)
                            gradient = torch.randn_like(expected)
                            compiled = compile_fx(gm, inputs, config=BackendConfig(require_full_aot=True))
                            actual = compiled(*inputs)
                        expected.backward(gradient)
                        with patch("torch.autograd.grad", side_effect=AssertionError("delegated derivative")):
                            actual.backward(gradient)
                    torch.testing.assert_close(actual, expected)
                    for value, ref in zip(inputs, refs):
                        torch.testing.assert_close(value.grad, ref.grad)

    def test_native_spatial_families_use_registered_derivatives(self):
        from deepopt import autocast
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model

        cases = [(nn.Conv1d(3, 4, 3, padding=1), (2, 3, 175)),
                 (nn.Conv2d(3, 4, 3, padding=1), (2, 3, 32, 32)),
                 (nn.ConvTranspose2d(3, 4, 3, stride=2), (2, 3, 16, 16)),
                 (nn.GroupNorm(1, 3), (2, 3, 32, 32)), (nn.BatchNorm2d(3), (2, 3, 32, 32)),
                 (nn.MaxPool2d(2), (2, 3, 32, 32)), (nn.AvgPool2d(2), (2, 3, 32, 32)),
                 (nn.AdaptiveAvgPool2d(1), (2, 3, 32, 32)), (nn.LPPool2d(2, 2), (2, 3, 32, 32)),
                 (nn.ReLU(), (2, 3, 32, 32))]
        for layer, shape in cases:
            with self.subTest(layer=type(layer).__name__):
                torch.compiler.reset()
                compression = {"enabled": True, "bits": 4, "group_size": 1024}
                model = prepare_fp8_model(nn.Sequential(layer), False, activation_compression=compression, config={})[0]
                reference = deepcopy(model)
                x = torch.randn(shape, requires_grad=True)
                xr = x.detach().clone().requires_grad_()
                with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=compression, config={}):
                    expected = reference(xr)
                gradient = torch.randn_like(expected)
                expected.backward(gradient)
                compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                    options={"require_full_aot": True}))
                with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=compression, config={}):
                    actual = model(x)
                actual.backward(gradient)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(x.grad, xr.grad)
                for a, b in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(a.grad, b.grad)
                for (_, a), (_, b) in zip(model.named_buffers(), reference.named_buffers()):
                    torch.testing.assert_close(a, b)
                evidence = compilation_report(model).graphs[0]
                self.assertGreater(evidence.native_autodiff_operations, 0)
                self.assertFalse(evidence.eager_autograd_operations)

    def test_fp8_blockwise_and_delayed_recipes_have_native_vjps(self):
        from deepopt import FP8Config, autocast
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model

        for recipe, blocks in (("current", (1, 4)), ("current", (2, 3)), ("delayed", None)):
            for compressed in (False, True):
                with self.subTest(recipe=recipe, blocks=blocks, compressed=compressed):
                    torch.compiler.reset()
                    settings = FP8Config(enabled=True, scaling_recipe=recipe,
                        block_shapes=dict.fromkeys(("activation", "weight", "grad"), blocks))
                    model = prepare_fp8_model(nn.Linear(13, 7), settings,
                        activation_compression={"enabled": compressed, "bits": 4, "group_size": 1024})[0]
                    reference = deepcopy(model)
                    compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                        options={"require_full_aot": True}))
                    for _ in range(2):
                        model.zero_grad(set_to_none=True)
                        reference.zero_grad(set_to_none=True)
                        x = torch.randn(80, 13, requires_grad=True)
                        xr = x.detach().clone().requires_grad_()
                        with autocast("cpu", enabled=False, fp8=settings):
                            expected = reference(xr)
                            actual = model(x)
                        gradient = torch.randn_like(expected)
                        expected.backward(gradient)
                        with patch("deepopt.opt.nn.autograd.linear._DOLinearFunction._backward", side_effect=AssertionError("legacy")):
                            actual.backward(gradient)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
                        for a, b in zip(model.parameters(), reference.parameters()):
                            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                        for (_, a), (_, b) in zip(model.named_buffers(), reference.named_buffers()):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        with torch.no_grad():
                            for a, b in zip(model.parameters(), reference.parameters()):
                                a.add_(a.grad, alpha=-0.01)
                                b.add_(b.grad, alpha=-0.01)
                    evidence = compilation_report(model).graphs[0]
                    self.assertFalse(evidence.eager_autograd_operations)
                    self.assertFalse(evidence.legacy_backward_operations)

    def test_native_fp8_gradient_transport_is_reused_between_linears(self):
        from deepopt import FP8Config, autocast
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model

        for recipe in ("current", "delayed"):
            with self.subTest(recipe=recipe):
                torch.compiler.reset()
                settings = FP8Config(enabled=True, scaling_recipe=recipe, backward_precision="auto")
                model = prepare_fp8_model(nn.Sequential(nn.Linear(7, 11), nn.Linear(11, 3)), settings,
                    activation_compression={"enabled": True, "bits": 4, "group_size": 1024})[0]
                reference = deepcopy(model)
                compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                    options={"require_full_aot": True}))
                x = torch.randn(147, 7, requires_grad=True)
                xr = x.detach().clone().requires_grad_()
                with autocast("cpu", enabled=False, fp8=settings):
                    expected = reference(xr)
                    actual = model(x)
                gradient = torch.randn_like(expected)
                expected.backward(gradient)
                with patch("deepopt.opt.nn.autograd.linear._DOLinearFunction._backward", side_effect=AssertionError("legacy")):
                    actual.backward(gradient)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
                for a, b in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                slots = model._deepopt_gradient_edges.values()
                self.assertTrue(any(slot.last_input_execution and slot.last_input_execution["reused_fp8"] for slot in slots))
                self.assertFalse(compilation_report(model).graphs[0].legacy_backward_operations)

    def test_mha_composite_has_native_derivatives_and_codec_saves(self):
        from deepopt import FP8Config, autocast
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model

        class Attention(nn.Module):
            def __init__(self, weights):
                super().__init__()
                self.attention = nn.MultiheadAttention(8, 2, batch_first=True)
                self.weights = weights

            def forward(self, x):
                return self.attention(x, x, x, need_weights=self.weights)

        for fp8 in (False, True):
            for weights in (False, True):
                with self.subTest(fp8=fp8, weights=weights):
                    torch.compiler.reset()
                    torch.manual_seed(378)
                    compression = {"enabled": True, "bits": 4, "group_size": 1024}
                    settings = FP8Config(enabled=True) if fp8 else False
                    model = prepare_fp8_model(Attention(weights), settings, activation_compression=compression, config={})[0]
                    reference = deepcopy(model)
                    compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                        options={"require_full_aot": True}))
                    x = torch.randn(2, 65, 8, requires_grad=True)
                    xr = x.detach().clone().requires_grad_()
                    with autocast("cpu", dtype=torch.bfloat16, fp8=settings, activation_compression=compression, config={}):
                        expected = reference(xr)
                        actual = model(x)
                    def loss(values):
                        return values[0].float().square().sum() + (values[1].float().square().sum() if values[1] is not None else 0)
                    loss(expected).backward()
                    with patch("deepopt.opt.nn.autograd.linear._DOLinearFunction._backward", side_effect=AssertionError("legacy")):
                        loss(actual).backward()
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(x.grad, xr.grad)
                    for a, b in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(a.grad, b.grad)
                    evidence = compilation_report(model).graphs[0]
                    self.assertFalse(evidence.eager_autograd_operations)
                    self.assertFalse(evidence.legacy_backward_operations)
                    self.assertGreater(evidence.native_autodiff_operations, 5)

    def test_saved_codec_payload_releases_before_earlier_layer_backward(self):
        import weakref
        from deepopt import autocast
        from deepopt.compiler.autodiff.semantic import BF16Linear

        graph = torch.fx.symbolic_trace(nn.Sequential(nn.Linear(7, 29), nn.GELU(), nn.Linear(29, 3)).bfloat16())
        x = torch.randn(2, 80, 7, dtype=torch.bfloat16, requires_grad=True)
        compression = {"enabled": True, "bits": 4, "group_size": 1024}
        last_payload = []
        observed = []
        class Saved:
            def __init__(self, value):
                self.value = value
        def pack(value):
            if value.dtype == torch.int32 and tuple(value.shape) == (5, 128):
                last_payload.append(weakref.ref(value))
            return Saved(value)
        def unpack(saved):
            return saved.value
        original = BF16Linear.dw
        def checked_dw(spec, *args):
            if spec.module_path == "0":
                observed.append(True)
                self.assertTrue(last_payload)
                self.assertIsNone(last_payload[0]())
            return original(spec, *args)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=compression, config={}):
            ShapeProp(graph).propagate(x)
            with patch.object(BF16Linear, "dw", checked_dw):
                compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True))
                with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                    y = compiled(x)
        y.sum().backward()
        self.assertTrue(observed)

    def test_derived_fusion_preserves_packed_state_and_bf16_rounding(self):
        from dataclasses import replace
        from deepopt.compiler.lowering import GraphLowering
        from deepopt.compiler.autodiff.joint import _clone_region
        from deepopt.compiler.autodiff.semantic import register_rules
        from deepopt.compiler.autodiff.optimization import forward, backward, backward_layouts
        from deepopt.compiler.ir import InputNode, GetAttrNode, OutputNode, register_uses
        from deepopt.compiler.registry import KernelCodegenContext
        from deepopt.compiler.scheduler import Scheduler
        from deepopt.compiler.codegen.wrapper import WrapperCodegen
        from deepopt.kernel_policy import KernelPolicy
        from deepopt.opt.policy import prepare_fp8_model

        model = prepare_fp8_model(nn.Sequential(nn.Linear(7, 11), nn.GELU(), nn.Linear(11, 3)).bfloat16(), False,
            activation_compression={"enabled": True, "bits": 4, "group_size": 1024}, config={})[0]
        reference = deepcopy(model)
        fx = torch.fx.Graph()
        source = fx.placeholder("x")
        value = source
        for name in ("0", "1", "2"):
            value = fx.call_module(name, (value,))
        fx.output(value)
        graph = torch.fx.GraphModule(model, fx)
        x = torch.randn(2, 80, 7, dtype=torch.bfloat16, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        ShapeProp(graph).propagate(x)
        registry = BackendRegistry()
        register_rules(registry)
        ir = GraphLowering(graph, [x], registry=registry).run()
        for node in ir.nodes:
            for value in node.outputs():
                value.meta = replace(value.meta, device=torch.device("cuda:0"))
        nodes = [node for node in ir.nodes if not isinstance(node, (InputNode, GetAttrNode, OutputNode))]
        context = KernelCodegenContext(kernel_policy=KernelPolicy(), bf16_gemm_backend="generated", fuse_deepopt_activations=True)
        local, inputs, _, _ = _clone_region(nodes, context, registry)
        trace = AOTAutoDiff(local, registry=registry).diff()
        output_node = local.nodes[-1]
        output_node.result = (output_node.result, trace.saved_tensors)
        register_uses(output_node)
        _, patterns = forward(local, context)
        patterns += backward(trace.graph)
        self.assertIn("bf16_gelu_producer_pack", patterns)
        self.assertIn("bf16_dx_packed_derivative_epilogue", patterns)
        # Exercise the selected numerical schedule using the CPU semantic
        # control. This validates bindings/rounding, not CUDA compilation.
        for ir_graph in (local, trace.graph):
            for node in ir_graph.nodes:
                for value in node.outputs():
                    if value.meta is not None:
                        value.meta = replace(value.meta, device=torch.device("cpu"))
        backward_layouts(trace.graph)
        fwd = WrapperCodegen(graph, Scheduler(local, registry=registry).run(), registry=registry, enable_autodiff=False).run().fn
        bwd = WrapperCodegen(graph, Scheduler(trace.graph, registry=registry).run(), registry=registry, enable_autodiff=False).run().fn
        runtime = {}
        for node in ir.nodes:
            if isinstance(node, InputNode):
                runtime[node.value] = x
            elif isinstance(node, GetAttrNode):
                owner = graph
                for component in node.target.split("."):
                    owner = getattr(owner, component)
                runtime[node.value] = owner
        with torch.no_grad():
            (actual,), saved = fwd(*(runtime[value] for value in inputs))
        expected = reference(xr)
        gradient = torch.randn_like(expected)
        expected.backward(gradient)
        with torch.no_grad():
            gradients = bwd(gradient, *saved)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for target, actual_gradient in zip(trace.grad_targets, gradients):
            source_value = runtime[inputs[int(target.key)]]
            if source_value is x:
                expected_gradient = xr.grad
            else:
                parameter_name = next(name for name, parameter in graph.named_parameters() if parameter is source_value)
                expected_gradient = reference.get_parameter(parameter_name).grad
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

    def test_functional_spatial_and_normalization_adapters_have_native_coverage(self):
        from deepopt import FP8Config, autocast
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model

        class Functional(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 3, 3, 3) / 10)
                self.bias = nn.Parameter(torch.randn(4) / 10)
                self.gamma, self.beta = nn.Parameter(torch.ones(4)), nn.Parameter(torch.zeros(4))
                self.output = nn.Parameter(torch.randn(2, 4) / 10)

            def forward(self, x):
                x = nn.functional.conv2d(x, self.weight, self.bias, padding=1)
                x = nn.functional.group_norm(x, 2, self.gamma, self.beta)
                x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
                return nn.functional.linear(x, self.output)

        settings = FP8Config(enabled=True, trace_mode="full", backward_precision="auto")
        compression = {"enabled": True, "bits": 4, "group_size": 1024}
        model = prepare_fp8_model(Functional(), settings, activation_compression=compression)[0]
        reference = deepcopy(model)
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
            options={"require_full_aot": True}))
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        with autocast("cpu", enabled=False, fp8=settings, activation_compression=compression):
            expected, actual = reference(xr), model(x)
        expected.square().sum().backward()
        with patch("deepopt.opt.nn.autograd.convolution.fp8._FP8ConvolutionFunction._backward", side_effect=AssertionError("legacy")), \
             patch("deepopt.opt.nn.autograd.normalization.function.NormalizationFunction.backward", side_effect=AssertionError("legacy")):
            actual.square().sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        report = compilation_report(model)
        self.assertTrue(all(not graph.legacy_backward_operations and not graph.eager_autograd_operations for graph in report.graphs))

    def test_fullgraph_diagnostics_use_compiler_module_names_without_hooks(self):
        from deepopt import FP8Config, autocast
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model
        from deepopt.opt.backends._shared.gemm_diagnostics import record_gemms

        settings = FP8Config(enabled=True)
        model = prepare_fp8_model(nn.Sequential(nn.Linear(7, 11), nn.SiLU(), nn.Linear(11, 3)), settings)[0]
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
            options={"require_full_aot": True}))
        x = torch.randn(2, 7, requires_grad=True)
        with record_gemms(model) as rows:
            self.assertFalse(any(module._forward_pre_hooks for module in model.modules()))
            with autocast("cpu", enabled=False, fp8=settings):
                value = model(x)
        value.sum().backward()
        self.assertEqual({row["module"] for row in rows}, {"0", "2"})
        self.assertEqual({row["role"] for row in rows}, {"forward", "dx", "dw"})

    def test_convolution_padding_modes_have_native_adjoints(self):
        from deepopt import FP8Config, autocast
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model

        for fp8 in (False, True):
            for mode in ("same", "reflect", "replicate", "circular"):
                with self.subTest(fp8=fp8, mode=mode):
                    torch.compiler.reset()
                    layer = (nn.Conv2d(3, 4, 2, padding="same") if mode == "same" else
                             nn.Conv2d(3, 4, 3, padding=1, padding_mode=mode))
                    compression = {"enabled": True, "bits": 4, "group_size": 1024}
                    settings = FP8Config(enabled=True) if fp8 else False
                    model = prepare_fp8_model(nn.Sequential(layer), settings, activation_compression=compression, config={})[0]
                    reference = deepcopy(model)
                    compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                        options={"require_full_aot": True}))
                    x = torch.randn(2, 3, 32, 32, requires_grad=True)
                    xr = x.detach().clone().requires_grad_()
                    with autocast("cpu", dtype=torch.bfloat16, fp8=settings, activation_compression=compression, config={}):
                        expected, actual = reference(xr), model(x)
                    gradient = torch.randn_like(expected)
                    expected.backward(gradient)
                    with patch("deepopt.opt.nn.autograd.convolution.fp8._FP8ConvolutionFunction._backward", side_effect=AssertionError("legacy")):
                        actual.backward(gradient)
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
                    torch.testing.assert_close(x.grad, xr.grad, rtol=0.02, atol=0.002)
                    for a, b in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(a.grad, b.grad, rtol=0.02, atol=0.002)

    def test_strided_activation_and_normalization_stay_native(self):
        from deepopt import autocast
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model

        compression = {"enabled": True, "bits": 4, "group_size": 1024}
        model = prepare_fp8_model(nn.Sequential(nn.SiLU(), nn.LayerNorm(1024), nn.Linear(1024, 3)), False,
                                  activation_compression=compression, config={})[0]
        reference = deepcopy(model)
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
            options={"require_full_aot": True}))
        x = torch.randn(2, 4, 1024).transpose(0, 1).requires_grad_()
        xr = x.detach().clone().requires_grad_()
        self.assertFalse(x.is_contiguous())
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=compression, config={}):
            expected, actual = reference(xr), model(x)
        gradient = torch.randn_like(expected)
        expected.backward(gradient)
        actual.backward(gradient)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
        self.assertFalse(compilation_report(model).graphs[0].eager_autograd_operations)

    def test_keyword_primitives_tensor_exponents_and_duplicate_indices(self):
        class Primitive(nn.Module):
            def forward(self, x, weight, exponent, indices):
                values = torch.mm(input=x, mat2=weight)
                values = torch.sigmoid(input=torch.add(input=values, other=0.25))
                values = torch.pow(input=values, exponent=exponent)
                return values[indices].mean()

        graph = torch.fx.symbolic_trace(Primitive())
        x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        exponent = torch.randn(5, dtype=torch.float64).exp().requires_grad_()
        indices = torch.tensor([0, 0, 3], dtype=torch.int64)
        xr, wr, er = (value.detach().clone().requires_grad_() for value in (x, weight, exponent))
        ShapeProp(graph).propagate(x, weight, exponent, indices)
        compiled = compile_fx(graph, [x, weight, exponent, indices], config=BackendConfig(require_full_aot=True))
        actual, expected = compiled(x, weight, exponent, indices), graph(xr, wr, er, indices)
        actual.backward()
        expected.backward()
        for a, b in ((x, xr), (weight, wr), (exponent, er)):
            torch.testing.assert_close(a.grad, b.grad)

    def test_attention_bias_gradients_gqa_and_dropout_use_native_rules(self):
        from torch.nn.attention import sdpa_kernel, SDPBackend

        class Attention(nn.Module):
            def forward(self, q, k, v, bias):
                return nn.functional.scaled_dot_product_attention(q, k, v, bias, dropout_p=0.2,
                                                                 scale=-0.3, enable_gqa=True)

        graph = torch.fx.symbolic_trace(Attention())
        inputs = (torch.randn(2, 4, 3, 8, dtype=torch.float64, requires_grad=True),
                  torch.randn(2, 2, 5, 8, dtype=torch.float64, requires_grad=True),
                  torch.randn(2, 2, 5, 6, dtype=torch.float64, requires_grad=True),
                  torch.randn(1, 1, 3, 5, dtype=torch.float64, requires_grad=True))
        refs = tuple(value.detach().clone().requires_grad_() for value in inputs)
        with sdpa_kernel(SDPBackend.MATH):
            ShapeProp(graph).propagate(*inputs)
            compiled = compile_fx(graph, inputs, config=BackendConfig(require_full_aot=True))
            torch.manual_seed(712)
            expected = graph(*refs)
            torch.manual_seed(712)
            actual = compiled(*inputs)
            gradient = torch.randn_like(expected)
            expected.backward(gradient)
            actual.backward(gradient)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(inputs, refs):
            torch.testing.assert_close(a.grad, b.grad)

    def test_empty_projection_work_and_fp8_validation_are_preserved(self):
        from deepopt import FP8Config, autocast
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model

        for fp8 in (False, True):
            for batch, width, output in ((0, 7, 3), (2, 0, 3), (2, 7, 0)):
                with self.subTest(fp8=fp8, shape=(batch, width, output)):
                    torch.compiler.reset()
                    settings = FP8Config(enabled=True) if fp8 else False
                    model = prepare_fp8_model(nn.Sequential(nn.Linear(width, output)), settings,
                        activation_compression={"enabled": True, "bits": 4, "group_size": 1024})[0]
                    reference = deepcopy(model)
                    compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                        options={"require_full_aot": True}))
                    x = torch.randn(batch, width, requires_grad=True)
                    xr = x.detach().clone().requires_grad_()
                    with autocast("cpu", dtype=torch.bfloat16, fp8=settings, activation_compression={"enabled": True, "group_size": 1024}):
                        if fp8:
                            with self.assertRaisesRegex(ValueError, "non-empty"):
                                reference(xr)
                            with self.assertRaisesRegex(Exception, "non-empty"):
                                model(x)
                            continue
                        expected, actual = reference(xr), model(x)
                    expected.sum().backward()
                    actual.sum().backward()
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(x.grad, xr.grad)
                    for a, b in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(a.grad, b.grad)
                    report = compilation_report(model).graphs[0]
                    self.assertFalse(report.legacy_backward_operations or report.eager_autograd_operations)

    def test_mha_differentiable_masks_keep_native_rounding_boundaries(self):
        from deepopt import FP8Config, autocast
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model

        class Masked(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiheadAttention(8, 2, batch_first=True)
                self.bias = nn.Parameter(torch.randn(65, 65) / 10)

            def forward(self, x):
                return self.attention(x, x, x, attn_mask=self.bias, need_weights=True)

        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                torch.compiler.reset()
                settings = FP8Config(enabled=True) if fp8 else False
                model = prepare_fp8_model(Masked(), settings, activation_compression={"enabled": True, "group_size": 1024})[0]
                reference = deepcopy(model)
                compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                    options={"require_full_aot": True}))
                x = torch.randn(2, 65, 8, requires_grad=True)
                xr = x.detach().clone().requires_grad_()
                with autocast("cpu", dtype=torch.bfloat16, fp8=settings, activation_compression={"enabled": True, "group_size": 1024}):
                    expected, actual = reference(xr), model(x)
                (expected[0].float().square().sum() + expected[1].float().square().sum()).backward()
                (actual[0].float().square().sum() + actual[1].float().square().sum()).backward()
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(model.bias.grad, reference.bias.grad, rtol=0.02, atol=0.002)
                torch.testing.assert_close(x.grad, xr.grad, rtol=0.02, atol=0.002)

    def test_native_activation_dx_epilogue_matches_unfused_backward(self):
        from deepopt.compiler.autodiff.semantic import BF16Linear, recorded
        from deepopt.compiler.autodiff.optimization import backward
        from deepopt.compiler.codegen.wrapper import WrapperCodegen
        from deepopt.compiler.native_linear import NativeLinearNode
        from deepopt.compiler.registry import KernelCodegenContext
        from deepopt.compiler.scheduler import Scheduler
        from deepopt.kernel_policy import KernelPolicy
        from deepopt.opt.nn.functional import _prepare_linear_plan

        xm = TensorMeta((4, 5), torch.bfloat16, "cuda:0", (5, 1), True)
        wm = TensorMeta((3, 5), torch.bfloat16, "cuda:0", (5, 1), True)
        ym = TensorMeta((4, 3), torch.bfloat16, "cuda:0", (3, 1), True)
        node = NativeLinearNode("bf16_linear", (TensorValue("x", xm), TensorValue("w", wm), None), {},
            TensorValue("y", ym), _prepare_linear_plan({"fp8": False}), "linear", "linear", True)
        spec = BF16Linear(node, KernelCodegenContext(kernel_policy=KernelPolicy(), bf16_gemm_backend="generated"))
        graph = IRGraph()
        builder = GraphBuilder(graph)
        from dataclasses import replace

        gradient = builder.input(0, replace(ym, device="cpu", requires_grad=False))
        weight = builder.input(1, replace(wm, device="cpu", requires_grad=False))
        value = builder.input(2, replace(xm, device="cpu", requires_grad=False))
        dx = builder.emit_native("bf16.gemm_dx", recorded(spec, spec.dx, "backward"), (gradient, weight), {}, value.value.meta)
        output = builder.emit("gelu_backward", dx, value, approximate="none")
        builder.output(output)
        registry = BackendRegistry()
        unfused = WrapperCodegen(nn.Module(), Scheduler(graph, registry=registry).run(), registry=registry, enable_autodiff=False).run().fn
        patterns = backward(graph)
        self.assertEqual(patterns, ("bf16_dx_gelu_epilogue",))
        fused = WrapperCodegen(nn.Module(), Scheduler(graph, registry=registry).run(), registry=registry, enable_autodiff=False).run().fn
        inputs = (torch.randn(4, 3, dtype=torch.bfloat16), torch.randn(3, 5, dtype=torch.bfloat16), torch.randn(4, 5, dtype=torch.bfloat16))
        with torch.no_grad():
            torch.testing.assert_close(fused(*inputs), unfused(*inputs), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
