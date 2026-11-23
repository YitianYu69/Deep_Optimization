"""Exact clamp state and activation-free nearest resize for the VAE path."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F

from deepopt.compiler import BackendConfig, CompilationReport, compile_fx


class _Function(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class NativeVAEPrimitiveTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def compile(self, model, x):
        graph = torch.fx.symbolic_trace(model)
        ShapeProp(graph).propagate(x)
        report = CompilationReport()
        compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True), report=report)
        self.assertFalse(report.graphs[0].eager_autograd_operations)
        self.assertFalse(report.graphs[0].legacy_backward_operations)
        return compiled, graph

    def test_scalar_clamp_forms_preserve_native_boundaries_and_nonfinite_gradients(self):
        forms = (
            lambda x: x.clamp(-20., 10.),
            lambda x: torch.clamp(input=x, min=-20., max=10.),
            lambda x: x.clip(-20., 10.),
            lambda x: torch.clip(x, -20., 10.),
            lambda x: torch.ops.aten.clamp.default(x, -20., 10.),
            lambda x: x.clamp(min=-20.),
            lambda x: torch.clamp_max(x, 10.),
            lambda x: torch.ops.aten.clamp_min.default(x, -20.),
            lambda x: x.clamp(10., 10.),
            lambda x: x.clamp(10., -20.),
        )
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            values = torch.tensor([-float("inf"), -21, -20, -19, 0, 9, 10, 11,
                                   float("inf"), float("nan")], dtype=dtype)
            for function in forms:
                for nonfinite in (False, True):
                    with self.subTest(dtype=dtype, function=function, nonfinite=nonfinite):
                        x = values.clone().requires_grad_()
                        reference = x.detach().clone().requires_grad_()
                        compiled, graph = self.compile(_Function(function), x)
                        saved = []
                        with torch.autograd.graph.saved_tensors_hooks(
                                lambda value: saved.append(value) or value, lambda value: value):
                            actual = compiled(x)
                        expected = graph(reference)
                        gradient = torch.full_like(actual, float("nan")) if nonfinite else torch.ones_like(actual)
                        actual.backward(gradient)
                        expected.backward(gradient)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0, equal_nan=True)
                        self.assertEqual(len(saved), 1)
                        self.assertEqual(saved[0].dtype, torch.int32)
                        self.assertEqual(saved[0].numel(), 1)

    def test_clamp_retains_only_lossless_branch_snapshot_for_strided_inputs(self):
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                lower, upper = torch.tensor(-20., dtype=dtype), torch.tensor(10., dtype=dtype)
                boundaries = torch.stack((torch.nextafter(lower, torch.tensor(-float("inf"), dtype=dtype)),
                    lower, torch.nextafter(lower, upper), torch.nextafter(upper, lower), upper,
                    torch.nextafter(upper, torch.tensor(float("inf"), dtype=dtype))))
                x = boundaries.repeat(17, 1).t().detach().requires_grad_()
                reference = x.detach().clone().requires_grad_()
                compiled, graph = self.compile(_Function(lambda x: x.clamp(-20., 10.)), x)
                actual, expected = compiled(x), graph(reference)
                region, = compiled.backend_artifact.joint_autograd_regions
                self.assertEqual(len(region.trace.saved_tensors), 1)
                self.assertEqual(region.trace.saved_tensors[0].meta.dtype, torch.int32)
                binding, = [item for item in region.trace.saved_value_specs
                            if item["representation"] == "lossless_clamp_mask"]
                self.assertEqual(binding["consumers"][0]["role"], "input_gradient_branch")
                with torch.no_grad():
                    x.fill_(0)
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)

    def test_tensor_bound_clamp_is_not_misclassified_as_scalar_clamp(self):
        class Model(nn.Module):
            def forward(self, x):
                return x.clamp(min=x.mean())

        graph = torch.fx.symbolic_trace(Model())
        x = torch.randn(3, 4, requires_grad=True)
        ShapeProp(graph).propagate(x)
        with self.assertRaisesRegex(NotImplementedError, "coverage incomplete"):
            compile_fx(graph, [x], config=BackendConfig(require_full_aot=True))

    def test_nearest_resize_forms_save_no_activation_and_preserve_original_scales(self):
        forms = (
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.UpsamplingNearest2d(size=(9, 11)),
            _Function(lambda x: F.interpolate(x, scale_factor=(1.7, 2.3), mode="nearest")),
            _Function(lambda x: F.interpolate(input=x, scale_factor=(1.7, 2.3), mode="nearest",
                                               recompute_scale_factor=True)),
            _Function(lambda x: F.interpolate(x, size=(3, 4), mode="nearest")),
            _Function(lambda x: torch.ops.aten.upsample_nearest2d.default(x, [8, 16], 1.7, 2.3)),
            _Function(lambda x: torch.ops.aten.upsample_nearest2d.vec(x, None, [1.7, 2.3])),
            _Function(lambda x: torch._C._nn.upsample_nearest2d(x, None, [1.7, 2.3])),
        )
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            for channels_last in (False, True):
                for form in forms:
                    with self.subTest(dtype=dtype, channels_last=channels_last, form=form):
                        x = torch.randn(2, 3, 5, 7, dtype=dtype)
                        x = x.contiguous(memory_format=torch.channels_last) if channels_last else x.transpose(-1, -2)
                        x.requires_grad_()
                        reference = x.detach().clone().requires_grad_()
                        compiled, graph = self.compile(nn.Sequential(form), x)
                        saved = []
                        with torch.autograd.graph.saved_tensors_hooks(
                                lambda value: saved.append(value) or value, lambda value: value):
                            actual = compiled(x)
                        expected = graph(reference)
                        gradient = torch.randn_like(actual)
                        actual.backward(gradient)
                        expected.backward(gradient)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                        self.assertEqual(saved, [])
                        region, = compiled.backend_artifact.joint_autograd_regions
                        self.assertEqual(region.trace.saved_tensors, ())
                        self.assertEqual(region.trace.saved_value_specs[0]["representation"], "metadata_only")

    def test_eager_clamp_uses_the_shared_lossless_residual_contract(self):
        from deepopt.opt.nn.primitive_state import scalar_clamp

        x = torch.linspace(-25, 15, 129, dtype=torch.bfloat16).requires_grad_()
        reference = x.detach().clone().requires_grad_()
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(
                lambda value: saved.append(value) or value, lambda value: value):
            actual = scalar_clamp(x, -20., 10.)
        expected = reference.clamp(-20., 10.)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].dtype, torch.int32)
        self.assertEqual(saved[0].numel(), 5)

    def test_scoped_eager_clamp_variants_match_native_and_keep_only_masks(self):
        from deepopt import autocast

        forms = (lambda x: x.clamp(-20., 10.), lambda x: torch.clamp(input=x, min=-20., max=10.),
                 lambda x: x.clip(-20., 10.), lambda x: torch.clip(x, -20., 10.),
                 lambda x: x.clamp_min(-20.), lambda x: torch.clamp_min(x, -20.),
                 lambda x: x.clamp_max(10.), lambda x: torch.clamp_max(x, 10.))
        for dtype in (torch.bfloat16, torch.float32):
            for function in forms:
                with self.subTest(dtype=dtype, function=function):
                    x = torch.tensor([-float("inf"), -20., -19., 0., 9., 10., float("inf"), float("nan")],
                                     dtype=dtype).requires_grad_()
                    reference = x.detach().clone().requires_grad_()
                    saved = []
                    with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True), config={}):
                        with torch.autograd.graph.saved_tensors_hooks(
                                lambda value: saved.append(value) or value, lambda value: value):
                            actual = function(x)
                    expected = function(reference)
                    gradient = torch.full_like(actual, float("inf"))
                    actual.backward(gradient)
                    expected.backward(gradient)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0, equal_nan=True)
                    self.assertEqual(len(saved), 1)
                    self.assertEqual(saved[0].dtype, torch.int32)

    def test_scoped_eager_resize_saves_geometry_only_and_matches_native(self):
        from deepopt import autocast

        forms = (nn.Upsample(scale_factor=2, mode="nearest"),
                 lambda x: F.interpolate(x, scale_factor=(1.7, 2.3), mode="nearest"),
                 lambda x: F.interpolate(x, scale_factor=(1.7, 2.3), mode="nearest", recompute_scale_factor=True),
                 lambda x: F.interpolate(input=x, size=(9, 11), mode="nearest"))
        for dtype in (torch.bfloat16, torch.float32):
            for function in forms:
                with self.subTest(dtype=dtype, function=function):
                    x = torch.randn(2, 3, 5, 7, dtype=dtype).contiguous(memory_format=torch.channels_last).requires_grad_()
                    reference = x.detach().clone().requires_grad_()
                    saved = []
                    with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True), config={}):
                        with torch.autograd.graph.saved_tensors_hooks(
                                lambda value: saved.append(value) or value, lambda value: value):
                            actual = function(x)
                    expected = function(reference)
                    gradient = torch.randn_like(actual)
                    actual.backward(gradient)
                    expected.backward(gradient)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                    self.assertEqual(saved, [])

    def test_resize_fake_cuda_precision_contract_preserves_amp_and_fp32_boundaries(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.utils._python_dispatch import TorchDispatchMode
        from deepopt._bf16_policy import bf16_policy_scope
        from deepopt.opt.nn.primitive_state import Nearest2dGeometry, nearest2d_call, nearest2d_precision

        # This proves metadata and autocast scope behavior on a CPU host. Real
        # CUDA forward/backward and allocation checks are separate below.
        forms = ((F.interpolate, {"scale_factor": (1.7, 2.3)}),
                 (torch.ops.aten.upsample_nearest2d.default,
                  {"output_size": (8, 16), "scales_h": 1.7, "scales_w": 2.3}),
                 (torch.ops.aten.upsample_nearest2d.vec, {"output_size": None, "scale_factors": (1.7, 2.3)}),
                 (torch._C._nn.upsample_nearest2d, {"output_size": None, "scale_factors": (1.7, 2.3)}))
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.is_bf16_supported", return_value=True), FakeTensorMode():
            for policy in ("torch_amp", "bf16_storage"):
                for dtype in (torch.bfloat16, torch.float32):
                    x = torch.empty(2, 3, 5, 7, device="cuda", dtype=dtype)
                    with bf16_policy_scope(policy, "cuda"), torch.autocast("cuda", dtype=torch.bfloat16):
                        precision = nearest2d_precision(dtype, x.device)
                        for function, keywords in forms:
                            with self.subTest(policy=policy, dtype=dtype, function=function):
                                output = nearest2d_call(function, (x,), keywords)
                                expected = (torch.bfloat16 if policy == "bf16_storage" and dtype == torch.bfloat16
                                            else torch.float32)
                                self.assertEqual(output.dtype, expected)
                                self.assertTrue(torch.is_autocast_enabled("cuda"))
                    with bf16_policy_scope("torch_amp", "cuda"), torch.autocast("cuda", dtype=torch.bfloat16):
                        # The forward choice remains frozen after policy exit.
                        self.assertEqual(precision.call(lambda: torch.is_autocast_enabled("cuda")),
                                         not (policy == "bf16_storage" and dtype == torch.bfloat16))
                        geometry = Nearest2dGeometry((2, 3, 5, 7), (10, 14), (2., 2.), dtype, precision)
                        gradient = torch.empty(2, 3, 10, 14, dtype=dtype, device="cuda")
                        native_backward = torch.ops.aten.upsample_nearest2d_backward.default
                        autocast_states = []
                        class AdjointMode(TorchDispatchMode):
                            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                                if function is native_backward:
                                    autocast_states.append(torch.is_autocast_enabled("cuda"))
                                return function(*args, **(kwargs or {}))
                        with AdjointMode():
                            result = geometry.backward(gradient)
                        self.assertEqual(result.dtype, dtype)
                        self.assertEqual(autocast_states, [False])
                        self.assertTrue(torch.is_autocast_enabled("cuda"))

    def test_scoped_storage_resize_covers_aten_and_compression_off(self):
        from deepopt import autocast

        forms = ((F.interpolate, {"scale_factor": (1.7, 2.3), "recompute_scale_factor": True}),
                 (torch.ops.aten.upsample_nearest2d.default,
                  {"output_size": (8, 16), "scales_h": 1.7, "scales_w": 2.3}),
                 (torch.ops.aten.upsample_nearest2d.vec, {"output_size": None, "scale_factors": (1.7, 2.3)}))
        for compression in (False, dict(enabled=True, bits=2, group_size=1024)):
            for function, keywords in forms:
                with self.subTest(compression=bool(compression), function=function):
                    x = torch.randn(2, 3, 5, 7, dtype=torch.bfloat16).transpose(-1, -2).requires_grad_()
                    reference = x.detach().clone().requires_grad_()
                    saved = []
                    with autocast("cpu", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                                  activation_compression=compression, config={}):
                        with torch.autograd.graph.saved_tensors_hooks(
                                lambda value: saved.append(value) or value, lambda value: value):
                            actual = function(x, **keywords)
                    expected = function(reference, **keywords)
                    self.assertEqual(type(actual.grad_fn).__name__, "_Nearest2dFunctionBackward")
                    gradient = torch.randn_like(actual)
                    actual.backward(gradient)
                    expected.backward(gradient)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                    self.assertEqual(saved, [])

    def test_eager_primitive_scope_overrides_no_grad_and_tensor_bound_fallback(self):
        from deepopt import autocast

        x = torch.randn(2, 3, 5, 7, dtype=torch.bfloat16, requires_grad=True)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True), config={}):
            with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=False, config={}):
                dense = x.clamp(-20., 10.)
            packed = x.clamp(-20., 10.)
            saved = []
            with torch.no_grad(), torch.autograd.graph.saved_tensors_hooks(
                    lambda value: saved.append(value) or value, lambda value: value):
                self.assertFalse(x.clamp(-20., 10.).requires_grad)
                self.assertFalse(F.interpolate(x, scale_factor=2, mode="nearest").requires_grad)
        self.assertNotEqual(type(dense.grad_fn), type(packed.grad_fn))
        self.assertEqual(saved, [])
        bound = torch.tensor(0., dtype=torch.bfloat16, requires_grad=True)
        reference, reference_bound = x.detach().clone().requires_grad_(), bound.detach().clone().requires_grad_()
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True), config={}):
            actual = x.clamp(min=bound)
        expected = reference.clamp(min=reference_bound)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
        torch.testing.assert_close(bound.grad, reference_bound.grad, rtol=0, atol=0)

    def test_builtin_padding_retains_geometry_only(self):
        model = _Function(lambda x: torch._C._nn.pad(x, (1, 2, 1, 2), "reflect"))
        x = torch.randn(1, 3, 5, 7, requires_grad=True)
        reference = x.detach().clone().requires_grad_()
        compiled, graph = self.compile(model, x)
        actual, expected = compiled(x), graph(reference)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
        region, = compiled.backend_artifact.joint_autograd_regions
        self.assertEqual(region.trace.saved_tensors, ())

    def test_exp_vjp_uses_promoted_forward_coefficient_domain(self):
        from deepopt.compiler.autodiff import AOTAutoDiff
        from deepopt.compiler.autodiff.primitives import register_rules
        from deepopt.compiler.codegen.wrapper import WrapperCodegen
        from deepopt.compiler.ir import IRGraph, TensorMeta
        from deepopt.compiler.registry import BackendRegistry
        from deepopt.compiler.scheduler import Scheduler
        from deepopt.compiler.trace.builder import GraphBuilder

        # Model CUDA AMP's BF16 operand/FP32 result contract on a CPU-only host.
        # This directly checks the derivative domain, rather than relying on CPU
        # autocast (which does not promote exp in the same way).
        graph = IRGraph()
        builder = GraphBuilder(graph)
        x = builder.input(0, TensorMeta((37,), torch.bfloat16, "cpu", (1,), True))
        output = builder.emit_native("autodiff_primitive_exp", lambda value: value.float().exp(),
            (x,), {}, TensorMeta((37,), torch.float32, "cpu", (1,), True))
        builder.output(output)
        registry = BackendRegistry()
        register_rules(registry)
        trace = AOTAutoDiff(graph, registry=registry).diff()
        self.assertEqual(trace.saved_tensors, (output.value,))
        backward = WrapperCodegen(nn.Module(), Scheduler(trace.graph, registry=registry).run(),
                                  registry=registry, enable_autodiff=False).run().fn
        values = torch.linspace(-2, 2, 37, dtype=torch.bfloat16)
        coefficient = values.float().exp()
        gradient = torch.linspace(.13, 1.71, 37)
        actual, = backward(gradient, coefficient)
        expected = (gradient * coefficient).to(torch.bfloat16)
        self.assertFalse(torch.equal(expected, (gradient * values.exp()).to(torch.bfloat16)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_other_resize_modes_do_not_claim_nearest_native_coverage(self):
        model = _Function(lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        graph = torch.fx.symbolic_trace(model)
        x = torch.randn(1, 2, 3, 5, requires_grad=True)
        ShapeProp(graph).propagate(x)
        with self.assertRaisesRegex(NotImplementedError, "coverage incomplete"):
            compile_fx(graph, [x], config=BackendConfig(require_full_aot=True))

    def test_real_dynamo_fullgraph_clamp_and_resize_use_native_vjps(self):
        from deepopt import autocast
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model

        class Model(nn.Module):
            def forward(self, x):
                return F.interpolate(x.clamp(-20., 10.), scale_factor=2., mode="nearest")

        model, reference = Model(), Model()
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                                           options={"require_full_aot": True}))
        x = torch.randn(1, 2, 5, 7, dtype=torch.bfloat16, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True), config={}):
            expected = reference(xr)
            with patch("deepopt.opt.nn.primitive_state._ScalarClampFunction.forward", side_effect=AssertionError("eager clamp owner")), \
                    patch("deepopt.opt.nn.primitive_state._Nearest2dFunction.forward", side_effect=AssertionError("eager resize owner")):
                actual = model(x)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
        self.assertTrue(compilation_report(model).graphs)
        for report in compilation_report(model).graphs:
            self.assertFalse(report.eager_autograd_operations)
            self.assertFalse(report.legacy_backward_operations)


@unittest.skipUnless(torch.cuda.is_available(), "Native VAE primitive BF16 qualification requires CUDA")
class NativeVAEPrimitiveCudaTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        if torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("BF16 qualification requires compute capability >= 8")

    def test_storage_conv_resize_depthwise_chain_has_native_bf16_edges(self):
        from copy import deepcopy
        from deepopt import autocast
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model

        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                torch.compiler.reset()
                compression = dict(enabled=compressed, bits=2, group_size=1024)
                config = dict(runtime=dict(precision="bf16", bf16_policy="bf16_storage",
                                           fp8=dict(enabled=False), activation_compression=compression))
                eager = nn.Sequential(nn.Conv2d(3, 8, 1), nn.Upsample(scale_factor=2, mode="nearest"),
                                      nn.Conv2d(8, 8, 3, padding=1, groups=8)).cuda()
                eager = prepare_fp8_model(eager, False, config=config)[0]
                native = deepcopy(eager)
                compile_model(native, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                    options=dict(require_full_aot=True)), config=config)
                x = torch.randn(2, 3, 5, 7, device="cuda", requires_grad=True)
                reference = x.detach().clone().requires_grad_()
                edges = []
                handle = eager[1].register_forward_hook(lambda module, args, output: edges.append((args[0].dtype, output.dtype)))
                with autocast("cuda", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                              activation_compression=compression, config={}):
                    expected, actual = eager(reference), native(x)
                handle.remove()
                self.assertEqual(edges, [(torch.bfloat16, torch.bfloat16)])
                self.assertEqual(actual.dtype, torch.bfloat16)
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
                expected.backward(gradient)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                for parameter, oracle in zip(native.parameters(), eager.parameters()):
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertEqual(parameter.grad.dtype, torch.float32)
                    torch.testing.assert_close(parameter.grad, oracle.grad, rtol=0, atol=0)
                for graph in compilation_report(native).graphs:
                    self.assertFalse(graph.eager_autograd_operations)
                    self.assertFalse(graph.legacy_backward_operations)

    def test_storage_resize_matches_bf16_native_geometry_for_all_supported_forms(self):
        from deepopt import autocast

        forms = (nn.Upsample(scale_factor=2, mode="nearest"),
                 _Function(lambda x: F.interpolate(x, scale_factor=(1.7, 2.3), mode="nearest")),
                 _Function(lambda x: F.interpolate(x, scale_factor=(1.7, 2.3), mode="nearest",
                                                    recompute_scale_factor=True)),
                 _Function(lambda x: F.interpolate(x, size=(3, 4), mode="nearest")),
                 _Function(lambda x: torch.ops.aten.upsample_nearest2d.default(x, [8, 16], 1.7, 2.3)),
                 _Function(lambda x: torch.ops.aten.upsample_nearest2d.vec(x, None, [1.7, 2.3])))
        for compressed in (False, True):
            for layout in ("contiguous", "channels_last", "strided"):
                for form in forms:
                    with self.subTest(compressed=compressed, layout=layout, form=form):
                        x = torch.randn(2, 3, 5, 7, device="cuda", dtype=torch.bfloat16)
                        if layout == "channels_last":
                            x = x.contiguous(memory_format=torch.channels_last)
                        elif layout == "strided":
                            x = x.transpose(-1, -2)
                        x.requires_grad_()
                        reference = x.detach().clone().requires_grad_()
                        eager_input = x.detach().clone().requires_grad_()
                        graph = torch.fx.symbolic_trace(nn.Sequential(form))
                        compression = dict(enabled=compressed, bits=2, group_size=1024)
                        with autocast("cuda", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                                      activation_compression=compression, config={}):
                            ShapeProp(graph).propagate(x)
                            report = CompilationReport()
                            compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True), report=report)
                            actual, eager = compiled(x), graph(eager_input)
                            self.assertTrue(torch.is_autocast_enabled("cuda"))
                        with torch.autocast("cuda", enabled=False):
                            expected = graph(reference)
                        self.assertEqual(actual.dtype, torch.bfloat16)
                        self.assertEqual(eager.dtype, torch.bfloat16)
                        # Integer cotangents make the gather sum exactly
                        # representable, independently of backend reduction order.
                        gradient = torch.randint(-1, 2, actual.shape, device="cuda").bfloat16()
                        actual.backward(gradient)
                        eager.backward(gradient)
                        expected.backward(gradient)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
                        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                        torch.testing.assert_close(eager_input.grad, reference.grad, rtol=0, atol=0)
                        self.assertEqual(x.grad.dtype, torch.bfloat16)
                        self.assertFalse(report.graphs[0].eager_autograd_operations)
                        self.assertFalse(report.graphs[0].legacy_backward_operations)
                        region, = compiled.backend_artifact.joint_autograd_regions
                        self.assertEqual(region.trace.saved_tensors, ())

    def test_storage_decoder_shapes_have_no_full_fp32_resize_or_adjoint_copy(self):
        from deepopt import autocast
        from deepopt.opt.nn.primitive_state import Nearest2dGeometry, nearest2d_precision

        if torch.cuda.get_device_properties(0).total_memory < 48 * 1024**3:
            self.skipTest("Original batch128 decoder-shape operators require at least 48 GiB CUDA memory")
        for channels, height in ((512, 56), (352, 112), (192, 224)):
            with self.subTest(channels=channels, height=height):
                shape = (128, channels, height // 2, height // 2)
                x = torch.ones(shape, device="cuda", dtype=torch.bfloat16)
                with autocast("cuda", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                              activation_compression=False, config={}):
                    precision = nearest2d_precision(x.dtype, x.device)
                    torch.cuda.synchronize()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    output = F.interpolate(x, scale_factor=2, mode="nearest")
                    torch.cuda.synchronize()
                    growth = torch.cuda.max_memory_allocated() - baseline
                self.assertEqual(output.shape, (128, channels, height, height))
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertLessEqual(growth, output.numel() * output.element_size() + 8 * 1024**2)
                del output
                gradient = torch.ones((128, channels, height, height), device="cuda", dtype=torch.bfloat16)
                spec = Nearest2dGeometry(shape, (height, height), (2., 2.), x.dtype, precision)
                # Measure the adjoint independently of autograd leaf copies.
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = spec.backward(gradient)
                torch.cuda.synchronize()
                growth = torch.cuda.max_memory_allocated() - baseline
                self.assertEqual(result.dtype, torch.bfloat16)
                self.assertLessEqual(growth, x.numel() * x.element_size() + 8 * 1024**2)
                self.assertEqual(result.amin().item(), 4.)
                self.assertEqual(result.amax().item(), 4.)
                del gradient, result, x

    def test_bf16_exp_uses_actual_fp32_autocast_result(self):
        from deepopt import autocast

        if torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("BF16 qualification requires compute capability >= 8")
        graph = torch.fx.symbolic_trace(_Function(torch.exp))
        x = torch.linspace(-2, 2, 37, device="cuda", dtype=torch.bfloat16).requires_grad_()
        reference = x.detach().clone().requires_grad_()
        with autocast("cuda", dtype=torch.bfloat16, fp8=False, activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True), config={}):
            ShapeProp(graph).propagate(x)
            compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True))
            actual, expected = compiled(x), graph(reference)
        self.assertEqual(actual.dtype, torch.float32)
        gradient = torch.linspace(.13, 1.71, 37, device="cuda")
        actual.backward(gradient)
        expected.backward(gradient)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)

    def test_bf16_clamp_and_scaled_resize_preserve_exact_native_derivatives(self):
        from deepopt import autocast

        if torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("BF16 qualification requires compute capability >= 8")

        class Model(nn.Module):
            def forward(self, x):
                return F.interpolate(x.clamp(-20., 10.), scale_factor=(1.7, 2.3), mode="nearest")

        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                x = torch.linspace(-25, 15, 2 * 3 * 5 * 7, device="cuda", dtype=dtype).reshape(2, 3, 5, 7)
                x = x.contiguous(memory_format=torch.channels_last).requires_grad_()
                reference = x.detach().clone().requires_grad_()
                graph = torch.fx.symbolic_trace(Model())
                with autocast("cuda", dtype=torch.bfloat16, fp8=False, activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True), config={}):
                    ShapeProp(graph).propagate(x)
                    report = CompilationReport()
                    compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True), report=report)
                    actual, expected = compiled(x), graph(reference)
                # Nearest gathers at most six contributors here. Integer upstream
                # values keep all sums exactly representable even in BF16.
                gradient = torch.randint(-1, 2, actual.shape, device=actual.device).to(actual.dtype)
                actual.backward(gradient)
                expected.backward(gradient)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                self.assertFalse(report.graphs[0].eager_autograd_operations)
                self.assertFalse(report.graphs[0].legacy_backward_operations)
                region, = compiled.backend_artifact.joint_autograd_regions
                self.assertEqual(len(region.trace.saved_tensors), 1)
                self.assertEqual(region.trace.saved_tensors[0].meta.dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
