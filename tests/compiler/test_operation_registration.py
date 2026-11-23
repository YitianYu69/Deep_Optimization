"""Public custom operation contracts and frozen BlurPool saved dependencies."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import SavedUse, autocast, register_operation
from deepopt.compiler.ir import TensorMeta
from deepopt.compiler import compilation_report
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model


def _square_state(ctx, inputs, output):
    if ctx.needs_grad(inputs["x"]):
        ctx.save("source", inputs["x"], use=SavedUse(allow_approximation=True))
        ctx.metadata("scale", inputs["scale"])


def _square_vjp(state, grad):
    return {"x": grad * state["source"] * (2 * state["scale"])}


@register_operation(name="test_square", fake=lambda x, scale=1: torch.empty_like(x),
                    saved_state=_square_state, vjp=_square_vjp)
def registered_square(x, scale=1):
    return x.square() * scale


@register_operation(name="test_composite", decomposition=True)
def composite(x, factor=2):
    return x * factor + x


def _positive_state(ctx, inputs, output):
    metadata = ctx.meta(inputs["x"])
    branch = ctx.compute(torch.gt, inputs["x"], 0,
        meta=TensorMeta(metadata.shape, torch.bool, metadata.device))
    ctx.mask("branch", branch)
    ctx.metadata("input_meta", metadata)


def _positive_vjp(state, grad):
    return {"x": state.call(torch.where, state["branch"], grad, 0, meta=state["input_meta"])}


@register_operation(name="test_positive", fake=torch.empty_like,
                    saved_state=_positive_state, vjp=_positive_vjp)
def registered_positive(x):
    return x.clamp_min(0)


def _owned_state(ctx, inputs, output):
    if ctx.needs_grad(inputs["x"]):
        ctx.save("kernel", inputs["kernel"], use=SavedUse(allow_approximation=True))
    if ctx.needs_grad(inputs["kernel"]):
        ctx.save("source", inputs["x"], use=SavedUse(allow_approximation=True))


def _owned_vjp(state, grad):
    return {**({"x": grad * state["kernel"]} if "kernel" in state else {}),
            **({"kernel": grad * state["source"]} if "source" in state else {})}


@register_operation(name="test_owned", fake=lambda x, kernel: torch.empty_like(x),
                    saved_state=_owned_state, vjp=_owned_vjp, state_operands=("kernel",))
def registered_owned(x, kernel):
    return x * kernel


class OperationRegistrationTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        torch.manual_seed(12)

    def compile(self, model):
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
            dynamic=False, options={"require_full_aot": True}))
        return model

    def assert_native(self, model):
        report = compilation_report(model)
        self.assertTrue(report.graphs)
        self.assertFalse(report.fallback_reasons)
        self.assertFalse(report.graphs[-1].eager_autograd_operations)
        self.assertFalse(report.graphs[-1].legacy_backward_operations)
        self.assertGreater(report.graphs[-1].native_autodiff_operations, 0)

    def test_explicit_registered_vjp_is_native_and_matches_eager_recipe(self):
        class Model(nn.Module):
            def forward(self, x):
                return registered_square(x=x, scale=3)

        model = self.compile(Model())
        x = torch.randn(3, 1041, dtype=torch.bfloat16, requires_grad=True)
        reference = x.detach().clone().requires_grad_()
        saved_eager, saved_native = [], []
        with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                      activation_compression={"enabled": True, "bits": 2, "group_size": 1024, "norm_xhat_recenter": True}, config={}):
            with torch.autograd.graph.saved_tensors_hooks(lambda v: saved_eager.append(v) or v, lambda v: v):
                expected = registered_square(reference, 3)
            with patch("deepopt.operations._OperationFunction.forward", side_effect=AssertionError("eager forward owner")):
                with torch.autograd.graph.saved_tensors_hooks(lambda v: saved_native.append(v) or v, lambda v: v):
                    actual = model(x)
        gradient = torch.randn_like(actual)
        with patch("deepopt.operations._OperationFunction.backward", side_effect=AssertionError("eager backward owner")):
            actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
        self.assertEqual([v.dtype for v in saved_eager], [torch.int32, torch.float32, torch.float32])
        self.assertEqual([v.dtype for v in saved_native], [v.dtype for v in saved_eager])
        self.assert_native(model)

    def test_decomposition_has_metadata_only_derivative_and_second_user_operation(self):
        class Model(nn.Module):
            def forward(self, x):
                return composite(x, factor=3)

        model = self.compile(Model())
        x = torch.randn(3, 9, requires_grad=True)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda v: saved.append(v) or v, lambda v: v):
            y = model(x)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 4))
        self.assertEqual(saved, [])
        self.assert_native(model)

    def test_auxiliary_lossless_mask_uses_same_eager_native_contract(self):
        class Model(nn.Module):
            def forward(self, x):
                return registered_positive(x)

        model = self.compile(Model())
        x = torch.randn(3, 1041, dtype=torch.bfloat16, requires_grad=True)
        reference = x.detach().clone().requires_grad_()
        eager_saved, native_saved = [], []
        with torch.autograd.graph.saved_tensors_hooks(lambda v: eager_saved.append(v) or v, lambda v: v):
            expected = registered_positive(reference)
        with torch.autograd.graph.saved_tensors_hooks(lambda v: native_saved.append(v) or v, lambda v: v):
            actual = model(x)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
        self.assertEqual([(v.dtype, v.numel()) for v in eager_saved], [(torch.int32, 98)])
        self.assertEqual([(v.dtype, v.numel()) for v in native_saved], [(torch.int32, 98)])
        self.assert_native(model)

    def test_blur_pool_frozen_and_trainable_kernel_masks(self):
        from examples.vae.mbconv_vae import BlurPool

        for need_x, need_weight in ((True, False), (False, True), (True, True), (False, False)):
            with self.subTest(mask=(need_x, need_weight)):
                torch.compiler.reset()
                layer = BlurPool(2)
                if need_weight:
                    layer.kernel.requires_grad_()
                reference = deepcopy(layer)
                self.compile(layer)
                x = torch.randn(1, 2, 27, 29, dtype=torch.bfloat16, requires_grad=need_x)
                xr = x.detach().clone().requires_grad_(need_x)
                saved = []
                with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                              activation_compression={"enabled": True, "bits": 2, "group_size": 1024, "norm_xhat_recenter": True}, config={}):
                    expected = reference(xr)
                    with torch.autograd.graph.saved_tensors_hooks(lambda v: saved.append(v) or v, lambda v: v):
                        actual = layer(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if need_x or need_weight:
                    gradient = torch.randn_like(actual)
                    actual.backward(gradient)
                    expected.backward(gradient)
                    torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
                    torch.testing.assert_close(layer.kernel.grad, reference.kernel.grad, rtol=0, atol=0)
                    self.assert_native(layer)
                if not need_weight:
                    self.assertFalse(any(v.shape == x.shape for v in saved))
                    self.assertFalse(any(v.dtype == torch.int32 for v in saved))
                else:
                    self.assertTrue(any(v.dtype == torch.int32 for v in saved))

    def test_exact_save_mutation_and_interleaved_scopes(self):
        x = torch.randn(2, 1041, dtype=torch.bfloat16, requires_grad=True)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=False, config={}):
            exact = registered_square(x)
        with torch.no_grad():
            x.add_(1)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace"):
            exact.sum().backward()
        source = x.detach().clone().requires_grad_()
        with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                      activation_compression={"enabled": True, "bits": 2, "group_size": 1024, "norm_xhat_recenter": True}, config={}):
            a = registered_square(source)
            with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=False, config={}):
                b = registered_square(source, scale=2)
        (a.sum() + b.sum()).backward(retain_graph=True)
        first = source.grad.clone()
        source.grad = None
        (a.sum() + b.sum()).backward()
        torch.testing.assert_close(first, source.grad, rtol=0, atol=0)

    def test_model_state_stays_exact_across_freeze_layout_and_scope_changes(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("kernel", torch.randn(3, 1041, dtype=torch.bfloat16))

            def forward(self, x):
                return registered_owned(x, self.kernel)

        model = self.compile(Model())
        control = deepcopy(Model())
        control.kernel.copy_(model.kernel)
        for train_kernel, compressed in ((False, True), (True, True), (False, False)):
            with self.subTest(train_kernel=train_kernel, compressed=compressed):
                model.kernel.requires_grad_(train_kernel)
                control.kernel.requires_grad_(train_kernel)
                x = torch.randn(1041, 3, dtype=torch.bfloat16).t().requires_grad_()
                xr = x.detach().clone().requires_grad_()
                eager_saved, native_saved = [], []
                with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                    activation_compression={"enabled": compressed, "bits": 2, "group_size": 1024, "norm_xhat_recenter": True}, config={}):
                    with torch.autograd.graph.saved_tensors_hooks(lambda v: eager_saved.append(v) or v, lambda v: v):
                        expected = control(xr)
                    with torch.autograd.graph.saved_tensors_hooks(lambda v: native_saved.append(v) or v, lambda v: v):
                        actual = model(x)
                    # Backward under an active scope must not intercept its own arithmetic.
                    actual.sum().backward()
                    expected.sum().backward()
                torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
                torch.testing.assert_close(model.kernel.grad, control.kernel.grad, rtol=0, atol=0)
                self.assertEqual(sum(v.numel() * v.element_size() for v in eager_saved),
                                 sum(v.numel() * v.element_size() for v in native_saved))
                self.assertTrue(any(v.data_ptr() == model.kernel.data_ptr() for v in native_saved))
                self.assertTrue(any(v.data_ptr() == control.kernel.data_ptr() for v in eager_saved))
                model.kernel.grad = control.kernel.grad = None

    def test_no_grad_and_inference_do_not_save_state(self):
        for scope in (torch.no_grad, torch.inference_mode):
            with scope():
                source = torch.randn(2, 7)
                saved = []
                with torch.autograd.graph.saved_tensors_hooks(lambda v: saved.append(v) or v, lambda v: v):
                    actual = registered_square(source, scale=2)
                torch.testing.assert_close(actual, source.square() * 2)
                self.assertEqual(saved, [])

    def test_nondifferentiable_output_skips_fake_grad_and_saved_state(self):
        def no_backward(*args):
            raise AssertionError("A boolean/integer output must not prepare or differentiate residuals")

        for dtype in (torch.bool, torch.int64):
            with self.subTest(dtype=dtype):
                @register_operation(name="test_nondifferentiable", fake=lambda x: torch.empty_like(x, dtype=dtype),
                                    saved_state=no_backward, vjp=no_backward)
                def operation(x):
                    return x > 0 if dtype == torch.bool else x.to(torch.int64)

                class Model(nn.Module):
                    def forward(self, x):
                        return operation(x)

                model = self.compile(Model())
                source = torch.randn(3, 7, requires_grad=True)
                saved = []
                with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
                    expected = operation(source)
                    actual = model(source)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, dtype)
                self.assertFalse(actual.requires_grad)
                self.assertEqual(saved, [])
                report = compilation_report(model).graphs[-1]
                self.assertFalse(report.eager_autograd_operations)
                self.assertFalse(report.legacy_backward_operations)

    def test_registration_errors_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "require fake"):
            register_operation(name="missing_vjp")
        with self.assertRaisesRegex(ValueError, "Choose"):
            register_operation(name="conflict", decomposition=True, vjp=lambda s, g: {})
        with self.assertRaisesRegex(ValueError, "no effects"):
            register_operation(name="impure", decomposition=True, effects=("rng",))
        tensor = torch.ones(1)
        with self.assertRaisesRegex(ValueError, "explicit operands"):
            @register_operation(name="hidden_tensor", decomposition=True)
            def hidden(x):
                return x + tensor
        with self.assertRaisesRegex(TypeError, "metadata"):
            from deepopt.operations import _metadata
            _metadata(tensor)


if __name__ == "__main__":
    unittest.main()
