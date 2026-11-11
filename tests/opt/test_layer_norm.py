"""LayerNorm suffix semantics, saved-xhat policy and cached precision transport."""

from copy import deepcopy
from contextvars import Context
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

import deepopt
from deepopt.opt.nn import functional as doF
from deepopt.opt.nn.normalization.modules import DOLayerNorm
from deepopt.opt.policy import normalization_plan, prepare_fp8_model
from deepopt.opt.precision import precision_report
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation, dequantize_activation
from deepopt.opt.quantization.quantizer import quantize_fp8


def policy(compressed=True, *, bits=2, group=17, block=None, recipe="current"):
    return normalization_plan(fp8={"enabled": True, "backend": "reference", "scaling_recipe": recipe,
                                    "block_shapes": {"activation": block}},
                               activation_compression={"enabled": compressed, "bits": bits, "group_size": group})


def oracle(x, shape, eps=1e-5):
    matrix = x.reshape(-1, torch.tensor(shape).prod().item())
    variance, mean = torch.var_mean(matrix, 1, correction=0)
    inverse = torch.rsqrt(variance + eps)
    return (matrix - mean[:, None]) * inverse[:, None], mean, inverse


class LayerNormTests(unittest.TestCase):
    def test_suffix_affine_options_and_all_input_output_precision_pairs(self):
        torch.manual_seed(241)
        for shape, norm in (((3, 33), (33,)), ((2, 3, 7), (3, 7)), ((7,), (7,))):
            for source_fp8 in (False, True):
                for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
                    for block in (None, (2, 8)):
                        for affine in (False, True):
                            with self.subTest(shape=shape, norm=norm, input=source_fp8, output=dtype, block=block, affine=affine):
                                x = torch.randn(shape, dtype=torch.bfloat16).requires_grad_()
                                source = to_fp8_activation(x, layout="last", block_shape=block) if source_fp8 else x
                                dense = source.dequantize(torch.float32) if source_fp8 else x.detach().float()
                                w = torch.randn(norm, requires_grad=True) if affine else None
                                b = torch.randn(norm, requires_grad=True) if affine else None
                                h, _, inverse = oracle(dense, norm)
                                y = h * w.detach().reshape(1, -1) + b.detach().reshape(1, -1) if affine else h
                                output = doF.layer_norm(source, norm, w, b, plan=policy(False, block=block), output_dtype=dtype)
                                y = y.reshape(shape)
                                if dtype == torch.bfloat16:
                                    torch.testing.assert_close(output, y.bfloat16(), rtol=0, atol=0)
                                else:
                                    q = quantize_fp8(y.reshape(-1, shape[-1]), fp8_dtype=dtype, block_shape=block)
                                    torch.testing.assert_close(output._data.float(), q.data.float(), rtol=0, atol=0)
                                    torch.testing.assert_close(output._scale_inv, q.scale_inv, rtol=0, atol=0)
                                dy = torch.randn_like(x)
                                output.backward(dy)
                                u = dy.float().reshape_as(h) * w.detach().reshape(1, -1) if affine else dy.float().reshape_as(h)
                                dx = (u - u.mean(1, keepdim=True) - h * (u * h).mean(1, keepdim=True)) * inverse[:, None]
                                torch.testing.assert_close(x.grad, dx.reshape(shape).bfloat16(), rtol=0, atol=0)
                                if affine:
                                    torch.testing.assert_close(w.grad, (dy.float().reshape_as(h) * h).sum(0).reshape(norm))
                                    torch.testing.assert_close(b.grad, dy.float().reshape_as(h).sum(0).reshape(norm))

    def test_compression_packs_only_pre_affine_xhat_and_matches_codec_backward(self):
        for bits in (1, 2, 4, 8):
            for group in (1, 17, 512, 17003):
                x = (torch.randn(2, 3, 19) + 50).requires_grad_()
                w, b = torch.randn(3, 19, requires_grad=True), torch.randn(3, 19, requires_grad=True)
                saved = []
                with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                    output = doF.layer_norm(x, (3, 19), w, b, plan=policy(True, bits=bits, group=group))
                h, _, inverse = oracle(x.detach(), (3, 19))
                effective = min(group, 57)
                q = quantize_pack_activation(h, effective, (1, effective), bits=bits)
                self.assertEqual(len(saved), 5)
                self.assertFalse(any(t is x or t.shape == x.shape for t in saved))
                for actual, expected in zip(saved[:3], (q.data, q.step, q.minimum)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                reconstructed = dequantize_unpack_activation(q, dtype=torch.float32)
                dy = torch.randn_like(x)
                output.backward(dy)
                u = dy.reshape_as(h) * w.detach().reshape(1, -1)
                dx = (u - u.mean(1, keepdim=True) - reconstructed * (u * reconstructed).mean(1, keepdim=True)) * inverse[:, None]
                torch.testing.assert_close(x.grad, dx.reshape_as(x))
                torch.testing.assert_close(w.grad, (dy.reshape_as(h) * reconstructed).sum(0).reshape_as(w))
                torch.testing.assert_close(b.grad, dy.sum(0))
                baseline = doF.layer_norm(x.detach(), (3, 19), w.detach(), b.detach(), plan=policy(False))
                torch.testing.assert_close(output, baseline, rtol=0, atol=0)

    def test_native_math_gradcheck_strided_affine_and_dtype_validation(self):
        for norm, shape in (((7,), (3, 7)), ((3, 7), (2, 3, 7))):
            x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
            w, b = torch.randn(norm, dtype=torch.float64, requires_grad=True), torch.randn(norm, dtype=torch.float64, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(lambda a, g, z: doF.layer_norm(a, norm, g, z, plan=policy(False)), (x, w, b)))
        x = torch.randn(2, 3, 7).transpose(0, 1).requires_grad_()
        w = torch.randn(7, 2).t().requires_grad_()
        z, g = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
        actual, expected = doF.layer_norm(x, (2, 7), w, plan=policy(False)), F.layer_norm(z, (2, 7), g)
        actual.sum().backward(); expected.sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, z.grad, rtol=2e-4, atol=2e-6)
        torch.testing.assert_close(w.grad, g.grad, rtol=2e-4, atol=2e-6)
        with self.assertRaises(RuntimeError):
            doF.layer_norm(torch.randn(2, 7), (7,), torch.ones(7, dtype=torch.bfloat16), plan=policy())
        for norm in ((), (1, 7), (21,), (-1,)):
            with self.assertRaises(ValueError):
                doF.layer_norm(torch.randn(2, 3, 7), norm, plan=policy())

    def test_empty_singleton_constant_and_bias_only_backward(self):
        for shape, norm in (((0, 7), (7,)), ((2, 0), (0,)), ((1,), (1,)), ((2, 3, 1), (1,))):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                x = torch.ones(shape, dtype=torch.bfloat16, requires_grad=True)
                w, b = torch.ones(norm, requires_grad=True), torch.zeros(norm, requires_grad=True)
                out = doF.layer_norm(x, norm, w, b, plan=policy(), output_dtype=dtype)
                dequantize_activation(out).sum().backward()
                torch.testing.assert_close(x.grad, torch.zeros_like(x))
                torch.testing.assert_close(w.grad, torch.zeros_like(w))
        x = torch.randn(2, 3, 7)
        b = torch.randn(3, 7, requires_grad=True)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            doF.layer_norm(x, (3, 7), bias=b, plan=policy()).sum().backward()
        self.assertEqual(saved, [])
        torch.testing.assert_close(b.grad, torch.full_like(b, 2))

    def test_fp8_channel_layout_and_fp8_upstream_gradient(self):
        x = torch.randn(2, 3, 5, 7, dtype=torch.bfloat16).transpose(2, 3).requires_grad_()
        source = to_fp8_activation(x, layout="channels", block_shape=(2, 8))
        norm = tuple(x.shape[-2:])
        out = doF.layer_norm(source, norm, plan=policy(False, block=(2, 8)), output_dtype=torch.float8_e5m2,
                             output_layout="channels")
        dy = to_fp8_activation(torch.randn_like(x), layout="last", fp8_dtype=torch.float8_e5m2)
        out.backward(dy)
        h, _, inverse = oracle(source.dequantize(torch.float32), norm)
        u = dy.dequantize(torch.float32).reshape_as(h)
        expected = (u - u.mean(1, keepdim=True) - h * (u * h).mean(1, keepdim=True)) * inverse[:, None]
        torch.testing.assert_close(x.grad, expected.reshape_as(x).bfloat16())

    def test_independent_gradient_requests_skip_unneeded_saved_state(self):
        for need_dx, need_dw, need_db in ((True, False, False), (False, True, False),
                                          (False, False, True), (True, True, False), (False, True, True)):
            x = torch.randn(3, 7, requires_grad=need_dx)
            w, b = torch.randn(7, requires_grad=need_dw), torch.randn(7, requires_grad=need_db)
            saved = []
            with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                y = doF.layer_norm(x, 7, w, b, plan=policy())
            dy = torch.randn_like(x)
            y.backward(dy)
            if need_dx or need_dw:
                h, _, inverse = oracle(x.detach(), (7,))
                q = quantize_pack_activation(h, 7, (1, 7), bits=2)
                h = dequantize_unpack_activation(q, dtype=torch.float32)
                if need_dx:
                    u = dy * w.detach()
                    dx = (u - u.mean(1, keepdim=True) - h * (u * h).mean(1, keepdim=True)) * inverse[:, None]
                    torch.testing.assert_close(x.grad, dx)
                if need_dw:
                    torch.testing.assert_close(w.grad, (dy * h).sum(0))
            if need_db:
                torch.testing.assert_close(b.grad, dy.sum(0))
            self.assertEqual(len(saved), 5 if need_dx else 3 if need_dw else 0)

    def test_no_grad_fake_meta_and_explicit_scales(self):
        for device in ("meta", "cpu"):
            with FakeTensorMode() if device == "cpu" else torch.device("meta"):
                x = torch.empty(2, 3, 7, device=device, dtype=torch.bfloat16, requires_grad=True)
                y = doF.layer_norm(x, (3, 7), plan=policy(), output_dtype=torch.float8_e4m3fn)
                dequantize_activation(y).sum().backward()
                self.assertEqual(tuple(y.shape), (2, 3, 7))
        info = {}
        x = torch.randn(3, 7, requires_grad=True)
        with torch.no_grad():
            doF.layer_norm(x, 7, plan=policy(), _execution_info=info)
        self.assertEqual(info["saved_state_bytes"], 0)
        self.assertFalse(info["compression"])
        scale = torch.full((1, 1), 2.)
        y = doF.layer_norm(x, 7, plan=policy(), output_dtype=torch.float8_e4m3fn, output_scale=scale)
        h, _, _ = oracle(x.detach(), (7,))
        torch.testing.assert_close(y._data.float(), (h * 2).clamp(-448, 448).to(torch.float8_e4m3fn).float(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            doF.layer_norm(x, 7, plan=policy(recipe="delayed"), output_dtype=torch.float8_e4m3fn)
        model, _ = prepare_fp8_model(nn.Sequential(nn.LayerNorm(7), nn.Linear(7, 1)),
                                     deepopt.FP8Config(enabled=True, scaling_recipe="delayed"))
        with self.assertRaisesRegex(ValueError, "output format must match"):
            doF.layer_norm(x, 7, plan=model[0].plan, output_dtype=torch.float8_e5m2,
                           _scaling_state=model[0].fp8_scaling)

    def test_preparation_sharing_optimizer_identity_and_one_time_bidirectional_plans(self):
        for mode in ("modules", "full"):
            for activation in (nn.ReLU, nn.Sigmoid, nn.SiLU, nn.GELU, nn.Tanh, nn.ReLU6):
                model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8, bias=False), activation(), nn.Linear(8, 3))
                parameters = tuple(model.parameters())
                model, _ = prepare_fp8_model(model, deepopt.FP8Config(enabled=True, trace_mode=mode, backward_precision="auto"), activation_compression=True)
                self.assertEqual(tuple(model.parameters()), parameters)
                self.assertIsInstance(model.get_submodule("1"), DOLayerNorm)
                self.assertIsNone(model.get_submodule("1").bias)
                self.assertFalse(precision_report(model)["fallbacks"])
                with patch("deepopt.opt.fx.tracer.DOTracer.trace", side_effect=AssertionError("retrace")):
                    for training in (True, False):
                        model.train(training)
                        model(torch.randn(3, 8)).sum().backward()
                self.assertEqual(precision_report(model)["analyses"], 1)
                self.assertTrue(any(e["precision"] == "fp8" for e in precision_report(model)["backward_edges"]))
        source = nn.LayerNorm((3, 7), elementwise_affine=False)
        model, _ = prepare_fp8_model(nn.Sequential(source, source), deepopt.FP8Config(enabled=False), activation_compression=True)
        self.assertIs(model[0], model[1])
        self.assertIsInstance(model[0], DOLayerNorm)

    def test_functional_scopes_wrappers_and_direct_cached_call_sites(self):
        for direct in (False, True):
            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.a, self.b = nn.Linear(8, 8), nn.Linear(8, 3)

                def forward(self, x):
                    return self.b((doF.layer_norm if direct else F.layer_norm)(self.a(x), (8,)))
            for mode in ("modules", "full"):
                model, _ = prepare_fp8_model(Model(), deepopt.FP8Config(enabled=True, trace_mode=mode, backward_precision="auto"), activation_compression=True)
                self.assertFalse(precision_report(model)["fallbacks"])
                model(torch.randn(3, 8)).sum().backward()
        x = torch.randn(3, 7, dtype=torch.bfloat16, requires_grad=True)
        with deepopt.autocast("cpu", enabled=False, fp8=False, activation_compression=True):
            F.layer_norm(x, (7,)).sum().backward()
        wrapped = to_fp8_activation(x, layout="last")
        self.assertNotIsInstance(F.layer_norm(wrapped, (7,)), FP8Activation)
        rank_one, _ = prepare_fp8_model(nn.Sequential(nn.LayerNorm(7), nn.Sigmoid()), True)
        rank_one(torch.randn(7)).sum().backward()
        self.assertFalse(precision_report(rank_one)["fallbacks"])

    def test_delayed_ownership_accumulation_recomputation_and_exact_resume(self):
        for recompute in (False, True):
            source = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.GELU(), nn.Linear(8, 3))
            settings = deepopt.FP8Config(enabled=True, scaling_recipe="delayed", backward_precision="auto", grad_dtype="e5m2")
            runtime = deepopt.TrainingRuntime(source, torch.optim.SGD(source.parameters(), lr=0.001), fp8=settings,
                                              checks=False, config={"runtime": {"activation_compression": {"enabled": True}}})
            from deepopt.opt.nn.autograd.normalization.layer import LayerNormFunction
            original = LayerNormFunction.backward
            with patch.object(LayerNormFunction, "backward", new=lambda ctx, grad: Context().run(original, ctx, grad)):
                for _ in range(2):
                    loss = runtime.compute(torch.randn(3, 8), lambda x: (deepopt.fp8_checkpoint(runtime.model, x) if recompute else runtime.model(x)).square().mean())
                    runtime.backward(loss)
                    self.assertEqual(runtime.model[1].fp8_scaling.updates[0].item(), 0)
            runtime.step()
            self.assertEqual(runtime.model[1].fp8_scaling.updates[0].item(), 1)
            state = deepcopy(runtime.model.state_dict())
            restored, _ = prepare_fp8_model(nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.GELU(), nn.Linear(8, 3)), settings, activation_compression=True)
            restored.load_state_dict(state)
            runtime.model.eval(); restored.eval()
            values = torch.randn(3, 8)
            torch.testing.assert_close(runtime.model(values), restored(values), rtol=0, atol=0)

    def test_compilation_keeps_layernorm_dispatch_eager_and_preserves_gradients(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        model, _ = prepare_fp8_model(nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.GELU(), nn.Linear(8, 3)),
                                     deepopt.FP8Config(enabled=True, backward_precision="auto"), activation_compression=True)
        oracle_model = deepcopy(model)
        model.compile(backend="eager")
        from deepopt.opt.nn.autograd.normalization.layer import LayerNormFunction
        original, records = LayerNormFunction.forward, []
        def observed(ctx, *args):
            records.append(torch.compiler.is_compiling())
            return original(ctx, *args)
        x = torch.randn(3, 8, requires_grad=True)
        z = x.detach().clone().requires_grad_()
        with patch.object(LayerNormFunction, "forward", new=staticmethod(observed)):
            y, expected = model(x), oracle_model(z)
        y.sum().backward(); expected.sum().backward()
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, z.grad, rtol=0, atol=0)
        self.assertGreaterEqual(len(records), 2)
        self.assertFalse(any(records))


if __name__ == "__main__":
    unittest.main()
