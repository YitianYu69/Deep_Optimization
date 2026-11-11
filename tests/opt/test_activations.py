"""Activation coefficients, precision transport and preparation as installed APIs."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

import deepopt
from deepopt.opt.nn import functional as doF
from deepopt.opt.nn.activation.elementwise_modules import CONVERSIONS
from deepopt.opt.policy import activation_plan, prepare_fp8_model
from deepopt.opt.precision import precision_report
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation, dequantize_activation
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.opt.backends.reference.activation.math import evaluate


VARIANTS = (("sigmoid", "none"), ("silu", "none"), ("gelu", "none"),
            ("gelu", "tanh"), ("tanh", "none"), ("relu6", "none"))


def policy(compressed=False, block=None, bits=2, group=17, recipe="current", backward="native"):
    return activation_plan(fp8={"enabled": True, "backend": "reference", "scaling_recipe": recipe,
                                "backward_precision": backward, "block_shapes": {"activation": block}},
                           activation_compression={"enabled": compressed, "bits": bits, "group_size": group})


def call(kind, x, approximate="none", **options):
    return getattr(doF, kind)(x, **({"approximate": approximate} if kind == "gelu" else {}), **options)


class ActivationTests(unittest.TestCase):
    def test_all_precision_pairs_and_bit_widths_use_preprocessed_coefficients(self):
        torch.manual_seed(91)
        for kind, approximate in VARIANTS:
            for block in (None, (2, 8)):
                for bits in (1, 2, 4, 8):
                    for source_fp8 in (False, True):
                        for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
                            with self.subTest(kind=kind, approximate=approximate, block=block, bits=bits, source=source_fp8, output=dtype):
                                x = torch.randn(3, 33, dtype=torch.bfloat16).requires_grad_()
                                source = to_fp8_activation(x, layout="last", block_shape=block) if source_fp8 else x
                                represented = source.dequantize(torch.float32) if source_fp8 else x.detach().float()
                                y, d = evaluate(represented, kind, approximate)
                                saved = []
                                with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                                    output = call(kind, source, approximate, plan=policy(True, block, bits), output_dtype=dtype)
                                if dtype == torch.bfloat16:
                                    torch.testing.assert_close(output, y.bfloat16(), rtol=0, atol=0)
                                else:
                                    q = quantize_fp8(y.float(), fp8_dtype=dtype, block_shape=block)
                                    torch.testing.assert_close(output._data.float(), q.data.float(), rtol=0, atol=0)
                                    torch.testing.assert_close(output._scale_inv, q.scale_inv, rtol=0, atol=0)
                                if kind == "relu6":
                                    self.assertEqual(len(saved), 1)
                                    self.assertEqual(saved[0].dtype, torch.int32)
                                    self.assertEqual(saved[0].numel(), (x.numel() + 31) // 32)
                                else:
                                    oracle = quantize_pack_activation(d, 17, block, bits=bits)
                                    self.assertEqual(len(saved), 3)
                                    for actual, expected in zip(saved, (oracle.data, oracle.step, oracle.minimum)):
                                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                                    d = dequantize_unpack_activation(oracle, dtype=torch.float32)
                                dy = torch.randn_like(x)
                                output.backward(dy)
                                expected_dx = torch.where(d, dy.float(), 0) if kind == "relu6" else dy.float() * d
                                torch.testing.assert_close(x.grad, expected_dx.bfloat16(), rtol=0, atol=0)

    def test_compression_does_not_change_forward_and_uncompressed_gradcheck(self):
        for kind, approximate in VARIANTS:
            x = (torch.randn(2, 5, dtype=torch.float64) + 0.3).requires_grad_()
            native = activation_plan(fp8=False, activation_compression=False)
            self.assertTrue(torch.autograd.gradcheck(lambda z: call(kind, z, approximate, plan=native), (x,)))
            raw = x.detach().bfloat16().requires_grad_()
            a = call(kind, raw, approximate, plan=policy(False))
            b = call(kind, raw, approximate, plan=policy(True))
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_special_values_and_relu6_thresholds_follow_native(self):
        values = torch.tensor([-float("inf"), -8, -0.0, 0, 1, 6, 7, float("inf"), float("nan")])
        dy = torch.tensor([2., 3, float("nan"), float("inf"), 5, float("nan"), float("inf"), 6, 7])
        for kind, approximate in VARIANTS:
            for compressed in ((False, True) if kind == "relu6" else (False,)):
                x, z = values.clone().requires_grad_(), values.clone().requires_grad_()
                output = call(kind, x, approximate, plan=activation_plan(False, activation_compression=compressed))
                native = getattr(F, kind)(z, **({"approximate": approximate} if kind == "gelu" else {}))
                torch.testing.assert_close(output, native, equal_nan=True)
                output.backward(dy)
                native.backward(dy)
                torch.testing.assert_close(x.grad, z.grad, equal_nan=True)

    def test_scalar_empty_strides_layouts_and_fp8_upstream(self):
        for kind, approximate in VARIANTS:
            for shape in ((), (0,), (2, 0, 5), (2, 3, 5, 7)):
                x = torch.randn(shape, dtype=torch.bfloat16)
                x = (x.transpose(2, 3) if len(shape) == 4 else x).requires_grad_()
                for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                    output = call(kind, x, approximate, plan=policy(True), output_dtype=dtype)
                    self.assertEqual(tuple(output.shape), tuple(x.shape))
                    dequantize_activation(output).sum().backward()
                    x.grad = None
            x = torch.randn(2, 3, 5, 7, dtype=torch.bfloat16).transpose(2, 3).requires_grad_()
            source = to_fp8_activation(x, layout="channels", block_shape=(2, 8))
            output = call(kind, source, approximate, plan=policy(False, (2, 8)),
                          output_dtype=torch.bfloat16, output_layout="last")
            dy = to_fp8_activation(torch.randn_like(x), layout="channels", fp8_dtype=torch.float8_e5m2)
            output.backward(dy)
            _, d = evaluate(source.dequantize(torch.float32), kind, approximate)
            expected = torch.where(d, dy.dequantize(torch.float32), 0) if kind == "relu6" else dy.dequantize(torch.float32) * d
            torch.testing.assert_close(x.grad, expected.bfloat16())

    def test_inplace_aliases_versions_leaf_errors_and_inference(self):
        for kind in ("sigmoid", "silu", "tanh", "relu6"):
            for compressed in (False, True):
                x = torch.randn(3, 33, dtype=torch.bfloat16, requires_grad=True)
                value = x + 0
                original = value.detach().clone()
                version = value._version
                out = call(kind, value, plan=policy(compressed), inplace=True)
                self.assertIs(out, value)
                self.assertGreater(value._version, version)
                torch.testing.assert_close(out, evaluate(original.float(), kind)[0].bfloat16())
                out.sum().backward()
                with self.assertRaisesRegex(RuntimeError, "leaf"):
                    call(kind, x, inplace=True, plan=policy())
                with torch.no_grad():
                    self.assertIs(call(kind, value, inplace=True, plan=policy(compressed)), value)
            value = to_fp8_activation(x, layout="last")
            if kind == "relu6":
                self.assertIs(call(kind, value, inplace=True, plan=policy()), value)
            else:
                with self.assertRaises(ValueError):
                    call(kind, value, inplace=True, plan=policy())
        with self.assertRaises(ValueError):
            doF.gelu(torch.ones(3), inplace=True)

    def test_meta_and_fake_tensor_payloads_do_not_read_values(self):
        for kind, approximate in VARIANTS:
            for device in ("meta", "cpu"):
                with FakeTensorMode() if device == "cpu" else torch.device("meta"):
                    x = torch.empty(3, 33, device=device, dtype=torch.bfloat16, requires_grad=True)
                    out = call(kind, x, approximate, plan=policy(True), output_dtype=torch.float8_e4m3fn)
                    self.assertIsInstance(out, FP8Activation)
                    self.assertEqual(tuple(out.shape), (3, 33))
                    dequantize_activation(out).sum().backward()

    def test_prepared_inplace_silu_and_relu6_keep_valid_native_boundaries(self):
        for source in (nn.SiLU, nn.ReLU6):
            model, _ = prepare_fp8_model(nn.Sequential(nn.GroupNorm(2, 8), source(inplace=True), nn.Conv2d(8, 8, 1)),
                                         deepopt.FP8Config(enabled=True, backward_precision="auto"),
                                         activation_compression=True)
            model(torch.randn(2, 8, 3, 3, requires_grad=True)).sum().backward()
            self.assertFalse(precision_report(model)["fallbacks"])

    def test_scoped_functionals_tensor_methods_and_wrapper_dispatch(self):
        for kind, approximate in VARIANTS:
            x = torch.randn(3, 33, dtype=torch.bfloat16, requires_grad=True)
            with deepopt.autocast("cpu", enabled=False, fp8=False, activation_compression=True):
                output = getattr(F, kind)(x, **({"approximate": approximate} if kind == "gelu" else {}))
            expected = call(kind, x, approximate, plan=activation_plan(False, activation_compression=True))
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            output.sum().backward()
            source = to_fp8_activation(x, layout="last")
            wrapped = getattr(F, kind)(source, **({"approximate": approximate} if kind == "gelu" else {}))
            self.assertIsInstance(wrapped, FP8Activation)
            if kind in ("sigmoid", "tanh"):
                self.assertIsInstance(getattr(source, kind)(), FP8Activation)
                value = x + 0
                with deepopt.autocast("cpu", enabled=False, fp8=False, activation_compression=True):
                    actual = getattr(torch, kind + "_")(value)
                self.assertIs(actual, value)
                actual.sum().backward()

    def test_prepared_modules_and_functionals_cache_both_precision_directions(self):
        for native, target in CONVERSIONS.items():
            for trace in ("modules", "full"):
                model = nn.Sequential(nn.Linear(8, 8), native(), nn.Linear(8, 3))
                settings = deepopt.FP8Config(enabled=True, trace_mode=trace, backward_precision="auto")
                model, _ = prepare_fp8_model(model, settings, activation_compression=True)
                self.assertIsInstance(model.get_submodule("1"), target)
                report = precision_report(model)
                self.assertFalse(report["fallbacks"])
                self.assertTrue(any(e["precision"] == "fp8" for e in report["backward_edges"]))
                with patch("deepopt.opt.fx.tracer.DOTracer.trace", side_effect=AssertionError("must not retrace")):
                    for _ in range(2):
                        model(torch.randn(3, 8)).sum().backward()
                self.assertEqual(precision_report(model)["analyses"], 1)
        for kind, approximate in VARIANTS:
            class Functional(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.a, self.b = nn.Linear(8, 8), nn.Linear(8, 3)

                def forward(self, x):
                    y = self.a(x)
                    return self.b(getattr(F, kind)(y, **({"approximate": approximate} if kind == "gelu" else {})))
            for trace in ("modules", "full"):
                model, _ = prepare_fp8_model(Functional(), deepopt.FP8Config(enabled=True, trace_mode=trace,
                                             backward_precision="auto"), activation_compression=True)
                self.assertFalse(precision_report(model)["fallbacks"])
                model(torch.randn(3, 8)).sum().backward()
                self.assertTrue(any(isinstance(m, tuple(CONVERSIONS.values())) for m in model.modules()))

    def test_sharing_compression_only_preparation_and_gelu_option(self):
        for source, target in CONVERSIONS.items():
            shared = source(approximate="tanh") if source is nn.GELU else source()
            class Shared(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.a = self.b = shared

                def forward(self, x):
                    return self.b(self.a(x))
            model, _ = prepare_fp8_model(Shared(), deepopt.FP8Config(enabled=False), activation_compression=True)
            self.assertIs(model.a, model.b)
            self.assertIsInstance(model.a, target)
            self.assertEqual(model.a.approximate, "tanh" if source is nn.GELU else "none")
            model(torch.randn(3, 8, requires_grad=True)).sum().backward()

    def test_direct_do_functionals_get_owned_forward_and_backward_slots(self):
        for kind, approximate in VARIANTS:
            class Direct(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.a, self.b = nn.Linear(8, 8), nn.Linear(8, 3)

                def forward(self, x):
                    return self.b(getattr(doF, kind)(self.a(x), **({"approximate": approximate} if kind == "gelu" else {})))
            for trace in ("modules", "full"):
                settings = deepopt.FP8Config(enabled=True, trace_mode=trace, backward_precision="auto", scaling_recipe="delayed")
                model, _ = prepare_fp8_model(Direct(), settings, activation_compression=True)
                report = precision_report(model)
                self.assertFalse(report["fallbacks"], report)
                with deepopt.fp8_scaling_scope():
                    model(torch.randn(3, 8, requires_grad=True)).sum().backward()
                activation = [m for m in model.modules() if isinstance(m, tuple(CONVERSIONS.values()))]
                self.assertTrue(activation)
                self.assertTrue(any(m.fp8_scaling is not None and m.fp8_scaling.updates[0] == 1 for m in activation))
                self.assertTrue(any(e["precision"] == "fp8" for e in report["backward_edges"]))

    def test_fixed_scales_delayed_resume_and_checkpoint_recomputation(self):
        for source, _ in CONVERSIONS.items():
            torch.manual_seed(17)
            model = nn.Sequential(source(), nn.Linear(8, 3))
            settings = deepopt.FP8Config(enabled=True, backward_precision="auto", scaling_recipe="delayed")
            model, _ = prepare_fp8_model(model, settings, activation_compression=True)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            runtime = deepopt.TrainingRuntime(model, optimizer, checks=False)
            x = torch.randn(3, 8, requires_grad=True)
            runtime.zero_grad()
            loss = runtime.compute(x, lambda z: deepopt.fp8_checkpoint(model, z).square().mean())
            runtime.backward(loss)
            runtime.step()
            state = deepcopy(model.state_dict())
            self.assertGreater(model[0].fp8_scaling.updates[0].item(), 0)
            restored, _ = prepare_fp8_model(nn.Sequential(source(), nn.Linear(8, 3)), settings, activation_compression=True)
            restored.load_state_dict(state)
            model.eval(); restored.eval()
            torch.testing.assert_close(model(x), restored(x))

    def test_explicit_output_scale_and_invalid_options(self):
        for kind, approximate in VARIANTS:
            x = torch.randn(3, 33, requires_grad=True)
            scale = torch.full((1, 1), 2.0)
            out = call(kind, x, approximate, plan=policy(True), output_dtype=torch.float8_e4m3fn, output_scale=scale)
            expected = (evaluate(x.detach(), kind, approximate)[0] * 2).clamp(-448, 448).to(torch.float8_e4m3fn)
            torch.testing.assert_close(out._data.float(), expected.float(), rtol=0, atol=0)
            torch.testing.assert_close(out._scale_inv, scale.reciprocal())
            with self.assertRaises((TypeError, ValueError)):
                call(kind, x, approximate, output_scale=torch.ones(1, 1))
        with self.assertRaises(ValueError):
            doF.gelu(torch.ones(3), approximate="bad")
        with self.assertRaisesRegex(ValueError, "finite coefficients"):
            doF.silu(torch.tensor([float("inf")], requires_grad=True), plan=policy(True))

    def test_compiled_surrounding_model_keeps_activation_kernels_eager(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        for source in CONVERSIONS:
            model, _ = prepare_fp8_model(nn.Sequential(nn.Linear(8, 8), source(), nn.ELU(), nn.Linear(8, 3)),
                                         deepopt.FP8Config(enabled=True, backward_precision="auto"),
                                         activation_compression=True)
            oracle = deepcopy(model)
            graphs = []
            def backend(graph, inputs):
                graphs.append(graph)
                return graph.forward
            model.compile(backend=backend)
            x, z = torch.randn(3, 8, requires_grad=True), torch.randn(3, 8, requires_grad=True)
            z.data.copy_(x.data)
            from deepopt.opt.nn.autograd.activation.elementwise import ActivationFunction

            original, compiling = ActivationFunction.forward, []
            def observed(ctx, *args):
                compiling.append(torch.compiler.is_compiling())
                return original(ctx, *args)
            with patch.object(ActivationFunction, "forward", new=staticmethod(observed)):
                actual, expected = model(x), oracle(z)
            actual.sum().backward(); expected.sum().backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, z.grad, rtol=0, atol=0)
            nodes = [node for graph in graphs for node in graph.graph.nodes]
            self.assertGreaterEqual(len(compiling), 2)
            self.assertFalse(any(compiling), compiling)
            self.assertFalse(any(n.target in (torch.isfinite, torch.amax, "amax") for n in nodes))


if __name__ == "__main__":
    unittest.main()
