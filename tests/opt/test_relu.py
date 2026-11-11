"""ReLU precision combinations, exact saved-mask gradients and preparation."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

import deepopt
from deepopt.opt.nn import functional as doF
from deepopt.opt.nn.activation.modules import DOReLU
from deepopt.opt.policy import prepare_fp8_model, relu_plan
from deepopt.opt.precision import precision_report
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation, dequantize_activation
from deepopt.opt.quantization.boolean import pack_boolean_mask
from deepopt.opt.quantization.quantizer import quantize_fp8


def plan(compressed=True, block=None, recipe="current"):
    return relu_plan(fp8={"enabled": True, "backend": "reference", "scaling_recipe": recipe,
                          "block_shapes": {"activation": block}},
                     activation_compression={"enabled": compressed, "bits": 4, "group_size": 17})


class ReLUTests(unittest.TestCase):
    def test_all_precision_pairs_and_mask_policies_match_explicit_oracles(self):
        for shape in ((3, 33), (2, 4, 3, 5)):
            for source_fp8 in (False, True):
                for output_dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
                    for block in (None, (4, 4)):
                        for compressed in (False, True):
                            with self.subTest(shape=shape, source_fp8=source_fp8, dtype=output_dtype, block=block, act=compressed):
                                torch.manual_seed(17)
                                x = torch.randn(shape, dtype=torch.bfloat16).requires_grad_()
                                source = to_fp8_activation(x, layout="last", block_shape=block) if source_fp8 else x
                                dense = source.dequantize(torch.float32) if source_fp8 else x.detach().float()
                                saved = []
                                with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
                                    actual = doF.relu(source, plan=plan(compressed, block), output_dtype=output_dtype)
                                expected = dense.relu()
                                if output_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                                    if source_fp8 and source._data.dtype == output_dtype:
                                        inverse = source._scale_inv
                                        data = torch.where(source._data.float() < 0, 0.0, source._data.float()).to(output_dtype)
                                        torch.testing.assert_close(actual._scale_inv, inverse, rtol=0, atol=0)
                                    else:
                                        q = quantize_fp8(expected.reshape(-1, shape[-1]), fp8_dtype=output_dtype, block_shape=block)
                                        data = q.data
                                        torch.testing.assert_close(actual._scale_inv, q.scale_inv, rtol=0, atol=0)
                                    torch.testing.assert_close(actual._data.float(), data.float(), rtol=0, atol=0)
                                else:
                                    self.assertNotIsInstance(actual, FP8Activation)
                                    torch.testing.assert_close(actual, expected.to(output_dtype), rtol=0, atol=0)
                                self.assertEqual(len(saved), 1)
                                predicate = ~(dense <= 0)
                                if compressed:
                                    self.assertEqual(saved[0].dtype, torch.int32)
                                    torch.testing.assert_close(saved[0], pack_boolean_mask(predicate.reshape(-1, shape[-1])).data)
                                    self.assertEqual(saved[0].numel(), (x.numel() + 31) // 32)
                                else:
                                    self.assertEqual(saved[0].dtype, torch.bool)
                                    torch.testing.assert_close(saved[0], predicate.reshape(-1))
                                gradient = torch.randn(shape, dtype=torch.bfloat16)
                                actual.backward(gradient)
                                torch.testing.assert_close(x.grad, torch.where(predicate, gradient, 0), rtol=0, atol=0)

    def test_special_values_match_native_forward_and_threshold_backward(self):
        values = torch.tensor([-float("inf"), -2, -0.0, 0.0, 1, float("inf"), float("nan")], dtype=torch.bfloat16)
        dy = torch.tensor([float("nan"), float("inf"), float("nan"), 3, 4, 5, 6], dtype=torch.bfloat16)
        for compressed in (False, True):
            x, native = values.clone().requires_grad_(), values.clone().requires_grad_()
            actual = doF.relu(x, plan=relu_plan(fp8=False, activation_compression=compressed))
            expected = F.relu(native)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            self.assertEqual(torch.signbit(actual).tolist(), torch.signbit(expected).tolist())
            actual.backward(dy)
            expected.backward(dy)
            torch.testing.assert_close(x.grad, native.grad, rtol=0, atol=0, equal_nan=True)

    def test_packed_mask_uses_pre_cast_predicate_when_fp8_output_underflows(self):
        x = torch.tensor([[1e-8, -1e-8, 0.0, 1.0]], dtype=torch.bfloat16, requires_grad=True)
        y = doF.relu(x, plan=plan(), output_dtype=torch.float8_e4m3fn, output_scale=torch.ones(1, 1))
        self.assertEqual(y._data.float()[0, 0].item(), 0.0)
        y.backward(torch.ones_like(x))
        torch.testing.assert_close(x.grad, torch.tensor([[1, 0, 0, 1]], dtype=torch.bfloat16))

    def test_scalar_empty_strided_channels_and_fp8_gradient_inputs(self):
        for shape in ((), (0,), (2, 0, 5), (3, 33)):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                x = torch.randn(shape, dtype=torch.bfloat16).requires_grad_()
                y = doF.relu(x, plan=plan(), output_dtype=dtype)
                self.assertEqual(tuple(y.shape), shape)
                dequantize_activation(y).sum().backward()
                torch.testing.assert_close(x.grad, (x > 0).to(x.dtype))
        source = torch.randn(2, 4, 5, 7, dtype=torch.bfloat16).transpose(2, 3).requires_grad_()
        q = to_fp8_activation(source, layout="channels", block_shape=(4, 4))
        output = doF.relu(q, plan=plan(block=(4, 4)), output_layout="last", output_dtype=torch.float8_e5m2)
        gradient = to_fp8_activation(torch.randn_like(source), layout="channels", fp8_dtype=torch.float8_e5m2)
        output.backward(gradient)
        expected = torch.where(q.dequantize() <= 0, 0, gradient.dequantize()).to(source.dtype)
        torch.testing.assert_close(dequantize_activation(source.grad), expected)

    def test_inference_saves_no_mask_and_both_inplace_paths_preserve_aliases(self):
        for fp8 in (False, True):
            x = torch.randn(3, 33, dtype=torch.bfloat16, requires_grad=True)
            value = to_fp8_activation(x, layout="last") if fp8 else x + 0
            expected = value.dequantize() if fp8 else value.detach().clone()
            alias = value.detach()
            output = doF.relu(value, inplace=True, plan=plan())
            self.assertIs(output, value)
            torch.testing.assert_close(dequantize_activation(alias), expected.relu().to(x.dtype))
            output.backward(torch.ones_like(x))
            torch.testing.assert_close(x.grad, (expected > 0).to(x.dtype))
            saved = []
            with torch.no_grad(), torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                doF.relu(value, plan=plan())
                self.assertIs(doF.relu(value, inplace=True, plan=plan()), value)
            self.assertEqual(saved, [])
            plain = value.detach()
            self.assertIs(doF.relu(plain, inplace=True, plan=plan()), plain)
        leaf = torch.randn(3, 4, requires_grad=True)
        snapshot = leaf.detach().clone()
        for value in (leaf, leaf[:, :2]):
            with self.assertRaisesRegex(RuntimeError, "leaf"):
                doF.relu(value, inplace=True)
        torch.testing.assert_close(leaf, snapshot)
        with self.assertRaisesRegex(ValueError, "change dtype"):
            doF.relu(snapshot, inplace=True, output_dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "overlapping"):
            doF.relu(torch.ones(1, 4).expand(3, 4), inplace=True)

    def test_inplace_invalidates_previous_saves(self):
        x = torch.randn(3, 4, requires_grad=True)
        y = x + 0
        previous = y.square()
        doF.relu(y, inplace=True)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace"):
            previous.sum().backward()
        class SaveFP8Data(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                ctx.save_for_backward(value._data)
                return value.dequantize()

            @staticmethod
            def backward(ctx, gradient):
                _ = ctx.saved_tensors
                return gradient

        x = torch.randn(3, 4, requires_grad=True)
        q = to_fp8_activation(x, layout="last")
        saved = SaveFP8Data.apply(q)
        doF.relu(q, inplace=True)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace"):
            saved.sum().backward()

    def test_ordinary_relu_apis_use_custom_mask_backward_in_scope_and_for_fp8(self):
        for call in (torch.relu, F.relu, lambda x: x.relu(), nn.ReLU()):
            for fp8 in (False, True):
                x = torch.randn(3, 33, dtype=torch.bfloat16, requires_grad=True)
                value = to_fp8_activation(x, layout="last") if fp8 else x
                with deepopt.autocast("cpu", enabled=False, fp8=False if not fp8 else {"enabled": True, "backend": "reference"},
                                      activation_compression=True):
                    output = call(value)
                self.assertIn("ReLUFunction", type(output.grad_fn).__name__)
                output.backward(torch.ones_like(x))
                dense = value.dequantize() if fp8 else x
                torch.testing.assert_close(x.grad, (dense > 0).to(x.dtype))

    def test_prepared_modules_and_functionals_analyze_once_and_select_output_per_callsite(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()
                self.linear = nn.Linear(8, 8)

            def forward(self, x):
                a = self.linear(self.relu(x))
                b = self.relu(x + 1)
                c = self.linear(F.relu(x + 2))
                d = self.linear((x + 3).relu())
                return a + b + c + d

        for trace in ("modules", "full"):
            model, _ = prepare_fp8_model(Model(), deepopt.FP8Config(enabled=True, backend="reference", trace_mode=trace),
                                         activation_compression=True)
            self.assertEqual(precision_report(model)["analyses"], 1)
            self.assertEqual(precision_report(model)["fallbacks"], ())
            self.assertTrue(any(isinstance(m, DOReLU) for m in model.modules()))
            with patch("deepopt.opt.fx.tracer.DOTracer.trace", side_effect=AssertionError("retrace")):
                for target in (model, deepcopy(model)):
                    target(torch.randn(3, 8, requires_grad=True)).sum().backward()
            relus = [m for m in model.modules() if isinstance(m, DOReLU)]
            self.assertTrue(any("last" in choices for m in relus for choices in m._precision_outputs.values()))

    def test_delayed_scaling_state_and_checkpoint_replay(self):
        model = nn.Sequential(nn.ReLU(), nn.Linear(8, 8))
        model, _ = prepare_fp8_model(model, deepopt.FP8Config(enabled=True, backend="reference", trace_mode="modules",
                                                           scaling_recipe="delayed"), activation_compression=True)
        x = torch.randn(3, 8, requires_grad=True)
        output = deepopt.fp8_checkpoint(model, x, use_reentrant=False)
        output.sum().backward()
        self.assertEqual(model[0].fp8_scaling.updates[0].item(), 1)
        self.assertEqual(model[0].execution_info["scale_mode"], "delayed")
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_fake_cuda_and_meta_never_load_real_kernels(self):
        for device in ("cpu", "cuda:0", "meta"):
            with FakeTensorMode():
                gradients = device != "cuda:0" or torch.version.cuda is not None
                x = torch.empty(2, 4, 3, 5, device=device, dtype=torch.bfloat16, requires_grad=gradients)
                y = doF.relu(x, plan=plan(block=(4, 4)), output_dtype=torch.float8_e4m3fn, output_layout="channels")
                self.assertIsInstance(y, FP8Activation)
                z = doF.relu(y, plan=plan(), output_dtype=torch.bfloat16)
                self.assertEqual(tuple(z.shape), tuple(x.shape))
                if gradients:
                    z.sum().backward()
                    self.assertEqual(tuple(x.grad.shape), tuple(x.shape))

    def test_compiled_prepared_model_retains_precision_choices_and_mask_backward(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        model, _ = prepare_fp8_model(nn.Sequential(nn.ReLU(), nn.Linear(8, 8)),
                                     deepopt.FP8Config(enabled=True, backend="reference", trace_mode="modules"),
                                     activation_compression=True)
        compiled = deepcopy(model)
        compiled.compile(backend="eager", fullgraph=False)
        x = torch.randn(3, 8, requires_grad=True)
        other = x.detach().clone().requires_grad_()
        expected = model(x)
        actual = compiled(other)
        expected.sum().backward()
        actual.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(other.grad, x.grad, rtol=0, atol=0)
        self.assertEqual(precision_report(compiled)["analyses"], 1)


if __name__ == "__main__":
    unittest.main()
