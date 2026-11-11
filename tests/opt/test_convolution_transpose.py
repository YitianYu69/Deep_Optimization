"""Installed-library transposed convolution contracts and independent oracles."""

import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

import deepopt
from deepopt.opt.nn import functional as doF
from deepopt.opt.nn.convolution import source_matrix, restore_matrix
from deepopt.opt.nn.convolution_transpose.geometry import ConvTransposeGeometry
from deepopt.opt.nn.convolution_transpose.modules import TRANSPOSE_TYPES
from deepopt.opt.policy import conv_plan, prepare_fp8_model
from deepopt.opt.precision import precision_report
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.opt.quantization.tensor import QuantizedTensor


CASES = (
    (1, (2, 4, 5), (4, 3, 3), dict(stride=2, padding=1, output_padding=1, groups=2)),
    (2, (2, 4, 3, 4), (4, 3, 4, 4), dict(stride=2, padding=1, groups=2)),
    (3, (1, 4, 3, 3, 4), (4, 1, 2, 2, 2), dict(stride=(2, 1, 2), dilation=(1, 2, 1), groups=4)),
    # Legal output padding >= stride; virtual ordinary output would be larger.
    (1, (1, 3, 4), (3, 2, 3), dict(stride=1, dilation=3, output_padding=2)),
)


def plan(fp8, compression=False, bits=2, blocks=None):
    return conv_plan(config={"runtime": {
        "fp8": {"enabled": fp8, "backend": "reference", "block_shapes": blocks or {}},
        "activation_compression": {"enabled": compression, "bits": bits, "group_size": 17},
    }})


def decode(value, block=None, compress=False, bits=2):
    q = quantize_fp8(source_matrix(value), block_shape=block)
    if compress:
        packed = quantize_pack_activation(q.data, 17, block, bits=bits)
        q = QuantizedTensor(dequantize_unpack_activation(packed, dtype=q.data.dtype), q.scale_inv, block)
    return restore_matrix(q.dequantize(), value.shape)


class ConvTransposeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(208)
        torch.set_num_threads(1)

    def test_native_modes_all_ranks_and_bits(self):
        for rank, shape, wshape, opts in CASES:
            for compression in (False, True):
                for bits in ((1, 2, 4, 8) if compression else (2,)):
                    with self.subTest(rank=rank, opts=opts, compression=compression, bits=bits):
                        x = torch.randn(shape, requires_grad=True)
                        w = torch.randn(wshape, requires_grad=True)
                        b = torch.randn(wshape[1] * opts.get("groups", 1), requires_grad=True)
                        native = getattr(F, f"conv_transpose{rank}d")
                        expected = native(x, w, b, **opts)
                        g = torch.randn_like(expected)
                        dx, dw, db = torch.autograd.grad(expected, (x, w, b), g)
                        saved = []
                        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                            y = getattr(doF, f"conv_transpose{rank}d")(x, w, b, plan=plan(False, compression, bits), **opts)
                        actual = torch.autograd.grad(y, (x, w, b), g)
                        if compression:
                            matrix = dequantize_unpack_activation(quantize_pack_activation(source_matrix(x.detach()), 17, bits=bits), dtype=x.dtype)
                            restored = restore_matrix(matrix, x.shape).contiguous().requires_grad_()
                            dw = torch.autograd.grad(native(restored, w, b, **opts), w, g)[0]
                            self.assertTrue(any(t.dtype == torch.int32 for t in saved))
                            self.assertFalse(any(t is x for t in saved))
                        torch.testing.assert_close(y, expected, rtol=0, atol=0)
                        for value, oracle in zip(actual, (dx, dw, db)):
                            torch.testing.assert_close(value, oracle, rtol=0, atol=0)

    def test_fp8_forward_and_gradients_independent_reference(self):
        for rank, shape, wshape, opts in CASES:
            for compression in (False, True):
                for block in (None, (2, 3)):
                    with self.subTest(rank=rank, opts=opts, compression=compression, block=block):
                        x = torch.randn(shape, requires_grad=True)
                        w = torch.randn(wshape, requires_grad=True)
                        b = torch.randn(wshape[1] * opts.get("groups", 1), requires_grad=True)
                        blocks = {key: block for key in ("activation", "weight", "grad")}
                        y = getattr(doF, f"do_conv_transpose{rank}d")(x, w, b, plan=plan(True, compression, blocks=blocks), **opts)
                        qx = decode(x.detach(), block)
                        qw = quantize_fp8(w.detach().reshape(wshape[0], -1), block_shape=block).dequantize().reshape(wshape)
                        native = getattr(F, f"conv_transpose{rank}d")
                        torch.testing.assert_close(y, native(qx, qw, b, **opts), rtol=0, atol=0)
                        g = torch.randn_like(y)
                        dx, dw, db = torch.autograd.grad(y, (x, w, b), g)
                        qg = decode(g, block)
                        rx = decode(x.detach(), block, compression).requires_grad_()
                        rw = qw.detach().requires_grad_()
                        expected_dx, expected_dw = torch.autograd.grad(native(rx, rw, None, **opts), (rx, rw), qg)
                        torch.testing.assert_close(dx, expected_dx)
                        torch.testing.assert_close(dw, expected_dw)
                        torch.testing.assert_close(db, g.sum((0, *range(2, g.ndim))), rtol=0, atol=0)

    def test_output_size_and_unbatched_module(self):
        for rank in (1, 2, 3):
            for batched in (False, True):
                with self.subTest(rank=rank, batched=batched):
                    original = getattr(nn, f"ConvTranspose{rank}d")(4, 6, 3, stride=2, padding=1, groups=2)
                    prepared, _ = prepare_fp8_model(copy.deepcopy(original), deepopt.FP8Config(enabled=False),
                                                   activation_compression={"enabled": True})
                    self.assertIsInstance(prepared, TRANSPOSE_TYPES)
                    x = torch.randn((*( (2,) if batched else ()), 4, *((3,) * rank)))
                    size = (6,) * rank
                    torch.testing.assert_close(prepared(x, output_size=size), original(x, output_size=size), rtol=0, atol=0)
                    with self.assertRaises(ValueError):
                        prepared(x, output_size=(20,) * rank)

    def test_conversion_parameter_identity_and_exclusion(self):
        for mode in ("modules", "full", "auto"):
            model = nn.Sequential(nn.ConvTranspose2d(4, 8, 4, stride=2, padding=1), nn.SiLU(), nn.Conv2d(8, 4, 1))
            weight = model[0].weight
            prepared, report = prepare_fp8_model(model, deepopt.FP8Config(enabled=True, backend="reference", trace_mode=mode))
            self.assertIs(prepared.get_submodule("0").weight, weight)
            self.assertIn("0", report.converted_modules)
            self.assertEqual(precision_report(prepared)["fallbacks"], ())
            x = torch.randn(2, 4, 3, 4, requires_grad=True)
            prepared(x).square().mean().backward()
            self.assertIsNotNone(weight.grad)
            self.assertEqual(precision_report(prepared)["analyses"], 1)
        original = nn.Sequential(nn.ConvTranspose1d(4, 4, 3))
        prepared, _ = prepare_fp8_model(original, deepopt.FP8Config(enabled=True, backend="reference", trace_mode="modules", intercept_linear=False, exclude_modules=("0",)))
        self.assertIsInstance(prepared[0], nn.ConvTranspose1d)

    def test_root_adapter_preserves_runtime_output_size_in_every_trace_mode(self):
        for rank in (1, 2, 3):
            for mode in ("modules", "full", "auto"):
                with self.subTest(rank=rank, mode=mode):
                    source = getattr(nn, f"ConvTranspose{rank}d")(4, 6, 3, stride=2, padding=1)
                    weight = source.weight
                    model, _ = prepare_fp8_model(source, deepopt.FP8Config(enabled=True, backend="reference", trace_mode=mode))
                    self.assertIs(model.weight, weight)
                    self.assertEqual(precision_report(model)["fallbacks"], ())
                    y = model(torch.randn(1, 4, *((3,) * rank)), output_size=(6,) * rank)
                    self.assertEqual(y.shape, (1, 6, *((6,) * rank)))
                    y.sum().backward()

    def test_fp8_chain_and_backward_transport(self):
        model = nn.Sequential(nn.Conv2d(4, 4, 1), nn.GroupNorm(2, 4), nn.ConvTranspose2d(4, 4, 4, stride=2, padding=1), nn.SiLU(), nn.Conv2d(4, 2, 1))
        prepared, _ = prepare_fp8_model(model, deepopt.FP8Config(enabled=True, backend="reference", trace_mode="modules", backward_precision="auto"))
        observed = []
        prepared[2].register_forward_hook(lambda m, args, out: observed.append((isinstance(args[0], FP8Activation), isinstance(out, FP8Activation))))
        prepared(torch.randn(2, 4, 3, 3, requires_grad=True)).float().square().mean().backward()
        self.assertEqual(observed, [(True, True)])
        self.assertEqual(precision_report(prepared)["fallbacks"], ())
        self.assertTrue(any(v["precision"] == "fp8" for v in precision_report(prepared)["backward_edges"]))

    def test_geometry_explicit_adjoint_bounds(self):
        for rank, shape, wshape, opts in CASES:
            x, w = torch.randn(shape), torch.randn(wshape)
            geom = ConvTransposeGeometry.from_tensors(x, w, **opts)
            expected = getattr(F, f"conv_transpose{rank}d")(x, w, **opts)
            self.assertEqual(geom.output_shape, tuple(expected.shape))
            self.assertEqual(geom.adjoint.input_shape, geom.output_shape)
            self.assertEqual(geom.adjoint.output_shape, geom.input_shape)
            self.assertEqual(geom.weight_matrix_shape, (wshape[0], w[0].numel()))

    def test_functional_owners_current_delayed_and_interception(self):
        class Functional(nn.Module):
            def __init__(self, rank, direct):
                super().__init__()
                self.rank, self.direct = rank, direct
                self.weight = nn.Parameter(torch.randn(4, 4, *((3,) * rank)))
                self.activation = nn.SiLU()

            def forward(self, x):
                call = getattr(doF, f"do_conv_transpose{self.rank}d") if self.direct else getattr(F, f"conv_transpose{self.rank}d")
                return self.activation(call(x, self.weight, stride=2, padding=1))

        for rank in (1, 2, 3):
            for recipe in ("current", "delayed"):
                for mode in ("modules", "full"):
                    for direct in (False, True):
                        with self.subTest(rank=rank, recipe=recipe, mode=mode, direct=direct):
                            model, _ = prepare_fp8_model(Functional(rank, direct), deepopt.FP8Config(
                                enabled=True, backend="reference", trace_mode=mode, scaling_recipe=recipe))
                            self.assertEqual(precision_report(model)["fallbacks"], ())
                            for _ in range(2):
                                model(torch.randn(1, 4, *((3,) * rank))).float().sum().backward()
                                model.zero_grad(set_to_none=True)

    def test_saved_fp8_input_reused_and_no_double_quantization(self):
        from unittest.mock import patch

        x = torch.randn(2, 4, 3, 3, requires_grad=True)
        qx = to_fp8_activation(x, layout="channels", block_shape=(2, 3))
        w = torch.randn(4, 3, 4, 4, requires_grad=True)
        for compress in (False, True):
            with patch("deepopt.opt.nn.autograd.convolution_transpose.fp8.quantize_fp8", wraps=quantize_fp8) as cast:
                y = doF.do_conv_transpose2d(qx, w, plan=plan(True, compress))
                # The only ordinary cast is the weight, not the FP8 source.
                self.assertEqual(cast.call_count, 1)
            y.sum().backward(retain_graph=True)

    def test_cache_version_state_roundtrip_and_delayed_scaling(self):
        from tests.opt.test_convolution_cache import _CountingWeightCache

        settings = deepopt.FP8Config(enabled=True, backend="reference", trace_mode="modules", scaling_recipe="delayed")
        original = nn.Sequential(nn.ConvTranspose2d(4, 4, 4, stride=2, padding=1), nn.SiLU())
        model, _ = prepare_fp8_model(original, settings, activation_compression={"enabled": True})
        model[0]._fp8_weight_cache = cache = _CountingWeightCache()
        x = torch.randn(2, 4, 3, 3)
        with torch.no_grad():
            first = model(x)
            model(x)
            # Delayed updates can change the recipe token; cache must follow it.
            self.assertGreaterEqual(cache.misses, 1)
            model[0].weight.add_(0.25)
            changed = model(x)
        self.assertFalse(torch.equal(first, changed))
        state = copy.deepcopy(model.state_dict())
        restored, _ = prepare_fp8_model(nn.Sequential(nn.ConvTranspose2d(4, 4, 4, stride=2, padding=1), nn.SiLU()),
                                        settings, activation_compression={"enabled": True})
        restored.load_state_dict(state)
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)

    def test_invalid_geometry(self):
        x, w = torch.randn(1, 4, 3), torch.randn(4, 2, 3)
        for kwargs in (dict(output_padding=2, stride=2), dict(groups=3), dict(padding=-1), dict(stride=0), dict(dilation=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, RuntimeError)):
                doF.do_conv_transpose1d(x, w, plan=plan(True), **kwargs)

    def test_bf16_amp_layout_partial_gradients_empty_batch(self):
        for rank in (1, 2, 3):
            for mask in ((True, False, False), (False, True, False), (False, False, True)):
                shape = (1, 4, *((3,) * rank))
                x = torch.randn(shape, requires_grad=mask[0])
                w = torch.randn((4, 2, *((3,) * rank)), requires_grad=mask[1])
                b = torch.randn(2, requires_grad=mask[2])
                if rank > 1:
                    fmt = torch.channels_last if rank == 2 else torch.channels_last_3d
                    x = x.detach().contiguous(memory_format=fmt).requires_grad_(mask[0])
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    y = getattr(doF, f"do_conv_transpose{rank}d")(x, w, b, plan=plan(False, True))
                self.assertEqual(y.dtype, torch.bfloat16)
                y.float().sum().backward()
                self.assertEqual(tuple(v.grad is not None for v in (x, w, b)), mask)
            empty = torch.randn((0, 4, *((3,) * rank)), requires_grad=True)
            y = getattr(doF, f"do_conv_transpose{rank}d")(empty, w.detach(), plan=plan(True))
            self.assertEqual(y.shape[0], 0)
            y.sum().backward()


if __name__ == "__main__":
    unittest.main()
