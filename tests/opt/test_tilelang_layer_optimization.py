"""Independent semantics for cooperative LayerNorm; CUDA evidence is separate."""

from contextlib import contextmanager, ExitStack
import importlib
import importlib.util
import os
import unittest
from unittest.mock import patch

import torch

from deepopt.opt.backends.tilelang.normalization import layer
from deepopt.opt.backends.tilelang.normalization.layer import fast
from deepopt.opt.nn.normalization.layer_geometry import LayerNormGeometry
from tests.opt.test_tilelang_operations import _Language, interpreted_kernels


class ReductionLanguage(_Language):
    @staticmethod
    def reduce_sum(source, output, dim=-1, clear=True):
        value = source.sum(dim=dim).reshape(output.shape)
        output.copy_(value if clear else output + value)

    @staticmethod
    def reduce_max(source, output, dim=-1, clear=True):
        value = source.amax(dim=dim).reshape(output.shape)
        output.copy_(value if clear else torch.maximum(output, value))


def clear_factories():
    for value in vars(fast).values():
        if hasattr(value, "cache_clear"):
            value.cache_clear()


@contextmanager
def interpreted_layer():
    language = ReductionLanguage()
    clear_factories()
    with interpreted_kernels(), ExitStack() as stack:
        for module in (fast, importlib.import_module("deepopt.opt.backends.tilelang.quantization.fp8"),
                       importlib.import_module("deepopt.opt.backends.tilelang.scaling.reduction")):
            stack.enter_context(patch.object(module, "runtime", lambda: (None, language)))
            stack.enter_context(patch.object(module, "compile_kernel", language.compile))
        stack.enter_context(patch.object(fast, "require_cuda", lambda value, *args: value.device))
        try:
            yield
        finally:
            clear_factories()


def independent_stats(x, geometry):
    values = x.detach().float().cpu().reshape(geometry.matrix_shape)
    # Accumulate independently, then retain the rounded FP32 center used by
    # both saved-mean reconstruction and the existing two-pass contract.
    mean = values.double().mean(1).float()
    variance = (values.double() - mean.double()[:, None]).square().mean(1).float()
    inverse = torch.rsqrt(variance + geometry.eps)
    return values, mean, inverse, (values - mean[:, None]) * inverse[:, None]


def independent_backward(dy, h, inverse, weight):
    dy = dy.detach().float().cpu().reshape(h.shape)
    u = dy if weight is None else dy * weight.detach().float().cpu().reshape(1, -1)
    dx = (u - u.mean(1, keepdim=True) - h * (u * h).mean(1, keepdim=True)) * inverse[:, None]
    return dx, (dy * h).sum(0), dy.sum(0)


class LayerOptimizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1711)

    kernel_context = staticmethod(interpreted_layer)

    def close(self, actual, expected, *, atol=5e-6, rtol=5e-6):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), atol=atol, rtol=rtol)

    def test_native_forward_ragged_multidimensional_suffix_and_affine(self):
        with self.kernel_context():
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                for width in (1, 7, 33, 257):
                    x = torch.randn(3, width, dtype=dtype)
                    g = LayerNormGeometry(tuple(x.shape), (width,))
                    w, b = torch.randn(width), torch.randn(width)
                    _, mean, inverse, h = independent_stats(x, g)
                    result, mu, inv, saved = fast.forward(x, w, b, g, dtype)
                    self.close(mu, mean)
                    self.close(inv, inverse, atol=3e-5)
                    self.close(result, (h * w.cpu() + b.cpu()).to(dtype), atol=.032 if dtype == torch.bfloat16 else .002 if dtype == torch.float16 else 5e-6,
                               rtol=0 if dtype != torch.float32 else 5e-6)
                    self.assertIsNone(saved)
            x = torch.randn(2, 3, 5)
            g = LayerNormGeometry(tuple(x.shape), (3, 5))
            w, b = torch.randn(5, 3).T, torch.randn(5, 3).T
            _, _, _, h = independent_stats(x, g)
            result, _, _, _ = fast.forward(x, w, b, g, torch.float32)
            self.close(result, h * w.cpu().reshape(1, -1) + b.cpu().reshape(1, -1))

    def test_constant_and_large_offset_tiny_variance_are_stable(self):
        with self.kernel_context():
            x = torch.full((3, 33), 100000., dtype=torch.float32)
            x[1, 1::2] += .0078125
            x[2] += torch.arange(33).remainder(3) * .0078125
            g = LayerNormGeometry(tuple(x.shape), (33,))
            _, mean, inverse, h = independent_stats(x, g)
            result, mu, inv, _ = fast.forward(x, None, None, g, torch.float32)
            self.close(mu, mean, atol=0, rtol=0)
            self.close(inv, inverse, atol=3e-5)
            self.close(result, h, atol=5e-6)
            self.assertTrue(torch.equal(result[0], torch.zeros_like(result[0])))

    def test_all_selective_gradients_and_zero_gamma(self):
        with self.kernel_context():
            x, dy = torch.randn(5, 17), torch.randn(5, 17, dtype=torch.bfloat16)
            w = torch.randn(17)
            w[::3] = 0
            g = LayerNormGeometry(tuple(x.shape), (17,))
            _, mean, inverse, h = independent_stats(x, g)
            mean, inverse = mean.to(x.device), inverse.to(x.device)
            expected = independent_backward(dy, h, inverse.cpu(), w)
            for flags in ((True, False, False), (False, True, False), (False, False, True),
                          (True, True, False), (True, False, True), (False, True, True), (True, True, True),
                          (False, False, False)):
                actual = fast.backward(dy, w, g, packed=None, x=x if flags[0] or flags[1] else None,
                    mean=mean if flags[0] or flags[1] else None, inverse=inverse if flags[0] or flags[1] else None,
                    need_dx=flags[0], need_dw=flags[1], need_db=flags[2], input_dtype=torch.float32)
                for result, oracle, needed in zip(actual[:3], expected, flags):
                    if needed:
                        self.close(result, oracle)
                    else:
                        self.assertIsNone(result)
                self.assertLessEqual(actual[3], fast.AFFINE_WORKSPACE_LIMIT)

    def test_large_offset_rounded_center_matches_native_output_and_existing_backward(self):
        with self.kernel_context():
            x = torch.tensor([[1e8, 1e8, 1e8 + 8]], dtype=torch.float32)
            dy = torch.tensor([[1., 2., 3.]])
            w, b = torch.ones(3), torch.zeros(3)
            g = LayerNormGeometry(tuple(x.shape), (3,))
            result, mean, inverse, _ = fast.forward(x, w, b, g, torch.float32)
            _, expected_mean, expected_inverse, h = independent_stats(x, g)
            self.close(mean, expected_mean, atol=0, rtol=0)
            self.close(inverse, expected_inverse, atol=1e-7)
            self.close(result, h, atol=1e-6)
            native = torch.nn.functional.layer_norm(x.cpu(), (3,), w.cpu(), b.cpu(), g.eps)
            self.close(result, native, atol=1e-6)
            gradients = fast.backward(dy, w, g, packed=None, x=x, mean=mean, inverse=inverse,
                need_dx=True, need_dw=True, need_db=True, input_dtype=torch.float32)
            for actual, expected in zip(gradients[:3], independent_backward(dy, h, inverse.cpu(), w)):
                self.close(actual, expected, atol=1e-6)
            self.close(gradients[1], (dy.cpu() * native).sum(0), atol=1e-6)
            # At this extreme native fused backward has cancellation artifacts.
            # Preserve the existing explicit reconstructed-xhat backward rule.
            from deepopt.opt.backends.reference.normalization import layer as reference
            existing = reference.backward(dy.cpu(), w.cpu(), g, packed=None, x=x.cpu(), mean=mean.cpu(),
                inverse=inverse.cpu(), need_dx=True, need_dw=True, need_db=True, input_dtype=torch.float32)
            for actual, expected in zip(gradients[:3], existing):
                self.close(actual, expected, atol=1e-6)

    def test_native_backward_matches_independent_autograd(self):
        with self.kernel_context():
            x, dy = torch.randn(4, 33), torch.randn(4, 33)
            weight, bias = torch.randn(33), torch.randn(33)
            g = LayerNormGeometry(tuple(x.shape), (33,))
            _, mean, inverse, _ = fast.forward(x, weight, bias, g, torch.float32)
            actual = fast.backward(dy, weight, g, packed=None, x=x, mean=mean, inverse=inverse,
                need_dx=True, need_dw=True, need_db=True, input_dtype=torch.float32)
            xr, wr, br = (value.detach().cpu().clone().requires_grad_() for value in (x, weight, bias))
            torch.nn.functional.layer_norm(xr, (33,), wr, br, g.eps).backward(dy.cpu())
            for result, expected in zip(actual[:3], (xr.grad, wr.grad, br.grad)):
                self.close(result, expected, atol=1e-5)

    def test_affine_workspace_is_bounded_and_partial_tails_are_deterministic(self):
        for rows in (1, 4096, 10**8):
            for cols in (1, 768, 8192):
                for count in (1, 2):
                    parts = fast.affine_parts(rows, cols, True, count == 2)
                    self.assertLessEqual(parts, fast.MAX_PARTS)
                    self.assertLessEqual(parts * cols * 4 * count, fast.AFFINE_WORKSPACE_LIMIT)
        with self.kernel_context(), patch.object(fast, "AFFINE_WORKSPACE_LIMIT", 512):
            x, dy = torch.randn(257, 17), torch.randn(257, 17)
            g = LayerNormGeometry(tuple(x.shape), (17,))
            _, mean, inverse, h = independent_stats(x, g)
            kwargs = dict(packed=None, x=x, mean=mean.to(x.device), inverse=inverse.to(x.device),
                need_dx=False, need_dw=True, need_db=True, input_dtype=torch.float32)
            first = fast.backward(dy, None, g, **kwargs)
            second = fast.backward(dy, None, g, **kwargs)
            expected = independent_backward(dy, h, inverse, None)
            for a, b, oracle in zip(first[1:3], second[1:3], expected[1:]):
                self.close(a, b, atol=0, rtol=0)
                self.close(a, oracle, atol=1e-5)
            self.assertLessEqual(first[3], 512)

    def test_native_path_avoids_full_size_float32_saved_xhat_and_weighted_dy(self):
        with self.kernel_context(), patch.object(fast, "dense", side_effect=AssertionError("native dense copy")):
            x, dy = torch.randn(4, 33, dtype=torch.bfloat16), torch.randn(4, 33, dtype=torch.bfloat16)
            g = LayerNormGeometry(tuple(x.shape), (33,))
            result, mean, inverse, h = fast.forward(x, None, None, g, torch.bfloat16)
            self.assertEqual(result.dtype, torch.bfloat16)
            self.assertIsNone(h)
            dx, _, _, _ = fast.backward(dy, None, g, packed=None, x=x, mean=mean, inverse=inverse,
                need_dx=True, need_dw=False, need_db=False, input_dtype=torch.bfloat16)
            self.assertEqual(dx.dtype, torch.bfloat16)

    def test_compressed_state_payload_moment_correction_and_backward(self):
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
        with self.kernel_context():
            x, dy = torch.randn(3, 17), torch.randn(3, 17)
            x[0].fill_(3.5)
            w, b = torch.randn(17), torch.randn(17)
            g = LayerNormGeometry(tuple(x.shape), (17,))
            _, _, _, h = independent_stats(x, g)
            for bits in (1, 2, 4, 8):
                shift, gain = torch.empty(3), torch.empty(3)
                output, _, state, _, _, info = layer._cooperative_forward(x, g, weight=w, bias=b,
                    dtype=torch.float32, block_shape=None, compression=dict(enabled=True, group_size=7, bits=bits),
                    check_finite=False, center_shift=shift, moment_gain=gain)
                expected = quantize_pack_activation(h.cpu(), 7, (1, 7), bits=bits)
                self.close(state.packed.data, expected.data, atol=0, rtol=0)
                self.close(state.packed.step, expected.step)
                self.close(state.packed.minimum, expected.minimum)
                self.close(output, h * w.cpu() + b.cpu())
                # Decode off-device, independently of the selected GPU codec.
                from deepopt.opt.quantization.activation import PackedActivation
                packed_cpu = PackedActivation(state.packed.data.cpu(), state.packed.step.cpu(), state.packed.minimum.cpu(),
                    state.packed.shape, state.packed.group_size, state.packed.block_shape, bits)
                decoded = dequantize_unpack_activation(packed_cpu, dtype=torch.float32)
                corrected = (decoded - shift.cpu()[:, None]) * gain.cpu()[:, None]
                self.close(corrected.mean(1), torch.zeros(3), atol=1e-6)
                self.close(corrected.square().mean(1), h.square().mean(1))
                result = fast.backward(dy, w, g, packed=state.packed, x=None, mean=None, inverse=state.inverse,
                    need_dx=True, need_dw=True, need_db=True, input_dtype=torch.float32, center_shift=shift, moment_gain=gain)
                for actual, oracle in zip(result[:3], independent_backward(dy, corrected, state.inverse.cpu(), w)):
                    self.close(actual, oracle, atol=1e-5)
                self.assertEqual(info["forward_variant"], "cooperative")

    def test_fp8_input_and_output_use_independent_preparation_fixtures(self):
        from deepopt.opt.quantization.quantizer import quantize_fp8
        from deepopt.opt.quantization.activation_tensor import FP8Activation
        with self.kernel_context():
            source = torch.linspace(-2, 3, 51, device="cpu").reshape(3, 17)
            for fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                q = quantize_fp8(source, fp8_dtype=fp8_dtype, block_shape=(2, 4))
                device = torch.empty(0).device
                x = FP8Activation(q.data.to(device), q.scale_inv.to(device), tuple(source.shape), (2, 4), "last", torch.bfloat16)
                g = LayerNormGeometry(tuple(source.shape), (17,))
                _, _, _, h = independent_stats(q.dequantize(), g)
                output, mean, inverse, _ = fast.forward(x, None, None, g, torch.float32)
                self.close(output, h, atol=5e-6)
                scales = torch.full((2, 5), 2.0, device=device)
                result = layer._cooperative_forward(x, g, weight=None, bias=None, dtype=fp8_dtype,
                    block_shape=(2, 4), compression=dict(enabled=False, group_size=7, bits=2), check_finite=False,
                    scale=scales, observe=True, capture_scale=True)
                self.close(result[0].float(), (h * 2).to(fp8_dtype).float(), atol=0, rtol=0)
                self.close(result[1], torch.full((2, 5), .5, device="cpu"), atol=0, rtol=0)
                self.close(result[3], torch.tensor([2., .5], device="cpu"), atol=0, rtol=0)
                self.assertIsNotNone(result[4])
                self.assertEqual(result[5]["decoded_input_workspace_bytes"], source.numel() * 4)
                self.assertEqual(result[5]["workspace_dtype"], "torch.float32")
                current = layer._cooperative_forward(x, g, weight=None, bias=None, dtype=fp8_dtype,
                    block_shape=(2, 4), compression=dict(enabled=False, group_size=7, bits=2), check_finite=False)
                from tests.opt.test_tilelang_amax import reference_quantize
                expected_data, expected_inverse = reference_quantize(h, fp8_dtype, (2, 4))
                self.close(current[0].float(), expected_data.float(), atol=0, rtol=0)
                self.close(current[1], expected_inverse, atol=1e-7, rtol=1e-6)
                qg = quantize_fp8(torch.randn(3, 17, device="cpu"), fp8_dtype=torch.float8_e5m2, block_shape=(2, 4))
                gradient = FP8Activation(qg.data.to(device), qg.scale_inv.to(device), tuple(source.shape),
                    (2, 4), "last", torch.bfloat16)
                gradients = fast.backward(gradient, None, g, packed=None, x=x, mean=mean, inverse=inverse,
                    need_dx=True, need_dw=True, need_db=True, input_dtype=torch.float32)
                for actual, expected in zip(gradients[:3], independent_backward(qg.dequantize(), h, inverse.cpu(), None)):
                    self.close(actual, expected)

    def test_compression_nonfinite_rejection_does_not_mutate_saved_correction(self):
        with self.kernel_context():
            x = torch.randn(3, 17)
            x[0, 0] = float("nan")
            g = LayerNormGeometry(tuple(x.shape), (17,))
            shift, gain = torch.full((3,), 19.), torch.full((3,), 23.)
            with self.assertRaisesRegex(ValueError, "finite normalized values"):
                layer._cooperative_forward(x, g, weight=None, bias=None, dtype=torch.float32,
                    block_shape=None, compression=dict(enabled=True, group_size=7, bits=2), check_finite=True,
                    center_shift=shift, moment_gain=gain)
            self.close(shift, torch.full((3,), 19.), atol=0, rtol=0)
            self.close(gain, torch.full((3,), 23.), atol=0, rtol=0)

    def test_gradient_output_conversion_is_not_absorbed(self):
        with self.kernel_context():
            x, dy = torch.randn(3, 17), torch.randn(3, 17)
            g = LayerNormGeometry(tuple(x.shape), (17,))
            _, mean, inverse, _ = fast.forward(x, None, None, g, torch.float32)
            class Output:
                calls = 0
                def convert(self, value):
                    self.calls += 1
                    self.value = value
                    return value.to(torch.bfloat16)
            target = Output()
            dx, _, _, _ = fast.backward(dy, None, g, packed=None, x=x, mean=mean, inverse=inverse,
                need_dx=True, need_dw=False, need_db=False, input_dtype=torch.bfloat16, gradient_output=target)
            self.assertEqual(target.calls, 1)
            self.assertEqual(target.value.dtype, torch.float32)
            self.assertEqual(dx.dtype, torch.bfloat16)

    def test_mixed_auxiliary_pins_remain_separate_dispatches(self):
        from deepopt.kernel_policy import kernel_backend, kernel_policy_scope
        calls = []
        def selected_module(family, path):
            calls.append((family, path, kernel_backend(family)))
            # Host routing probe: execute the same semantic language adapter
            # after observing the selected route; no Triton compiler is loaded.
            return importlib.import_module("deepopt.opt.backends.tilelang." + path)
        policy = dict(backend="mixed", preferred="tilelang",
            families={"activation_codec": "triton", "quantization": "triton", "scaling": "triton"})
        with self.kernel_context(), kernel_policy_scope(policy), \
                patch("deepopt.opt.backends.dispatch.backend_module", selected_module):
            x, dy = torch.randn(3, 17), torch.randn(3, 17)
            g = LayerNormGeometry(tuple(x.shape), (17,))
            shift, gain = torch.empty(3), torch.empty(3)
            _, _, state, _, _, _ = layer._cooperative_forward(x, g, weight=None, bias=None,
                dtype=torch.float8_e4m3fn, block_shape=None, compression=dict(enabled=True, group_size=7, bits=2),
                check_finite=True, center_shift=shift, moment_gain=gain, scale=torch.ones(1, 1),
                observe=True, capture_scale=True)
            fast.backward(dy, None, g, packed=state.packed, x=None, mean=None, inverse=state.inverse,
                need_dx=True, need_dw=True, need_db=True, input_dtype=torch.float32, center_shift=shift, moment_gain=gain)
        for family in ("activation_codec", "quantization", "scaling"):
            self.assertTrue(any(f == family and backend == "triton" for f, _, backend in calls), calls)

    def test_public_cuda_entry_reports_cooperative_variants(self):
        with self.kernel_context():
            x = torch.randn(3, 17)
            if x.device.type != "cuda":
                self.skipTest("public cooperative selection requires CUDA")
            g = LayerNormGeometry(tuple(x.shape), (17,))
            output, _, state, _, _, info = layer.forward(x, g, weight=None, bias=None,
                dtype=torch.bfloat16, block_shape=None, compression=dict(enabled=False, group_size=7, bits=2), check_finite=False)
            self.assertEqual(info["forward_variant"], "cooperative")
            result = layer.backward(torch.randn_like(x), None, g, packed=None, x=x, mean=state.mean,
                inverse=state.inverse, need_dx=True, need_dw=True, need_db=True,
                input_dtype=torch.bfloat16, execution_info=info)
            self.assertEqual(info["backward_variant"], "cooperative")
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assertEqual(result[0].dtype, torch.bfloat16)

    def test_eligibility_preserves_generic_layout_and_width_cases(self):
        self.assertFalse(fast.supported(torch.empty(2, 3, 5), LayerNormGeometry((2, 3, 5), (5,), layout="channels")))
        self.assertFalse(fast.supported(torch.empty(2, 8193), LayerNormGeometry((2, 8193), (8193,))))
        self.assertFalse(fast.supported(torch.empty(2, 0), LayerNormGeometry((2, 0), (0,))))
        self.assertFalse(fast.supported(torch.empty(3, 2).T, LayerNormGeometry((2, 3), (3,))))
        self.assertFalse(fast.supported(torch.empty(2, 3, dtype=torch.float64), LayerNormGeometry((2, 3), (3,))))


class LayerOptimizationFrontendTests(unittest.TestCase):
    def test_real_primfunc_construction(self):
        if importlib.util.find_spec("tilelang") is None:
            if os.getenv("DEEPOPT_REQUIRE_TILELANG") == "1":
                self.fail("TileLang frontend is required")
            self.skipTest("optional TileLang frontend is not installed")
        clear_factories()
        with patch.object(fast, "compile_kernel", lambda function, device: function):
            for dtype in ("float32", "bfloat16", "float16"):
                self.assertIsNotNone(fast._forward(3, 33, dtype, "float32", "float32", dtype, True, True, True, 1e-5, "cuda:0"))
            for compressed in (False, True):
                self.assertIsNotNone(fast._dx(3, 33, "float32", "bfloat16", "float32", "bfloat16", not compressed, compressed, compressed, True, "cuda:0"))
                for dw, db in ((True, True), (True, False), (False, True)):
                    self.assertIsNotNone(fast._affine_partial(71, 33, 3, "float32", "bfloat16", not compressed, compressed, compressed, dw, db, "cuda:0"))
            self.assertIsNotNone(fast._affine_finish(33, 3, 2, "cuda:0"))
        clear_factories()


class LayerOptimizationCudaTests(LayerOptimizationTests):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or importlib.util.find_spec("tilelang") is None or torch.cuda.get_device_capability() < (9, 0):
            error = RuntimeError if os.getenv("DEEPOPT_REQUIRE_TILELANG") == "1" else unittest.SkipTest
            raise error("cooperative LayerNorm CUDA tests require TileLang and NVIDIA SM90+")

    @staticmethod
    @contextmanager
    def kernel_context():
        from deepopt.kernel_policy import kernel_policy_scope
        with torch.device("cuda"), kernel_policy_scope("tilelang"):
            yield


if __name__ == "__main__":
    unittest.main()
