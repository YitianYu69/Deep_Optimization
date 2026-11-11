"""Real CUDA compilation, fusion, precision edges and changing-value pool replay."""

import os
import unittest

import torch
from torch import nn

import deepopt
from deepopt.opt.nn import functional as D
from deepopt.opt.nn.pooling.spec import SPECS
from deepopt.opt.policy import pooling_plan, prepare_fp8_model
from deepopt.opt.precision import precision_report
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation
from tests.opt.test_pooling import case
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


if FP8_CUDA_SKIP_REASON and os.getenv("DEEPOPT_REQUIRE_CUDA_POOLING") == "1":
    raise RuntimeError(FP8_CUDA_SKIP_REASON)


def policy(enabled=True, bits=2, group=512, blocks=None):
    return pooling_plan({"enabled": True, "backend": "triton", "block_shapes": {"activation": blocks}},
                        activation_compression=dict(enabled=enabled, bits=bits, group_size=group))


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class PoolingCUDATests(unittest.TestCase):
    def test_parameterless_delayed_pooling_resolves_forward_and_backward_device_once(self):
        cfg = deepopt.FP8Config(enabled=True, backend="triton", backward_precision="auto", scaling_recipe="delayed", grad_dtype="e5m2")
        model, _ = prepare_fp8_model(nn.Sequential(nn.AvgPool1d(2), nn.AvgPool1d(2), nn.ReLU()), cfg, activation_compression=True)
        with deepopt.autocast("cuda", enabled=False, fp8=cfg, activation_compression=True, check_finite=False):
            model(torch.randn(2, 3, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)).sum().backward()
        self.assertEqual(model[0].fp8_scaling.scales.device.type, "cuda")
        self.assertEqual(precision_report(model)["analyses"], 1)

    def test_large_global_reductions_and_anisotropic_strided_windows(self):
        for kind, options in (("adaptive_avg_pool1d", dict(output_size=1)),
                              ("adaptive_max_pool1d", dict(output_size=1)),
                              ("lp_pool1d", dict(norm_type=2, kernel_size=17003))):
            x = torch.randn(1, 2, 17003, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            z = x.detach().float().requires_grad_()
            info = {}
            with deepopt.autocast("cuda", enabled=False, check_finite=False):
                actual = getattr(D, "do_" + kind)(x, **options, plan=policy(False), output_dtype=torch.float32, _execution_info=info)
            expected = SPECS[kind].native(z, **options)
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
            self.assertTrue(info["global_reduction"])
            actual.sum().backward(); expected.sum().backward()
            torch.testing.assert_close(x.grad.float(), z.grad, rtol=0.03, atol=0.002)
        for kind, shape, options in (
            ("max_pool2d", (2, 4, 9, 7), dict(kernel_size=(2, 3), stride=(2, 1), padding=(0, 1), dilation=(2, 1), ceil_mode=True, return_indices=True)),
            ("avg_pool3d", (2, 4, 5, 7, 9), dict(kernel_size=(2, 3, 2), stride=(1, 2, 2), padding=(0, 1, 0), ceil_mode=True, count_include_pad=False, divisor_override=7)),
        ):
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).transpose(-1, -2).requires_grad_()
            z = x.detach().float().requires_grad_()
            with deepopt.autocast("cuda", enabled=False, check_finite=False):
                actual = getattr(D, "do_" + kind)(x, **options, plan=policy(), output_dtype=torch.float32)
            expected = SPECS[kind].native(z, **options)
            if isinstance(actual, tuple):
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                actual, expected = actual[0], expected[0]
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)
            dy = torch.randn_like(actual)
            actual.backward(dy); expected.backward(dy)
            torch.testing.assert_close(x.grad.float(), z.grad, rtol=0.03, atol=0.01)

    def test_all_families_batched_unbatched_and_fp8_operand_formats(self):
        for kind, spec in SPECS.items():
            for unbatched in (False, True):
                for fmt in (None, torch.float8_e4m3fn, torch.float8_e5m2):
                    with self.subTest(kind=kind, unbatched=unbatched, fmt=fmt):
                        x, options = case(kind, dtype=torch.bfloat16, unbatched=unbatched, device="cuda")
                        source = to_fp8_activation(x, layout="channels", fp8_dtype=fmt, block_shape=(1, 32)) if fmt is not None else x
                        represented = source.dequantize(torch.float32) if isinstance(source, FP8Activation) else x.float()
                        native_options = dict(options)
                        if "max" in spec.family:
                            options["return_indices"] = native_options["return_indices"] = True
                        if "_random_samples" in native_options:
                            native_options["_random_samples"] = native_options["_random_samples"].float()
                        z = represented.detach().requires_grad_()
                        expected = spec.native(z, **native_options)
                        with deepopt.autocast("cuda", enabled=False, check_finite=False):
                            actual = getattr(D, "do_" + kind)(source, **options, plan=policy(), output_dtype=torch.bfloat16)
                        if isinstance(actual, tuple):
                            torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                            actual, expected = actual[0], expected[0]
                        torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)
                        dy = torch.randn_like(actual)
                        actual.backward(dy); expected.backward(dy.float())
                        if spec.family != "power":
                            torch.testing.assert_close(x.grad.float(), z.grad, rtol=0.03, atol=0.025)
                        self.assertTrue(torch.isfinite(x.grad).all())

    def test_lp_amp_dtype_compression_invariance_and_fused_channel_coefficients(self):
        for bits in (1, 2, 4, 8):
            x = torch.randn(2, 32, 8, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            source = to_fp8_activation(x, layout="channels", block_shape=(1, 32))
            info = {}
            with deepopt.autocast("cuda", enabled=False, check_finite=False):
                before = D.do_lp_pool2d(source, 2, 2, plan=policy(False), output_dtype=torch.float8_e4m3fn, output_layout="channels")
                after = D.do_lp_pool2d(source, 2, 2, plan=policy(True, bits), output_dtype=torch.float8_e4m3fn,
                                       output_layout="channels", _execution_info=info)
            torch.testing.assert_close(before._data.float(), after._data.float(), rtol=0, atol=0)
            torch.testing.assert_close(before._scale_inv, after._scale_inv, rtol=0, atol=0)
            self.assertTrue(info["coefficient_fused"])
            self.assertEqual(info["packing_launches"], 0)
            after.backward(torch.ones(after.shape, device="cuda", dtype=torch.bfloat16))
        x = torch.rand(2, 3, 6, 6, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = torch.nn.functional.lp_pool2d(x, 2, 2)
            actual = D.do_lp_pool2d(x, 2, 2, plan=policy())
        self.assertEqual(actual.dtype, expected.dtype)

    def test_profiles_no_separate_private_index_pack_or_dense_fp8_decode(self):
        for kind in ("max_pool2d", "avg_pool2d", "adaptive_max_pool2d", "adaptive_avg_pool2d"):
            x, options = case(kind, dtype=torch.bfloat16, device="cuda")
            def step():
                x.grad = None
                with deepopt.autocast("cuda", enabled=False, check_finite=False):
                    source = to_fp8_activation(x, layout="channels", block_shape=(1, 32))
                    y = getattr(D, "do_" + kind)(source, **options, plan=policy())
                y.backward(torch.ones_like(y))
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as trace:
                step()
            names = [event.name for event in trace.events() if event.device_type == torch.autograd.DeviceType.CUDA]
            self.assertFalse(any("_dequantize_unpack" in name or name.startswith("_pack") or name.startswith("_unpack") for name in names), names)

    def test_cached_pool_norm_activation_convolution_backward_edges(self):
        cfg = deepopt.FP8Config(enabled=True, backend="triton", backward_precision="auto", grad_dtype="e5m2",
                               block_shapes={"activation": (1, 32), "weight": None, "grad": (1, 32)})
        model = nn.Sequential(nn.Conv2d(32, 32, 1), nn.GroupNorm(4, 32), nn.ReLU(), nn.MaxPool2d(2),
                              nn.AvgPool2d(2), nn.Conv2d(32, 32, 1)).cuda()
        model, _ = prepare_fp8_model(model, cfg, activation_compression=True)
        with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=cfg, check_finite=False, activation_compression=True):
            result = model(torch.randn(2, 32, 8, 16, device="cuda"))
            result.square().mean().backward()
        report = precision_report(model)
        self.assertEqual(report["analyses"], 1)
        self.assertFalse(report["fallbacks"])
        self.assertTrue(model[3].execution_info["compression_active"])
        self.assertEqual(model[4].execution_info["saved_state_bytes"], 0)
        pools = {name: info for name, info in report["backward_execution"].items() if "_3" in name or "_4" in name}
        self.assertTrue(any(info is not None and info["fused"] for info in pools.values()), report)

    def test_changing_value_and_fractional_window_cuda_graph_replay(self):
        for kind in ("max_pool2d", "avg_pool2d", "adaptive_max_pool2d", "adaptive_avg_pool2d",
                     "lp_pool2d", "fractional_max_pool2d", "max_unpool2d"):
            x, options = case(kind, dtype=torch.bfloat16, device="cuda")
            g = policy()
            dy = None
            def step():
                with deepopt.autocast("cuda", enabled=False, check_finite=False):
                    y = getattr(D, "do_" + kind)(x, **options, plan=g, output_dtype=torch.float8_e4m3fn)
                y.backward(dy)
                return y
            with torch.no_grad(), deepopt.autocast("cuda", enabled=False, check_finite=False):
                shape = getattr(D, "do_" + kind)(x, **options, plan=g).shape
            dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    x.grad = None
                    step()
            torch.cuda.current_stream().wait_stream(stream)
            x.grad = None
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = step()
            graph_grad = x.grad
            for _ in range(3):
                with torch.no_grad():
                    x.copy_(torch.randn_like(x) * 2)
                    if "_random_samples" in options:
                        options["_random_samples"].uniform_()
                    x.grad = graph_grad
                    x.grad.zero_()
                graph.replay()
                actual, actual_scale, dx = captured._data.clone(), captured._scale_inv.clone(), x.grad.clone()
                x.grad = None
                eager = step()
                torch.testing.assert_close(actual.float(), eager._data.float(), rtol=0, atol=0)
                torch.testing.assert_close(actual_scale, eager._scale_inv, rtol=0, atol=0)
                torch.testing.assert_close(dx, x.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
