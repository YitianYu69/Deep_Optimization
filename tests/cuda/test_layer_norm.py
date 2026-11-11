"""Real CUDA LayerNorm compilation, fusion, transport, AMP and replay acceptance."""

import os
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

import deepopt
from deepopt.opt.nn import functional as doF
from deepopt.opt.policy import normalization_plan, prepare_fp8_model
from deepopt.opt.precision import precision_report
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation, dequantize_activation
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


if FP8_CUDA_SKIP_REASON and os.getenv("DEEPOPT_REQUIRE_CUDA_LAYER_NORM") == "1":
    raise RuntimeError(FP8_CUDA_SKIP_REASON)


def policy(compressed=True, *, bits=2, group=512, block=None, recenter=False):
    return normalization_plan(fp8={"enabled": True, "backend": "triton", "block_shapes": {"activation": block}},
                               activation_compression={"enabled": compressed, "group_size": group, "bits": bits,
                                                       "norm_xhat_recenter": recenter})


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class LayerNormCUDATests(unittest.TestCase):
    def test_native_amp_and_affine_dtype_behavior(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for affine_dtype in (dtype, torch.float32):
                x = torch.randn(3, 33, dtype=dtype, device="cuda", requires_grad=True)
                w = torch.randn(33, dtype=affine_dtype, device="cuda", requires_grad=True)
                b = torch.randn(33, dtype=affine_dtype, device="cuda", requires_grad=True)
                z, g, beta = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
                with deepopt.autocast("cuda", dtype=torch.bfloat16,
                                      fp8=deepopt.FP8Config(enabled=True, backend="triton"), check_finite=False):
                    actual = doF.layer_norm(x, (33,), w, b, plan=policy(False))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = F.layer_norm(z, (33,), g, beta)
                self.assertEqual(actual.dtype, expected.dtype)
                actual.sum().backward(); expected.sum().backward()
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.003)
                torch.testing.assert_close(x.grad, z.grad, rtol=0.02, atol=0.005)

    def test_suffix_precisions_strides_and_compressed_xhat_gradients(self):
        for norm in ((7,), (5, 7)):
            for block in (None, (1, 8)):
                for source_dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
                    for output_dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
                        x = torch.randn(2, 3, 5, 14, device="cuda", dtype=torch.bfloat16)[..., ::2].requires_grad_()
                        source = (to_fp8_activation(x, layout="channels", fp8_dtype=source_dtype, block_shape=(2, 8))
                                  if source_dtype != torch.bfloat16 else x)
                        represented = source.dequantize(torch.float32) if isinstance(source, FP8Activation) else x.detach().float()
                        w = torch.randn(norm, device="cuda", requires_grad=True)
                        b = torch.randn(norm, device="cuda", requires_grad=True)
                        matrix = represented.cpu().reshape(-1, w.numel())
                        variance, mean = torch.var_mean(matrix, 1, correction=0)
                        inverse = torch.rsqrt(variance + 1e-5)
                        h = (matrix - mean[:, None]) * inverse[:, None]
                        group = min(17, w.numel())
                        packed = quantize_pack_activation(h, group, (1, group), bits=4)
                        decoded = dequantize_unpack_activation(packed, dtype=torch.float32)
                        with deepopt.autocast("cuda", enabled=False, check_finite=False):
                            y = doF.layer_norm(source, norm, w, b, plan=policy(bits=4, group=17, block=block),
                                               output_dtype=output_dtype, output_layout="channels")
                        expected = (h * w.detach().cpu().reshape(1, -1) + b.detach().cpu().reshape(1, -1)).reshape(x.shape)
                        torch.testing.assert_close(dequantize_activation(y).float().cpu(), expected,
                                                   rtol=0.15 if output_dtype != torch.bfloat16 else 0.02,
                                                   atol=0.025 if output_dtype != torch.bfloat16 else 0.005)
                        dy = torch.randn_like(x)
                        y.backward(dy)
                        u = dy.float().cpu().reshape_as(h) * w.detach().cpu().reshape(1, -1)
                        dx = (u - u.mean(1, keepdim=True) - decoded * (u * decoded).mean(1, keepdim=True)) * inverse[:, None]
                        torch.testing.assert_close(x.grad.float().cpu(), dx.reshape(x.shape), rtol=0.05, atol=0.03)
                        torch.testing.assert_close(w.grad.cpu(), (dy.float().cpu().reshape_as(h) * decoded).sum(0).reshape(norm), rtol=0.04, atol=0.03)

    def test_large_rows_groups_all_bits_and_partial_gradient_requests(self):
        for bits in (1, 2, 4, 8):
            for group in (17, 512, 17003):
                x = torch.randn(3, 17003, device="cuda", requires_grad=True)
                w = torch.ones(17003, device="cuda", requires_grad=True)
                with deepopt.autocast("cuda", enabled=False, check_finite=False):
                    y = doF.layer_norm(x, (17003,), w, plan=policy(bits=bits, group=group))
                y.sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(torch.isfinite(w.grad).all())
        bias = torch.zeros(33, device="cuda", requires_grad=True)
        info = {}
        doF.layer_norm(torch.randn(3, 33, device="cuda"), (33,), bias=bias, plan=policy(), _execution_info=info).sum().backward()
        self.assertEqual(info["saved_state_bytes"], 0)
        torch.testing.assert_close(bias.grad, torch.full_like(bias, 3))

    def test_native_and_provided_scale_small_row_forward_is_fused(self):
        for output_dtype in (torch.bfloat16, torch.float8_e4m3fn):
            x = torch.randn(32, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            info = {}
            scale = torch.ones(1, 1, device="cuda") if output_dtype != torch.bfloat16 else None
            def step():
                x.grad = None
                with deepopt.autocast("cuda", enabled=False, check_finite=False):
                    doF.layer_norm(x, (768,), plan=policy(), output_dtype=output_dtype, output_scale=scale,
                                   _execution_info=info).backward(torch.ones_like(x))
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                step()
            names = [e.name for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            self.assertEqual(sum("_row_forward" in n for n in names), 1, names)
            self.assertFalse(any("_dequantize_unpack" in n or "_quantize_pack_groups" in n for n in names), names)
            self.assertEqual(info["statistics_launches"], 0)
            self.assertEqual(info["scale_launches"], 0)

    def test_compression_grouping_never_changes_forward_statistics_or_output(self):
        for width in (768, 5003, 17003):
            x = torch.randn(3, width, device="cuda", requires_grad=True)
            weight, bias = torch.randn(width, device="cuda"), torch.randn(width, device="cuda")
            for dtype in (torch.float32, torch.bfloat16, torch.float8_e4m3fn):
                with deepopt.autocast("cuda", enabled=False, check_finite=False):
                    baseline = doF.layer_norm(x, (width,), weight, bias, plan=policy(False), output_dtype=dtype)
                    for group in (1, 17, 512):
                        result = doF.layer_norm(x, (width,), weight, bias, plan=policy(True, group=group, bits=1), output_dtype=dtype)
                        if isinstance(result, FP8Activation):
                            torch.testing.assert_close(result._data.float(), baseline._data.float(), rtol=0, atol=0)
                            torch.testing.assert_close(result._scale_inv, baseline._scale_inv, rtol=0, atol=0)
                        else:
                            torch.testing.assert_close(result, baseline, rtol=0, atol=0)

    def test_planned_layernorm_activation_linear_reuses_fp8_and_fuses_dx(self):
        cfg = deepopt.FP8Config(enabled=True, backend="triton", backward_precision="auto", grad_dtype="e5m2",
                               block_shapes={"activation": None, "weight": None, "grad": (1, 32)})
        source = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 32)).cuda()
        model, _ = prepare_fp8_model(source, cfg, activation_compression=True)
        with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=cfg, activation_compression=True, check_finite=False):
            y = model(torch.randn(16, 32, device="cuda", requires_grad=True))
        with patch.object(FP8Activation, "dequantize", side_effect=AssertionError("dense FP8 reconstruction")):
            y.sum().backward()
        reports = precision_report(model)["backward_execution"]
        self.assertTrue(all(v["fused"] for v in reports.values() if v is not None), reports)
        self.assertEqual(precision_report(model)["analyses"], 1)

    def test_current_block_and_fixed_scale_cuda_graph_replay(self):
        for block, supplied, recenter in ((block, supplied, recenter) for block, supplied in
                                         ((None, False), ((1, 32), False), (None, True)) for recenter in (False, True)):
            x = torch.randn(3, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            dy = torch.randn_like(x)
            scale = torch.ones(1, 1, device="cuda") if supplied else None
            p = policy(block=block, recenter=recenter)
            def step():
                with deepopt.autocast("cuda", enabled=False, check_finite=False):
                    y = doF.layer_norm(x, (768,), plan=p, output_dtype=torch.float8_e4m3fn, output_scale=scale)
                y.backward(dy)
                return y
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
            graph_gradient = x.grad
            for _ in range(3):
                with torch.no_grad():
                    x.copy_(torch.randn_like(x))
                    x.grad = graph_gradient
                    x.grad.zero_()
                graph.replay()
                actual, grad = captured._data.clone(), x.grad.clone()
                x.grad = None
                eager = step()
                torch.testing.assert_close(actual.float(), eager._data.float(), rtol=0, atol=0)
                torch.testing.assert_close(grad, x.grad, rtol=0, atol=0)

    def test_delayed_checkpoint_recomputation_accumulates_once(self):
        cfg = deepopt.FP8Config(enabled=True, backend="triton", backward_precision="auto", scaling_recipe="delayed")
        model = nn.Sequential(nn.Linear(32, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 32)).cuda()
        runtime = deepopt.TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=0.001), device="cuda", precision="bf16",
                                          fp8=cfg, checks=False, config={"runtime": {"activation_compression": {"enabled": True}}})
        for _ in range(2):
            loss = runtime.compute(torch.randn(16, 32, device="cuda"), lambda x: deepopt.fp8_checkpoint(runtime.model, x).square().mean())
            runtime.backward(loss)
            self.assertEqual(runtime.model[1].fp8_scaling.updates[0].item(), 0)
        runtime.step()
        self.assertEqual(runtime.model[1].fp8_scaling.updates[0].item(), 1)


if __name__ == "__main__":
    unittest.main()
