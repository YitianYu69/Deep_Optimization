"""Hardware acceptance for fused correction and dynamic device replay state."""

import os
import unittest

import torch
from torch import nn

import deepopt
from deepopt.opt.nn import functional as DF
from deepopt.opt.policy import normalization_plan
from deepopt.opt.policy import prepare_fp8_model
from deepopt.opt.precision import precision_report
from deepopt.opt.quantization.activation_tensor import FP8Activation
from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


if FP8_CUDA_SKIP_REASON and os.getenv("DEEPOPT_REQUIRE_CUDA_NORM_RECENTER") == "1":
    raise RuntimeError(FP8_CUDA_SKIP_REASON)


def policy(recenter, group=512, bits=2):
    return normalization_plan({"enabled": True, "backend": "triton"}, activation_compression={
        "enabled": True, "group_size": group, "bits": bits, "norm_xhat_recenter": recenter})


def norm(kind, x, weight, bias, p, info):
    if kind == "layer":
        return DF.do_layer_norm(x, x.shape[-1:], weight, bias, plan=p, _execution_info=info)
    if kind == "group":
        return DF.do_group_norm(x, 2, weight, bias, plan=p, _execution_info=info)
    return DF.do_batch_norm(x, None, None, weight, bias, True, plan=p, _execution_info=info)


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class NormXhatRecenterCUDATests(unittest.TestCase):
    def test_planned_fp8_input_output_and_dx_fuse_corrected_xhat(self):
        from unittest.mock import patch

        settings = deepopt.FP8Config(enabled=True, backend="triton", backward_precision="auto", grad_dtype="e5m2",
                                    block_shapes={"activation": None, "weight": None, "grad": (1, 32)})
        source = nn.Sequential(nn.Linear(32, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 32)).cuda()
        compression = dict(enabled=True, group_size=17, bits=2, norm_xhat_recenter=True)
        model, _ = prepare_fp8_model(source, settings, activation_compression=compression)
        with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=settings, activation_compression=compression, check_finite=False):
            y = model(torch.randn(16, 32, device="cuda", requires_grad=True))
        with patch.object(FP8Activation, "dequantize", side_effect=AssertionError("dense FP8 reconstruction")):
            y.sum().backward()
        self.assertTrue(model[1].execution_info["norm_xhat_recenter"])
        self.assertEqual(model[1].execution_info["recenter_launches"], 0)
        report = precision_report(model)
        self.assertEqual(report["analyses"], 1)
        self.assertTrue(all(info["fused"] for info in report["backward_execution"].values() if info is not None))

    def test_codec_invariance_corrected_gradient_and_bounded_reductions(self):
        for kind in ("batch", "group", "layer"):
            for bits in (1, 2, 4, 8):
                for size in (17, 512, 17003):
                    shape = (3, 17003) if kind == "layer" else (2, 4, 1, 2501)
                    x = torch.randn(shape, device="cuda", requires_grad=True)
                    features = shape[-1] if kind == "layer" else 4
                    w = torch.randn(features, device="cuda", requires_grad=True)
                    b = torch.randn(features, device="cuda", requires_grad=True)
                    dy = torch.randn_like(x)
                    results, states = [], []
                    for enabled in (False, True):
                        saved, info = [], {}
                        with deepopt.autocast("cuda", enabled=False, check_finite=False), torch.autograd.graph.saved_tensors_hooks(
                                lambda t: saved.append(t) or t, lambda t: t):
                            results.append(norm(kind, x, w, b, policy(enabled, size, bits), info))
                        states.append(saved)
                    torch.testing.assert_close(*results, rtol=0, atol=0)
                    for a, z in zip(states[0][:3], states[1][:3]):
                        torch.testing.assert_close(a, z, rtol=0, atol=0)
                    matrix = (x.detach().cpu().reshape(-1, features) if kind == "layer" else
                              x.detach().cpu().movedim(1, 0).reshape(4, -1) if kind == "batch" else
                              x.detach().cpu().reshape(4, -1))
                    variance, mean = torch.var_mean(matrix, 1, correction=0)
                    inverse = torch.rsqrt(variance + 1e-5)
                    h = (matrix - mean[:, None]) * inverse[:, None]
                    group = min(size, matrix.shape[1])
                    data, step, low = (v.cpu() for v in states[1][:3])
                    packed = PackedActivation(data, step, low, tuple(matrix.shape), group, (1, group), bits)
                    decoded = dequantize_unpack_activation(packed, dtype=torch.float32)
                    shift = states[1][-1].cpu()
                    corrected = decoded - shift[:, None]
                    torch.testing.assert_close(corrected.mean(1), h.mean(1), rtol=0, atol=3e-5)
                    results[1].backward(dy)
                    restored = (corrected.reshape(shape) if kind != "batch" else
                                corrected.reshape(4, shape[0], *shape[2:]).movedim(0, 1))
                    expected_dw = ((dy.cpu() * restored).sum(0) if kind == "layer" else
                                   (dy.cpu() * restored).sum((0, 2, 3)))
                    torch.testing.assert_close(w.grad.cpu(), expected_dw, rtol=2e-4, atol=3e-4)
                    self.assertTrue(torch.isfinite(x.grad).all())
                    self.assertEqual(info["recenter_state_bytes"], matrix.shape[0] * 4)
                    if kind != "layer" or size == 17:
                        self.assertGreater(info["recenter_launches"], 0)

    def test_complete_domains_add_no_launch_or_dense_decode_pass(self):
        for kind in ("batch", "group", "layer"):
            x = torch.randn((3, 33) if kind == "layer" else (2, 4, 4, 4), device="cuda", requires_grad=True)
            features = x.shape[-1] if kind == "layer" else 4
            w = torch.ones(features, device="cuda", requires_grad=True)
            counts = []
            for enabled in (False, True):
                info = {}
                def step():
                    x.grad = w.grad = None
                    with deepopt.autocast("cuda", enabled=False, check_finite=False):
                        y = norm(kind, x, w, None, policy(enabled), info)
                    y.backward(torch.ones_like(y))
                for _ in range(3):
                    step()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as trace:
                    step()
                names = [e.name for e in trace.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                self.assertFalse(any("_merge_center" in n or "_dequantize_unpack" in n for n in names), names)
                counts.append(len(names))
                self.assertEqual(info["recenter_launches"], 0)
                self.assertEqual(info["recenter_scratch_bytes"], 0)
            self.assertEqual(*counts)

    def test_changing_value_graph_replay_updates_shift_and_rstd(self):
        for kind in ("batch", "group", "layer"):
            x = torch.randn((3, 768) if kind == "layer" else (2, 4, 8, 8), device="cuda", requires_grad=True)
            dy = torch.randn_like(x)
            p = policy(True, group=17)
            def step():
                with deepopt.autocast("cuda", enabled=False, check_finite=False):
                    y = norm(kind, x, None, None, p, {})
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
            graph_grad = x.grad
            for _ in range(3):
                with torch.no_grad():
                    x.copy_(torch.randn_like(x) * 2 + 10)
                    x.grad = graph_grad
                    x.grad.zero_()
                graph.replay()
                actual, dx = captured.clone(), x.grad.clone()
                x.grad = None
                expected = step()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(dx, x.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
