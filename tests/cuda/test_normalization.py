"""Real CUDA normalization, xhat compression and planned FP8 pipeline tests."""

from copy import deepcopy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

import deepopt
from deepopt.opt.nn import functional as DF
from deepopt.opt.nn.normalization import NormGeometry
from deepopt.opt.policy import normalization_plan, prepare_fp8_model
from deepopt.opt.quantization.activation import PackedActivation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_tensor import to_fp8_activation
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class NormalizationCUDATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2049)

    def test_bf16_and_fp8_input_forward_and_packed_xhat_backward(self):
        for kind in ("batch", "group"):
            for fp8 in (False, True):
                for bits in (1, 2, 4, 8):
                    with self.subTest(kind=kind, fp8=fp8, bits=bits):
                        source = torch.randn(2, 4, 5, 7, device="cuda", dtype=torch.bfloat16).requires_grad_()
                        x = to_fp8_activation(source, block_shape=(4, 4)) if fp8 else source
                        w, bias = torch.randn(4, device="cuda", requires_grad=True), torch.randn(4, device="cuda", requires_grad=True)
                        settings = deepopt.FP8Config(enabled=True, backend="triton")
                        plan = normalization_plan(settings, activation_compression={"enabled": True, "group_size": 17, "bits": bits})
                        saved = []
                        with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
                            y = (DF.batch_norm(x, None, None, w, bias, True, plan=plan) if kind == "batch" else
                                 DF.group_norm(x, 2, w, bias, plan=plan))
                        dense = x.dequantize(torch.float32) if fp8 else x.float()
                        expected = (F.batch_norm(dense, None, None, w, bias, True) if kind == "batch" else
                                    F.group_norm(dense, 2, w, bias)).bfloat16()
                        torch.testing.assert_close(y, expected, rtol=0.01, atol=0.01)
                        self.assertEqual(saved[0].dtype, torch.int32)
                        self.assertFalse(any(value is x or value is source or tuple(value.shape) == tuple(source.shape) for value in saved))
                        geometry = NormGeometry(kind, tuple(source.shape), 2)
                        packed = PackedActivation(saved[0], saved[1], saved[2], geometry.matrix_shape, 17, (1, 17), bits)
                        h = dequantize_unpack_activation(packed, torch.float32)
                        inv = saved[3]
                        gradient = torch.randn_like(y)
                        y.backward(gradient)
                        u = geometry.matrix(gradient.float() * w.detach().reshape(1, 4, 1, 1))
                        dx = geometry.restore((u - u.mean(1, keepdim=True) - h * (u * h).mean(1, keepdim=True)) * inv[:, None])
                        torch.testing.assert_close(source.grad, dx.bfloat16(), rtol=0.02, atol=0.02)
                        torch.testing.assert_close(w.grad, (gradient.float() * geometry.restore(h)).sum((0, 2, 3)), rtol=2e-4, atol=2e-4)
                        torch.testing.assert_close(bias.grad, gradient.float().sum((0, 2, 3)), rtol=2e-4, atol=2e-4)

    def test_pipeline_graph_replays_preserve_running_state_and_cached_plan(self):
        settings = deepopt.FP8Config(enabled=True, backend="triton", intercept_linear=False)
        original = nn.Sequential(nn.BatchNorm2d(4, momentum=None), nn.ReLU(), nn.Conv2d(4, 8, 3, padding=1)).cuda()
        eager, _ = prepare_fp8_model(deepcopy(original), settings, activation_compression=True)
        model, _ = prepare_fp8_model(original, settings, activation_compression=True)
        sample = torch.randn(2, 4, 5, 7, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        captured = None
        try:
            with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=settings, activation_compression=True,
                                  cache_enabled=False, check_finite=False):
                captured = deepopt.CUDAGraphCore("cuda").capture_model(model, (sample,), warmup=3, autograd=True)
                self.assertEqual(model[0].num_batches_tracked.item(), 0)
                for _ in range(2):
                    a = torch.randn_like(sample, requires_grad=True)
                    b = a.detach().clone().requires_grad_()
                    actual, expected = model(a), eager(b)
                    gradient = torch.randn_like(actual)
                    ga = torch.autograd.grad(actual, (a, *model.parameters()), gradient)
                    gb = torch.autograd.grad(expected, (b, *eager.parameters()), gradient)
                    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
                    for left, right in zip(ga, gb):
                        torch.testing.assert_close(left, right, rtol=2e-3, atol=2e-3)
                self.assertEqual(model[0].num_batches_tracked.item(), 2)
                torch.testing.assert_close(model[0].running_mean, eager[0].running_mean)
                torch.testing.assert_close(model[0].running_var, eager[0].running_var)
                self.assertEqual(model._deepopt_precision_plan["analyses"], 1)
                self.assertEqual(captured.statistics["forward_replays"], 2)
        finally:
            if captured is not None:
                captured.close()

    def test_large_group_packing_uses_bounded_chunks(self):
        x = torch.randn(2, 4, 33, 35, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        plan = normalization_plan(deepopt.FP8Config(enabled=True, backend="triton"),
                                  activation_compression={"enabled": True, "group_size": 8192, "bits": 2})
        # GroupNorm's single normalization domain has 4620 values, selecting
        # hierarchical min/max plus chunked xhat encoding rather than a huge CTA.
        y = DF.group_norm(x, 1, plan=plan)
        y.float().square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_group_norm_retains_native_amp_output_precision(self):
        x = torch.randn(2, 4, 5, 7, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            native = F.group_norm(x, 2)
            output = DF.group_norm(x, 2, plan=normalization_plan(False, activation_compression=True))
        self.assertEqual(output.dtype, native.dtype)
        torch.testing.assert_close(output, native, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
