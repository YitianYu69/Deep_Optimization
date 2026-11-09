"""Real native scaled GEMM and warmed eager launch integration on SM90+."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import FP8Config, TrainingRuntime
from deepopt.opt.backends import gemm_nt
from deepopt.opt.do_nn.AOTAutograd_Func import linear_autograd_func
from deepopt.opt.quantization.quantizer import quantize_fp8
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class NativeBackwardCUDATests(unittest.TestCase):
    def test_native_uses_the_same_prepared_operands_scales_and_output_types(self):
        for formats in ((torch.float8_e4m3fn, torch.float8_e4m3fn),
                        (torch.float8_e5m2, torch.float8_e4m3fn),
                        (torch.float8_e4m3fn, torch.float8_e5m2)):
            a = quantize_fp8(torch.randn(33, 64, device="cuda"), fp8_dtype=formats[0])
            b = quantize_fp8(torch.randn(48, 64, device="cuda"), fp8_dtype=formats[1])
            for backend in ("native_fast", "native_precise"):
                for dtype in (torch.float32, torch.bfloat16, torch.float16):
                    with self.subTest(formats=formats, backend=backend, dtype=dtype), torch.autocast("cuda", enabled=False):
                        expected = torch._scaled_mm(a.data, b.data.T, scale_a=a.scale_inv, scale_b=b.scale_inv,
                                                    out_dtype=dtype, use_fast_accum=backend == "native_fast")
                        expected = expected[0] if isinstance(expected, tuple) else expected
                        actual = gemm_nt(a, b, backend=backend, out_dtype=dtype)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_warm_triton_launch_bypasses_autotuner_and_binds_new_buffers(self):
        from deepopt.opt.backends import triton_fp8

        a = quantize_fp8(torch.randn(33, 128, device="cuda"))
        b = quantize_fp8(torch.randn(65, 128, device="cuda"))
        gemm_nt(a, b, backend="triton")
        # New payloads and scale snapshots reuse only layout/config decisions.
        a = quantize_fp8(torch.randn(33, 128, device="cuda") * .1)
        b = quantize_fp8(torch.randn(65, 128, device="cuda") * 3.)
        with patch.object(triton_fp8._gemm_nt_kernel, "run", side_effect=AssertionError("autotuner on warm launch")):
            actual = gemm_nt(a, b, backend="triton")
        expected = a.dequantize() @ b.dequantize().T
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)
        fallback = gemm_nt(a, b, backend="native_fast")
        torch.testing.assert_close(fallback, actual, rtol=0, atol=0)

    def test_warm_delayed_conversion_bypasses_autotuner_and_keeps_current_scale(self):
        from deepopt.opt.backends import triton_delayed

        options = dict(normal=True, transposed=True, bias=False, observe=True, check_finite=False)
        value = torch.ones(33, 65, device="cuda")
        scale = torch.ones(1, device="cuda")
        triton_delayed.convert(value, scale, torch.float8_e4m3fn, **options)
        value = torch.full((33, 65), 3., device="cuda")
        scale = torch.full((1,), 2., device="cuda")
        with patch.object(triton_delayed._convert, "run", side_effect=AssertionError("autotuner on warm conversion")):
            normal, transposed, snapshot, _, _ = triton_delayed.convert(value, scale, torch.float8_e4m3fn, **options)
        torch.testing.assert_close(normal.float(), torch.full_like(value, 6.), rtol=0, atol=0)
        torch.testing.assert_close(transposed.float(), torch.full_like(value.T, 6.), rtol=0, atol=0)
        torch.testing.assert_close(snapshot, torch.tensor([2., .5], device="cuda"), rtol=0, atol=0)

    def test_native_backward_keeps_original_fp32_optimizer_parameters_in_all_execution_modes(self):
        for recipe in ("current", "delayed"):
            for backend in ("inherit", "native_fast", "native_precise"):
                for mode in ("max_speed", "balanced", "analyze"):
                    with self.subTest(recipe=recipe, backend=backend, mode=mode):
                        model = nn.Linear(64, 64, device="cuda")
                        parameters = tuple(model.parameters())
                        optimizer = torch.optim.SGD(parameters, lr=.001)
                        options = FP8Config(enabled=True, backend="triton", backward_backend=backend,
                                            scaling_recipe=recipe, intercept_linear=False)
                        runtime = TrainingRuntime(model, optimizer, fp8=options, precision="bf16", execution_mode=mode)
                        with patch.object(linear_autograd_func, "gemm_nt", wraps=gemm_nt) as dispatch:
                            loss = runtime.compute(torch.randn(80, 64, device="cuda", requires_grad=True),
                                                   lambda x: runtime.model(x).float().square().mean())
                            runtime.backward(loss)
                        backward = "triton" if backend == "inherit" else backend
                        self.assertEqual([(call.kwargs["backend"], call.kwargs["out_dtype"])
                                          for call in dispatch.call_args_list],
                                         [("triton", torch.bfloat16), (backward, torch.float32),
                                          (backward, torch.bfloat16)])
                        runtime.step()
                        self.assertEqual(tuple(runtime.model.parameters()), parameters)
                        for parameter in parameters:
                            self.assertEqual(parameter.dtype, torch.float32)
                            self.assertEqual(parameter.grad.dtype, torch.float32)
                            self.assertTrue(torch.isfinite(parameter).all())
