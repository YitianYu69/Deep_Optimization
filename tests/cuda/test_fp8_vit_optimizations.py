"""Real-GPU coverage for long reductions, paired layouts and disk tuning reuse."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import DiagnosticPolicy, TrainingRuntime
from deepopt.opt.backends import gemm_nt
from deepopt.opt.quantization.quantizer import quantize_fp8, _quantize_tensorwise_pair
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class ViTFP8OptimizationTests(unittest.TestCase):
    def test_weight_only_backward_writes_only_the_consumed_gradient_layout(self):
        from deepopt.opt.nn.autograd import linear as implementation
        from deepopt.opt.quantization.quantizer import _quantize_tensorwise_transpose
        from deepopt.opt.policy import fp8_plan
        from deepopt import FP8Config
        from deepopt.opt.nn.functional import do_linear

        x = torch.randn(129, 65, device="cuda")
        weight = torch.randn(97, 65, device="cuda", requires_grad=True)
        grad = torch.randn(129, 97, device="cuda")

        def oracle(a, b, **kwargs):
            return gemm_nt(a, b, **{**kwargs, "backend": "reference"})

        with patch.object(implementation, "gemm_nt", new=oracle), \
             patch.object(implementation, "_quantize_tensorwise_transpose", wraps=_quantize_tensorwise_transpose) as transpose:
            do_linear(x, weight, plan=fp8_plan(FP8Config(enabled=True, backend="triton"))).backward(grad)
        self.assertEqual(transpose.call_count, 1)
        qx = quantize_fp8(x).dequantize()
        qg = quantize_fp8(grad).dequantize()
        torch.testing.assert_close(weight.grad, qg.T @ qx, rtol=1e-4, atol=1e-4)

    def test_native_accumulation_covers_long_cancellation_magnitudes_and_zero_reductions(self):
        import triton
        from deepopt.opt.backends import triton_fp8

        config = triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64, "NUM_STAGES": 2,
                                "SPLIT_K": 1, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=2)
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for magnitude in (0., .001, 100.):
                x = torch.randn(17, 8193, device="cuda") * magnitude
                x[:, 1:8192:2] = -x[:, :8192:2]
                w = torch.randn(33, 8193, device="cuda") * magnitude
                a, b = quantize_fp8(x, fp8_dtype=dtype), quantize_fp8(w, fp8_dtype=dtype)
                with patch.object(triton_fp8._gemm_nt_kernel, "configs", [config]):
                    result = gemm_nt(a, b, backend="triton", num_stages=2)
                expected = a.dequantize().double() @ b.dequantize().double().T
                self.assertTrue(torch.isfinite(result).all())
                relative = (result.double() - expected).norm() / expected.norm().clamp_min(1e-30)
                self.assertLess(relative.item(), .05)

    def test_pair_matches_tensorwise_oracle_and_initializes_both_padding_regions(self):
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            source = torch.randn(65, 129, device="cuda").T
            expected = quantize_fp8(source.cpu(), fp8_dtype=dtype)
            normal, transposed = _quantize_tensorwise_pair(source, fp8_dtype=dtype)
            torch.testing.assert_close(normal.data.float().cpu(), expected.data.float(), rtol=0, atol=0)
            torch.testing.assert_close(transposed.data.float().cpu(), expected.data.T.float(), rtol=0, atol=0)
            self.assertIs(normal.scale_inv, transposed.scale_inv)
            for value in (normal, transposed):
                rows, cols = value.data.shape
                stride = value.data.stride(0)
                padded = value.data.as_strided((rows, stride), (stride, 1))
                self.assertEqual(stride % 16, 0)
                self.assertEqual(value.data.data_ptr() % 16, 0)
                self.assertTrue(torch.all(padded[:, cols:].float() == 0).item())

    def test_pair_training_keeps_independent_forward_backward_check_cadences(self):
        from deepopt.opt.backends import triton_quantization
        from deepopt.opt.nn.autograd import linear as linear_autograd_func

        model = nn.Linear(65, 97, device="cuda")
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.001), device="cuda",
                                  fp8={"enabled": True, "backend": "triton"},
                                  diagnostics=DiagnosticPolicy(loss_finite_every=2, gradient_finite_every=3))
        original = triton_quantization.quantize_tensorwise_pair
        records, layouts, phase = [], [], None

        def pair(*args, **kwargs):
            records.append((runtime.mode_controller.completed_updates + 1, phase, args[-1]))
            layouts.append(kwargs.get("transpose_only", False))
            return original(*args, **kwargs)

        def oracle(a, b, **kwargs):
            return gemm_nt(a, b, **{**kwargs, "backend": "reference"})

        with patch.object(triton_quantization, "quantize_tensorwise_pair", new=pair), \
             patch.object(linear_autograd_func, "gemm_nt", new=oracle):
            for update in range(6):
                phase = "forward"
                loss = runtime.compute(torch.randn(129, 65, device="cuda", requires_grad=update % 2 == 0),
                                       lambda value: runtime.model(value).square().mean())
                phase = "backward"
                runtime.backward(loss)
                runtime.step()
        self.assertEqual({phase for _, phase, _ in records}, {"forward", "backward"})
        self.assertIn(True, layouts)
        for update, phase, checked in records:
            self.assertEqual(checked, update % (2 if phase == "forward" else 3) == 0)

    def test_prepared_pair_requires_no_pack_and_aligned_transposes_need_no_memset(self):
        import triton
        from deepopt.opt.backends import triton_fp8

        a, _ = _quantize_tensorwise_pair(torch.randn(129, 65, device="cuda") * .1)
        b, _ = _quantize_tensorwise_pair(torch.randn(97, 65, device="cuda") * .1)
        config = triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32,
                                "NUM_STAGES": 2, "SPLIT_K": 1, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=2)
        with patch.object(triton_fp8._gemm_nt_kernel, "configs", [config]), \
             patch.object(triton_fp8._pack_rows, "run", side_effect=AssertionError("unnecessary pack")):
            actual = gemm_nt(a, b, backend="triton", num_stages=2)
        torch.testing.assert_close(actual.cpu(), a.dequantize().cpu() @ b.dequantize().cpu().T,
                                   rtol=5e-4, atol=5e-4)
        source = quantize_fp8(torch.randn(64, 128, device="cuda")).data.T
        with patch("torch.zeros", side_effect=AssertionError("unnecessary zero fill")):
            packed = triton_fp8._pack_fp8_rows(source, 16)
        torch.testing.assert_close(packed.float(), source.float(), rtol=0, atol=0)

    def test_dense_split_k_with_native_accumulation_matches_quantized_reference(self):
        import triton
        from deepopt.opt.backends import triton_fp8

        a = quantize_fp8(torch.randn(385, 8193, device="cuda") * .1)
        b = quantize_fp8(torch.randn(513, 8193, device="cuda") * .1)
        bias = torch.randn(513, device="cuda") * .1
        expected = a.dequantize().double() @ b.dequantize().double().T + bias.double()
        configs = [triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64,
                                   "NUM_STAGES": 2, "SPLIT_K": 4, "GROUP_SIZE_M": group},
                                  num_warps=8, num_stages=2) for group in (1, 8)]

        def bench(call, quantiles):
            call()
            return [1.] * len(quantiles)

        kernel = triton_fp8._gemm_nt_kernel
        with patch.object(kernel, "configs", configs), patch.object(kernel, "cache", {}), \
             patch.object(kernel, "do_bench", new=bench):
            actual = gemm_nt(a, b, backend="triton", bias=bias, num_stages=2)
            relative = (actual.double() - expected).norm() / expected.norm()
            self.assertLess(relative.item(), .05)
            self.assertEqual(len(kernel.cache), 1)

    def test_tuning_results_survive_a_fresh_process_without_config_hooks(self):
        from deepopt.opt.backends.triton.gemm.fp8 import _gemm_nt_kernel

        if not getattr(_gemm_nt_kernel, "cache_results", False):
            self.skipTest("This Triton runtime does not support persistent tuning results")
        root = Path(__file__).resolve().parents[2]
        code = textwrap.dedent('''
            import json, sys, torch, triton
            from deepopt.opt.backends import gemm_nt, triton_fp8
            from deepopt.opt.quantization.quantizer import quantize_fp8
            kernel = triton_fp8._gemm_nt_kernel
            kernel.configs = [triton.Config({"BLOCK_M": size, "BLOCK_N": size, "BLOCK_K": 32,
                "NUM_STAGES": 2, "SPLIT_K": 1, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=2)
                for size in (32, 64)]
            assert all(config.pre_hook is None for config in kernel.configs)
            calls = []
            def bench(call, quantiles):
                calls.append(1)
                call()
                return [1.] * len(quantiles)
            kernel.do_bench = bench
            if sys.argv[1] == "cached":
                def fail(*args, **kwargs):
                    raise AssertionError("Fresh process benchmarked instead of loading disk results")
                kernel._bench = fail
            a = quantize_fp8(torch.ones(64, 64, device="cuda"), check_finite=False)
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                result = gemm_nt(a, a, backend="triton", num_stages=2, out_dtype=dtype)
                torch.testing.assert_close(result.cpu(), torch.full((64, 64), 64., dtype=dtype))
            json.dumps(list(kernel.cache))
            print("CACHE_RESULT=" + json.dumps({"benchmarks": len(calls)}))
        ''')
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "PYTHONPATH": str(root / "src"), "TRITON_CACHE_DIR": directory}
            for phase in ("cold", "cached"):
                result = subprocess.run([sys.executable, "-c", code, phase], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                record = next(line for line in result.stdout.splitlines() if line.startswith("CACHE_RESULT="))
                self.assertEqual(json.loads(record.split("=", 1)[1])["benchmarks"], 6 if phase == "cold" else 0)
