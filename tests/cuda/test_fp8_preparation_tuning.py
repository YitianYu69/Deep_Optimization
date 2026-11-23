"""Real-GPU equivalence and replay checks for new preparation/schedule candidates."""

import unittest
from unittest.mock import patch

import torch

from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON, quantize_gemm_operand


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class FP8PreparationCUDATests(unittest.TestCase):
    def test_cold_selection_during_capture_uses_baseline_without_caching_a_winner(self):
        from deepopt.opt.backends.triton.gemm import fp8
        from deepopt.opt.backends.triton.quantization import fp8 as quantization
        from deepopt.opt.backends._shared.paired_cast import clear_paired_cast_cache, _paired_cast_cache
        from deepopt.opt.backends._shared.gemm_diagnostics import record_gemms
        from deepopt.opt.backends import gemm_nt

        values = torch.randn(65, 97, device="cuda", dtype=torch.bfloat16)
        a = quantize_gemm_operand(values)
        b = quantize_gemm_operand(torch.randn(33, 97, device="cuda"))
        reference = gemm_nt(a, b, backend="triton", autotune=False)
        pair_reference = quantization.quantize_tensorwise_pair(values, torch.float8_e4m3fn, False, autotune=False)
        torch.cuda.synchronize()
        previous = dict(fp8._gemm_nt_kernel.cache)
        fp8._gemm_nt_kernel.cache.clear()
        clear_paired_cast_cache()
        try:
            graph = torch.cuda.CUDAGraph()
            with patch.object(fp8._gemm_nt_kernel, "run", side_effect=AssertionError("cold graph autotuning")):
                with record_gemms() as records:
                    with torch.cuda.graph(graph):
                        actual = gemm_nt(a, b, backend="triton", autotune=True)
                        pair_actual = quantization.quantize_tensorwise_pair(values, torch.float8_e4m3fn, False, autotune=True)
                graph.replay()
                torch.cuda.synchronize()
            self.assertEqual(records[0]["kernel_launch"]["selection"], "capture_baseline")
            self.assertFalse(fp8._gemm_nt_kernel.cache)
            self.assertFalse(_paired_cast_cache)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            for actual_value, reference_value in zip(pair_actual, pair_reference):
                torch.testing.assert_close(actual_value.float(), reference_value.float(), rtol=0, atol=0)
        finally:
            fp8._gemm_nt_kernel.cache.clear()
            fp8._gemm_nt_kernel.cache.update(previous)
            clear_paired_cast_cache()

    def test_each_pair_tile_preserves_bytes_scales_and_odd_tails(self):
        from deepopt.opt.backends._shared.paired_cast import PAIRED_CAST_CANDIDATES
        from deepopt.opt.backends.triton.quantization import fp8

        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for transposed_input in (False, True):
                values = torch.randn(65, 97, device="cuda", dtype=torch.bfloat16)
                if transposed_input:
                    values = values.T
                reference = fp8.quantize_tensorwise_pair(values, dtype, False, autotune=False)
                for transpose_only in (False, True):
                    for config in PAIRED_CAST_CANDIDATES:
                        with self.subTest(dtype=dtype, transposed_input=transposed_input,
                                          transpose_only=transpose_only, config=config):
                            with patch("deepopt.opt.backends._shared.paired_cast.select_paired_cast", return_value=config):
                                actual = fp8.quantize_tensorwise_pair(values, dtype, False,
                                                                      transpose_only=transpose_only, autotune=False)
                            if not transpose_only:
                                torch.testing.assert_close(actual[0].float(), reference[0].float(), rtol=0, atol=0)
                            else:
                                self.assertIsNone(actual[0])
                            torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)
                            torch.testing.assert_close(actual[2].float(), reference[2].float(), rtol=0, atol=0)

    def test_persistent_and_one_tile_schedule_match_and_replay_new_values(self):
        import triton
        from deepopt.opt.backends.triton.gemm import fp8
        from deepopt.opt.backends._shared.gemm_diagnostics import record_gemms
        from deepopt.opt.backends import gemm_nt

        # More logical tiles than H100 SMs exercises repeated CTA ownership.
        a = quantize_gemm_operand(torch.randn(2049, 97, device="cuda"))
        b = quantize_gemm_operand(torch.randn(1025, 97, device="cuda"), fp8_dtype=torch.float8_e5m2)
        bias = torch.randn(1025, device="cuda", dtype=torch.float32)
        baseline = fp8._fixed_config(None, None, 3)
        persistent = triton.Config({**baseline.kwargs, "PERSISTENT": True},
                                    num_warps=baseline.num_warps, num_stages=baseline.num_stages)
        expected = gemm_nt(a, b, backend="triton", bias=bias, autotune=False)
        with patch.object(fp8, "_fixed_config", return_value=persistent):
            with record_gemms() as records:
                actual = gemm_nt(a, b, backend="triton", bias=bias, autotune=False)
            self.assertEqual(records[0]["kernel_launch"]["schedule"], "persistent")
            launch = records[0]["kernel_launch"]
            self.assertLess(launch["grid"][0], triton.cdiv(2049, launch["tile"][0]) * triton.cdiv(1025, launch["tile"][1]))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replayed = gemm_nt(a, b, backend="triton", bias=bias, autotune=False)
            torch.cuda.current_stream().wait_stream(stream)
            for multiplier in (0.5, 2.0):
                a.scale_inv.fill_(multiplier)
                graph.replay()
                torch.cuda.synchronize()
                reference = gemm_nt(a, b, backend="triton", bias=bias, autotune=False)
                torch.testing.assert_close(replayed, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
