"""Disabled FP8 tuning must bypass autotuners on cold forward/backward calls."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import FP8Config
from deepopt.opt.backends import gemm_nt
from deepopt.opt.policy import prepare_fp8_model
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON, quantize_gemm_operand


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class UntunedFP8CUDATests(unittest.TestCase):
    def test_cold_gemm_bypasses_search_and_does_not_populate_tuning_cache(self):
        from deepopt.opt.backends.triton.gemm import fp8

        for blocks in ((None, None), ((1, 32), (32, 32)), ((4, 3), None)):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(blocks=blocks, dtype=dtype):
                    a = quantize_gemm_operand(
                        torch.randn(33, 96, device="cuda"), block_shape=blocks[0]
                    )
                    b = quantize_gemm_operand(
                        torch.randn(65, 96, device="cuda"),
                        fp8_dtype=torch.float8_e5m2,
                        block_shape=blocks[1],
                    )
                    bias = torch.randn(65, device="cuda", dtype=dtype)
                    previous = dict(fp8._gemm_nt_kernel.cache)
                    with (
                        patch.object(
                            fp8._gemm_nt_kernel,
                            "run",
                            side_effect=AssertionError("autotuner invoked"),
                        ),
                        patch.object(
                            fp8,
                            "selected_config",
                            side_effect=AssertionError("tuned cache read"),
                        ),
                    ):
                        actual = gemm_nt(
                            a,
                            b,
                            backend="triton",
                            out_dtype=dtype,
                            bias=bias,
                            autotune=False,
                        )
                    expected = (a.dequantize() @ b.dequantize().T + bias.float()).to(
                        dtype
                    )
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.03)
                    self.assertEqual(fp8._gemm_nt_kernel.cache, previous)

    def test_cold_delayed_conversion_keeps_scale_transpose_observations_and_bias(self):
        from deepopt.opt.backends.triton.quantization import delayed

        value = torch.randn(65, 97, device="cuda")
        scale = torch.tensor([2.0], device="cuda")
        previous = dict(delayed._convert.cache)
        with (
            patch.object(
                delayed._convert,
                "run",
                side_effect=AssertionError("converter autotuner invoked"),
            ),
            patch.object(
                delayed,
                "selected_config",
                side_effect=AssertionError("tuned cache read"),
            ),
        ):
            normal, transpose, snapshot, observations, bias = delayed.convert(
                value,
                scale,
                torch.float8_e4m3fn,
                normal=True,
                transposed=True,
                observe=True,
                bias=True,
                check_finite=False,
                autotune=False,
            )
        expected = (value * 2).clamp(-448, 448).to(torch.float8_e4m3fn)
        torch.testing.assert_close(normal.float(), expected.float(), rtol=0, atol=0)
        torch.testing.assert_close(
            transpose.float(), expected.T.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            snapshot, torch.tensor([2.0, 0.5], device="cuda"), rtol=0, atol=0
        )
        torch.testing.assert_close(
            observations.max(), value.abs().max(), rtol=0, atol=0
        )
        torch.testing.assert_close(bias, value.sum(0), rtol=1e-4, atol=1e-4)
        self.assertEqual(delayed._convert.cache, previous)

    def test_model_forward_and_backward_never_enter_either_autotuner(self):
        from deepopt.opt.backends.triton.gemm import fp8
        from deepopt.opt.backends.triton.quantization import delayed

        for recipe in ("current", "delayed"):
            with self.subTest(recipe=recipe):
                model, _ = prepare_fp8_model(
                    nn.Sequential(
                        nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 16)
                    ).cuda(),
                    FP8Config(
                        enabled=True,
                        backend="triton",
                        autotune=False,
                        scaling_recipe=recipe,
                        grad_dtype="e5m2",
                        out_dtype="bf16",
                    ),
                )
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                original = next(model.parameters()).detach().clone()
                with (
                    patch.object(
                        fp8._gemm_nt_kernel,
                        "run",
                        side_effect=AssertionError("GEMM autotuner invoked"),
                    ),
                    patch.object(
                        delayed._convert,
                        "run",
                        side_effect=AssertionError("converter autotuner invoked"),
                    ),
                ):
                    for _ in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = (
                                model(
                                    torch.randn(
                                        16, 32, device="cuda", requires_grad=True
                                    )
                                )
                                .float()
                                .square()
                                .mean()
                            )
                        loss.backward()
                        self.assertTrue(
                            all(
                                torch.isfinite(p.grad).all() for p in model.parameters()
                            )
                        )
                        optimizer.step()
                self.assertFalse(torch.equal(original, next(model.parameters())))


if __name__ == "__main__":
    unittest.main()
