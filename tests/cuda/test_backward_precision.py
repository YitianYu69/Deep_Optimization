"""Real CUDA planned FP8 gradients, fused epilogues and complete pipeline replay."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn

import deepopt
from deepopt.opt.backends import gemm_nt
from deepopt.opt.policy import prepare_fp8_model
from deepopt.opt.precision import precision_report
from deepopt.opt.precision.gradient import GradientEdgePlan
from deepopt.opt.quantization.gradient import GradientSlot
from deepopt.opt.quantization.tensor import QuantizedTensor
from deepopt.opt.quantization.activation_tensor import FP8Activation
from deepopt.opt.quantization.quantizer import quantize_fp8
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class BackwardPrecisionCUDATests(unittest.TestCase):
    def test_gemm_block_epilogue_writes_fp8_scales_without_a_separate_cast(self):
        rows, cols, reduction = 67, 64, 64
        a = ((torch.arange(rows, device="cuda") % 4 - 2)[:, None].expand(rows, reduction).contiguous()).to(torch.float8_e4m3fn)
        b = ((torch.arange(cols, device="cuda") % 2 + 1)[:, None].expand(cols, reduction).contiguous()).to(torch.float8_e4m3fn)
        qa = QuantizedTensor(a, torch.full((1, 1), 0.25, device="cuda"))
        qb = QuantizedTensor(b, torch.full((1, 1), 0.5, device="cuda"))
        slot = GradientSlot(GradientEdgePlan("fp8", "e5m2", "last", (1, 32)), {}, device="cuda")

        def step():
            with slot.output((rows, cols), torch.bfloat16, check_finite=False) as out:
                return gemm_nt(qa, qb, backend="triton", gradient_output=out)

        step()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            result = step()
        expected = quantize_fp8((qa.dequantize() @ qb.dequantize().T).cpu(), fp8_dtype=torch.float8_e5m2, block_shape=(1, 32))
        torch.testing.assert_close(result._data.float().cpu(), expected.data.float(), rtol=0, atol=0)
        torch.testing.assert_close(result._scale_inv.cpu(), expected.scale_inv, rtol=2e-7, atol=0)
        self.assertTrue(slot.last_execution["fused"])
        names = [event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA]
        self.assertFalse(any("_cast_scaled" in name or "_fused_blocks" in name for name in names), names)
        self.assertEqual(sum("_gemm_nt_kernel" in name for name in names), 1, names)

    def test_delayed_gemm_output_snapshot_is_independent_of_state_updates(self):
        a = QuantizedTensor(torch.ones(16, 4096, device="cuda", dtype=torch.float8_e4m3fn), torch.ones(1, 1, device="cuda"))
        b = QuantizedTensor(torch.ones(16, 4096, device="cuda", dtype=torch.float8_e4m3fn), torch.ones(1, 1, device="cuda"))
        slot = GradientSlot(GradientEdgePlan("fp8", "e5m2", "last", None, "delayed"),
                            {"history_length": 4, "reduce_amax": False}, device="cuda")
        slot.scaling.scales[2] = 0.5
        with slot.output((16, 16), torch.bfloat16, check_finite=False) as out:
            result = gemm_nt(a, b, backend="triton", gradient_output=out)
        torch.testing.assert_close(result._data.float(), torch.full((16, 16), 2048.0, device="cuda"), rtol=0, atol=0)
        self.assertEqual(result._scale_inv.item(), 2)
        self.assertEqual(slot.scaling.updates[2].item(), 1)

    def test_all_spatial_ranks_and_normalization_accept_fp8_gradients_without_dense_decode(self):
        for rank in (1, 2, 3):
            conv = getattr(nn, f"Conv{rank}d")
            for kernel in (1, 3):
                model = nn.Sequential(conv(32, 32, kernel, padding=kernel // 2), nn.GroupNorm(4, 32), nn.ReLU(),
                                      conv(32, 32, kernel, padding=kernel // 2)).cuda()
                cfg = deepopt.FP8Config(enabled=True, backend="triton", backward_precision="auto", grad_dtype="e5m2",
                                        block_shapes={"activation": None, "weight": None, "grad": (1, 32)})
                model, _ = prepare_fp8_model(model, cfg, activation_compression=True)
                x = torch.randn(2, 32, *((3,) * rank), device="cuda", requires_grad=True)
                with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=cfg, activation_compression=True, check_finite=False):
                    output = model(x)
                with patch.object(FP8Activation, "dequantize", side_effect=AssertionError("dense FP8 gradient reconstruction")):
                    output.sum().backward()
                execution = precision_report(model)["backward_execution"]
                self.assertEqual(sum(info is not None for info in execution.values()), 3)
                self.assertTrue(all(info["fused"] for info in execution.values() if info is not None), execution)
                self.assertNotIsInstance(x.grad, FP8Activation)

    def test_cached_decisions_and_fp8_gradient_scales_replay_in_cuda_graph(self):
        cfg = deepopt.FP8Config(enabled=True, backend="triton", backward_precision="auto", grad_dtype="e5m2",
                                block_shapes={"activation": None, "weight": None, "grad": (1, 32)})
        model, _ = prepare_fp8_model(nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32)).cuda(), cfg,
                                     activation_compression=True)
        x = torch.randn(16, 32, device="cuda", requires_grad=True)
        dy = torch.ones_like(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                x.grad = None
                with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=cfg, activation_compression=True, check_finite=False):
                    model(x).backward(dy)
        torch.cuda.current_stream().wait_stream(stream)
        model.zero_grad(set_to_none=True)
        x.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=cfg, activation_compression=True, check_finite=False):
                captured = model(x)
                captured.backward(dy)
        for _ in range(2):
            reference = deepcopy(model)
            reference.zero_grad(set_to_none=True)
            with torch.no_grad():
                x.copy_(torch.randn_like(x))
                x.grad.zero_()
                for parameter in model.parameters():
                    parameter.grad.zero_()
            xx = x.detach().clone().requires_grad_()
            with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=cfg, activation_compression=True, check_finite=False):
                expected = reference(xx)
                expected.backward(dy)
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, xx.grad, rtol=1e-5, atol=1e-6)
        self.assertEqual(precision_report(model)["analyses"], 1)
