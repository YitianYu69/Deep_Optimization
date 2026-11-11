"""Real CUDA numerical, finite-policy, and launch-path checks for fused FP8."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import DiagnosticPolicy, TrainingRuntime
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.opt.quantization.tensor import QuantizedTensor
from deepopt.opt.scaling.amax import AmaxScaling
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class FusedFP8QuantizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(192)

    def test_standalone_scaling_fuses_conversion_for_strided_inputs_and_large_blocks(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.int32):
            x = (torch.randn(257, 259, device="cuda") * 3).to(dtype).T
            for blocks in (None, (1, 32), (32, 32), (256, 256), (1024, 1024)):
                expected = AmaxScaling(blocks).compute(x.cpu())
                for checks in (False, True):
                    with patch.object(torch.Tensor, "float", side_effect=AssertionError("FP32 copy")), \
                         patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("dense input copy")), \
                         patch("torch.isfinite", side_effect=AssertionError("separate finite scan")):
                        actual = AmaxScaling(blocks).compute(x, check_finite=checks)
                    torch.testing.assert_close(actual.scale.cpu(), expected.scale, rtol=1e-6, atol=0)
                    torch.testing.assert_close(actual.scale_inv.cpu(), expected.scale_inv, rtol=1e-6, atol=0)

    def test_standalone_scaling_rejects_nonfinite_inputs_without_a_separate_scan(self):
        for blocks in (None, (1, 32), (256, 256)):
            for value in (float("nan"), float("inf"), float("-inf")):
                x = torch.ones(257, 259, device="cuda", dtype=torch.bfloat16).T
                x[-1, -1] = value
                with patch("torch.isfinite", side_effect=AssertionError("separate finite scan")), \
                     self.assertRaisesRegex(ValueError, "finite inputs"):
                    AmaxScaling(blocks).compute(x)

    def test_dequantization_fuses_cast_and_strided_scale_lookup(self):
        for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):
            for blocks in (None, (2, 3), (32, 32)):
                cpu = quantize_fp8(torch.randn(37, 65), fp8_dtype=fp8, block_shape=blocks)
                values = torch.empty(37, 130, device="cuda", dtype=fp8)[:, ::2]
                values.copy_(cpu.data.to("cuda"))
                rows, cols = cpu.scale_inv.shape
                inverse = torch.empty(rows, cols * 2, device="cuda")[:, ::2]
                inverse.copy_(cpu.scale_inv.to("cuda"))
                quantized = QuantizedTensor(values, inverse, blocks)
                expected = cpu.dequantize()
                with patch.object(torch.Tensor, "float", side_effect=AssertionError("FP32 copy")), \
                     patch.object(torch.Tensor, "repeat_interleave", side_effect=AssertionError("expanded scales")), \
                     patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("dense copy")), \
                     patch.object(torch.Tensor, "item", side_effect=AssertionError("scalar read")):
                    actual = quantized.dequantize()
                    transposed = quantized.transpose(contiguous=False).dequantize()
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(transposed.cpu(), expected.T, rtol=0, atol=0)

    def assert_quantization_matches_cpu(self, x, blocks, fp8_dtype=torch.float8_e4m3fn):
        expected = quantize_fp8(x.cpu(), block_shape=blocks, fp8_dtype=fp8_dtype)
        actual = quantize_fp8(x, block_shape=blocks, fp8_dtype=fp8_dtype)
        torch.testing.assert_close(actual.data.float().cpu(), expected.data.float(), rtol=0, atol=0)
        torch.testing.assert_close(actual.scale_inv.cpu(), expected.scale_inv, rtol=1e-6, atol=0)
        self.assertTrue(actual.data.is_contiguous())

    def test_blockwise_and_small_tensorwise_values_scales_dtypes_and_strides(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.int32):
            x = (torch.randn(37, 65, device="cuda") * 3).to(dtype)
            for view in (x, x.T, x[1::2, 1::2], x[:1].expand(7, -1)):
                for blocks in (None, (1, 32), (32, 32), (128, 128)):
                    for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):
                        with self.subTest(dtype=dtype, stride=view.stride(), blocks=blocks, fp8=fp8):
                            self.assert_quantization_matches_cpu(view, blocks, fp8)

    def test_zero_tiny_and_rounding_boundary_inputs(self):
        x = torch.tensor([[0., -0., 1., -1., 1.0625, -1.0625, 448.],
                          [0., 0., 0., 0., 0., 0., 0.],
                          [1e-37, -1e-37, 1e-20, 0., 0., 0., 0.]], device="cuda")
        for blocks in (None, (1, 8), (2, 4), (32, 32)):
            for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):
                self.assert_quantization_matches_cpu(x, blocks, fp8)

    def test_large_tensorwise_reduction_and_noncontiguous_inputs(self):
        for shape in ((129, 97), (2051, 2049)):
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).T
            self.assert_quantization_matches_cpu(x, None)

    def test_oversized_scale_blocks_use_chunked_fusion_and_clipped_edge_tiles(self):
        for shape in ((37, 65), (257, 259)):
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).T
            for blocks in ((256, 128), (128, 512), (1024, 1024)):
                for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):
                    self.assert_quantization_matches_cpu(x, blocks, fp8)

    def test_unchecked_quantization_avoids_reference_intermediates_and_scalar_reads(self):
        for shape, blocks in (((37, 65), (1, 32)), ((37, 65), None), ((129, 97), None),
                              ((257, 259), (256, 128))):
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16).T
            with patch.object(AmaxScaling, "compute", side_effect=AssertionError("scale-only dispatch")), \
                 patch.object(torch.Tensor, "float", side_effect=AssertionError("FP32 intermediate")), \
                 patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("dense input copy")), \
                 patch.object(torch.Tensor, "repeat_interleave", side_effect=AssertionError("expanded scales")), \
                 patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar read")), \
                 patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("host scalar branch")):
                quantize_fp8(x, block_shape=blocks, check_finite=False)

    def test_strict_checks_reject_nonfinite_values_in_every_reduction_path(self):
        for shape, blocks in (((37, 65), (1, 32)), ((37, 65), None), ((129, 97), None),
                              ((257, 259), (256, 128))):
            for value in (float("nan"), float("inf"), float("-inf")):
                x = torch.ones(shape, device="cuda")
                x[-1, -1] = value
                with self.subTest(shape=shape, blocks=blocks, value=value), \
                     self.assertRaisesRegex(ValueError, "finite inputs"):
                    quantize_fp8(x.T, block_shape=blocks)

    def test_runtime_modes_control_fused_forward_and_backward_independently(self):
        from deepopt.opt.backends import triton_quantization

        original = triton_quantization.quantize
        original_scale = triton_quantization.amax_scale
        policies = ({"activation": (1, 32), "weight": (32, 32), "grad": (1, 32)},
                    dict.fromkeys(("activation", "weight", "grad")),
                    dict.fromkeys(("activation", "weight", "grad"), (256, 256)))
        for blocks in policies:
            model = nn.Linear(257, 129, device="cuda")
            # Reference GEMM isolates quantization from autotuning while real
            # CUDA kernels cover single-pass, tensorwise and chunked blocks.
            runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.001), device="cuda",
                                      fp8={"enabled": True, "block_shapes": blocks},
                                      diagnostics=DiagnosticPolicy(loss_finite_every=2, gradient_finite_every=3))
            records, phase = [], None

            def quantize(*args):
                records.append((runtime.mode_controller.completed_updates + 1, phase, args[-1]))
                return original(*args)

            def scale(*args):
                records.append((runtime.mode_controller.completed_updates + 1, phase, args[-1]))
                return original_scale(*args)

            def step(value):
                AmaxScaling(blocks["activation"]).compute(value)
                return runtime.model(value).square().mean()

            with patch.object(triton_quantization, "quantize", new=quantize), \
                 patch.object(triton_quantization, "amax_scale", new=scale):
                for _ in range(6):
                    phase = "forward"
                    loss = runtime.compute(torch.randn(129, 257, device="cuda", requires_grad=True),
                                           step)
                    phase = "backward"
                    runtime.backward(loss)
                    runtime.step()
            for update, operation, checked in records:
                self.assertEqual(checked, update % (2 if operation == "forward" else 3) == 0)
            self.assertEqual({operation for _, operation, _ in records}, {"forward", "backward"})

    def test_max_speed_schedule_and_explicit_finite_override_reach_fused_path(self):
        from deepopt.opt.backends import triton_quantization

        model = nn.Linear(65, 10, device="cuda")
        runtime = TrainingRuntime(model, torch.optim.SGD(model.parameters(), lr=.001), device="cuda",
                                  fp8=True, execution_mode="max_speed",
                                  mode_schedule={2: "analyze", 3: "max_speed"})
        original = triton_quantization.quantize
        records = []

        def quantize(*args):
            records.append((runtime.mode_controller.completed_updates + 1, args[-1]))
            return original(*args)

        with patch.object(triton_quantization, "quantize", new=quantize):
            context = runtime.autocast()
            for _ in range(3):
                with context:
                    loss = runtime.model(torch.ones(3, 65, device="cuda")).square().mean()
                runtime.backward(loss)
                runtime.step()
        for update, checked in records:
            self.assertEqual(checked, update == 2)
        with runtime.autocast(), self.assertRaisesRegex(ValueError, "finite inputs"):
            quantize_fp8(torch.full((2, 3), float("nan"), device="cuda"), check_finite=True)
