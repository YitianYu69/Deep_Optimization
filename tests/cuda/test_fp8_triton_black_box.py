"""Numerical FP8 CUDA kernel and autograd validation; never mocked GPU evidence."""

from copy import deepcopy
from itertools import product
import unittest

import torch

from deepopt.opt import transform_model
from deepopt.opt.backends import gemm_nt
from deepopt.opt.do_nn.functional import do_linear
from deepopt.opt.quantization.quantizer import quantize_fp8
from tests.cuda.fp8_support import FP8_CUDA_SKIP_REASON
from tests.opt.full_model_support import (
    assert_relative_error, fp8_plan, logits_and_loss, make_gpt2, make_llama,
    make_timm_vit, make_vit, move_inputs,
)


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class TritonFP8BlackBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(251)
        self.device = torch.device("cuda", torch.cuda.current_device())

    def test_gemm_matches_dequantized_oracle_for_partial_tiles_and_scale_boundaries(self):
        # K=97 and scale width 96 exercise a final partial tile and gcd(BLOCK_K, 96).
        shapes = ((13, 19, 7), (129, 137, 97), (32, 65, 160))
        scales = (
            (None, None), ((32, 96), (64, 64)), (None, (32, 32)), ((64, 64), None),
            ((1, 32), (32, 32)), ((32, 1), (32, 32)), ((32, 32), (32, 1)),
            ((3, 5), (7, 9)), (None, (7, 9)), ((3, 5), None),
        )
        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        for m, n, k in shapes:
            for a_block, b_block in scales:
                for a_dtype, b_dtype in product(fp8_dtypes, repeat=2):
                    with self.subTest(shape=(m, n, k), blocks=(a_block, b_block),
                                      dtypes=(a_dtype, b_dtype)):
                        a = quantize_fp8(torch.randn(m, k, device=self.device) * 0.3,
                                         fp8_dtype=a_dtype, block_shape=a_block)
                        b = quantize_fp8(torch.randn(n, k, device=self.device) * 0.3,
                                         fp8_dtype=b_dtype, block_shape=b_block)
                        bias = torch.randn(n, device=self.device) * 0.1
                        actual = gemm_nt(a, b, backend="triton", bias=bias)
                        # CPU FP32 oracle avoids CUDA TF32 settings affecting the comparison.
                        expected = a.dequantize().cpu() @ b.dequantize().cpu().T + bias.cpu()
                        self.assertEqual(actual.device, self.device)
                        self.assertFalse(actual.requires_grad)
                        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=3e-4)

    def test_output_dtypes_pipeline_stages_and_biasless_gemm(self):
        for a_block, b_block in ((None, None), ((32, 1), (7, 9))):
            a = quantize_fp8(torch.randn(9, 73, device=self.device) * 0.2,
                             fp8_dtype=torch.float8_e5m2, block_shape=a_block)
            b = quantize_fp8(torch.randn(17, 73, device=self.device) * 0.2,
                             block_shape=b_block)
            expected = a.dequantize().cpu() @ b.dequantize().cpu().T
            for dtype, tolerance in ((torch.float32, 3e-4), (torch.float16, 2e-3), (torch.bfloat16, 1e-2)):
                for stages in (1, 2, 3, 4):
                    with self.subTest(blocks=(a_block, b_block), dtype=dtype, stages=stages):
                        actual = gemm_nt(a, b, backend="triton", out_dtype=dtype, num_stages=stages)
                        self.assertEqual(actual.dtype, dtype)
                        torch.testing.assert_close(actual.cpu(), expected.to(dtype),
                                                   rtol=tolerance, atol=tolerance)

    def test_linear_forward_and_all_gradients_match_quantized_reference(self):
        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        roles = ("activation", "weight", "grad")
        layouts = ((None, None, None), ((32, 96),) * 3, ((1, 32), (32, 32), (32, 32)))
        cases = tuple(product(layouts, product(fp8_dtypes, repeat=3)))
        hybrid_formats = (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e5m2)
        cases += (
            (((1, 32),) * 3, hybrid_formats),
            (((3, 5), (7, 9), (5, 3)), hybrid_formats),
        )
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for blocks, formats in cases:
                with self.subTest(dtype=dtype, blocks=blocks, formats=formats):
                    plan = fp8_plan("triton")
                    plan["fp8_training"]["fp8_dtypes"] = dict(zip(roles, formats))
                    plan["block_shapes"] = dict(zip(roles, blocks))
                    reference_plan = deepcopy(plan)
                    reference_plan["fp8_training"]["backend"] = "reference"
                    x = (torch.randn(2, 17, 65, device=self.device, dtype=dtype) * 0.2).requires_grad_()
                    weight = (torch.randn(37, 65, device=self.device, dtype=dtype) * 0.2).requires_grad_()
                    bias = (torch.randn(37, device=self.device, dtype=dtype) * 0.1).requires_grad_()
                    ref_x, ref_w, ref_b = [value.detach().clone().requires_grad_() for value in (x, weight, bias)]
                    actual = do_linear(x, weight, bias, plan=plan)
                    expected = do_linear(ref_x, ref_w, ref_b, plan=reference_plan)
                    grad = torch.randn_like(actual) * 0.1
                    gradients = torch.autograd.grad(actual, (x, weight, bias), grad)
                    reference_gradients = torch.autograd.grad(expected, (ref_x, ref_w, ref_b), grad)
                    tolerance = 4e-4 if dtype == torch.float32 else (3e-3 if dtype == torch.float16 else 2e-2)
                    for value, baseline in zip((actual, *gradients), (expected, *reference_gradients)):
                        self.assertTrue(torch.isfinite(value).all().item())
                        torch.testing.assert_close(value, baseline, rtol=tolerance, atol=tolerance)


@unittest.skipIf(FP8_CUDA_SKIP_REASON, FP8_CUDA_SKIP_REASON)
class TritonFullModelBlackBoxTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(173)

    def _exercise(self, factory, mode):
        source, inputs, labels = factory()
        source = source.cuda().train()
        reference, _ = transform_model(deepcopy(source), plan=fp8_plan(), trace_mode=mode)
        model, report = transform_model(source, plan=fp8_plan("triton"), trace_mode=mode)
        self.assertTrue(report.converted_modules or report.function_calls)
        inputs, labels = move_inputs(inputs, "cuda"), labels.cuda()
        actual, loss = logits_and_loss(model, inputs, labels)
        expected, reference_loss = logits_and_loss(reference, inputs, labels)
        assert_relative_error(self, actual, expected, 0.02, "Triton vs reference FP8 logits")
        assert_relative_error(self, loss, reference_loss, 0.01, "Triton vs reference FP8 loss")
        loss.backward()
        reference_loss.backward()
        actual_grads, expected_grads = [], []
        expected_parameters = dict(reference.named_parameters())
        self.assertEqual(dict(model.named_parameters()).keys(), expected_parameters.keys())
        for name, parameter in model.named_parameters():
            reference_parameter = expected_parameters[name]
            self.assertEqual(parameter.grad is None, reference_parameter.grad is None, name)
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                actual_grads.append(parameter.grad.flatten())
                expected_grads.append(reference_parameter.grad.flatten())
        self.assertTrue(actual_grads)
        assert_relative_error(self, torch.cat(actual_grads), torch.cat(expected_grads),
                              0.05, "Triton vs reference FP8 gradients")
        before = [value.detach().clone() for value in model.parameters()]
        torch.optim.SGD(model.parameters(), lr=0.01).step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))
        self.assertTrue(all(torch.isfinite(value).all().item() for value in model.parameters()))
        torch.cuda.synchronize()

    def test_torchvision_vit_real_triton_forward_backward_and_update(self):
        self._exercise(make_vit, "full")

    def test_timm_vit_real_triton_forward_backward_and_update(self):
        self._exercise(make_timm_vit, "modules")

    def test_hf_gpt2_real_triton_forward_backward_and_update(self):
        self._exercise(make_gpt2, "modules")

    def test_hf_llama_real_triton_forward_backward_and_update(self):
        self._exercise(make_llama, "auto")


if __name__ == "__main__":
    unittest.main()
