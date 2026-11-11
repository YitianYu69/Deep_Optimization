"""Real CUDA validation for native attention with packed Q/K/V saves."""

import importlib.util
import unittest

import torch
from torch.nn import functional as F

from deepopt.opt.nn import functional as doF
from tests.opt.test_attention_compression import (
    _math_backend, _oracle_hooks, compression_plan, record_attention_packs,
)


_CUDA_REASON = (
    "An NVIDIA CUDA GPU and Triton are required for activation codec validation"
    if not torch.cuda.is_available() or torch.version.hip is not None
    or importlib.util.find_spec("triton") is None else ""
)


@unittest.skipIf(_CUDA_REASON, _CUDA_REASON)
class AttentionCompressionCUDATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(372)

    def test_math_forward_and_gradients_match_explicit_compressed_saved_tensor_reference(self):
        shapes = ((1, 2, 3, 7), (1, 2, 5, 7), (1, 2, 5, 11))
        for bits in (1, 2, 4, 8):
            with self.subTest(bits=bits), _math_backend():
                inputs = tuple(torch.randn(shape, device="cuda", requires_grad=True) for shape in shapes)
                reference = tuple(value.detach().clone().requires_grad_() for value in inputs)
                with _oracle_hooks(reference, bits=bits, group_size=17):
                    expected = F.scaled_dot_product_attention(*reference, scale=.31)
                with record_attention_packs() as records:
                    actual = doF.do_scaled_dot_product_attention(
                        *inputs, scale=.31, plan=compression_plan(attention_bits=bits))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
                expected.backward(gradient)
                self.assertEqual({value.bits for value in records}, {bits})
                for value, wanted in zip(inputs, reference):
                    torch.testing.assert_close(value.grad, wanted.grad, rtol=2e-5, atol=2e-6)

    def test_flash_attention_backend_and_forward_are_preserved(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if not torch.backends.cuda.is_flash_attention_available():
            self.skipTest("PyTorch was built without FlashAttention")
        if torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("FlashAttention requires compute capability >= 8")
        for attention_bits, wanted_bits in ((None, 4), (2, 2), (1, 1)):
            with self.subTest(attention_bits=attention_bits), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                query = torch.randn(2, 2, 17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                key = torch.randn(2, 2, 23, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                value = torch.randn_like(key, requires_grad=True)
                expected = F.scaled_dot_product_attention(query, key, value)
                with record_attention_packs() as records:
                    actual = doF.do_scaled_dot_product_attention(
                        query, key, value, plan=compression_plan(attention_bits=attention_bits))
                self.assertEqual(type(actual.grad_fn), type(expected.grad_fn))
                self.assertIn("FlashAttention", type(actual.grad_fn).__name__)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.backward(torch.randn_like(actual))
                self.assertTrue(records)
                self.assertEqual({packed.bits for packed in records}, {wanted_bits})
                for tensor in (query, key, value):
                    self.assertTrue(torch.isfinite(tensor.grad).all())
                    self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_native_torch_attention_is_compressed_inside_bf16_amp_scope(self):
        import deepopt

        query = torch.randn(1, 2, 13, 32, device="cuda", requires_grad=True)
        key = torch.randn(1, 2, 19, 32, device="cuda", requires_grad=True)
        value = torch.randn_like(key, requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = F.scaled_dot_product_attention(query, key, value)
        with record_attention_packs() as records:
            with deepopt.autocast(
                "cuda", dtype=torch.bfloat16, fp8=False,
                activation_compression={"enabled": True, "bits": 1},
            ):
                actual = F.scaled_dot_product_attention(query, key, value)
            actual.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(records)
        self.assertEqual({packed.bits for packed in records}, {4})
        for tensor in (query, key, value):
            self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main()
