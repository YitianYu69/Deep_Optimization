"""Native SDPA preserves PyTorch's selected backend in its saved-state VJP."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from deepopt.compiler.autodiff.attention import SDPA
from deepopt.compiler.ir import TensorMeta, TensorValue


def specification(inputs, *, mask=None, **overrides):
    arguments = tuple(TensorValue(name, TensorMeta(tuple(value.shape), value.dtype, value.device,
                          tuple(value.stride()), value.requires_grad)) if value is not None else None
                      for name, value in zip(("q", "k", "v", "mask"), (*inputs, mask)))
    q, _, v = inputs
    output = TensorValue("attention", TensorMeta((*q.shape[:-1], v.shape[-1]), q.dtype, q.device))
    options = {"compression": None, "scale": None, "dropout_p": 0., "is_causal": False,
               "enable_gqa": False, "backend": "native", **overrides}
    return SDPA(SimpleNamespace(args=arguments, value=output, options=options), None)


class SDPABackendTests(unittest.TestCase):
    def inputs(self, *, device="cpu", dtype=torch.float32):
        return tuple(torch.randn(2, 3, length, 8, device=device, dtype=dtype, requires_grad=True)
                     for length in (5, 7, 7))

    def cudnn_forward_result(self, inputs):
        q, k, _ = inputs
        return (torch.randn_like(q), q.new_empty(q.shape[:-1]), torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.int32), q.shape[-2], k.shape[-2],
                torch.tensor(17, dtype=torch.int64), torch.tensor(31, dtype=torch.int64), q.new_empty(0))

    def test_cudnn_enum_selects_native_forward_and_saved_state_selects_its_vjp(self):
        inputs = self.inputs()
        program = specification(inputs)
        native = self.cudnn_forward_result(inputs)
        gradients = tuple(torch.full_like(value, index + 1) for index, value in enumerate(inputs))
        with patch.object(torch.ops.aten._fused_sdp_choice, "default", return_value=int(SDPBackend.CUDNN_ATTENTION)), \
                patch.object(torch.ops.aten._scaled_dot_product_cudnn_attention, "default", return_value=native) as forward:
            output, *saved = program.forward(*inputs, None)
        self.assertIs(output, native[0])
        self.assertEqual(int(saved[-1]), int(SDPBackend.CUDNN_ATTENTION))
        forward.assert_called_once()
        upstream = torch.randn_like(output)
        with patch.object(torch.ops.aten._fused_sdp_choice, "default", side_effect=AssertionError("VJP reselected the backend")), \
                patch.object(torch.ops.aten._scaled_dot_product_cudnn_attention_backward, "default", return_value=gradients) as backward:
            actual = program.gradients(upstream, output, None, *saved)
        backward.assert_called_once()
        self.assertIs(backward.call_args.args[6], native[6])
        self.assertIs(backward.call_args.args[7], native[7])
        self.assertEqual(backward.call_args.args[11:13], (inputs[0].shape[-2], inputs[1].shape[-2]))
        for value, expected in zip(actual, gradients):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_overrideable_backend_is_rejected_without_calling_cudnn(self):
        inputs = self.inputs()
        program = specification(inputs)
        with patch.object(torch.ops.aten._fused_sdp_choice, "default", return_value=int(SDPBackend.OVERRIDEABLE)), \
                patch.object(torch.ops.aten._scaled_dot_product_cudnn_attention, "default", side_effect=AssertionError("cuDNN called")), \
                self.assertRaisesRegex(NotImplementedError, "Overrideable SDPA"):
            program.forward(*inputs, None)

    def test_unknown_backend_and_invalid_saved_backend_fail_closed(self):
        inputs = self.inputs()
        program = specification(inputs)
        for identifier in (int(SDPBackend.ERROR), 8675309):
            with self.subTest(identifier=identifier), \
                    patch.object(torch.ops.aten._fused_sdp_choice, "default", return_value=identifier), \
                    self.assertRaisesRegex(NotImplementedError, "SDPA.*implementation"):
                program.forward(*inputs, None)
        with sdpa_kernel(SDPBackend.MATH):
            output, *saved = program.forward(*inputs, None)
        for identifier in (int(SDPBackend.ERROR), int(SDPBackend.OVERRIDEABLE), 8675309):
            with self.subTest(saved_identifier=identifier), self.assertRaisesRegex(NotImplementedError, "SDPA"):
                program.gradients(torch.ones_like(output), output, None, *saved[:-1], torch.tensor(identifier))

    def test_efficient_backend_preserves_forward_residuals_and_native_backward(self):
        inputs = self.inputs()
        program = specification(inputs)
        q = inputs[0]
        residuals = (torch.randn_like(q), q.new_empty(q.shape[:-1]), torch.tensor(13), torch.tensor(23))
        gradients = tuple(torch.randn_like(value) for value in inputs)
        with patch.object(torch.ops.aten._fused_sdp_choice, "default", return_value=int(SDPBackend.EFFICIENT_ATTENTION)), \
                patch.object(torch.ops.aten._scaled_dot_product_efficient_attention, "default", return_value=residuals):
            output, *saved = program.forward(*inputs, None)
        self.assertEqual(int(saved[-1]), int(SDPBackend.EFFICIENT_ATTENTION))
        with patch.object(torch.ops.aten._scaled_dot_product_efficient_attention_backward, "default", return_value=(*gradients, None)) as backward:
            actual = program.gradients(torch.randn_like(output), output, None, *saved)
        backward.assert_called_once()
        self.assertIs(backward.call_args.args[7], residuals[2])
        self.assertIs(backward.call_args.args[8], residuals[3])
        for value, expected in zip(actual, gradients):
            torch.testing.assert_close(value, expected)

    def test_cpu_flash_and_math_outputs_and_qkv_vjps_match_torch(self):
        torch.manual_seed(813)
        for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.MATH):
            for causal in (False, True):
                with self.subTest(backend=backend, causal=causal):
                    inputs = self.inputs(dtype=torch.float64)
                    references = tuple(value.detach().clone().requires_grad_() for value in inputs)
                    program = specification(inputs, is_causal=causal)
                    with sdpa_kernel(backend):
                        expected = torch.nn.functional.scaled_dot_product_attention(*references, is_causal=causal)
                        output, *saved = program.forward(*inputs, None)
                        upstream = torch.randn_like(expected)
                        expected.backward(upstream)
                        actual = program.gradients(upstream, output, None, *saved)
                    self.assertEqual(int(saved[-1]), 11 if backend == SDPBackend.FLASH_ATTENTION else int(SDPBackend.MATH))
                    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
                    for value, reference in zip(actual, references):
                        torch.testing.assert_close(value, reference.grad, rtol=1e-11, atol=1e-12)

    def test_flash_attn_2_retains_separate_saved_implementation_and_vjp(self):
        inputs = self.inputs()
        program = specification(inputs, backend="flash_attn_2")
        transposed = tuple(value.transpose(1, 2) for value in inputs)
        buffers = tuple(torch.zeros_like(value) for value in transposed)
        forward = Mock(return_value=(transposed[0].clone(), torch.empty(2, 3, 5), None, torch.tensor(29)))

        def backward(*args, **kwargs):
            for index, buffer in enumerate(args[6:9]):
                buffer.fill_(index + 1)

        kernels = forward, Mock(side_effect=backward)
        with patch.object(torch.ops.aten._fused_sdp_choice, "default", return_value=int(SDPBackend.MATH)), \
                patch("deepopt.opt.nn.attention_backend._validate_and_cast", return_value=inputs), \
                patch("deepopt.opt.nn.attention_backend._kernels", return_value=kernels), \
                patch("deepopt.opt.nn.attention_backend._gradient_buffers", return_value=buffers):
            output, *saved = program.forward(*inputs, None)
            self.assertEqual(int(saved[-1]), 10)
            actual = program.gradients(torch.ones_like(output), output, None, *saved)
        kernels[1].assert_called_once()
        for index, value in enumerate(actual):
            torch.testing.assert_close(value, torch.full_like(inputs[index], index + 1), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "cuDNN SDPA qualification requires a CUDA GPU")
    def test_cuda_cudnn_outputs_and_qkv_vjps_match_torch(self):
        if not torch.backends.cudnn.is_available():
            self.skipTest("cuDNN is unavailable")
        # A contiguous head width and lengths supported by native cuDNN SDPA.
        inputs = tuple(torch.randn(2, 4, 64, 64, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3))
        references = tuple(value.detach().clone().requires_grad_() for value in inputs)
        program = specification(inputs)
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            try:
                choice = int(torch.ops.aten._fused_sdp_choice.default(*inputs, None, 0., False))
            except RuntimeError as error:
                self.skipTest(f"cuDNN SDPA cannot run on this CUDA device: {error}")
            if choice != int(SDPBackend.CUDNN_ATTENTION):
                self.skipTest("The target CUDA device did not select cuDNN SDPA")
            expected = torch.nn.functional.scaled_dot_product_attention(*references)
            output, *saved = program.forward(*inputs, None)
            self.assertEqual(int(saved[-1]), int(SDPBackend.CUDNN_ATTENTION))
            upstream = torch.randn_like(expected)
            expected.backward(upstream)
            actual = program.gradients(upstream, output, None, *saved)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)
        for value, reference in zip(actual, references):
            torch.testing.assert_close(value, reference.grad, rtol=3e-3, atol=3e-3)


if __name__ == "__main__":
    unittest.main()
