"""Native rounded statistics are inputs to the staged VJP, not recomputed."""
import unittest

import torch

from deepopt.opt.nn.normalization.native_group import NativeGroupContract
from deepopt.opt.backends.reference.normalization import native_group as reference


class NativeGroupStagedTests(unittest.TestCase):
    device = 'cpu'

    def executor(self):
        if self.device == 'cpu':
            return reference
        from deepopt.opt.backends.triton.normalization import native_group
        return native_group

    def test_native_forward_saved_statistics_and_fixed_upstream_masks(self):
        for dtype, affine_dtype in ((torch.bfloat16, torch.bfloat16),
                                     (torch.bfloat16, torch.float32), (torch.float32, torch.float32)):
            for layout in ('contiguous', 'channels_last', 'strided'):
                for needs in ((True, True, True), (True, False, False), (False, True, False), (False, False, True)):
                    with self.subTest(dtype=dtype, affine=affine_dtype, layout=layout, needs=needs):
                        torch.manual_seed(47)
                        x = torch.randn(2, 4, 3, 10 if layout == 'strided' else 5, device=self.device, dtype=dtype)
                        if layout == 'strided':
                            x = x[..., ::2]
                        if layout == 'channels_last':
                            x = x.contiguous(memory_format=torch.channels_last)
                        w = torch.randn(4, device=self.device, dtype=affine_dtype)
                        b = torch.randn_like(w)
                        contract = NativeGroupContract(tuple(x.shape), 2, 1e-5, dtype, affine_dtype, dtype, affine_dtype)
                        y, mean, inv = torch.ops.aten.native_group_norm.default(x, w, b, 2, 4, 15, 2, 1e-5)
                        actual, saved_mean, saved_inv, source, gamma = self.executor().forward(x, w, b, contract)
                        tol = dict(rtol=0.02, atol=0.02) if dtype == torch.bfloat16 else dict(rtol=2e-5, atol=2e-6)
                        torch.testing.assert_close(actual, y, **tol)
                        self.assertEqual(saved_mean.dtype, mean.dtype)
                        torch.testing.assert_close(saved_mean.reshape_as(mean), mean, **tol)
                        torch.testing.assert_close(saved_inv.reshape_as(inv), inv, **tol)
                        dy = torch.randn_like(y)
                        expected = torch.ops.aten.native_group_norm_backward.default(dy, x, mean, inv, w, 2, 4, 15, 2, list(needs))
                        # Exact captured native statistics isolate the backward
                        # contract from permissible parallel forward reductions.
                        got = self.executor().backward(dy, source if any(needs[:2]) else None,
                            mean if any(needs[:2]) else None, inv if any(needs[:2]) else None,
                            gamma if needs[0] else None, contract, needs=needs)
                        for a, e, need in zip(got, expected, needs):
                            if need:
                                torch.testing.assert_close(a, e, **tol)
                            else:
                                self.assertIsNone(a)

    def test_activation_rounding_and_saved_statistic_perturbation(self):
        torch.manual_seed(81)
        x = torch.randn(2, 4, 3, 17, device=self.device, dtype=torch.bfloat16)
        w, b = torch.randn(4, device=self.device, dtype=x.dtype), torch.randn(4, device=self.device, dtype=x.dtype)
        c = NativeGroupContract(tuple(x.shape), 2, 1e-5, x.dtype, x.dtype, x.dtype, x.dtype)
        raw, mean, inv, source, gamma = self.executor().forward(x, w, b, c)
        for activation in (('silu', 'none'), ('gelu', 'none'), ('gelu', 'tanh')):
            pre = torch.empty_like(raw)
            info = {}
            y, *_ = self.executor().forward(x, w, b, c, activation=activation, preactivation=pre, execution_info=info)
            function = torch.nn.functional.silu if activation[0] == 'silu' else lambda t: torch.nn.functional.gelu(t, approximate=activation[1])
            torch.testing.assert_close(pre, raw, rtol=0, atol=0)
            torch.testing.assert_close(y, function(raw), rtol=0.02, atol=0.02)
            leaf = pre.detach().requires_grad_()
            dy = torch.randn_like(y)
            adjusted, = torch.autograd.grad(function(leaf), leaf, dy)
            actual = self.executor().backward(dy, source, mean, inv, gamma, c,
                derivative_input=pre, derivative_activation=activation)
            expected = self.executor().backward(adjusted, source, mean, inv, gamma, c)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=0.02, atol=0.04)
            changed = self.executor().backward(dy, source, mean, inv * 0.5, gamma, c)
            original = self.executor().backward(dy, source, mean, inv, gamma, c)
            self.assertFalse(torch.equal(changed[0], original[0]))
            self.assertFalse(info['cuda_numerically_qualified'])


@unittest.skipUnless(torch.cuda.is_available(), 'Requires actual target CUDA execution')
class NativeGroupStagedCudaTests(NativeGroupStagedTests):
    device = 'cuda'
