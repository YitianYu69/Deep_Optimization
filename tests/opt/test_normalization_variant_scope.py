"""Kernel variant ownership follows the saved autograd invocation."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from deepopt.opt.backends._shared.normalization import normalization_kernel_scope, normalization_variant, layer_backward_programs
from deepopt.opt.backends.reference.normalization import layer as layer_reference
from deepopt.opt.nn.autograd.normalization import layer as layer_autograd
from deepopt.opt.nn.functional import do_layer_norm
from deepopt.opt.policy import normalization_plan


class NormalizationVariantTests(unittest.TestCase):
    def test_new_program_policy_can_populate_the_gpu_and_bounds_all_partial_storage(self):
        self.assertEqual(layer_backward_programs(6304, 768, 4, 264), 264)
        self.assertEqual(layer_backward_programs(6304, 768, 4, 0, num_sms=132), 528)
        self.assertEqual(layer_backward_programs(3, 768, 4, 512), 1)
        for width in (33, 768, 3072, 4096):
            p = layer_backward_programs(100864, width, 1, 4096)
            size, n = p, p
            while n > 1:
                n = (n + 127) // 128
                size += n
            self.assertLessEqual(size * 2 * width * 4, 8 * 2**20)

    def test_checkpoint_recomputation_restores_variant_before_native_saved_tensor_consumers(self):
        from deepopt.opt.scaling.delayed import fp8_checkpoint

        observed = []
        plan = normalization_plan(False, activation_compression=True)
        def compute(value):
            observed.append(normalization_variant())
            return do_layer_norm(value, (17,), plan=plan).relu()
        x = torch.randn(3, 17, requires_grad=True)
        with normalization_kernel_scope("candidate", row_chunk=8, max_programs=128):
            y = fp8_checkpoint(compute, x)
        with normalization_kernel_scope("baseline"):
            y.sum().backward()
            self.assertEqual(normalization_variant(), "baseline")
        self.assertGreaterEqual(len(observed), 2)
        self.assertTrue(all(value == "candidate" for value in observed))

    def test_backward_restores_forward_variant_even_outside_its_scope(self):
        observed = []
        def backward(*args, **kwargs):
            observed.append(normalization_variant())
            return layer_reference.backward(*args, **kwargs)
        compute = SimpleNamespace(forward=layer_reference.forward, backward=backward)
        with patch.object(layer_autograd, "executor", return_value=compute):
            x = torch.randn(3, 17, requires_grad=True)
            with normalization_kernel_scope("candidate"):
                y = do_layer_norm(x, (17,), plan=normalization_plan(False, activation_compression=True))
            with normalization_kernel_scope("baseline"):
                y.square().mean().backward()
                self.assertEqual(normalization_variant(), "baseline")
        self.assertEqual(observed, ["candidate"])
        self.assertEqual(normalization_variant(), "auto")
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_variants_fail_and_nested_exception_restores_context(self):
        with self.assertRaises(ValueError), normalization_kernel_scope("unknown"):
            pass
        with self.assertRaises(RuntimeError):
            with normalization_kernel_scope("candidate"):
                raise RuntimeError("failure")
        self.assertEqual(normalization_variant(), "auto")


if __name__ == "__main__":
    unittest.main()
