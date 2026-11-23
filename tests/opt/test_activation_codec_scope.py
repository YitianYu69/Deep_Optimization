"""Codec experiments have bounded lifetimes, including checkpoint replay."""

import unittest
import torch

from deepopt.opt.backends._shared.activation_codec import (activation_codec_scope, activation_packing,
                                                         silu_saved_state, activation_layout_policy)
from deepopt.opt.scaling.delayed import fp8_checkpoint


class ActivationCodecScopeTests(unittest.TestCase):
    def test_nested_scope_and_failed_validation_do_not_leak(self):
        with activation_codec_scope("batched", groups_per_program=8):
            self.assertEqual(activation_packing(), ("batched", 8))
            with self.assertRaises(ValueError), activation_codec_scope("invalid"):
                pass
            self.assertEqual(activation_packing(), ("batched", 8))
        self.assertEqual(activation_packing(), ("auto", 4))

    def test_checkpoint_restores_the_forward_codec_during_recomputation(self):
        seen = []
        def compute(x):
            seen.append(activation_packing())
            return x.sin().relu()
        x = torch.randn(7, requires_grad=True)
        with activation_codec_scope("batched", groups_per_program=8):
            y = fp8_checkpoint(compute, x)
        y.sum().backward()
        self.assertGreaterEqual(len(seen), 2)
        self.assertTrue(all(value == ("batched", 8) for value in seen))
        self.assertEqual(activation_packing(), ("auto", 4))

    def test_silu_state_is_opt_in_validated_and_owned_by_checkpoint(self):
        self.assertEqual(silu_saved_state(), "derivative")
        with self.assertRaises(ValueError), activation_codec_scope(silu_state="auto"):
            pass
        seen = []
        def compute(x):
            seen.append((silu_saved_state(), activation_layout_policy()))
            return x.sin().relu()
        x = torch.randn(7, requires_grad=True)
        with activation_codec_scope(silu_state="recompute", activation_layout="channels_last"):
            y = fp8_checkpoint(compute, x)
        y.sum().backward()
        self.assertGreaterEqual(len(seen), 2)
        self.assertTrue(all(value == ("recompute", "channels_last") for value in seen))
        self.assertEqual(silu_saved_state(), "derivative")
        self.assertEqual(activation_layout_policy(), "auto")
