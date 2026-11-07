"""Finite additive masks must remain finite through low-precision projections."""

import unittest

import torch
from torch import nn

from deepopt.opt.do_nn.mha_module import DOMultiheadAttention
from tests.opt.fp8_support import fp8_plan


class FP8AttentionMaskTests(unittest.TestCase):
    def test_finite_common_offsets_preserve_attention_for_both_execution_paths(self):
        for dtype in (torch.float16, torch.bfloat16):
            original = nn.MultiheadAttention(2, 1, batch_first=True, dtype=dtype).eval()
            with torch.no_grad():
                original.in_proj_weight.copy_(torch.eye(2).repeat(3, 1))
                original.in_proj_bias.zero_()
                original.out_proj.weight.copy_(torch.eye(2))
                original.out_proj.bias.zero_()
            adapter = DOMultiheadAttention(original, plan=fp8_plan())
            query = torch.zeros(1, 1, 2, dtype=dtype)
            key = torch.zeros(1, 2, 2, dtype=dtype)
            value = torch.tensor([[[1., 2.], [3., 4.]]], dtype=dtype)
            cases = (
                {"attn_mask": torch.full((1, 2), -100000., dtype=torch.float32)},
                {"key_padding_mask": torch.full((1, 2), -100000., dtype=torch.float32)},
                {"attn_mask": torch.full((1, 2), -40000., dtype=torch.float16),
                 "key_padding_mask": torch.full((1, 2), -40000., dtype=torch.float16)},
            )
            for need_weights in (False, True):
                with torch.no_grad():
                    expected, expected_weights = adapter(query, key, value, need_weights=need_weights)
                for masks in cases:
                    with self.subTest(dtype=dtype, weights=need_weights, masks=tuple(masks)), torch.no_grad():
                        actual, weights = adapter(query, key, value, need_weights=need_weights, **masks)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        if need_weights:
                            torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
                        else:
                            self.assertIsNone(weights)

    def test_boolean_mask_still_excludes_keys_when_combined_with_large_finite_bias(self):
        original = nn.MultiheadAttention(2, 1, batch_first=True, dtype=torch.float16).eval()
        adapter = DOMultiheadAttention(original, plan=fp8_plan())
        query = torch.zeros(1, 1, 2, dtype=torch.float16)
        key = torch.zeros(1, 2, 2, dtype=torch.float16)
        value = torch.tensor([[[1., 2.], [3., 4.]]], dtype=torch.float16)
        boolean = torch.tensor([[True, False]])
        for need_weights in (False, True):
            with self.subTest(weights=need_weights), torch.no_grad():
                expected, _ = adapter(query, key, value, need_weights=need_weights, attn_mask=boolean)
                actual, weights = adapter(query, key, value, need_weights=need_weights,
                                          attn_mask=boolean,
                                          key_padding_mask=torch.full((1, 2), -100000.))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if need_weights:
                    torch.testing.assert_close(weights, torch.tensor([[[0., 1.]]], dtype=torch.float16))


if __name__ == "__main__":
    unittest.main()
