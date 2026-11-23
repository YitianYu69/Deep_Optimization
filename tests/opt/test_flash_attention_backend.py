"""Optional backend routing, saved-state contract, and configuration ownership."""

from dataclasses import asdict
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import deepopt
from deepopt._config_resolution import ResolvedConfig, resolve_config
from deepopt.config import AttentionConfig, ConfigError, attention_config_from_dict
from deepopt.cuda_graph._validation import operation_policy_spec
from deepopt.opt._autocast import _get_attention_backend
from deepopt.opt.nn import attention_backend as backend
from deepopt.opt.nn import functional as doF
from deepopt.opt.policy import prepare_fp8_model


def fake_forward(q, k, v, dropout, scale, *args):
    assert dropout == 0
    q, k, v = (x.transpose(1, 2).float() for x in (q, k, v))
    scores = q @ k.transpose(-2, -1) * scale
    lse = scores.logsumexp(-1)
    out = (scores.softmax(-1) @ v).to(torch.bfloat16).transpose(1, 2)
    return out, lse, torch.empty(0), torch.zeros(2, dtype=torch.int64)


def fake_backward(dout, q, k, v, out, lse, dq, dk, dv, dropout, scale, *args, **kwargs):
    q, k, v, dout, out = (x.transpose(1, 2).float() for x in (q, k, v, dout, out))
    p = (q @ k.transpose(-2, -1) * scale - lse[..., None]).exp()
    ds = p * (dout @ v.transpose(-2, -1) - (dout * out).sum(-1, keepdim=True))
    dq.copy_((ds @ k * scale).transpose(1, 2))
    dk.copy_((ds.transpose(-2, -1) @ q * scale).transpose(1, 2))
    dv.copy_((p.transpose(-2, -1) @ dout).transpose(1, 2))


class FlashAttentionBackendTests(unittest.TestCase):
    def setUp(self):
        deepopt.reset_config()
        self.addCleanup(deepopt.reset_config)
        torch.manual_seed(141)

    def test_config_validation_roundtrip_and_provenance(self):
        for name in ("native", "flash_attn_2"):
            self.assertEqual(attention_config_from_dict({"backend": name}), AttentionConfig(name))
            value = resolve_config({"runtime": {"attention": {"backend": name}}})
            self.assertEqual(value["runtime"]["attention"], asdict(AttentionConfig(name)))
        self.assertNotIn("attention", resolve_config({"runtime": {"checks": False}})["runtime"])
        for invalid in (True, None, [], {"backend": "flash"}, {"unexpected": True}, {"backend": []}):
            with self.subTest(invalid=invalid), self.assertRaises(ConfigError):
                attention_config_from_dict(invalid)
        self.assertIs(deepopt.AttentionConfig, AttentionConfig)

    def test_scope_routes_without_compression_and_nested_scope_restores(self):
        q = torch.randn(1, 2, 3, 8)
        with patch.object(backend, "flash_attention_2", return_value=q) as flash:
            with deepopt.autocast("cpu", enabled=False, fp8=False, activation_compression=False,
                                 attention_backend="flash_attn_2"):
                self.assertIs(F.scaled_dot_product_attention(q, q, q), q)
                with deepopt.autocast("cpu", enabled=False):
                    self.assertEqual(_get_attention_backend("cpu"), "flash_attn_2")
                with deepopt.autocast("cpu", enabled=False, attention_backend="native"):
                    self.assertEqual(_get_attention_backend("cpu"), "native")
                    self.assertIsNot(F.scaled_dot_product_attention(q, q, q), q)
                self.assertEqual(_get_attention_backend("cpu"), "flash_attn_2")
        flash.assert_called_once()
        self.assertIsNone(flash.call_args.kwargs["compression"])
        self.assertIsNone(_get_attention_backend("cpu"))

    def test_graph_signature_tracks_backend_without_changing_compression(self):
        before = operation_policy_spec()
        with deepopt.autocast("cuda", enabled=False, attention_backend="flash_attn_2"):
            after = operation_policy_spec()
            self.assertNotEqual(before, after)
            self.assertEqual(before[1], after[1])
        self.assertEqual(before, operation_policy_spec())

    def test_native_path_does_not_load_optional_extension(self):
        q = torch.randn(1, 2, 3, 8, requires_grad=True)
        with patch.object(backend, "_kernels", side_effect=AssertionError("optional import")):
            doF.do_scaled_dot_product_attention(q, q, q).sum().backward()
        self.assertIsNotNone(q.grad)
        self.assertNotIn("flash_attn_2_cuda", sys.modules)

    def test_unsupported_inputs_fail_before_import(self):
        q = torch.randn(1, 2, 3, 8)
        with patch.object(backend, "_kernels", side_effect=AssertionError("optional import")):
            with self.assertRaisesRegex(ValueError, "NVIDIA CUDA"):
                backend.flash_attention_2(q, q, q)
            for options in ({"is_causal": True}, {"enable_gqa": True}, {"attn_mask": torch.ones(3, 3)}):
                with self.assertRaisesRegex(ValueError, "unmasked"):
                    backend.flash_attention_2(q, q, q, **options)

    def test_preparation_preserves_mha_parameters_and_rejects_conflicts(self):
        original = torch.nn.MultiheadAttention(16, 2, batch_first=True)
        parameters = list(original.parameters())
        config = {"runtime": {"attention": {"backend": "flash_attn_2"}}}
        model, _ = prepare_fp8_model(original, False, config=config)
        self.assertEqual(list(model.parameters()), parameters)
        self.assertEqual(model.attention_backend, "flash_attn_2")
        again, _ = prepare_fp8_model(model, False, config=ResolvedConfig())
        self.assertIs(again, model)
        with self.assertRaisesRegex(ValueError, "backend differs"):
            prepare_fp8_model(model, False, config={"runtime": {"attention": {"backend": "native"}}})
        q = torch.randn(2, 3, 16)
        with self.assertRaisesRegex(ValueError, "need_weights=False"):
            model(q, q, q)
        with patch.object(backend, "flash_attention_2", return_value=torch.zeros(2, 2, 3, 8)):
            model(q, q, q, need_weights=False)[0].sum().backward()

    def test_fake_kernel_saved_state_and_partial_gradients(self):
        # A CPU stand-in checks state/layout/lifetime, not CUDA kernel accuracy.
        for bits in (None, 1, 2, 4, 8):
            for requires in ((True, True, True), (True, False, False), (False, False, True)):
                with self.subTest(bits=bits, requires=requires):
                    qkv = tuple(torch.randn(2, 3, 5, 8, dtype=torch.bfloat16).requires_grad_(r) for r in requires)
                    policy = None if bits is None else {"enabled": True, "group_size": 17, "attention_bits": bits}
                    packs = []
                    original_pack = backend._pack_operand

                    def record(x, b, group):
                        packs.append((tuple(x.shape), b))
                        return original_pack(x, b, group)

                    with patch.object(backend, "_validate_and_cast", side_effect=lambda q, k, v, *a: (q, k, v)), \
                            patch.object(backend, "_kernels", return_value=(fake_forward, fake_backward)), \
                            patch.object(backend, "_pack_operand", side_effect=record):
                        y = backend.flash_attention_2(*qkv, compression=policy)
                        y.float().square().sum().backward()
                    self.assertEqual(packs, [((2, 5, 3, 8), bits or 0)] * 3)
                    for x, needed in zip(qkv, requires):
                        self.assertEqual(x.grad is not None, needed)
                        if needed:
                            self.assertTrue(torch.isfinite(x.grad).all())

    def test_no_grad_skips_codec_and_observer_records_route(self):
        q = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16, requires_grad=True)
        records = []
        with patch.object(backend, "_validate_and_cast", side_effect=lambda q, k, v, *a: (q, k, v)), \
                patch.object(backend, "_kernels", return_value=(fake_forward, fake_backward)), \
                patch.object(backend, "_pack_operand", side_effect=AssertionError("unnecessary packing")), \
                backend.observe_attention(lambda inputs, out, info: records.append(info)), torch.no_grad():
            backend.flash_attention_2(q, q, q, compression={"enabled": True, "attention_bits": 4})
        self.assertEqual(records[0]["backend"], "flash_attn_2")
        self.assertEqual(records[0]["packing"], "none")

    def test_backward_keeps_valid_outer_strides_and_dense_gradient_layout(self):
        qkv = tuple(torch.randn(2, 3, 5, 8, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
        dy = torch.randn(2, 3, 5, 8, dtype=torch.bfloat16)
        seen = []
        def record(dout, q, k, v, out, lse, dq, dk, dv, *args, **kwargs):
            seen.append((dout.data_ptr(), tuple(x.stride() for x in (dq, dk, dv))))
            return fake_backward(dout, q, k, v, out, lse, dq, dk, dv, *args, **kwargs)
        with patch.object(backend, "_validate_and_cast", side_effect=lambda q, k, v, *a: (q, k, v)), \
                patch.object(backend, "_kernels", return_value=(fake_forward, record)):
            backend.flash_attention_2(*qkv, compression={"enabled": True, "attention_bits": 4, "group_size": 17}).backward(dy)
        self.assertEqual(seen[0][0], dy.data_ptr())
        self.assertEqual(seen[0][1], tuple(x.transpose(1, 2).stride() for x in qkv))

    def test_gradient_layout_does_not_reproduce_interleaved_storage_gaps(self):
        value = torch.empty(2, 5, 3, 3, 8)
        qkv = value.unbind(2)
        gradients = backend._gradient_buffers(qkv, tuple(x.stride() for x in qkv))
        self.assertTrue(all(x.is_contiguous() for x in gradients))
        self.assertEqual(sum(x.untyped_storage().nbytes() for x in gradients), value.numel() * value.element_size())

    def test_k_is_packed_before_transpose_in_legacy_bshd_order(self):
        qkv = tuple(torch.randn(2, 3, 5, 8, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
        calls = []
        original_pack = backend._pack_operand
        def record(value, bits, group):
            result = original_pack(value, bits, group)
            calls.append((value.detach().clone(), result))
            return result
        with patch.object(backend, "_validate_and_cast", side_effect=lambda q, k, v, *a: (q, k, v)), \
                patch.object(backend, "_kernels", return_value=(fake_forward, fake_backward)), \
                patch.object(backend, "_pack_operand", side_effect=record):
            output = backend.flash_attention_2(*qkv, compression={"enabled": True, "attention_bits": 4, "group_size": 17})
            output.sum().backward()
        self.assertEqual(len(calls), 3)  # Backward only decodes; it never re-packs K.
        packed_input, (actual_values, actual_header) = calls[1]
        canonical_k = qkv[1].detach().transpose(1, 2)
        torch.testing.assert_close(packed_input, canonical_k, rtol=0, atol=0)
        expected_values, expected_header = original_pack(canonical_k, 4, 17)
        self.assertEqual(actual_header, expected_header)
        for actual, expected in zip(actual_values, expected_values):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        transposed_values, _ = original_pack(qkv[1].detach().transpose(-2, -1), 4, 17)
        self.assertFalse(torch.equal(actual_values[0], transposed_values[0]))


if __name__ == "__main__":
    unittest.main()
