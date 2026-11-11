"""Attention keeps native forward semantics and compresses only saved Q/K/V."""

from contextlib import contextmanager
import gc
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from deepopt.opt.nn import functional as doF
from deepopt.opt.policy import linear_plan
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation


def compression_plan(bits=2, attention_bits=None, enabled=True, group_size=17):
    return linear_plan(False, activation_compression={
        "enabled": enabled, "group_size": group_size,
        "bits": bits, "attention_bits": attention_bits,
    })


@contextmanager
def record_attention_packs():
    """Observe real payloads rather than inferring bit width from tensor dtype."""
    records = []

    def record(value, *args, **kwargs):
        packed = quantize_pack_activation(value, *args, **kwargs)
        records.append(packed)
        return packed

    with patch("deepopt.opt.quantization.attention.quantize_pack_activation", side_effect=record):
        yield records


def _math_backend():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH)


def _oracle_hooks(inputs, *, bits, group_size):
    """Compress the native backend's Q/K/V saves, preserving exact softmax saves.

    Test shapes deliberately give Q, K and V unique element counts that differ
    from probabilities, the output, and logsumexp. This independent selection
    does not use DeepOpt's tensor tags or saved-tensor selection logic.
    """
    sizes = {value.numel() for value in inputs}

    def save(value):
        if value.is_floating_point() and value.numel() in sizes:
            packed = quantize_pack_activation(value.detach().reshape(-1, value.shape[-1]),
                                     group_size=group_size, bits=bits)
            return (packed, value.shape, value.dtype)
        return value

    def restore(value):
        if isinstance(value, tuple):
            packed, shape, dtype = value
            return dequantize_unpack_activation(packed, dtype=dtype).reshape(shape)
        return value

    return torch.autograd.graph.saved_tensors_hooks(save, restore)


class AttentionCompressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(926)

    @staticmethod
    def inputs(*, requires_grad=(True, True, True)):
        # Q/K/V counts 42, 70, 110 differ from scores (30), output (66), LSE (6).
        shapes = ((1, 2, 3, 7), (1, 2, 5, 7), (1, 2, 5, 11))
        return tuple(torch.randn(shape).requires_grad_(trainable)
                     for shape, trainable in zip(shapes, requires_grad))

    def test_native_forward_and_math_backward_match_independent_saved_tensor_oracle(self):
        for requested_bits, expected_bits in ((None, 4), (2, 2), (1, 1)):
            with self.subTest(attention_bits=requested_bits), _math_backend():
                inputs = self.inputs()
                reference_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
                mask = torch.randn(3, 5) * .05
                gradient = torch.randn(1, 2, 3, 11)
                with _oracle_hooks(reference_inputs, bits=expected_bits, group_size=17):
                    expected = F.scaled_dot_product_attention(*reference_inputs, attn_mask=mask, scale=.31)
                with record_attention_packs() as records:
                    actual = doF.do_scaled_dot_product_attention(
                        *inputs, attn_mask=mask, scale=.31,
                        plan=compression_plan(attention_bits=requested_bits))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(type(actual.grad_fn), type(expected.grad_fn))
                actual.backward(gradient)
                expected.backward(gradient)
                self.assertTrue(records)
                self.assertEqual({value.bits for value in records}, {expected_bits})
                for value, reference in zip(inputs, reference_inputs):
                    torch.testing.assert_close(value.grad, reference.grad, rtol=1e-5, atol=1e-6)

    def test_default_attention_floor_and_explicit_override(self):
        for global_bits, override, expected in ((1, None, 4), (2, None, 4), (4, None, 4),
                                                (8, None, 8), (8, 2, 2), (2, 1, 1), (2, 8, 8)):
            with self.subTest(global_bits=global_bits, override=override):
                with record_attention_packs() as records:
                    result = doF.scaled_dot_product_attention(
                        *self.inputs(), plan=compression_plan(global_bits, override))
                    result.sum().backward()
                self.assertTrue(records)
                self.assertEqual({packed.bits for packed in records}, {expected})

    def test_cpu_flash_backend_and_gradients_match_storage_selected_reference(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        shapes = ((1, 2, 3, 8), (1, 2, 5, 8), (1, 2, 5, 8))
        for override, bits in ((None, 4), (1, 1)):
            with self.subTest(attention_bits=override), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                values = tuple(torch.randn(shape, requires_grad=True) for shape in shapes)
                reference = tuple(value.detach().clone().requires_grad_() for value in values)
                source_storages = {value.untyped_storage()._cdata for value in reference}
                oracle_payloads = []

                def save(value):
                    # Flash saves the output with Q's shape. Storage identity
                    # distinguishes it from Q without involving DeepOpt tags.
                    if value.untyped_storage()._cdata in source_storages:
                        packed = quantize_pack_activation(value.detach().reshape(-1, value.shape[-1]),
                                                 group_size=17, bits=bits)
                        oracle_payloads.append(packed)
                        return packed, value.shape, value.dtype
                    return value

                def restore(value):
                    if isinstance(value, tuple):
                        packed, shape, dtype = value
                        return dequantize_unpack_activation(packed, dtype=dtype).reshape(shape)
                    return value

                try:
                    with torch.autograd.graph.saved_tensors_hooks(save, restore):
                        expected = F.scaled_dot_product_attention(*reference)
                except RuntimeError as error:
                    message = str(error).lower()
                    if any(text in message for text in ("not implemented", "not supported", "no available kernel")):
                        self.skipTest(f"Native CPU FlashAttention is unavailable: {error}")
                    raise
                with record_attention_packs() as records:
                    actual = doF.do_scaled_dot_product_attention(
                        *values, plan=compression_plan(attention_bits=override))
                self.assertIn("FlashAttention", type(actual.grad_fn).__name__)
                self.assertEqual(type(actual.grad_fn), type(expected.grad_fn))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(len(records), 3)
                self.assertEqual(len(oracle_payloads), 3)
                self.assertEqual({packed.bits for packed in records}, {bits})
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
                expected.backward(gradient)
                for value, wanted in zip(values, reference):
                    torch.testing.assert_close(value.grad, wanted.grad, rtol=1e-5, atol=1e-6)

    def test_fake_tensor_attention_uses_metadata_codec_without_cuda_or_triton(self):
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
        from deepopt.opt.quantization.attention import AttentionCompression

        with FakeTensorMode(), _math_backend():
            values = tuple(torch.empty(shape, requires_grad=True) for shape in
                           ((1, 2, 3, 8), (1, 2, 5, 8), (1, 2, 5, 11)))
            with record_attention_packs() as records:
                result = doF.do_scaled_dot_product_attention(
                    *values, plan=compression_plan(attention_bits=1))
            self.assertIsInstance(result, FakeTensor)
            self.assertEqual(result.shape, (1, 2, 3, 11))
            self.assertTrue(result.requires_grad)
            self.assertTrue(records)
            self.assertTrue(all(isinstance(packed.data, FakeTensor) for packed in records))
            self.assertEqual({packed.bits for packed in records}, {1})

            # CUDA storage tagging and the codec need no physical GPU. Exercise
            # saved-value handling directly; CPU-only PyTorch cannot execute a
            # native CUDA SDPA autograd kernel even with metadata-only tensors.
            query = torch.empty(1, 2, 3, 8, device="cuda", dtype=torch.bfloat16)
            key = torch.empty(1, 2, 5, 8, device="cuda", dtype=torch.bfloat16)
            value = torch.empty_like(key)
            with AttentionCompression(query, key, value, group_size=17, bits=4) as scope:
                saved = scope._quantize_pack(query)
            restored = scope._dequantize_unpack(saved)
            self.assertEqual(saved.packed.bits, 4)
            self.assertEqual(saved.packed.data.shape, (3, 3))
            self.assertIsInstance(restored, FakeTensor)
            self.assertEqual(restored.shape, query.shape)
            self.assertEqual(restored.device.type, "cuda")
            self.assertEqual(restored.dtype, torch.bfloat16)

    def test_disabled_and_no_grad_do_not_pack(self):
        values = self.inputs()
        with record_attention_packs() as records:
            result = doF.do_scaled_dot_product_attention(*values, plan=compression_plan(enabled=False))
            with torch.no_grad():
                inference = doF.do_scaled_dot_product_attention(*values, plan=compression_plan())
            frozen = doF.do_scaled_dot_product_attention(
                *(value.detach() for value in values), plan=compression_plan())
        self.assertFalse(records)
        self.assertFalse(inference.requires_grad)
        self.assertFalse(frozen.requires_grad)
        torch.testing.assert_close(result, F.scaled_dot_product_attention(*values), rtol=0, atol=0)

    def test_saved_tags_do_not_retain_dense_intermediate_qkv_storage(self):
        from torch.multiprocessing.reductions import StorageWeakRef

        leaves = self.inputs()
        # Addition's backward needs no saved output, isolating SDPA's retention.
        query, key, value = (tensor + 0 for tensor in leaves)
        storage_refs = [StorageWeakRef(tensor.untyped_storage()) for tensor in (query, key, value)]
        with _math_backend():
            output = doF.do_scaled_dot_product_attention(
                query, key, value, plan=compression_plan(attention_bits=2))
        del query, key, value
        gc.collect()
        self.assertTrue(all(reference.expired() for reference in storage_refs))
        output.sum().backward()
        self.assertTrue(all(torch.isfinite(tensor.grad).all() for tensor in leaves))

    def test_dropout_masks_and_partial_gradient_requirements(self):
        for requires_grad in ((False, False, True), (True, False, False), (False, True, True)):
            with self.subTest(requires_grad=requires_grad), _math_backend():
                values = self.inputs(requires_grad=requires_grad)
                reference = tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in values)
                mask = torch.tensor([[True, True, False, True, True]]).expand(3, -1)
                torch.manual_seed(71)
                expected = F.scaled_dot_product_attention(*reference, attn_mask=mask, dropout_p=.2)
                torch.manual_seed(71)
                actual = doF.do_scaled_dot_product_attention(
                    *values, attn_mask=mask, dropout_p=.2, plan=compression_plan(attention_bits=1))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.sum().backward()
                for value, trainable in zip(values, requires_grad):
                    self.assertEqual(value.grad is not None, trainable)
                    if trainable:
                        self.assertTrue(torch.isfinite(value.grad).all())

    def test_trainable_mask_with_frozen_qkv_still_compresses_saved_values(self):
        values = self.inputs(requires_grad=(False, False, False))
        mask = torch.randn(3, 5, requires_grad=True)
        reference_mask = mask.detach().clone().requires_grad_()
        with _math_backend():
            with _oracle_hooks(values, bits=4, group_size=17):
                expected = F.scaled_dot_product_attention(*values, attn_mask=reference_mask)
            with record_attention_packs() as records:
                actual = doF.do_scaled_dot_product_attention(
                    *values, attn_mask=mask, plan=compression_plan())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        self.assertTrue(records)
        self.assertEqual({packed.bits for packed in records}, {4})
        torch.testing.assert_close(mask.grad, reference_mask.grad, rtol=0, atol=0)

    def test_noncontiguous_inputs_causal_attention_and_grouped_query(self):
        query = torch.randn(1, 4, 8, 6).transpose(-1, -2).requires_grad_()
        key = torch.randn(1, 2, 8, 6).transpose(-1, -2).requires_grad_()
        value = torch.randn(1, 2, 11, 6).transpose(-1, -2).requires_grad_()
        with _math_backend():
            expected = F.scaled_dot_product_attention(query, key, value, is_causal=True, enable_gqa=True)
            actual = doF.do_scaled_dot_product_attention(
                query, key, value, is_causal=True, enable_gqa=True, plan=compression_plan())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        for tensor in (query, key, value):
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_deepopt_scope_intercepts_native_sdpa_even_without_linear_interception(self):
        import deepopt

        settings = deepopt.FP8Config(enabled=True, backend="reference", intercept_linear=False)
        inputs = self.inputs()
        with record_attention_packs() as records:
            with deepopt.autocast("cpu", enabled=False, fp8=settings,
                                 activation_compression={"enabled": True, "bits": 1, "attention_bits": 2}):
                actual = F.scaled_dot_product_attention(*inputs)
            actual.sum().backward()
        self.assertTrue(records)
        self.assertEqual({value.bits for value in records}, {2})

    def test_projection_inputs_keep_global_bits(self):
        plan = compression_plan(bits=1)
        x = torch.randn(6, 8, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(8, 8, dtype=torch.bfloat16, requires_grad=True)
        linear_payloads = []

        def record(value, *args, **kwargs):
            packed = quantize_pack_activation(value, *args, **kwargs)
            linear_payloads.append(packed)
            return packed

        with patch("deepopt.opt.nn.autograd.native_linear.quantize_pack_activation", side_effect=record):
            projected = doF.do_linear(x, weight, plan=plan)
        with record_attention_packs() as attention_payloads:
            query = projected.reshape(1, 2, 3, 8)
            output = doF.do_scaled_dot_product_attention(query, query, query, plan=plan)
            output.sum().backward()
        self.assertTrue(linear_payloads)
        self.assertEqual({packed.bits for packed in linear_payloads}, {1})
        self.assertTrue(attention_payloads)
        self.assertEqual({packed.bits for packed in attention_payloads}, {4})

    def test_multihead_adapter_supports_both_weight_modes_and_shared_qkv_storage(self):
        from deepopt.opt.nn.attention import DOMultiheadAttention
        from tests.opt.fp8_support import fp8_plan

        for need_weights in (False, True):
            for unbatched in (False, True):
                with self.subTest(need_weights=need_weights, unbatched=unbatched):
                    source = torch.nn.MultiheadAttention(8, 2, batch_first=True)
                    plan = fp8_plan()
                    adapter = DOMultiheadAttention(source, plan=plan)
                    query = torch.randn((3, 8) if unbatched else (2, 3, 8), requires_grad=True)
                    import deepopt
                    with deepopt.autocast("cpu", enabled=False, fp8=False, activation_compression=False):
                        expected, expected_weights = adapter(query, query, query, need_weights=need_weights)
                    with record_attention_packs() as records:
                        with deepopt.autocast(
                            "cpu", enabled=False, fp8=False,
                            activation_compression={"enabled": True, "bits": 2, "attention_bits": 1},
                        ):
                            actual, weights = adapter(query, query, query, need_weights=need_weights)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if need_weights:
                        torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
                    else:
                        self.assertIsNone(weights)
                    self.assertTrue(records)
                    self.assertEqual({packed.bits for packed in records}, {1})
                    actual.sum().backward()
                    self.assertTrue(torch.isfinite(query.grad).all())

    def test_multihead_cross_attention_noncontiguous_inputs_and_weight_gradients(self):
        import deepopt
        from deepopt.opt.nn.attention import DOMultiheadAttention
        from tests.opt.fp8_support import fp8_plan

        for need_weights in (False, True):
            with self.subTest(need_weights=need_weights):
                source = torch.nn.MultiheadAttention(8, 2, kdim=6, vdim=10, batch_first=True)
                adapter = DOMultiheadAttention(source, plan=fp8_plan())
                query = torch.randn(2, 8, 3).transpose(1, 2).requires_grad_()
                key = torch.randn(2, 6, 5).transpose(1, 2).requires_grad_()
                value = torch.randn(2, 10, 5).transpose(1, 2).requires_grad_()
                mask = torch.zeros(3, 5, dtype=torch.bool)
                mask[:, -1] = True
                with deepopt.autocast("cpu", enabled=False, fp8=False, activation_compression=False):
                    expected, expected_weights = adapter(query, key, value, need_weights=need_weights,
                                                         attn_mask=mask)
                with record_attention_packs() as records:
                    with deepopt.autocast(
                        "cpu", enabled=False, fp8=False,
                        activation_compression={"enabled": True, "bits": 1},
                    ):
                        actual, weights = adapter(query, key, value, need_weights=need_weights,
                                                  attn_mask=mask)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                loss = actual.square().sum()
                if need_weights:
                    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
                    loss = loss + weights.square().sum()
                self.assertTrue(records)
                self.assertEqual({packed.bits for packed in records}, {4})
                loss.backward()
                for tensor in (query, key, value, *adapter.parameters()):
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main()
