"""Faithful SiLU input recomputation is explicit and keeps compression policy."""

import unittest

import torch

from deepopt.opt.backends._shared.activation_codec import activation_codec_scope
from deepopt.opt.nn import functional as doF
from deepopt.opt.policy import activation_plan


class SiLUSavedStateTests(unittest.TestCase):
    def test_storage_inplace_snapshot_preserves_destination_and_original_derivative(self):
        from deepopt._bf16_policy import bf16_policy_scope
        from deepopt.opt.backends.reference.activation.math import evaluate
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation

        for packed in (False, True):
            for layout in ("contiguous", "channels_last", "strided", "empty"):
                with self.subTest(packed=packed, layout=layout):
                    raw = torch.randn((2, 8, 8, 8) if layout == "channels_last" else (0, 1024) if layout == "empty" else (4, 2048), dtype=torch.bfloat16)
                    if layout == "channels_last":
                        raw = raw.to(memory_format=torch.channels_last)
                    leaf = raw.detach().requires_grad_()
                    base = leaf + 0
                    x = base[:, ::2] if layout == "strided" else base
                    original = x.detach().clone()
                    saved = []
                    plan = activation_plan(False, activation_compression=dict(enabled=packed, bits=2, group_size=1024))
                    with bf16_policy_scope("bf16_storage", "cpu"), torch.autograd.graph.saved_tensors_hooks(
                            lambda value: saved.append(value) or value, lambda value: value):
                        y = doF.silu(x, inplace=True, plan=plan)
                    self.assertIs(y, x)
                    self.assertEqual(len(saved), 3 if packed else 1)
                    self.assertEqual(saved[0].dtype, torch.int32 if packed else torch.bfloat16)
                    self.assertNotEqual(saved[0].untyped_storage()._cdata, x.untyped_storage()._cdata)
                    values, coefficient = evaluate(original.float(), "silu", "none")
                    torch.testing.assert_close(y, values.bfloat16(), rtol=0, atol=0)
                    if packed and original.numel():
                        coefficient = dequantize_unpack_activation(quantize_pack_activation(
                            coefficient.reshape(-1, coefficient.shape[-1]), 1024, bits=2), dtype=torch.float32).reshape(original.shape)
                    gradient = torch.randn_like(y)
                    with torch.no_grad():
                        y.fill_(0)
                    y.backward(gradient, retain_graph=True)
                    actual = leaf.grad[:, ::2] if layout == "strided" else leaf.grad
                    torch.testing.assert_close(actual, (gradient.float() * coefficient).bfloat16(), rtol=0, atol=0)
                    leaf.grad = None
                    y.backward(gradient)
                    actual = leaf.grad[:, ::2] if layout == "strided" else leaf.grad
                    torch.testing.assert_close(actual, (gradient.float() * coefficient).bfloat16(), rtol=0, atol=0)

    def test_storage_inplace_leaf_rejection_precedes_mutation(self):
        from deepopt._bf16_policy import bf16_policy_scope

        for view in (False, True):
            leaf = torch.randn(4, 1024, dtype=torch.bfloat16, requires_grad=True)
            original = leaf.detach().clone()
            x = leaf[:, :512] if view else leaf
            with bf16_policy_scope("bf16_storage", "cpu"), self.assertRaisesRegex(RuntimeError, "leaf Variable"):
                doF.silu(x, inplace=True, plan=activation_plan(False, activation_compression=False))
            torch.testing.assert_close(leaf, original, rtol=0, atol=0)

    def test_recompute_preserves_forward_and_backward_and_halves_logical_saved_bytes(self):
        for channels_last in (False, True):
            raw = torch.randn(2, 23, 7, 7, dtype=torch.bfloat16)
            if channels_last:
                raw = raw.contiguous(memory_format=torch.channels_last)
            dy = torch.randn_like(raw)
            outputs, saved_bytes = [], []
            for state in ("derivative", "recompute"):
                x = raw.detach().clone().requires_grad_()
                saved, info = [], {}
                with activation_codec_scope(silu_state=state), torch.autograd.graph.saved_tensors_hooks(
                        lambda value: saved.append(value) or value, lambda value: value):
                    y = doF.silu(x, plan=activation_plan(False, activation_compression=False),
                                 _execution_info=info)
                # Backward occurs after leaving the selector scope.
                y.backward(dy)
                outputs.append((y.detach(), x.grad))
                saved_bytes.append(sum(t.numel() * t.element_size() for t in saved))
                self.assertEqual(info["saved_state"], "input_recompute" if state == "recompute" else "derivative")
                self.assertEqual(saved[0].dtype, torch.bfloat16 if state == "recompute" else torch.float32)
            self.assertEqual(saved_bytes[0], 2 * saved_bytes[1])
            for actual, expected in zip(*outputs):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_recompute_request_keeps_packed_derivatives_and_inplace_semantics(self):
        for packed, inplace in ((True, False), (False, True)):
            raw = torch.randn(3, 37, dtype=torch.bfloat16, requires_grad=True)
            x = raw + 0 if inplace else raw
            saved, info = [], {}
            with activation_codec_scope(silu_state="recompute"), torch.autograd.graph.saved_tensors_hooks(
                    lambda value: saved.append(value) or value, lambda value: value):
                y = doF.silu(x, inplace=inplace,
                    plan=activation_plan(False, activation_compression={"enabled": packed, "bits": 2, "group_size": 32}),
                    _execution_info=info)
            self.assertEqual(info["saved_state"], "derivative")
            self.assertEqual(saved[0].dtype, torch.int32 if packed else torch.float32)
            y.sum().backward()
            self.assertTrue(torch.isfinite(raw.grad).all())


if __name__ == "__main__":
    unittest.main()
