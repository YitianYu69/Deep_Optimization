"""FP8 saved-activation codec semantics, autograd ownership, and scale snapshots."""

from copy import deepcopy
from itertools import product
import unittest

import torch
from torch import nn

from deepopt import FP8Config, fp8_checkpoint, fp8_scaling_scope, reset_config
from deepopt.opt.backends import gemm_nt
from deepopt.opt.nn.functional import do_linear
from deepopt.opt.policy import fp8_plan, prepare_fp8_model
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_fp8 import quantize_pack_fp8_activation
from deepopt.opt.quantization.quantizer import quantize_fp8
from deepopt.opt.quantization.tensor import QuantizedTensor


def _decoded_operand(x, blocks, group_size, dtype=torch.float8_e4m3fn):
    saved_transpose = blocks is not None and blocks[0] == 1 and blocks[1] > 1
    q = quantize_fp8(x.T if saved_transpose else x, fp8_dtype=dtype, block_shape=blocks)
    packed = quantize_pack_activation(q.data.float(), group_size=group_size, block_shape=blocks)
    transposing = not saved_transpose
    restored = dequantize_unpack_activation(packed, dtype=dtype, transpose=transposing)
    restored_blocks = (blocks[1], blocks[0]) if blocks is not None and transposing else blocks
    return QuantizedTensor(restored, q.scale_inv.T if transposing else q.scale_inv, restored_blocks)


class FP8ActivationCompressionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(871)
        reset_config()
        self.addCleanup(reset_config)

    def test_fused_api_keeps_forward_values_and_groupwise_compression_follows_fp8_rectangles(self):
        x = torch.randn(7, 11).T
        for dtype, blocks in product((torch.float8_e4m3fn, torch.float8_e5m2),
                                     (None, (1, 4), (2, 3), (32, 32))):
            q, packed = quantize_pack_fp8_activation(x, fp8_dtype=dtype, block_shape=blocks, group_size=17)
            original = quantize_fp8(x, fp8_dtype=dtype, block_shape=blocks)
            torch.testing.assert_close(q.data.float(), original.data.float(), rtol=0, atol=0)
            torch.testing.assert_close(q.scale_inv, original.scale_inv, rtol=0, atol=0)
            expected = quantize_pack_activation(original.data.float(), group_size=17, block_shape=blocks)
            torch.testing.assert_close(packed.data, expected.data, rtol=0, atol=0)
            if blocks is None:
                self.assertEqual(tuple(packed.data.shape), (5, 2))
            else:
                groups = ((x.shape[0] + blocks[0] - 1) // blocks[0]) * ((x.shape[1] + blocks[1] - 1) // blocks[1])
                self.assertEqual(tuple(packed.data.shape), (groups, (blocks[0] * blocks[1] + 15) // 16))

    def test_only_weight_gradient_uses_lossy_activation_and_matches_explicit_decoded_oracle(self):
        for blocks, dtype, size in product((None, (1, 4), (2, 3)),
                                           (torch.float32, torch.bfloat16), (17, 512)):
            with self.subTest(blocks=blocks, dtype=dtype, group_size=size):
                x = torch.randn(2, 3, 7, dtype=dtype, requires_grad=True)
                w = torch.randn(4, 7, requires_grad=True)
                b = torch.randn(4, requires_grad=True)
                grad = torch.randn(2, 3, 4, dtype=dtype)
                plan = fp8_plan(FP8Config(enabled=True, block_shapes=dict.fromkeys(("activation", "weight", "grad"), blocks)))
                baseline = do_linear(x, w, b, plan=plan)
                baseline.backward(grad)
                dx, db = x.grad.clone(), b.grad.clone()
                x.grad = w.grad = b.grad = None
                plan["fp8_training"]["activation_compression"] = {"enabled": True, "group_size": size}
                result = do_linear(x, w, b, plan=plan)
                result.backward(grad)
                torch.testing.assert_close(result, baseline, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, dx, rtol=0, atol=0)
                torch.testing.assert_close(b.grad, db, rtol=0, atol=0)
                qx = _decoded_operand(x.detach().reshape(-1, 7), blocks, size)
                qg = quantize_fp8(grad.reshape(-1, 4).T, block_shape=blocks) if blocks and blocks[0] == 1 else \
                    quantize_fp8(grad.reshape(-1, 4), block_shape=blocks).transpose(contiguous=False)
                expected = gemm_nt(qg, qx, backend="reference", out_dtype=dtype)
                torch.testing.assert_close(w.grad, expected.float(), rtol=0, atol=0)

    def test_saved_hooks_own_all_codec_tensors_and_no_dense_activation(self):
        x = torch.randn(128, 64, requires_grad=True)
        weight = torch.randn(32, 64, requires_grad=True)
        plan = fp8_plan(FP8Config(enabled=True), activation_compression={"enabled": True})
        stored, restored = [], []

        def save(value):
            stored.append(value)
            return value.clone()

        def restore(value):
            restored.append(value)
            return value

        with torch.autograd.graph.saved_tensors_hooks(save, restore):
            result = do_linear(x, weight, plan=plan)
        self.assertEqual([t.dtype for t in stored], [torch.int32, torch.float32,
                                                    torch.float8_e4m3fn, torch.float32,
                                                    torch.float32, torch.float32])
        self.assertEqual(tuple(stored[0].shape), (16, 32))
        self.assertTrue(all(tuple(t.shape) != tuple(x.shape) for t in stored))
        codec_bytes = sum(t.numel() * t.element_size() for t in (stored[0], stored[4], stored[5]))
        self.assertEqual(codec_bytes, x.numel() // 4 + 16 * 8)
        self.assertLess(codec_bytes, x.numel())
        result.sum().backward()
        self.assertEqual(len(restored), 6)

    def test_no_codec_is_saved_without_a_weight_gradient_or_without_autograd(self):
        for need_x, need_w in ((True, False), (False, False)):
            x = torch.randn(5, 7, requires_grad=need_x)
            w = torch.randn(4, 7, requires_grad=need_w)
            stored = []
            plan = fp8_plan(FP8Config(enabled=True), activation_compression=True)
            with torch.autograd.graph.saved_tensors_hooks(lambda t: stored.append(t) or t, lambda t: t):
                result = do_linear(x, w, plan=plan)
            self.assertFalse(any(t.dtype == torch.int32 for t in stored))
            if need_x:
                result.sum().backward()

    def test_delayed_compression_retains_forward_scale_and_observations(self):
        settings = FP8Config(enabled=True, scaling_recipe="delayed")
        source = nn.Linear(7, 4)
        dense, _ = prepare_fp8_model(deepcopy(source), settings)
        compressed, _ = prepare_fp8_model(deepcopy(source), settings, activation_compression=True)
        dense.fp8_scaling.scales.copy_(torch.tensor([32., 16., 8.]))
        compressed.fp8_scaling.scales.copy_(dense.fp8_scaling.scales)
        x = torch.randn(5, 7, requires_grad=True)
        xc = x.detach().clone().requires_grad_()
        grad = torch.randn(5, 4)
        with fp8_scaling_scope():
            reference = dense(x)
            result = compressed(xc)
            reference.backward(grad)
            result.backward(grad)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        torch.testing.assert_close(xc.grad, x.grad, rtol=0, atol=0)
        torch.testing.assert_close(compressed.bias.grad, dense.bias.grad, rtol=0, atol=0)
        for name in ("scales", "history", "updates", "generations"):
            torch.testing.assert_close(getattr(compressed.fp8_scaling, name), getattr(dense.fp8_scaling, name))
        dtype = torch.float8_e4m3fn
        qx = (xc.detach() * 32).clamp(-448, 448).to(dtype)
        packed = quantize_pack_activation(qx.float())
        restored = dequantize_unpack_activation(packed, dtype=dtype, transpose=True)
        qg = QuantizedTensor((grad * 8).clamp(-448, 448).to(dtype).T.contiguous(), torch.tensor([[1 / 8]]))
        expected = gemm_nt(qg, QuantizedTensor(restored, torch.tensor([[1 / 32]])), backend="reference")
        torch.testing.assert_close(compressed.weight.grad, expected, rtol=0, atol=0)

    def test_activation_checkpoint_recomputation_preserves_compressed_gradients(self):
        for recipe in ("current", "delayed"):
            settings = FP8Config(enabled=True, scaling_recipe=recipe)
            original = nn.Linear(7, 4)
            eager, _ = prepare_fp8_model(deepcopy(original), settings, activation_compression=True)
            checkpointed, _ = prepare_fp8_model(deepcopy(original), settings, activation_compression=True)
            x = torch.randn(5, 7, requires_grad=True)
            other = x.detach().clone().requires_grad_()
            with fp8_scaling_scope():
                eager(x).square().sum().backward()
                fp8_checkpoint(checkpointed, other).square().sum().backward()
            torch.testing.assert_close(other.grad, x.grad, rtol=0, atol=0)
            torch.testing.assert_close(checkpointed.weight.grad, eager.weight.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
