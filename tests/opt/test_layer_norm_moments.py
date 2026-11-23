"""Mandatory compressed LayerNorm moment restoration and gradient consumers."""

import unittest
from copy import deepcopy
from pathlib import Path
import tempfile

import torch
from torch import nn
import deepopt
from deepopt.engine.checkpointing.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint

from deepopt.opt.nn import functional as DF
from deepopt.opt.policy import normalization_plan, prepare_fp8_model
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation


def restored_xhat(h, group, bits):
    """Independent oracle: zero mean and the original forward second moment."""
    group = min(group, h.shape[1])
    payload = quantize_pack_activation(h, group, (1, group), bits=bits)
    decoded = dequantize_unpack_activation(payload, dtype=torch.float32)
    shift = decoded.mean(1)
    centered = decoded - shift[:, None]
    variance = centered.square().mean(1)
    gain = torch.where(variance > 0,
                       torch.sqrt(h.square().mean(1) / variance.clamp_min(1e-12)), 0.0)
    return centered * gain[:, None], shift, gain


class LayerNormMomentsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1412)

    def test_forced_for_all_bits_recenter_settings_and_partial_gradients(self):
        for bits in (1, 2, 4, 8):
            for recenter in (False, True):
                for need_x, need_w in ((True, True), (True, False), (False, True)):
                    with self.subTest(bits=bits, recenter=recenter, dx=need_x, dw=need_w):
                        x = (torch.randn(3, 65) * .02 + .1).requires_grad_(need_x)
                        w = torch.randn(65, requires_grad=need_w)
                        b = torch.randn(65, requires_grad=True)
                        dy = torch.randn_like(x)
                        policy = normalization_plan(False, activation_compression={
                            "enabled": True, "bits": bits, "group_size": 17,
                            "norm_xhat_recenter": recenter})
                        saved, info = [], {}
                        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                            y = DF.layer_norm(x, (65,), w, b, eps=.003, plan=policy, _execution_info=info)
                        variance, mean = torch.var_mean(x.detach(), 1, correction=0)
                        rstd = torch.rsqrt(variance + .003)
                        h = (x.detach() - mean[:, None]) * rstd[:, None]
                        corrected, shift, gain = restored_xhat(h, 17, bits)
                        torch.testing.assert_close(saved[-2], shift, rtol=0, atol=0)
                        torch.testing.assert_close(saved[-1], gain, rtol=0, atol=0)
                        self.assertTrue(info["layer_norm_xhat_moment_correction"])
                        self.assertEqual(info["moment_state_bytes"], 3 * 4)
                        self.assertFalse(any(t is x for t in saved))
                        y.backward(dy)
                        if need_x:
                            u = dy * w.detach()
                            expected = (u - u.mean(1, keepdim=True)
                                        - corrected * (u * corrected).mean(1, keepdim=True)) * rstd[:, None]
                            torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
                        if need_w:
                            torch.testing.assert_close(w.grad, (dy * corrected).sum(0), rtol=0, atol=0)
                        torch.testing.assert_close(b.grad, dy.sum(0), rtol=0, atol=0)
                        torch.testing.assert_close(y, torch.nn.functional.layer_norm(
                            x.detach(), (65,), w.detach(), b.detach(), .003), rtol=2e-6, atol=2e-6)

    def test_centered_second_moment_matches_original_on_vit_rows(self):
        x = torch.randn(32, 768, requires_grad=True)
        model, _ = prepare_fp8_model(nn.LayerNorm(768, eps=1e-6), False,
            activation_compression={"enabled": True, "bits": 2, "group_size": 1024,
                                    "norm_xhat_recenter": False})
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            model(x)
        variance, mean = torch.var_mean(x.detach(), 1, correction=0)
        h = (x.detach() - mean[:, None]) * torch.rsqrt(variance + 1e-6)[:, None]
        z, _, _ = restored_xhat(h, 768, 2)
        torch.testing.assert_close(z.mean(1), torch.zeros(32), rtol=0, atol=1e-6)
        torch.testing.assert_close(z.square().mean(1), h.square().mean(1), rtol=2e-6, atol=2e-6)
        self.assertEqual(len(saved), 7)
        torch.testing.assert_close(saved[-1], restored_xhat(h, 768, 2)[2], rtol=0, atol=0)

    def test_constant_singleton_bias_only_empty_and_no_grad(self):
        policy = normalization_plan(False, activation_compression={"enabled": True, "bits": 2})
        for shape in ((3, 1), (3, 7), (0, 7)):
            x = torch.full(shape, 4., requires_grad=True)
            w = torch.ones(shape[-1], requires_grad=True)
            info = {}
            DF.layer_norm(x, (shape[-1],), w, plan=policy, _execution_info=info).sum().backward()
            torch.testing.assert_close(x.grad, torch.zeros_like(x))
            torch.testing.assert_close(w.grad, torch.zeros_like(w))
            self.assertEqual(info["moment_state_bytes"], 4 * shape[0])
        for enabled in (True, False):
            saved, info = [], {}
            b = torch.zeros(7, requires_grad=True)
            with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                with torch.set_grad_enabled(enabled):
                    y = DF.layer_norm(torch.randn(3, 7), (7,), bias=b, plan=policy, _execution_info=info)
            self.assertEqual(saved, [])
            self.assertFalse(info["layer_norm_xhat_moment_correction"])
            if enabled:
                y.sum().backward()
                torch.testing.assert_close(b.grad, torch.full_like(b, 3.))

    def test_old_compression_recipe_is_rejected_before_loading(self):
        def trainer():
            model = nn.Sequential(nn.LayerNorm(7), nn.Linear(7, 2))
            return deepopt.Trainer(model, torch.optim.SGD(model.parameters(), lr=.001),
                device="cpu", loss_fn=nn.MSELoss(), config={"runtime": {"activation_compression": {"enabled": True}}})
        source, target = trainer(), trainer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_checkpoint(path, source, generators={}, history=[])
            payload = read_checkpoint(path)
        restore_checkpoint(payload, target, generators={}, batch_size_mode="manual")
        before = deepcopy(target.model.state_dict())
        payload["trainer_settings"].pop("activation_compression_recipe")
        with self.assertRaisesRegex(ValueError, "Activation compression numerical recipe"):
            restore_checkpoint(payload, target, generators={}, batch_size_mode="manual")
        for name, value in target.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
