"""Installed-library contracts for optional normalization codec mean correction."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

import deepopt
from deepopt.config import activation_compression_config_from_dict, ConfigError
from deepopt.engine.checkpointing.checkpoint import _resume_config_for_comparison
from deepopt.engine.checkpointing.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from deepopt.opt.nn import functional as DF
from deepopt.opt.nn.normalization.geometry import NormGeometry
from deepopt.opt.nn.normalization.layer_geometry import LayerNormGeometry
from deepopt.opt.nn.normalization.recenter import corrected_xhat
from deepopt.opt.policy import normalization_plan, prepare_fp8_model
from deepopt.opt.quantization.activation import dequantize_unpack_activation, quantize_pack_activation
from deepopt.opt.quantization.activation_tensor import FP8Activation, to_fp8_activation


def compression(recenter=True, **kwargs):
    return dict(enabled=True, group_size=17, bits=2, norm_xhat_recenter=recenter, **kwargs)


def plan(recenter=True, *, fp8=False, bits=2, group=17, enabled=True):
    settings = deepopt.FP8Config(enabled=True, backend="reference") if fp8 else False
    return normalization_plan(settings, activation_compression=dict(
        enabled=enabled, group_size=group, bits=bits, norm_xhat_recenter=recenter))


class NormXhatRecenterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2571)

    def test_all_norm_domains_preserve_mean_payload_and_forward_and_use_corrected_codec_gradients(self):
        for kind in ("batch", "eval", "group", "layer"):
            for bits in (1, 2, 4, 8):
                for group in (1, 17, 512):
                    for fp8 in (False, True):
                        with self.subTest(kind=kind, bits=bits, group=group, fp8=fp8):
                            self.check_norm(kind, bits, group, fp8)

    def check_norm(self, kind, bits, group, fp8):
        source = (torch.randn(2, 4, 3, 5) * 3 + 50).bfloat16().requires_grad_()
        x = to_fp8_activation(source, block_shape=(4, 4)) if fp8 else source
        dense = x.dequantize(torch.float32).detach() if fp8 else source.detach().float()
        geometry = (LayerNormGeometry(tuple(x.shape), (3, 5)) if kind == "layer" else
                    NormGeometry("batch" if kind in ("batch", "eval") else "group", tuple(x.shape), 2,
                                 batch_stats=kind != "eval"))
        shape = (3, 5) if kind == "layer" else (4,)
        weight, bias = torch.randn(shape, requires_grad=True), torch.randn(shape, requires_grad=True)
        running_mean, running_var = torch.randn(4), torch.rand(4) + 0.5

        def call(policy, w, b):
            if kind == "layer":
                return DF.layer_norm(x, shape, w, b, plan=policy, output_dtype=torch.float8_e4m3fn if fp8 else None)
            if kind in ("batch", "eval"):
                return DF.do_batch_norm(x, running_mean.clone(), running_var.clone(), w, b, kind == "batch",
                                     plan=policy, _output_layout="channels" if fp8 else None)
            return DF.do_group_norm(x, 2, w, b, plan=policy, _output_layout="channels" if fp8 else None)

        states = []
        outputs = []
        for enabled in (False, True):
            saved = []
            with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                outputs.append(call(plan(enabled, fp8=fp8, bits=bits, group=group), weight, bias))
            states.append(saved)
        for before, after in zip(states[0][:3], states[1][:3]):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        for output in outputs:
            self.assertEqual(isinstance(output, FP8Activation), fp8)
        if fp8:
            torch.testing.assert_close(outputs[0]._data.float(), outputs[1]._data.float(), rtol=0, atol=0)
            torch.testing.assert_close(outputs[0]._scale_inv, outputs[1]._scale_inv, rtol=0, atol=0)
        else:
            torch.testing.assert_close(*outputs, rtol=0, atol=0)
        matrix = geometry.matrix(dense)
        if kind == "eval":
            mean, inverse = running_mean, torch.rsqrt(running_var + 1e-5)
        else:
            variance, mean = torch.var_mean(matrix, 1, correction=0)
            inverse = torch.rsqrt(variance + 1e-5)
        h = (matrix - mean[:, None]) * inverse[:, None]
        size = min(group, geometry.elements)
        payload = quantize_pack_activation(h, size, (1, size), bits=bits)
        decoded = dequantize_unpack_activation(payload, dtype=torch.float32)
        shift = (decoded - h).mean(1)
        torch.testing.assert_close(states[1][-1], shift, rtol=0, atol=0)
        corrected = decoded - shift[:, None]
        torch.testing.assert_close(corrected.mean(1), h.mean(1), rtol=3e-6, atol=1e-5)
        self.assertEqual(len(states[1]), len(states[0]) + 1)
        self.assertFalse(any(t is source for t in states[1]))
        dy = torch.randn_like(source)
        outputs[1].backward(dy)
        if kind == "layer":
            weighted = geometry.matrix(dy.float()) * weight.detach().reshape(1, -1)
            dw = (geometry.matrix(dy.float()) * corrected).sum(0).reshape(shape)
            db = geometry.matrix(dy.float()).sum(0).reshape(shape)
        else:
            weighted = geometry.matrix(dy.float() * weight.detach().reshape(1, 4, 1, 1))
            dw = (dy.float() * geometry.restore(corrected)).sum((0, 2, 3))
            db = dy.float().sum((0, 2, 3))
        if kind != "eval":
            weighted = weighted - weighted.mean(1, keepdim=True) - corrected * (weighted * corrected).mean(1, keepdim=True)
        dx = geometry.restore(weighted * inverse[:, None]).bfloat16()
        torch.testing.assert_close(source.grad, dx, rtol=0, atol=0)
        torch.testing.assert_close(weight.grad, dw, rtol=0, atol=0)
        torch.testing.assert_close(bias.grad, db, rtol=0, atol=0)

    def test_strict_config_and_legacy_checkpoint_comparison(self):
        config = activation_compression_config_from_dict({"enabled": True})
        self.assertFalse(config.norm_xhat_recenter)
        for invalid in (0, 1, None, "true", []):
            with self.assertRaisesRegex(ConfigError, "norm_xhat_recenter"):
                activation_compression_config_from_dict({"norm_xhat_recenter": invalid})
        legacy = {"runtime": {"activation_compression": {"enabled": True, "bits": 2}}}
        current = {"runtime": {"activation_compression": asdict(config)}}
        self.assertEqual(_resume_config_for_comparison(legacy), _resume_config_for_comparison(current))
        current["runtime"]["activation_compression"]["norm_xhat_recenter"] = True
        self.assertNotEqual(_resume_config_for_comparison(legacy), _resume_config_for_comparison(current))
        self.assertNotIn("norm_xhat_recenter", legacy["runtime"]["activation_compression"])

    def test_saved_state_weight_only_bias_only_no_grad_and_disabled_compression(self):
        for recenter in (False, True):
            for need_x, need_w, need_b in ((False, True, False), (False, False, True), (True, False, False)):
                x = torch.randn(2, 19, requires_grad=need_x)
                weight, bias = torch.ones(19, requires_grad=need_w), torch.zeros(19, requires_grad=need_b)
                saved = []
                with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                    y = DF.layer_norm(x, (19,), weight, bias, plan=plan(recenter))
                if need_w:
                    self.assertEqual(len(saved), 5 if recenter else 3)
                    if recenter:
                        self.assertEqual(tuple(saved[-2].shape), (2,))  # rstd snapshot
                elif need_x:
                    self.assertEqual(len(saved), 6 if recenter else 5)
                else:
                    self.assertEqual(saved, [])
                y.sum().backward()
        x = torch.randn(2, 19, requires_grad=True)
        saved = []
        with torch.no_grad(), torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            DF.layer_norm(x, (19,), plan=plan())
        self.assertEqual(saved, [])
        empty = torch.empty(0, 19, requires_grad=True)
        info = {}
        DF.do_layer_norm(empty, (19,), plan=plan(), _execution_info=info).sum().backward()
        self.assertEqual(info["recenter_state_bytes"], 0)
        self.assertFalse(info["norm_xhat_recenter"])
        self.assertEqual(tuple(empty.grad.shape), (0, 19))
        torch.testing.assert_close(DF.layer_norm(x, (19,), plan=plan(True, enabled=False)),
                                   DF.layer_norm(x, (19,), plan=plan(False, enabled=False)), rtol=0, atol=0)

    def test_resume_rejects_policy_change_before_loading_and_accepts_legacy_false(self):
        def trainer(recenter):
            model = nn.Sequential(nn.LayerNorm(19), nn.Linear(19, 2))
            return deepopt.Trainer(model, torch.optim.SGD(model.parameters(), lr=0.001), device="cpu",
                                   loss_fn=nn.MSELoss(), config={"runtime": {"activation_compression": compression(recenter)}})
        source = trainer(False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            save_checkpoint(path, source, generators={}, history=[])
            payload = read_checkpoint(path)
        target = trainer(True)
        before = deepcopy(target.model.state_dict())
        with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
            restore_checkpoint(payload, target, generators={}, batch_size_mode="manual")
        for name, value in target.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        payload["config"]["runtime"]["activation_compression"].pop("norm_xhat_recenter")
        legacy_target = trainer(False)
        restore_checkpoint(payload, legacy_target, generators={}, batch_size_mode="manual")
        for a, b in zip(source.model.parameters(), legacy_target.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_delayed_recomputation_accumulation_and_weight_only_report(self):
        settings = deepopt.FP8Config(enabled=True, scaling_recipe="delayed", backward_precision="auto", grad_dtype="e5m2")
        for recompute in (False, True):
            source = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.GELU(), nn.Linear(8, 3))
            runtime = deepopt.TrainingRuntime(source, torch.optim.SGD(source.parameters(), lr=0.001), fp8=settings,
                                              checks=False, config={"runtime": {"activation_compression": compression()}})
            for _ in range(2):
                loss = runtime.compute(torch.randn(3, 8), lambda x: (
                    deepopt.fp8_checkpoint(runtime.model, x) if recompute else runtime.model(x)).square().mean())
                runtime.backward(loss)
                self.assertEqual(runtime.model[1].fp8_scaling.updates[0].item(), 0)
            runtime.step()
            self.assertEqual(runtime.model[1].fp8_scaling.updates[0].item(), 1)
        module, _ = prepare_fp8_model(nn.LayerNorm(19), False, activation_compression=compression())
        module(torch.randn(3, 19)).sum().backward()
        self.assertEqual(module.execution_info["recenter_extra_rstd_bytes"], 12)

    def test_saved_shift_version_checks_and_rstd_guard(self):
        x = torch.randn(3, 19, requires_grad=True)
        y = DF.layer_norm(x, (19,), plan=plan())
        shift = y.grad_fn.saved_tensors[-1]
        shift.add_(1)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            y.sum().backward()
        value = torch.randn(5, 7)
        inverse = torch.tensor([1., 2., 0., float("inf"), float("nan")])
        shift = torch.arange(5).float() + 1
        result = corrected_xhat(value, inverse, shift)
        torch.testing.assert_close(result[:2], value[:2] - shift[:2, None])
        torch.testing.assert_close(result[2:], value[2:])

    def test_checked_execution_rejects_invalid_rstd_and_unchecked_retains_decoded_values(self):
        for kind in ("layer", "batch", "group"):
            x = torch.tensor([-1e30, 1e30]).repeat(8).reshape(2, 4, 1, 2).requires_grad_()
            def call():
                if kind == "layer":
                    return DF.layer_norm(x, (1, 2), plan=plan())
                if kind == "batch":
                    return DF.batch_norm(x, None, None, training=True, plan=plan())
                return DF.group_norm(x, 2, plan=plan())
            with deepopt.autocast("cpu", enabled=False, fp8=False, check_finite=True):
                with self.assertRaisesRegex(ValueError, "finite"):
                    call()
            with deepopt.autocast("cpu", enabled=False, fp8=False, check_finite=False):
                y = call()
                y.sum().backward()
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_global_scope_prepared_adapters_reports_fake_and_compile(self):
        for module, x in ((nn.LayerNorm(19), torch.randn(3, 19)),
                          (nn.GroupNorm(2, 4), torch.randn(2, 4, 3, 5)),
                          (nn.BatchNorm2d(4), torch.randn(2, 4, 3, 5))):
            x = x.bfloat16().requires_grad_()
            prepared, _ = prepare_fp8_model(deepcopy(module), False, activation_compression=compression())
            y = prepared(x)
            y.sum().backward()
            info = prepared.execution_info
            domains = 3 if isinstance(module, nn.LayerNorm) else 4
            self.assertEqual(info["recenter_state_bytes"], domains * 4)
            self.assertEqual(info["recenter_scratch_bytes"], 0)
            self.assertEqual(info["recenter_launches"], 0)
            expected_dx = x.grad.clone()
            x.grad = None
            with deepopt.autocast("cpu", enabled=False, fp8=False, activation_compression=compression()):
                scoped = module(x)
            scoped.sum().backward()
            torch.testing.assert_close(y, scoped, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)
        with FakeTensorMode():
            x = torch.randn(3, 19, requires_grad=True)
            DF.layer_norm(x, (19,), plan=plan()).sum().backward()
        x = torch.randn(3, 19, requires_grad=True)
        compiled = torch.compile(lambda t: DF.layer_norm(t, (19,), plan=plan()), backend="eager")
        torch.testing.assert_close(compiled(x), DF.layer_norm(x, (19,), plan=plan()))


if __name__ == "__main__":
    unittest.main()
