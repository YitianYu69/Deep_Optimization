"""Pooling preparation, numerical-state ownership and exact training resume."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

import deepopt
from deepopt.config import ConfigError, fp8_config_from_dict
from deepopt.engine.checkpointing.checkpoint import save_checkpoint, read_checkpoint, restore_checkpoint
from deepopt.opt.nn import functional as D
from deepopt.opt.policy import prepare_fp8_model, pooling_plan
from deepopt.opt.precision import precision_report


COMPRESSION = dict(enabled=True, bits=2, group_size=17)


class FunctionalPools(nn.Module):
    def __init__(self, direct=False):
        super().__init__()
        self.before, self.after = nn.Conv2d(3, 4, 1), nn.Conv2d(4, 2, 1)
        self.direct = direct

    def forward(self, x):
        value = self.before(x)
        if self.direct:
            value, indices = D.do_max_pool2d_with_indices(value, 2, return_indices=False)
            value = D.do_max_unpool2d(value, indices, 2, output_size=x.shape[-2:])
        else:
            value, indices = F.max_pool2d(value, 2, return_indices=True)
            value = F.max_unpool2d(value, indices, 2, output_size=x.shape[-2:])
        return self.after(value)


class PoolingIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9153)
        # Native unpool reference execution does not support deterministic
        # mode. Keep these scenarios independent of earlier Trainer tests.
        deterministic = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic, warn_only=warn_only)
        torch.use_deterministic_algorithms(False)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_functional_tuple_indices_and_direct_controls_in_both_trace_modes(self):
        for mode in ("modules", "full"):
            for direct in (False, True):
                cfg = deepopt.FP8Config(enabled=True, trace_mode=mode, backward_precision="auto")
                model, _ = prepare_fp8_model(FunctionalPools(direct), cfg, activation_compression=COMPRESSION)
                model(torch.randn(2, 3, 6, 6)).square().mean().backward()
                report = precision_report(model)
                self.assertFalse(report["fallbacks"], report)
                self.assertEqual(report["analyses"], 1)
                self.assertTrue(any(edge["precision"] == "fp8" and "pool" in edge["producer"]
                                    for edge in report["edges"] if edge["mode"] == "train"), report)
                self.assertEqual(sum(value is not None for name, value in report["backward_execution"].items()
                                     if name.startswith("train_") and "pool" in name), 2, report)
        cfg = deepopt.FP8Config(enabled=True, trace_mode="full")
        class Explicit(nn.Module):
            def forward(self, x):
                y, _ = D.do_max_pool2d_with_indices(x, 2, plan=pooling_plan(cfg, activation_compression=True), output_dtype=torch.bfloat16)
                return y
        model, _ = prepare_fp8_model(Explicit(), cfg, activation_compression=True)
        self.assertEqual(model(torch.randn(2, 3, 6, 6)).dtype, torch.bfloat16)

    def test_delayed_pooling_recompute_accumulation_and_exact_checkpoint_resume(self):
        for fractional in (False, True):
            cfg = deepopt.FP8Config(enabled=True, backward_precision="auto", scaling_recipe="delayed", grad_dtype="e5m2")
            def build():
                pool = nn.FractionalMaxPool2d(2, output_size=(3, 3)) if fractional else nn.LPPool2d(2, 2)
                model = nn.Sequential(nn.Conv2d(3, 4, 1), pool, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(4, 2))
                return deepopt.Trainer(model, torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9), device="cpu",
                                       loss_fn=F.mse_loss, fp8=cfg, checks=False,
                                       config={"runtime": {"activation_compression": COMPRESSION}})
            trainer = build()
            for _ in range(2):
                loss = trainer.runtime.compute(torch.randn(2, 3, 6, 6), lambda x: deepopt.fp8_checkpoint(trainer.model, x).square().mean())
                trainer.runtime.backward(loss)
                self.assertEqual(trainer.model[1].fp8_scaling.updates[0].item(), 0)
            trainer.runtime.step()
            self.assertEqual(trainer.model[1].fp8_scaling.updates[0].item(), 1)
            # Exercise Trainer checkpoint/progress ownership separately from the
            # preceding low-level accumulation boundary.
            trainer = build()
            batch = [(torch.randn(2, 3, 6, 6), torch.randn(2, 2))]
            trainer.fit(batch, epochs=1, progress_bar=False)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "state.pt"
                save_checkpoint(path, trainer, generators={}, history=[{}])
                restored = build()
                restore_checkpoint(read_checkpoint(path), restored, generators={}, batch_size_mode="manual")
            state = torch.get_rng_state().clone()
            trainer.fit(batch, epochs=2, progress_bar=False)
            torch.set_rng_state(state)
            restored.fit(batch, epochs=2, progress_bar=False)
            for a, b in zip(trainer.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_preparation_preserves_fractional_rng_and_explicit_sample_buffers(self):
        for mode in ("modules", "full"):
            for fixed in (False, True):
                samples = torch.rand(2, 3, 2) if fixed else None
                source = nn.Sequential(nn.FractionalMaxPool2d(2, output_size=(3, 3), _random_samples=samples), nn.AdaptiveAvgPool2d(1))
                state = torch.get_rng_state().clone()
                model, _ = prepare_fp8_model(source, deepopt.FP8Config(enabled=True, trace_mode=mode), activation_compression=True)
                torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
                self.assertIs(model[0]._random_samples if mode == "modules" else model.get_submodule("0")._random_samples, samples)
                model(torch.randn(2, 3, 6, 6, requires_grad=True)).sum().backward()

    def test_parameterless_delayed_state_device_initialization_in_both_backward_modes(self):
        for mode in ("native", "auto"):
            cfg = deepopt.FP8Config(enabled=True, scaling_recipe="delayed", backward_precision=mode)
            model, _ = prepare_fp8_model(nn.Sequential(nn.AvgPool1d(2), nn.AvgPool1d(2), nn.ReLU()), cfg, activation_compression=True)
            self.assertTrue(model._deepopt_gradient_dynamic_device)
            with deepopt.autocast("cpu", enabled=False, fp8=cfg, activation_compression=True, check_finite=False):
                model(torch.randn(2, 3, 8, dtype=torch.bfloat16, requires_grad=True)).sum().backward()
            self.assertFalse(model._deepopt_gradient_dynamic_device)
            self.assertEqual(precision_report(model)["analyses"], 1)

    def test_interception_override_strictness_and_native_bf16_compression(self):
        self.assertIsNone(fp8_config_from_dict({}).intercept_pooling)
        for value in (0, 1, "true", []):
            with self.assertRaisesRegex(ConfigError, "intercept_pooling"):
                fp8_config_from_dict(dict(intercept_pooling=value))
        x = torch.randn(2, 3, 6, 6, dtype=torch.bfloat16, requires_grad=True)
        with deepopt.autocast("cpu", enabled=False, fp8=False, activation_compression=True):
            y = F.avg_pool2d(x, 2)
        self.assertTrue(all(t is None for t in y.grad_fn.saved_tensors))
        y.sum().backward()
        model, _ = prepare_fp8_model(nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(3, 3, 1)), True, activation_compression=True)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            prepare_fp8_model(model, deepopt.FP8Config(enabled=True, intercept_pooling=False))

    def test_tuple_value_fanout_retains_native_partial_gradient_boundaries(self):
        class Branches(nn.Module):
            def __init__(self):
                super().__init__()
                self.before = nn.Conv2d(3, 4, 1)
                self.pool = nn.MaxPool2d(2, return_indices=True)
                self.a, self.b = nn.Conv2d(4, 2, 1), nn.Conv2d(4, 2, 1)

            def forward(self, x):
                value, _ = self.pool(self.before(x))
                return self.a(value) + self.b(value)
        model, _ = prepare_fp8_model(Branches(), deepopt.FP8Config(enabled=True, backward_precision="auto"), activation_compression=True)
        model(torch.randn(2, 3, 6, 6)).sum().backward()
        edges = [edge for edge in precision_report(model)["backward_edges"] if edge["mode"] == "train" and edge["producer"] in ("a", "b")]
        self.assertEqual(len(edges), 2)
        self.assertTrue(all(edge["precision"] == "native" and "fan-out" in edge["reason"] for edge in edges), edges)

    def test_meta_registration_fallback_and_optional_native_lp3d_wrapper(self):
        from deepopt.opt.nn.pooling.geometry import _output_shape

        _output_shape.cache_clear()
        self.addCleanup(_output_shape.cache_clear)
        native, probes = F.avg_pool2d, []
        def no_meta(x, *args, **kwargs):
            if x.device.type == "meta":
                raise NotImplementedError("simulated older Meta registration")
            if not x.numel():
                probes.append(tuple(x.shape))
            return native(x, *args, **kwargs)
        x = torch.randn(2, 3, 6, 6, requires_grad=True)
        with patch.object(F, "avg_pool2d", no_meta):
            y = D.do_avg_pool2d(x, 2, plan=pooling_plan(False, activation_compression=True))
        torch.testing.assert_close(y, native(x, 2))
        self.assertEqual(probes, [(0, 3, 6, 6)])
        x = torch.rand(1, 2, 4, 4, 4, requires_grad=True)
        _output_shape.cache_clear()
        expected = F.lp_pool3d(x, 2, 2)
        with patch.object(F, "lp_pool3d", None):
            actual = D.do_lp_pool3d(x, 2, 2, plan=pooling_plan(True, activation_compression=False))
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()


if __name__ == "__main__":
    unittest.main()
