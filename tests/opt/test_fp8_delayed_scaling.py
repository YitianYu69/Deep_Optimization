"""Delayed numerical state, invocation ownership, and CPU reference behavior."""

from copy import deepcopy
from dataclasses import asdict
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from deepopt import FP8Config, DelayedScalingConfig, fp8_scaling_scope, fp8_checkpoint, autocast, TrainingRuntime
from deepopt._config_resolution import ResolvedConfig
from deepopt.config import ConfigError, fp8_config_from_dict
from deepopt.opt.policy import fp8_plan, prepare_fp8_model
from deepopt.opt.do_nn.functional import do_linear
from deepopt.opt.do_nn.AOTAutograd_Func import linear_autograd_func
from deepopt.opt.scaling_strategy.delayed import DelayedScalingState, delayed_states


def settings(**kwargs):
    return FP8Config(enabled=True, scaling_recipe="delayed", delayed_scaling=DelayedScalingConfig(history_length=3), **kwargs)


def prepared(inputs=5, outputs=3):
    return prepare_fp8_model(nn.Linear(inputs, outputs), settings())[0]


class DelayedScalingTests(unittest.TestCase):
    def test_passive_cuda_graph_options_allow_eager_delayed_runtime(self):
        from deepopt import CUDAGraphCore

        for graph_options in ({}, {"warmup": 3, "autograd": True}):
            source = nn.Linear(5, 3)
            runtime = TrainingRuntime(source, torch.optim.SGD(source.parameters(), lr=.001), fp8=settings(),
                                      config=ResolvedConfig({"runtime": {"cuda_graph": graph_options}}), checks=False)
            loss = runtime.compute(torch.ones(7, 5), lambda x: runtime.model(x).sum())
            runtime.backward(loss)
            runtime.step()
            self.assertEqual(runtime.model.fp8_scaling.updates.tolist(), [1, 1, 1])
            with self.assertRaisesRegex(ValueError, "eager preprocessing"):
                CUDAGraphCore(device="cuda", config=ResolvedConfig({})).capture_model(runtime.model, (torch.ones(7, 5),))

    def test_warmed_runtime_reuses_state_discovery_and_validated_recipe(self):
        from deepopt.opt.do_nn import functional

        source = nn.Linear(5, 3)
        runtime = TrainingRuntime(source, torch.optim.SGD(source.parameters(), lr=.001), fp8=settings(), checks=False)
        runtime.compute(torch.ones(7, 5), lambda x: runtime.model(x).sum())
        runtime.zero_grad()
        with patch("deepopt.opt.scaling_strategy.delayed.delayed_states", side_effect=AssertionError("rescanned model")), \
             patch.object(functional, "asdict", side_effect=AssertionError("rebuilt recipe dictionary")):
            loss = runtime.compute(torch.ones(7, 5), lambda x: runtime.model(x).sum())
            runtime.backward(loss)
            runtime.step()

    def test_nested_activation_checkpoint_is_rejected_before_history_changes(self):
        model = prepared()
        with self.assertRaisesRegex(ValueError, "Nested fp8_checkpoint"):
            fp8_checkpoint(lambda x: fp8_checkpoint(model, x), torch.randn(7, 5, requires_grad=True))
        self.assertEqual(model.fp8_scaling.updates.tolist(), [0, 0, 0])

    def test_observation_compaction_preserves_one_history_update(self):
        state = prepared().fp8_scaling
        with fp8_scaling_scope():
            for index in range(600):
                state.quantize(torch.full((1, 1), float(index % 9 + 1)), "activation")
            self.assertEqual(state.updates.tolist(), [0, 0, 0])
        self.assertEqual(state.updates.tolist(), [1, 0, 0])
        self.assertEqual(state.history[0, 0].item(), 9.)

    def test_ema_evaluation_preserves_history_and_outstanding_autograd(self):
        from deepopt.engine.training.ema import ExponentialMovingAverage

        model = prepared()
        ema = ExponentialMovingAverage(model, .9)
        x = torch.randn(7, 5, requires_grad=True)
        value = model(x)
        history = model.fp8_scaling.history.clone()
        model.eval()
        with ema.average_parameters(model), torch.no_grad():
            model(x)
        torch.testing.assert_close(model.fp8_scaling.history, history, rtol=0, atol=0)
        model.train()
        value.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_abandoned_window_reobserves_cached_weights_without_advancing_history(self):
        original = nn.Linear(5, 3)
        runtime = TrainingRuntime(original, torch.optim.SGD(original.parameters(), lr=.001), fp8=settings(), checks=False)
        runtime.compute(torch.ones(7, 5), lambda batch: runtime.model(batch))
        runtime.zero_grad()
        state = runtime.model.fp8_scaling
        self.assertEqual(state.updates.tolist(), [0, 0, 0])
        loss = runtime.compute(torch.ones(7, 5), lambda batch: runtime.model(batch).sum())
        runtime.backward(loss)
        runtime.step()
        self.assertEqual(state.updates.tolist(), [1, 1, 1])

    def test_fake_cuda_layout_probe_needs_no_gpu_kernel_or_scalar_read(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode(), patch.dict("sys.modules", {"deepopt.opt.backends.triton_delayed": None}):
            state = DelayedScalingState({}, dict.fromkeys(("activation", "weight", "grad"), torch.float8_e4m3fn), device="cuda")
            normal, transposed, bias = state.quantize(torch.empty(65, 129, device="cuda"), "grad",
                                                      transposed=True, bias=True, check_finite=False)
        self.assertEqual(normal.data.stride(), (144, 1))
        self.assertEqual(transposed.data.stride(), (80, 1))
        self.assertEqual(bias.dtype, torch.float32)

    def test_delayed_attention_keeps_separate_projection_state_and_parameter_identity(self):
        source = nn.MultiheadAttention(8, 2, batch_first=True)
        parameters = tuple(source.parameters())
        model, _ = prepare_fp8_model(source, settings())
        self.assertEqual(tuple(model.parameters()), parameters)
        query = torch.randn(2, 3, 8, requires_grad=True)
        key = torch.randn(2, 4, 8, requires_grad=True)
        with fp8_scaling_scope():
            output, _ = model(query, key, key)
            output.sum().backward()
        for role in ("q", "k", "v", "out"):
            self.assertEqual(model._projection_scaling[role].updates.tolist(), [1, 1, 1])
        self.assertEqual(model._projection_scaling["qkv"].updates.tolist(), [0, 0, 0])
        self.assertTrue(torch.isfinite(query.grad).all())

    def test_runtime_freezes_weight_scales_and_reuses_weights_across_microbatches(self):
        original = nn.Linear(5, 3)
        runtime = TrainingRuntime(original, torch.optim.SGD(original.parameters(), lr=.001), fp8=settings(), checks=False)
        model = runtime.model
        state = model.fp8_scaling
        with patch.object(state, "quantize", wraps=state.quantize) as convert:
            for magnitude in (1., 3.):
                loss = runtime.compute(torch.ones(7, 5) * magnitude, lambda batch: runtime.model(batch).sum())
                runtime.backward(loss)
                self.assertEqual(state.updates.tolist(), [0, 0, 0])
                torch.testing.assert_close(state.scales, torch.ones(3), rtol=0, atol=0)
            self.assertEqual(len([call for call in convert.call_args_list if call.args[1] == "weight"]), 1)
            runtime.step()
        self.assertEqual(state.updates.tolist(), [1, 1, 1])
        torch.testing.assert_close(state.scales[0], torch.tensor(448 / 3))
        loss = runtime.compute(torch.ones(7, 5), lambda batch: runtime.model(batch).sum())
        runtime.backward(loss)
        runtime.zero_grad()
        self.assertEqual(state.updates.tolist(), [1, 1, 1])

    def test_configuration_is_typed_and_rejects_incompatible_recipes(self):
        value = fp8_config_from_dict({"enabled": True, "scaling_recipe": "delayed",
                                      "delayed_scaling": {"history_length": 3, "margin": 1}})
        self.assertIsInstance(value.delayed_scaling, DelayedScalingConfig)
        self.assertEqual(fp8_config_from_dict(asdict(value)), value)
        for raw in ({"scaling_recipe": "unknown"},
                    {"scaling_recipe": "delayed", "block_shapes": {"activation": [1, 32]}},
                    {"delayed_scaling": {"history_length": 0}}, {"delayed_scaling": {"history_length": True}},
                    {"delayed_scaling": {"margin": 1.5}}, {"delayed_scaling": {"amax_algorithm": "unknown"}},
                    {"delayed_scaling": {"reduce_amax": 1}}):
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                fp8_config_from_dict(raw)

    def test_cpu_scale_history_initialization_zero_and_nonfinite_rules(self):
        for algorithm in ("max", "most_recent"):
            state = DelayedScalingState({"history_length": 3, "margin": 1, "amax_algorithm": algorithm},
                                        dict.fromkeys(("activation", "weight", "grad"), torch.float8_e4m3fn), device="cpu")
            history, previous = [0., 0., 0.], 1.
            for index, maximum in enumerate((2., 8., 4., 1., 0., 0., 0., float("inf"))):
                values = torch.tensor([[maximum, -maximum]])
                normal, transposed, _ = state.quantize(values, "activation", transposed=True, check_finite=False)
                expected = (values * previous).clamp(-448, 448).to(torch.float8_e4m3fn)
                torch.testing.assert_close(normal.data.float(), expected.float(), equal_nan=True, rtol=0, atol=0)
                torch.testing.assert_close(transposed.data.float(), expected.T.float(), equal_nan=True, rtol=0, atol=0)
                self.assertAlmostEqual(normal.scale_inv.item(), 1 / previous, places=6)
                history[index % 3] = maximum
                selected = max(history) if algorithm == "max" else maximum
                if selected > 0 and selected != float("inf"):
                    previous = 448 / selected / 2
                self.assertAlmostEqual(state.scales[0].item(), previous, places=5)
                self.assertEqual(state.updates.tolist(), [index + 1, 0, 0])

    def test_nested_boundaries_freeze_scales_and_advance_once_per_role(self):
        model = prepared()
        original = model.fp8_scaling.scales.clone()
        with fp8_scaling_scope():
            model(torch.ones(7, 5))
            with fp8_scaling_scope(), autocast("cpu", enabled=False, fp8=settings()):
                model(torch.ones(7, 5) * 3)
            torch.testing.assert_close(model.fp8_scaling.scales, original)
            self.assertEqual(model.fp8_scaling.updates.tolist(), [0, 0, 0])
            with self.assertRaisesRegex(RuntimeError, "boundary"):
                model.state_dict()
        self.assertEqual(model.fp8_scaling.updates.tolist(), [1, 1, 0])
        torch.testing.assert_close(model.fp8_scaling.scales[0], torch.tensor(448 / 3))

    def test_weight_cache_tracks_scale_generations_and_parameter_versions(self):
        model = prepared()
        state = model.fp8_scaling
        with patch.object(state, "quantize", wraps=state.quantize) as convert:
            with fp8_scaling_scope():
                model(torch.ones(7, 5))
                model(torch.ones(7, 5))
            weights = [call for call in convert.call_args_list if call.args[1] == "weight"]
            self.assertEqual(len(weights), 1)
            model(torch.ones(7, 5))
            weights = [call for call in convert.call_args_list if call.args[1] == "weight"]
            self.assertEqual(len(weights), 2)
            self.assertEqual(len(model._fp8_weight_cache._entries), 1)
            with torch.no_grad():
                model.weight.add_(1)
            model(torch.ones(7, 5))
            self.assertEqual(len([call for call in convert.call_args_list if call.args[1] == "weight"]), 3)

    def test_multiple_outstanding_forwards_keep_inverse_scales_immutable(self):
        model = prepared()
        x = torch.randn(7, 5, requires_grad=True)
        snapshots = []
        original = linear_autograd_func.gemm_nt

        def capture(a, b, **kwargs):
            snapshots.extend((value.scale_inv, value.scale_inv.clone()) for value in (a, b))
            return original(a, b, **kwargs)

        with patch.object(linear_autograd_func, "gemm_nt", new=capture):
            first, second = model(x), model(x * 4)
        for saved, expected in snapshots:
            torch.testing.assert_close(saved, expected, rtol=0, atol=0)
            self.assertNotEqual(saved.untyped_storage().data_ptr(), model.fp8_scaling.scales.untyped_storage().data_ptr())
        (first.sum() + second.sum()).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_activation_checkpoint_replays_original_scales_without_double_advancement(self):
        model, oracle = prepared(), None
        oracle = deepcopy(model)
        x = torch.randn(7, 5, requires_grad=True)
        expected_x = x.detach().clone().requires_grad_()
        result = fp8_checkpoint(model, x)
        expected = oracle(expected_x)
        result.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, expected_x.grad, rtol=0, atol=0)
        torch.testing.assert_close(model.weight.grad, oracle.weight.grad, rtol=0, atol=0)
        self.assertEqual(model.fp8_scaling.updates.tolist(), [1, 1, 1])
        with self.assertRaisesRegex(RuntimeError, "fp8_checkpoint"):
            checkpoint(model, x, use_reentrant=False).sum().backward()

    def test_checkpoint_restores_nested_current_and_dense_scope_overrides(self):
        for override in (False, FP8Config(enabled=True)):
            model, x = prepared(), torch.randn(7, 5, requires_grad=True)
            oracle = deepcopy(model)
            expected_x = x.detach().clone().requires_grad_()
            with autocast("cpu", enabled=False, fp8=override):
                value = fp8_checkpoint(model, x)
                expected = oracle(expected_x)
            value.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            torch.testing.assert_close(x.grad, expected_x.grad, rtol=0, atol=0)
            self.assertEqual(model.fp8_scaling.updates.tolist(), [0, 0, 0])

    def test_eval_and_dtype_moves_preserve_fp32_state(self):
        model = prepared()
        model(torch.ones(7, 5)).sum().backward()
        model.eval()
        before = deepcopy(model.state_dict())
        model(torch.ones(7, 5)).sum().backward()
        for name, value in model.state_dict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        scales = model.fp8_scaling.scales.clone()
        model.bfloat16()
        self.assertEqual(model.fp8_scaling.history.dtype, torch.float32)
        self.assertEqual(model.fp8_scaling.updates.dtype, torch.int64)
        torch.testing.assert_close(scales, model.fp8_scaling.scales, rtol=0, atol=0)
        self.assertFalse(model._fp8_weight_cache._entries)

    def test_state_restore_reproduces_next_update_and_clears_disposable_cache(self):
        model = prepared()
        x = torch.randn(7, 5)
        model(x).sum().backward()
        restored = prepared()
        restored.load_state_dict(deepcopy(model.state_dict()))
        self.assertFalse(restored._fp8_weight_cache._entries)
        model.zero_grad(set_to_none=True)
        actual, expected = restored(x), model(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(restored.weight.grad, model.weight.grad, rtol=0, atol=0)
        for value in (restored, model):
            value.zero_grad(set_to_none=True)
        restored(x).sum().backward()
        model(x).sum().backward()
        torch.testing.assert_close(restored.weight.grad, model.weight.grad, rtol=0, atol=0)

    def test_functional_call_sites_own_histories_without_duplicate_parameters(self):
        class Functional(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 5))

            def forward(self, x):
                return F.linear(x, self.weight) + F.linear(x * 3, self.weight)

        source = Functional()
        model, report = prepare_fp8_model(source, settings(trace_mode="full"))
        states = delayed_states(model)
        self.assertEqual(len(states), 2)
        self.assertEqual(list(model.parameters()), [source.weight])
        with fp8_scaling_scope():
            model(torch.ones(7, 5)).sum().backward()
        self.assertNotEqual(states[0].scales[0].item(), states[1].scales[0].item())
        self.assertEqual(len(report.function_calls), 2)
        with self.assertRaisesRegex(ValueError, "prepared adapter"):
            do_linear(torch.ones(7, 5), source.weight, plan=fp8_plan(settings()))

    def test_finite_policy_and_failed_boundary_do_not_advance_histories(self):
        model = prepared()
        with self.assertRaisesRegex(ValueError, "finite"):
            model(torch.full((7, 5), float("nan")))
        self.assertEqual(model.fp8_scaling.updates.tolist(), [0, 0, 0])
        with autocast("cpu", enabled=False, fp8=settings(), check_finite=False):
            model(torch.full((7, 5), float("nan")))
        self.assertEqual(model.fp8_scaling.scales[0].item(), 1.)


if __name__ == "__main__":
    unittest.main()
