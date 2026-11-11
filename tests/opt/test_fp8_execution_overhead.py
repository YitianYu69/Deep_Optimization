"""Preserve FP8 semantics while removing repeated host and inference work."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import pickle
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode

from deepopt import FP8Config, TrainingRuntime, autocast
from deepopt.opt import _autocast
from deepopt.opt._execution import finite_checks_enabled
from deepopt.opt.nn import functional
from deepopt.opt.nn.autograd import linear as impl
from deepopt.opt.nn.linear import DOLinear
from deepopt.opt.policy import fp8_plan
from deepopt.opt.quantization.quantizer import quantize_fp8


def rowwise_plan():
    return fp8_plan(FP8Config(enabled=True, grad_dtype="e5m2", block_shapes={
        "activation": (1, 32), "weight": (32, 32), "grad": (1, 32)}))


class FP8ExecutionOverheadTests(unittest.TestCase):
    def test_device_metadata_is_cached_per_explicit_device_without_tensor_storage(self):
        from types import SimpleNamespace
        from deepopt.opt.backends._shared.cuda import device_metadata

        device_metadata.cache_clear()
        self.addCleanup(device_metadata.cache_clear)
        properties = [SimpleNamespace(major=9, minor=0, multi_processor_count=132),
                      SimpleNamespace(major=10, minor=0, multi_processor_count=148)]
        with patch.object(torch.cuda, "get_device_properties", side_effect=properties) as query:
            for _ in range(3):
                self.assertEqual(device_metadata(0), (90, 132))
                self.assertEqual(device_metadata(1), (100, 148))
        self.assertEqual(query.call_count, 2)

    def setUp(self):
        torch.manual_seed(1401)

    def test_no_grad_and_inference_mode_skip_backward_only_transpose(self):
        x = torch.randn(37, 65, requires_grad=True)
        weight = torch.randn(10, 65, requires_grad=True)
        expected = functional.do_linear(x, weight, plan=rowwise_plan()).detach()
        for scope in (torch.no_grad, torch.inference_mode):
            saved = []
            with scope(), torch.autograd.graph.saved_tensors_hooks(saved.append, lambda value: value), \
                 patch.object(impl, "quantize_fp8", wraps=impl.quantize_fp8) as quantize:
                actual = functional.do_linear(x, weight, plan=rowwise_plan())
            self.assertFalse(actual.requires_grad)
            self.assertEqual([tuple(call.args[0].shape) for call in quantize.call_args_list], [(37, 65), (10, 65)])
            self.assertEqual(saved, [])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prevalidated_plans_are_owned_immutable_and_survive_copy_and_pickle(self):
        caller = rowwise_plan()
        plan = functional._prepare_fp8_plan(caller)
        caller["fp8_training"]["backend"] = "triton"
        self.assertEqual(plan["fp8_training"]["backend"], "reference")
        self.assertIs(functional._prepare_fp8_plan(plan), plan)
        self.assertIs(deepcopy(plan), plan)
        for value in (plan, plan["fp8_training"], plan["fp8_training"]["fp8_dtypes"], plan["block_shapes"]):
            with self.assertRaises(TypeError):
                value["unknown"] = True
            with self.assertRaises(TypeError):
                value.update({"unknown": True})
        restored = pickle.loads(pickle.dumps(plan))
        self.assertEqual(restored, plan)
        self.assertIs(functional._prepare_fp8_plan(restored), restored)

    def test_runtime_reuses_resolved_policy_and_core_scope(self):
        runtime = TrainingRuntime(nn.Sequential(nn.Linear(7, 4)), fp8=True, checks=False)
        context = runtime._precision.autocast()
        with patch.object(_autocast, "resolve_config", side_effect=AssertionError("resolved in hot path")), \
             patch.object(functional, "validate_fp8_plan", side_effect=AssertionError("validated in hot path")), \
             patch("deepopt.opt.policy.normalize_fp8_config", side_effect=AssertionError("normalized in hot path")):
            for _ in range(3):
                self.assertIs(runtime._precision.autocast(), context)
                runtime.compute(torch.ones(2, 7), lambda batch: runtime.model(batch))
        self.assertIsNone(_autocast._get_fp8_plan("cpu"))

    def test_reusable_decorator_does_not_resolve_configuration_per_call(self):
        @autocast("cpu", enabled=False, fp8=True)
        def project(x, weight):
            return F.linear(x, weight)

        with patch.object(_autocast, "resolve_config", side_effect=AssertionError("resolved again")):
            for _ in range(3):
                project(torch.ones(2, 7), torch.ones(4, 7))

    def test_adapter_operations_bypass_our_interception_but_keep_native_linear_coverage(self):
        model = DOLinear.from_linear(nn.Linear(7, 4), plan=fp8_plan(True))
        intercepted = []
        original = _autocast._FP8LinearMode.__torch_function__

        def dispatch(self, func, types, args=(), kwargs=None):
            intercepted.append(func)
            return original(self, func, types, args, kwargs)

        with autocast("cpu", enabled=False, fp8=True), \
             patch.object(_autocast._FP8LinearMode, "__torch_function__", new=dispatch):
            model(torch.ones(2, 7))
            self.assertNotIn(torch.isfinite, intercepted)
            self.assertNotIn(torch.clamp, intercepted)
            with patch.object(functional, "do_linear", wraps=functional.do_linear) as native:
                F.linear(torch.ones(2, 7), torch.ones(4, 7))
            self.assertEqual(native.call_count, 1)

    def test_prepared_only_mode_preserves_nested_enable_disable_and_functional_opt_in(self):
        model = DOLinear.from_linear(nn.Linear(7, 4), plan=fp8_plan(True))
        x = torch.randn(2, 7)
        dense = F.linear(x, model.weight, model.bias)
        quantized = model(x)
        with autocast("cpu", enabled=False, fp8={"intercept_linear": False}):
            self.assertIsNone(torch.overrides._get_current_function_mode())
            torch.testing.assert_close(model(x), quantized, rtol=0, atol=0)
            torch.testing.assert_close(F.linear(x, model.weight, model.bias), dense, rtol=0, atol=0)
            with autocast("cpu", enabled=False, fp8=True):
                torch.testing.assert_close(F.linear(x, model.weight, model.bias), quantized, rtol=0, atol=0)
                with autocast("cpu", enabled=False, fp8={"intercept_linear": False}):
                    torch.testing.assert_close(F.linear(x, model.weight, model.bias), dense, rtol=0, atol=0)
            with autocast("cpu", enabled=False, fp8=False):
                torch.testing.assert_close(model(x), dense, rtol=0, atol=0)
            torch.testing.assert_close(model(x), quantized, rtol=0, atol=0)

    def test_user_torch_function_modes_remain_active(self):
        seen = []

        class UserMode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                seen.append(func)
                return func(*args, **(kwargs or {}))

        model = DOLinear.from_linear(nn.Linear(7, 4), plan=fp8_plan(True))
        for outer, inner in ((UserMode(), autocast("cpu", enabled=False, fp8=True)),
                             (autocast("cpu", enabled=False, fp8=True), UserMode())):
            seen.clear()
            with outer, inner:
                model(torch.ones(2, 7))
            self.assertIn(torch.isfinite, seen)

    def test_fast_checks_are_saved_for_backward_and_nested_strict_scope_restores(self):
        x = torch.randn(2, 7, requires_grad=True)
        weight = torch.randn(4, 7, requires_grad=True)
        with autocast("cpu", enabled=False, fp8=True, check_finite=False):
            self.assertFalse(finite_checks_enabled())
            with patch("torch.isfinite", side_effect=AssertionError("synchronous finite check")):
                output = F.linear(x, weight)
            with autocast("cpu", enabled=False, fp8=True, check_finite=True):
                self.assertTrue(finite_checks_enabled())
                with self.assertRaisesRegex(ValueError, "finite"):
                    F.linear(torch.full((2, 7), float("nan")), weight)
            self.assertFalse(finite_checks_enabled())
        self.assertTrue(finite_checks_enabled())
        with patch("torch.isfinite", side_effect=AssertionError("backward lost fast policy")):
            output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(weight.grad)

    def test_runtime_mode_transition_controls_quantization_checks(self):
        runtime = TrainingRuntime(nn.Linear(7, 4), fp8=True, checks=True, execution_mode="max_speed")
        with torch.no_grad(), patch("torch.isfinite", side_effect=AssertionError("max_speed finite check")):
            runtime.compute(torch.ones(2, 7), lambda batch: runtime.model(batch))
        runtime.set_execution_mode("analyze")
        with runtime.autocast():
            self.assertTrue(finite_checks_enabled())
            with self.assertRaisesRegex(ValueError, "finite"):
                runtime.model(torch.full((2, 7), float("inf")))

    def test_weight_cache_reuses_and_invalidates_for_updates_loads_and_replacement(self):
        model = DOLinear.from_linear(nn.Linear(7, 4), plan=fp8_plan(True))
        x = torch.randn(2, 7)
        with torch.no_grad(), patch.object(impl, "quantize_fp8", wraps=impl.quantize_fp8) as quantize:
            first = model(x)
            second = model(x)
            self.assertEqual(quantize.call_count, 3)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            model.weight.add_(1)
            model(x)
            self.assertEqual(quantize.call_count, 5)
            fresh = deepcopy(model.state_dict())
            fresh["weight"].add_(1)
            model.load_state_dict(fresh)
            model(x)
            self.assertEqual(quantize.call_count, 7)
            model.weight = nn.Parameter(torch.randn_like(model.weight))
            actual = model(x)
            self.assertEqual(quantize.call_count, 9)
            torch.testing.assert_close(actual, functional.do_linear(x, model.weight, model.bias, plan=model.plan))
        self.assertEqual(set(model.state_dict()), {"weight", "bias"})
        self.assertFalse(deepcopy(model)._fp8_weight_cache._entries)
        model.clear_fp8_cache()
        self.assertFalse(model._fp8_weight_cache._entries)

    def test_weight_cache_handles_inference_mode_then_training_and_optimizer_steps(self):
        model = DOLinear.from_linear(nn.Linear(7, 4), plan=rowwise_plan())
        x = torch.randn(2, 7)
        with torch.inference_mode():
            model(x)
            model(x)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(x.requires_grad_()).sum().backward()
            optimizer.step()
        for entry in model._fp8_weight_cache._entries.values():
            self.assertFalse(entry[1].data.is_inference())

    def test_weight_cache_separates_policy_dtype_and_unsafe_inference_tensors(self):
        model = DOLinear.from_linear(nn.Linear(7, 4), plan=fp8_plan(True))
        x = torch.randn(2, 7)
        for options in ({}, {"weight_dtype": "e5m2"}, {"block_shapes": {"weight": (2, 3)}}):
            with autocast("cpu", enabled=False, fp8=options):
                actual = model(x)
                expected = functional.do_linear(x, model.weight, model.bias, plan=fp8_plan(True))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        model.to(dtype=torch.bfloat16)
        actual = model(x.to(torch.bfloat16))
        expected = functional.do_linear(x.to(torch.bfloat16), model.weight, model.bias, plan=model.plan)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with torch.inference_mode():
            inference_model = DOLinear.from_linear(nn.Linear(7, 4), plan=fp8_plan(True))
            inference_model(x)
            self.assertFalse(inference_model._fp8_weight_cache._entries)

    def test_cache_enabled_false_forces_weight_requantization(self):
        model = DOLinear.from_linear(nn.Linear(7, 4), plan=fp8_plan(True))
        with autocast("cpu", enabled=False, cache_enabled=False, fp8=True), \
             patch.object(impl, "quantize_fp8", wraps=impl.quantize_fp8) as quantize:
            for _ in range(2):
                model(torch.ones(2, 7))
        self.assertEqual(quantize.call_count, 4)

    def test_transpose_view_reuses_storage_and_matches_materialized_transpose(self):
        original = quantize_fp8(torch.randn(37, 65), block_shape=(2, 3))
        view = original.transpose(contiguous=False)
        self.assertEqual(view.data.data_ptr(), original.data.data_ptr())
        self.assertEqual(view.scale_inv.data_ptr(), original.scale_inv.data_ptr())
        self.assertFalse(view.data.is_contiguous())
        torch.testing.assert_close(view.dequantize(), original.transpose().dequantize(), rtol=0, atol=0)

    def test_shared_adapter_cache_preserves_thread_local_policies(self):
        model = DOLinear.from_linear(nn.Linear(7, 4), plan=fp8_plan(True))
        x = torch.randn(2, 7)

        def run(e5m2):
            with autocast("cpu", enabled=False, cache_enabled=not e5m2,
                          fp8={"weight_dtype": "e5m2" if e5m2 else "e4m3fn"}):
                expected = functional.do_linear(x, model.weight, model.bias, plan=model.plan)
                for _ in range(10):
                    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run, (False, True)))
