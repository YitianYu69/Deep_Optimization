"""Native backward dispatch, numerical policy, and metadata-only eligibility."""

from contextlib import nullcontext
from dataclasses import asdict
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from deepopt import FP8Config
from deepopt.config import ConfigError, fp8_config_from_dict
from deepopt.opt._recipe import fp8_recipe
from deepopt.opt.backends import gemm_nt
from deepopt.opt.backends import native_fp8
from deepopt.opt.nn import functional
from deepopt.opt.nn.autograd import linear as linear_autograd_func
from deepopt.opt.policy import fp8_plan
from deepopt.opt.quantization.tensor import QuantizedTensor


def operand(rows=32, cols=64, dtype=torch.float8_e4m3fn, device="cpu"):
    return QuantizedTensor(torch.empty(rows, cols, dtype=dtype, device=device), torch.ones(1, 1, device=device))


class NativeBackwardTests(unittest.TestCase):
    def test_configuration_and_recipe_explicitly_record_changed_accumulation(self):
        for backend in ("native_fast", "native_precise"):
            value = FP8Config(enabled=True, backend="triton", backward_backend=backend)
            self.assertEqual(fp8_config_from_dict(asdict(value)), value)
            self.assertEqual(fp8_plan(value)["fp8_training"]["backward_backend"], backend)
            recipe = fp8_recipe("triton", value)
            self.assertEqual(recipe["version"], 5)
            self.assertEqual(recipe["weight_gradient_output"], "forward-dtype")
            self.assertEqual(recipe["backward_backend"], backend)
            self.assertIn("torch_version", recipe)
            self.assertEqual(recipe["backward_fallback"], "triton-unsupported-layout")
        self.assertNotEqual(fp8_recipe("triton", FP8Config(backend="triton", backward_backend="native_fast")),
                            fp8_recipe("triton", FP8Config(backend="triton", backward_backend="native_precise")))
        self.assertEqual(fp8_recipe("reference", FP8Config())["version"], 5)
        for options in ({"backward_backend": "unknown"}, {"backward_backend": "native_fast"},
                        {"backend": "triton", "backward_backend": "native_precise", "block_shapes": {"grad": [1, 32]}}):
            with self.subTest(options=options), self.assertRaises(ConfigError):
                fp8_config_from_dict(options)

    def test_eligibility_handles_tails_strides_and_e5_pairs_without_reading_values(self):
        self.assertIsNone(native_fp8.unsupported_reason(operand(rows=17), operand(rows=48)))
        self.assertIn("divisible", native_fp8.unsupported_reason(operand(cols=63), operand(cols=63)))
        self.assertIn("divisible", native_fp8.unsupported_reason(operand(), operand(rows=47)))
        self.assertIn("E5M2", native_fp8.unsupported_reason(operand(dtype=torch.float8_e5m2), operand(dtype=torch.float8_e5m2)))
        strided = QuantizedTensor(torch.empty(64, 32, dtype=torch.float8_e4m3fn).T, torch.ones(1, 1))
        self.assertIn("row-major", native_fp8.unsupported_reason(strided, operand()))
        broadcast = QuantizedTensor(torch.empty(1, 64, dtype=torch.float8_e4m3fn).expand(32, -1), torch.ones(1, 1))
        self.assertIn("row-major", native_fp8.unsupported_reason(broadcast, operand()))
        with self.assertRaisesRegex(ValueError, "reduction"):
            native_fp8.unsupported_reason(operand(), operand(cols=32))

    def test_native_executor_passes_existing_buffers_scales_and_output_dtype(self):
        with FakeTensorMode(), patch.object(native_fp8, "device_metadata", return_value=(90, 132)), \
             patch.object(torch.cuda, "device", return_value=nullcontext()), \
             patch.object(torch, "autocast", return_value=nullcontext()):
            a, b = operand(device="cuda:0"), operand(rows=48, device="cuda:0")
            for backend in ("native_fast", "native_precise"):
                result = torch.empty(32, 48, device="cuda:0", dtype=torch.float32)
                with patch.object(torch, "_scaled_mm", return_value=result) as scaled:
                    self.assertIs(gemm_nt(a, b, backend=backend, out_dtype=torch.float32), result)
                self.assertIs(scaled.call_args.args[0], a.data)
                self.assertEqual(scaled.call_args.args[1].stride(), b.data.T.stride())
                self.assertIs(scaled.call_args.kwargs["scale_a"], a.scale_inv)
                self.assertIs(scaled.call_args.kwargs["scale_b"], b.scale_inv)
                self.assertEqual(scaled.call_args.kwargs["use_fast_accum"], backend == "native_fast")
                self.assertEqual(scaled.call_args.kwargs["out_dtype"], torch.float32)
                with patch.object(torch, "_scaled_mm", return_value=(result, torch.empty((), device="cuda:0"))):
                    self.assertIs(gemm_nt(a, b, backend=backend), result)

    def test_unsupported_metadata_falls_back_but_native_execution_errors_propagate(self):
        fallback = Mock()
        with FakeTensorMode(), patch.object(native_fp8, "device_metadata", return_value=(90, 132)), \
             patch.object(torch.cuda, "device", return_value=nullcontext()), \
             patch.object(torch, "autocast", return_value=nullcontext()), \
             patch.dict(sys.modules, {"deepopt.opt.backends.triton.gemm.fp8": SimpleNamespace(gemm_nt=fallback)}):
            gemm_nt(operand(device="cuda:0"), operand(rows=17, device="cuda:0"), backend="native_fast")
            self.assertEqual(fallback.call_count, 1)
            with patch.object(torch, "_scaled_mm", side_effect=RuntimeError("native failure")):
                with self.assertRaisesRegex(RuntimeError, "native failure"):
                    gemm_nt(operand(device="cuda:0"), operand(device="cuda:0"), backend="native_fast")
            self.assertEqual(fallback.call_count, 1)

    def test_only_backward_products_change_executor_and_keep_fp32_masters(self):
        for recipe in ("current", "delayed"):
            for backend in ("native_fast", "native_precise"):
                with self.subTest(recipe=recipe, backend=backend):
                    options = FP8Config(enabled=True, backend="triton", backward_backend=backend, scaling_recipe=recipe)
                    model = torch.nn.Linear(32, 48)
                    weight = model.weight
                    from deepopt.opt.policy import prepare_fp8_model

                    model, _ = prepare_fp8_model(model, options)
                    records = []

                    def gemm(a, b, **kwargs):
                        records.append((kwargs["backend"], kwargs["out_dtype"]))
                        from deepopt.opt.backends.reference.gemm import gemm_nt as oracle

                        return oracle(a, b, out_dtype=kwargs["out_dtype"], bias=kwargs.get("bias"))

                    for out_dtype in (torch.float32, torch.bfloat16, torch.float16):
                        records.clear()
                        with patch.object(linear_autograd_func, "gemm_nt", new=gemm):
                            # Real CPU autograd with an oracle GEMM; the public
                            # native executor still requires NVIDIA CUDA.
                            value = linear_autograd_func._DOLinearFunction.apply(
                                torch.randn(16, 32, requires_grad=True), model.weight, model.bias,
                                model.plan, out_dtype, True, None, model.fp8_scaling, True)
                            value.sum().backward()
                        self.assertEqual(records, [("triton", out_dtype), (backend, torch.float32), (backend, out_dtype)])
                        self.assertEqual(weight.grad.dtype, torch.float32)
                    self.assertIs(model.weight, weight)
                    self.assertEqual(weight.dtype, torch.float32)

    def test_manual_plans_are_validated_and_cpu_execution_is_rejected(self):
        plan = fp8_plan(FP8Config())
        plan["fp8_training"]["backward_backend"] = "native_fast"
        with self.assertRaisesRegex(ValueError, "backend=triton"):
            functional.validate_fp8_plan(plan)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            gemm_nt(operand(), operand(), backend="native_fast")
