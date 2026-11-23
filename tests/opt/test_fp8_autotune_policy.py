"""FP8 autotune configuration, saved backward policy, and untuned launch choices."""

import sys
import unittest
from contextvars import Context
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from deepopt import FP8Config
from deepopt.config import ConfigError, fp8_config_from_dict
from deepopt.opt.backends import gemm_nt
from deepopt.opt.nn.autograd import linear as linear_autograd
from deepopt.opt.nn.functional import validate_fp8_plan
from deepopt.opt.policy import fp8_plan, prepare_fp8_model
from deepopt.opt.scaling.delayed import delayed_states


class AutotunePolicyTests(unittest.TestCase):
    def test_default_and_round_trip(self):
        self.assertIs(FP8Config().autotune, True)
        for enabled in (False, True):
            settings = fp8_config_from_dict({"enabled": True, "autotune": enabled})
            self.assertIs(settings.autotune, enabled)
            self.assertEqual(fp8_config_from_dict(asdict(settings)), settings)
            plan = validate_fp8_plan(fp8_plan(settings))
            self.assertIs(plan["fp8_training"]["autotune"], enabled)

    def test_invalid_values_are_rejected_at_config_plan_and_backend_boundaries(self):
        operand = linear_autograd.QuantizedTensor(
            torch.ones(2, 2).to(torch.float8_e4m3fn), torch.ones(1, 1)
        )
        for value in (0, 1, None, "false", "off", []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ConfigError, "autotune.*boolean"):
                    fp8_config_from_dict({"autotune": value})
                plan = fp8_plan(FP8Config(enabled=True))
                plan["fp8_training"]["autotune"] = value
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "autotune.*boolean"
                ):
                    validate_fp8_plan(plan)
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "autotune.*boolean"
                ):
                    gemm_nt(operand, operand, backend="reference", autotune=value)

    def test_forward_and_backward_retain_the_selected_policy_without_contextvars(self):
        source = nn.Linear(5, 7)
        outcomes = []
        for enabled in (False, True):
            model, _ = prepare_fp8_model(
                deepcopy(source), FP8Config(enabled=True, autotune=enabled)
            )
            inputs = torch.ones(3, 5, requires_grad=True)
            with patch.object(linear_autograd, "gemm_nt", wraps=gemm_nt) as execute:
                output = model(inputs)
                Context().run(output.sum().backward)
            self.assertEqual(execute.call_count, 3)
            self.assertTrue(
                all(
                    call.kwargs["autotune"] is enabled
                    for call in execute.call_args_list
                )
            )
            outcomes.append((output.detach(), inputs.grad, model.weight.grad))
        for actual, expected in zip(*outcomes):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_delayed_scaling_owners_inherit_disabled_autotune(self):
        model, _ = prepare_fp8_model(
            nn.Sequential(nn.Linear(5, 7), nn.GELU(), nn.Linear(7, 3)),
            FP8Config(
                enabled=True,
                autotune=False,
                scaling_recipe="delayed",
                backward_precision="auto",
            ),
        )
        model(torch.ones(4, 5, requires_grad=True)).sum().backward()
        states = list(delayed_states(model))
        self.assertTrue(states)
        self.assertTrue(all(state.autotune is False for state in states))

    def test_convolution_forward_and_backward_retain_disabled_autotune(self):
        from deepopt.opt.backends import convolution, convolution_transpose

        for source, executor in (
            (nn.Conv2d(2, 3, 3, padding=1), convolution),
            (nn.ConvTranspose2d(2, 3, 3, padding=1), convolution_transpose),
        ):
            with self.subTest(source=type(source).__name__):
                model, _ = prepare_fp8_model(
                    source, FP8Config(enabled=True, autotune=False)
                )
                with (
                    patch.object(
                        executor, "forward", wraps=executor.forward
                    ) as forward,
                    patch.object(
                        executor, "input_gradient", wraps=executor.input_gradient
                    ) as dx,
                    patch.object(
                        executor, "weight_gradient", wraps=executor.weight_gradient
                    ) as dw,
                ):
                    output = model(torch.ones(1, 2, 5, 5, requires_grad=True))
                    Context().run(output.sum().backward)
                for operation in (forward, dx, dw):
                    self.assertEqual(operation.call_count, 1)
                    self.assertIs(operation.call_args.kwargs["autotune"], False)

    def test_native_fallback_preserves_disabled_autotune(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        from deepopt.opt.backends.native import fp8 as native
        from deepopt.opt.quantization.tensor import QuantizedTensor

        calls = []

        def fallback(*args, **kwargs):
            calls.append(kwargs)
            return torch.empty(16, 16, device="cuda")

        with (
            FakeTensorMode(),
            patch.object(native, "device_metadata", return_value=(90, 132)),
            patch.dict(
                sys.modules,
                {
                    "deepopt.opt.backends.triton.gemm.fp8": SimpleNamespace(
                        gemm_nt=fallback
                    )
                },
            ),
        ):
            operand = QuantizedTensor(
                torch.empty(16, 33, device="cuda", dtype=torch.float8_e4m3fn),
                torch.ones(1, 1, device="cuda"),
            )
            native.gemm_nt(operand, operand, autotune=False)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["autotune"], False)

    def test_fixed_gemm_choices_are_compatible_and_respect_stage_pinning(self):
        from deepopt.opt.backends.triton.gemm.autotune import fixed_gemm_parameters

        for blocks in (
            (None, None),
            ((1, 32), (32, 64)),
            ((32, 1), None),
            ((3, 96), (1, 32)),
        ):
            for stages in ("auto", 1, 2, 3, 4, 5):
                choice = fixed_gemm_parameters(*blocks, num_stages=stages)
                self.assertEqual(choice["SPLIT_K"], 1)
                self.assertEqual(choice["GROUP_SIZE_M"], 1)
                self.assertEqual(
                    choice["NUM_STAGES"], 3 if stages == "auto" else stages
                )
                self.assertEqual(choice["num_stages"], choice["NUM_STAGES"])
                if all(block is None or block[1] % 32 == 0 for block in blocks):
                    for block in blocks:
                        if block is not None:
                            self.assertEqual(block[1] % choice["BLOCK_K"], 0)
                else:
                    self.assertLessEqual(choice["BLOCK_M"], 64)
                    self.assertLessEqual(choice["BLOCK_N"], 64)
                    self.assertEqual(choice["BLOCK_K"], 32)


if __name__ == "__main__":
    unittest.main()
