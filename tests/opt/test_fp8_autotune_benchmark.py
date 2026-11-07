"""Benchmark operands follow production autograd; search diagnostics retain each key."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from deepopt.config import FP8Config, fp8_config_from_dict
from deepopt.opt.backends import gemm_nt
from deepopt.opt.do_nn.functional import do_linear, validate_fp8_plan
from deepopt.opt.do_nn.AOTAutograd_Func import linear_autograd_func


spec = importlib.util.spec_from_file_location(
    "fp8_autotune_benchmark", Path(__file__).resolve().parents[2] / "benchmarks/benchmark_fp8_autotune.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class FP8AutotuneBenchmarkTests(unittest.TestCase):
    def test_capture_matches_actual_autograd_for_both_orientation_policies(self):
        for activation, grad_blocks in (((1, 32), (32, 32)), ((32, 32), (1, 32)), (None, None)):
            plan = {
                "fp8_training": {"backend": "reference", "out_dtype": torch.bfloat16,
                                 "fp8_dtypes": {"activation": torch.float8_e4m3fn,
                                                "weight": torch.float8_e4m3fn, "grad": torch.float8_e5m2}},
                "block_shapes": {"activation": activation, "weight": (32, 32), "grad": grad_blocks},
            }
            x = torch.randn(37, 65, dtype=torch.bfloat16, requires_grad=True)
            weight = torch.randn(10, 65, requires_grad=True)
            bias = torch.randn(10, requires_grad=True)
            grad = torch.randn(37, 10, dtype=torch.bfloat16)
            actual = []
            original = linear_autograd_func.gemm_nt

            def record(a, b, **kwargs):
                actual.append((a, b, kwargs.get("bias"), kwargs["out_dtype"]))
                return original(a, b, **kwargs)

            with patch.object(linear_autograd_func, "gemm_nt", new=record):
                do_linear(x, weight, bias, plan=plan).backward(grad)
            captured = benchmark._prepare_gemm_products(x, weight, bias, grad, plan)
            self.assertEqual([row[0] for row in captured], ["forward", "input_gradient", "weight_gradient"])
            for expected, (_, *observed) in zip(actual, captured):
                for left, right in zip(expected[:2], observed[:2]):
                    self.assertEqual(left.block_shape, right.block_shape)
                    self.assertEqual(left.data.dtype, right.data.dtype)
                    torch.testing.assert_close(left.data.float(), right.data.float(), rtol=0, atol=0)
                    torch.testing.assert_close(left.scale_inv, right.scale_inv, rtol=0, atol=0)
                self.assertIs(expected[2], observed[2])
                self.assertEqual(expected[3], observed[3])
            self.assertIsNone(x.grad)
            self.assertIsNone(weight.grad)
            self.assertIsNone(bias.grad)
            if activation == (1, 32):
                self.assertEqual(captured[2][2].block_shape, (1, 32))

    def test_records_each_key_failed_trials_and_restores_tuner(self):
        good = SimpleNamespace(kwargs={"BLOCK_M": 16, "SPLIT_K": 4}, num_warps=2, num_stages=2)
        bad = SimpleNamespace(kwargs={"BLOCK_M": 32, "SPLIT_K": 1}, num_warps=4, num_stages=3)

        class Tuner:
            keys = ("M",)
            arg_names = ("M", "a_ptr")

            def __init__(self):
                self.nargs = {"M": 32, "a_ptr": SimpleNamespace(dtype="fp8")}
                self.cache = {}

            def _bench(self, *args, config, **kwargs):
                return [0.2, 0.1, 0.3] if config is good else [float("inf")] * 3

        tuner = Tuner()
        modules = {"deepopt.opt.backends.triton_fp8": SimpleNamespace(_gemm_nt_kernel=tuner),
                   "deepopt.opt.backends._triton_autotune": SimpleNamespace(AUTOTUNE_KEY=("M",))}
        with patch.dict(sys.modules, modules):
            with benchmark._record_autotune_trials() as trials:
                for m in (32, 64):
                    tuner.nargs["M"] = m
                    tuner._bench(config=good)
                    tuner._bench(config=bad)
                    tuner.cache[m, "fp8"] = good
            self.assertNotIn("_bench", tuner.__dict__)
            records = benchmark._autotune_diagnostics(trials)
        self.assertEqual([row["key"]["M"] for row in records], [32, 64])
        for row in records:
            self.assertEqual(len(row["candidates"]), 2)
            self.assertEqual(row["candidates"][0]["quantiles_ms"], [0.2, 0.1, 0.3])
            self.assertEqual(row["candidates"][1]["quantiles_ms"], [None] * 3)
            self.assertEqual(row["candidates"][1]["status"], "unavailable")
        json.dumps(records, allow_nan=False)

    def test_auto_and_explicit_stages_round_trip_through_public_configuration(self):
        self.assertEqual(FP8Config().num_stages, "auto")
        for stages in ("auto", 1, 2, 3, 4):
            parsed = fp8_config_from_dict({"num_stages": stages})
            self.assertEqual(parsed.num_stages, stages)
            plan = {"fp8_training": {"num_stages": stages, "fp8_dtypes": {
                role: torch.float8_e4m3fn for role in ("activation", "weight", "grad")}},
                "block_shapes": dict.fromkeys(("activation", "weight", "grad"))}
            self.assertEqual(validate_fp8_plan(plan)["fp8_training"]["num_stages"], stages)
            from deepopt.opt.quantization.quantizer import quantize_fp8
            operand = quantize_fp8(torch.ones(2, 3))
            torch.testing.assert_close(gemm_nt(operand, operand, backend="reference", num_stages=stages),
                                       torch.full((2, 2), 3.0))
