"""Optional TE comparisons pin their dependency and expose matching limits."""

import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("te_linear_benchmark", Path(__file__).resolve().parents[2] /
                                             "benchmarks/benchmark_fp8_te_linear.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class TELinearBenchmarkTests(unittest.TestCase):
    def test_matched_math_uses_fp32_projection_and_common_bias_with_original_parameters(self):
        import torch

        projection = torch.nn.Linear(5, 3, bias=False)
        bias = torch.nn.Parameter(torch.randn(3))
        model = benchmark.matched_projection(projection, bias, torch)
        self.assertEqual({id(value) for value in model.parameters()}, {id(projection.weight), id(bias)})
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.randn(7, 5, dtype=dtype, requires_grad=True)
            result = model(x)
            expected = torch.nn.functional.linear(x.float(), projection.weight, bias).to(torch.bfloat16)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            result.sum().backward()
            self.assertEqual(x.grad.dtype, dtype)
            self.assertEqual(projection.weight.grad.dtype, torch.float32)
            self.assertEqual(result.dtype, torch.bfloat16)

    def test_matched_te_recipe_disables_split_accumulation_for_all_gemms(self):
        recipes = SimpleNamespace(Format=SimpleNamespace(E4M3="e4m3"),
                                  DelayedScaling=lambda **kwargs: SimpleNamespace(**kwargs),
                                  MMParams=lambda **kwargs: SimpleNamespace(**kwargs))
        args = SimpleNamespace(implementation="te-delayed", format="e4m3fn", history_length=3,
                               amax_algorithm="max", margin=0, math_policy="matched-fp32")
        recipe = benchmark.te_recipe(args, recipes)
        for role in ("fprop", "dgrad", "wgrad"):
            self.assertFalse(getattr(recipe, f"fp8_gemm_{role}").use_split_accumulator)

    def test_vit_contract_has_real_dtypes_tails_and_shared_dense_head(self):
        rows = benchmark.cases(SimpleNamespace(batches=[512, 336, 272], tokens=197))
        self.assertEqual(len(rows), 15)
        self.assertEqual(rows[0], (512, "qkv", 100864, 768, 2304, "fp32", 12))
        self.assertEqual(rows[1][-2:], ("bf16", 12))
        self.assertEqual(rows[2][-2:], ("fp32", 12))
        self.assertEqual(rows[-1], (272, "head", 272, 768, 10, "bf16", 1))

    def test_te_recipes_explicitly_match_history_formula_and_format(self):
        recipes = SimpleNamespace(Format=SimpleNamespace(E4M3="e4m3", HYBRID="hybrid"),
                                  Float8CurrentScaling=lambda **kwargs: kwargs, DelayedScaling=lambda **kwargs: kwargs)
        args = SimpleNamespace(implementation="te-delayed", format="e4m3fn", history_length=32, amax_algorithm="max", margin=1)
        self.assertEqual(benchmark.te_recipe(args, recipes),
                         {"fp8_format": "e4m3", "amax_history_len": 32, "amax_compute_algo": "max", "margin": 1, "reduce_amax": True})
        args.implementation, args.format = "te-current", "hybrid"
        self.assertEqual(benchmark.te_recipe(args, recipes), {"fp8_format": "hybrid", "use_power_2_scales": False})

    def test_summary_separates_performance_ratio_from_numerical_acceptance(self):
        results = [{"status": "ok", "implementation": name,
                    "timings": [{"sample_batch": 512, "vit_weight": 12, "warm_median_ms": time}]}
                   for name, time in (("te-delayed", 1.), ("deepopt-delayed", 1.1))]
        summary = benchmark.summary(results)
        self.assertAlmostEqual(summary["deepopt_delayed_over_te_delayed"]["512"], 1.1)
        self.assertIn("internal GEMM", summary["acceptance"])
        self.assertEqual(summary["target_ratio"], 1.05)

    def test_parent_runs_each_implementation_in_a_fresh_environment_process(self):
        calls = []

        def worker(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=0, stderr="", stdout=benchmark.PREFIX +
                                   json.dumps({"status": "unavailable", "implementation": command[-2]}))

        output = io.StringIO()
        with patch.object(sys, "argv", ["benchmark", "--implementation", "all"]), \
             patch.object(benchmark.subprocess, "run", side_effect=worker), patch("sys.stdout", new=output):
            benchmark.main()
        report = json.loads(output.getvalue())
        self.assertEqual(len(calls), 5)
        self.assertEqual(report["settings"]["te_version"], "2.9.0")
        for command, kwargs in calls:
            self.assertEqual(command[0], sys.executable)
            self.assertIn("TRITON_CACHE_DIR", kwargs["env"])
            self.assertNotIn("cwd", kwargs)
