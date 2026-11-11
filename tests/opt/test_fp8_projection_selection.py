"""Selective FP8 preparation preserves dense paths, aliases and parameters."""

from dataclasses import asdict
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import FP8Config, TrainingRuntime, autocast
from deepopt.config import ConfigError, fp8_config_from_dict
from deepopt.opt.nn.linear import DOLinear
from deepopt.opt.nn.attention import DOMultiheadAttention
from deepopt.opt.policy import prepare_fp8_model
from deepopt.opt.backends import gemm_nt


class Projections(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 8)
        self.attention = nn.ModuleDict({"proj": nn.Linear(8, 8)})
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        return self.head(self.attention["proj"](self.qkv(x)))


class ProjectionSelectionTests(unittest.TestCase):
    def test_named_globs_keep_attention_and_head_dense_in_every_trace_mode(self):
        for trace in ("modules", "full", "auto"):
            with self.subTest(trace=trace):
                source = Projections()
                parameters = tuple(source.parameters())
                options = FP8Config(enabled=True, trace_mode=trace, intercept_linear=False,
                                    exclude_modules=("attention.*", "head"))
                model, report = prepare_fp8_model(source, options)
                self.assertIsInstance(model.qkv, DOLinear)
                self.assertIs(type(model.attention.proj if trace == "full" else model.attention["proj"]), nn.Linear)
                self.assertIs(type(model.head), nn.Linear)
                self.assertEqual(tuple(model.parameters()), parameters)
                self.assertIn("attention.proj", report.excluded_modules)
                with autocast("cpu", enabled=False, fp8=options):
                    model(torch.randn(2, 8)).sum().backward()
                self.assertTrue(all(parameter.grad is not None for parameter in parameters))

    def test_exclusion_wins_for_shared_module_aliases(self):
        source = Projections()
        source.head = source.qkv
        model, report = prepare_fp8_model(source, FP8Config(enabled=True, intercept_linear=False, exclude_modules=("head",)))
        self.assertIs(model.qkv, model.head)
        self.assertIs(type(model.head), nn.Linear)
        self.assertIn("qkv", report.excluded_modules)
        self.assertIn("head", report.excluded_modules)

    def test_excluded_subtree_with_functional_calls_is_a_dense_leaf(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(8, 8))

            def forward(self, x):
                return nn.functional.linear(x, self.weight)

        source = nn.Sequential(Block(), nn.Linear(8, 3))
        model, report = prepare_fp8_model(source, FP8Config(enabled=True, trace_mode="full", intercept_linear=False,
                                                          exclude_modules=("0",)))
        self.assertIs(model.get_submodule("0"), source[0])
        self.assertEqual(report.function_calls, ())
        self.assertIsInstance(model.get_submodule("1"), DOLinear)
        model(torch.randn(2, 8)).sum().backward()

    def test_native_mha_can_exclude_only_the_output_projection(self):
        for trace in ("modules", "full"):
            with self.subTest(trace=trace):
                source = nn.Sequential(nn.MultiheadAttention(8, 2, batch_first=True))
                options = FP8Config(enabled=True, intercept_linear=False, scaling_recipe="delayed", trace_mode=trace,
                                    exclude_modules=("0.out_proj",))
                model, report = prepare_fp8_model(source, options)
                attention = model.get_submodule("0")
                self.assertIsInstance(attention, DOMultiheadAttention)
                self.assertFalse(attention._fp8_out_projection)
                self.assertNotIn("out", attention._projection_scaling)
                self.assertIn("0.out_proj", report.excluded_modules)
                query = torch.randn(2, 3, 8, requires_grad=True)
                attention(query, query, query)[0].sum().backward()
                self.assertTrue(torch.isfinite(attention.out_proj.weight.grad).all())

    def test_existing_graph_module_excludes_inlined_functional_projection(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(8, 8))

            def forward(self, x):
                return nn.functional.linear(x, self.weight)

        source = torch.fx.symbolic_trace(nn.Sequential(Block(), nn.Linear(8, 3)))
        for node in source.graph.nodes:
            node.meta.clear()  # Attribute ownership remains available without trace metadata.
        model, report = prepare_fp8_model(source, FP8Config(enabled=True, trace_mode="full", intercept_linear=False,
                                                          exclude_modules=("0",)))
        self.assertEqual(report.function_calls, ())
        self.assertIsInstance(model.get_submodule("1"), DOLinear)
        self.assertTrue(any(node.target is nn.functional.linear for node in model.graph.nodes))
        model(torch.randn(2, 8)).sum().backward()

    def test_root_exclusion_and_unmatched_glob_preserve_expected_conversion(self):
        for pattern, converted in (("<root>", False), ("missing", True)):
            original = nn.Linear(8, 3)
            model, _ = prepare_fp8_model(original, FP8Config(enabled=True, intercept_linear=False, exclude_modules=(pattern,)))
            self.assertEqual(isinstance(model, DOLinear), converted)
            self.assertIs(model.weight, original.weight)

    def test_exclusion_does_not_silently_leave_an_already_converted_child_active(self):
        child, _ = prepare_fp8_model(nn.Linear(8, 3), FP8Config(enabled=True))
        source = nn.Sequential(child)
        with self.assertRaisesRegex(ValueError, "original eager model"):
            prepare_fp8_model(source, FP8Config(enabled=True, intercept_linear=False, exclude_modules=("0",)))

    def test_config_rejects_ambiguous_interception_and_snapshots_selection(self):
        for options in ({"exclude_modules": "head"}, {"exclude_modules": [""]}, {"exclude_modules": ["head"]}):
            with self.assertRaises(ConfigError):
                fp8_config_from_dict(options)
        options = FP8Config(enabled=True, intercept_linear=False, exclude_modules=("head",))
        self.assertEqual(fp8_config_from_dict(asdict(options)), options)
        source = Projections()
        runtime = TrainingRuntime(source, torch.optim.SGD(source.parameters(), lr=.001), fp8=options, execution_mode="max_speed")
        with patch("deepopt.opt.nn.autograd.linear.gemm_nt", wraps=gemm_nt) as gemm:
            loss = runtime.compute(torch.randn(2, 8), lambda x: runtime.model(x).sum())
            runtime.backward(loss)
            runtime.step()
        self.assertEqual(gemm.call_count, 5)  # Two FP8 forwards, dW for both, and dX for the second.
