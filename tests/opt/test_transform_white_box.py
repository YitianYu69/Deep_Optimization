"""FX matching, replacement ownership, and transformation control flow."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import fx, nn
from torch.nn import functional as F

from deepopt.opt import transform_model
from deepopt.opt.nn.functional import do_linear, validate_fp8_plan
from deepopt.opt.nn.linear import DOLinear
from deepopt.opt.fx import ConversionRegistry, DOTracer, default_registry, match_graph, rewrite_graph


def plan():
    return validate_fp8_plan({
        "fp8_training": {"fp8_dtypes": {
            "activation": torch.float8_e4m3fn,
            "weight": torch.float8_e4m3fn,
            "grad": torch.float8_e5m2,
        }},
        "block_shapes": {role: None for role in ("activation", "weight", "grad")},
    })


def trace(model, registry):
    return fx.GraphModule(model, DOTracer(extra_leaf_types=registry.leaf_types).trace(model))


class GraphFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)
        self.activation = nn.ReLU()
        self.weight = nn.Parameter(torch.eye(4))

    def forward(self, x):
        x = self.activation(self.layer(x))
        return F.linear(input=x, weight=self.weight).sum(dim=-1)


class TransformationWhiteBoxTests(unittest.TestCase):
    def test_matcher_only_returns_registered_module_and_supported_function_nodes(self):
        registry = default_registry()
        graph_module = trace(GraphFixture(), registry)
        before = str(graph_module.graph)
        matches = match_graph(graph_module, registry)
        self.assertEqual([(node.op, node.target) for node in matches],
                         [("call_module", "layer"), ("call_module", "activation"), ("call_function", F.linear)])
        self.assertEqual(str(graph_module.graph), before)
        self.assertIsInstance(graph_module.layer, nn.Linear)

    def test_registry_uses_exact_types_and_rejects_duplicate_rules(self):
        class SpecialLinear(nn.Linear):
            pass

        registry = default_registry()
        self.assertIsNone(registry.lookup(SpecialLinear(4, 4)))
        rule = registry.lookup(nn.Linear(4, 4))
        self.assertIs(rule.target_type, DOLinear)
        self.assertIn(DOLinear, registry.leaf_types)
        with self.assertRaises(ValueError):
            registry.register(nn.Linear, DOLinear, rule.converter)

    def test_tracer_retains_registered_leaf_and_preconverted_modules(self):
        registry = default_registry()
        source = nn.Linear(4, 4)
        converted = DOLinear(source, plan=plan())

        class AttentionBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = source
                self.converted = converted
                self.attention = nn.MultiheadAttention(4, 2, batch_first=True)

            def forward(self, x):
                x = self.converted(self.layer(x))
                return self.attention(x, x, x)[0]

        graph_module = trace(AttentionBlock(), registry)
        calls = [node.target for node in graph_module.graph.nodes if node.op == "call_module"]
        self.assertEqual(calls, ["layer", "converted", "attention"])

    def test_rewriter_changes_only_supplied_matches(self):
        registry = default_registry()
        graph_module = trace(GraphFixture(), registry)
        matches = match_graph(graph_module, registry)
        returned = rewrite_graph(graph_module, matches[:1], registry, plan=plan())
        self.assertIs(returned, graph_module)
        self.assertIsInstance(graph_module.layer, DOLinear)
        self.assertIs(matches[2].target, F.linear)
        self.assertIsInstance(graph_module.activation, nn.ReLU)

    def test_function_rewrite_retains_weight_nodes_and_renames_keyword(self):
        registry = default_registry()
        graph_module = trace(GraphFixture(), registry)
        function_node = next(node for node in match_graph(graph_module, registry)
                             if node.op == "call_function")
        input_node = function_node.kwargs["input"]
        weight_node = function_node.kwargs["weight"]
        checked_plan = plan()
        rewrite_graph(graph_module, [function_node], registry, plan=checked_plan)
        self.assertIs(function_node.target, do_linear)
        self.assertNotIn("input", function_node.kwargs)
        self.assertIs(function_node.kwargs["x"], input_node)
        self.assertIs(function_node.kwargs["weight"], weight_node)
        self.assertEqual(function_node.kwargs["plan"], checked_plan)
        # Execute the regenerated forward, not just the edited graph structure.
        self.assertEqual(graph_module(torch.ones(2, 4)).shape, (2,))

    def test_positional_function_call_keeps_args(self):
        class Functional(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.eye(4))

            def forward(self, x):
                return F.linear(x, self.weight)

        registry = default_registry()
        graph_module = trace(Functional(), registry)
        node, = match_graph(graph_module, registry)
        args = node.args
        rewrite_graph(graph_module, [node], registry, plan=plan())
        self.assertEqual(node.args, args)
        self.assertEqual(graph_module(torch.ones(2, 4)).shape, (2, 4))

    def test_shared_cache_calls_converter_once_across_graphs(self):
        calls = []

        def converter(module, *, plan):
            calls.append(module)
            return DOLinear(module, plan=plan)

        registry = ConversionRegistry()
        registry.register(nn.Linear, DOLinear, converter)
        shared = nn.Linear(4, 4)
        graphs = [trace(nn.Sequential(shared), registry), trace(nn.Sequential(shared), registry)]
        cache = {}
        for graph_module in graphs:
            rewrite_graph(graph_module, match_graph(graph_module, registry), registry,
                          plan=plan(), replacements=cache)
        self.assertEqual(calls, [shared])
        self.assertIs(graphs[0].get_submodule("0"), graphs[1].get_submodule("0"))
        self.assertIs(cache[id(shared)].weight, shared.weight)

    def test_duplicate_call_sites_share_one_replacement(self):
        class Repeated(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(4, 4)

            def forward(self, x):
                return self.layer(x) + self.layer(x + 1)

        calls = []
        registry = ConversionRegistry()

        def converter(module, *, plan):
            calls.append(module)
            return DOLinear(module, plan=plan)

        registry.register(nn.Linear, DOLinear, converter)
        transformed, report = transform_model(Repeated(), plan=plan(), registry=registry)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report.converted_modules, ("layer", "layer"))
        self.assertEqual(transformed(torch.ones(2, 4)).shape, (2, 4))

    def test_converter_errors_do_not_trigger_auto_fallback(self):
        registry = ConversionRegistry()

        def fail(module, *, plan):
            raise RuntimeError("converter failed")

        registry.register(nn.Linear, DOLinear, fail)
        with patch("deepopt.opt.transform._transform_submodules") as fallback:
            with self.assertRaisesRegex(RuntimeError, "converter failed"):
                transform_model(nn.Sequential(nn.Linear(4, 4)), plan=plan(),
                                trace_mode="auto", registry=registry)
        fallback.assert_not_called()

    def test_plan_is_owned_independently_from_caller(self):
        caller_plan = plan()
        before = deepcopy(caller_plan)
        transformed, _ = transform_model(GraphFixture(), plan=caller_plan)
        caller_plan["fp8_training"]["backend"] = "triton"
        caller_plan["block_shapes"]["activation"] = (32, 32)
        self.assertEqual(transformed.layer.plan, before)
        function_node = next(node for node in transformed.graph.nodes if node.target is do_linear)
        self.assertEqual(function_node.kwargs["plan"], before)

    def test_custom_registry_rewrites_registered_module(self):
        class Shift(nn.Module):
            def forward(self, x):
                return x + 1

        class DoubleShift(nn.Module):
            def forward(self, x):
                return x + 2

        registry = ConversionRegistry()
        registry.register(Shift, DoubleShift, lambda module, *, plan: DoubleShift())
        transformed, report = transform_model(nn.Sequential(Shift()), plan=plan(), registry=registry)
        self.assertIsInstance(transformed.get_submodule("0"), DoubleShift)
        torch.testing.assert_close(transformed(torch.zeros(2)), torch.full((2,), 2.))
        self.assertEqual(report.converted_modules, ("0",))

    def test_fx_and_opt_exports_resolve_existing_symbols(self):
        import deepopt.opt as opt
        import deepopt.opt.fx as opt_fx

        for module in (opt, opt_fx):
            for name in module.__all__:
                with self.subTest(module=module.__name__, name=name):
                    self.assertIsNotNone(getattr(module, name))
        self.assertNotIn("RewriteResult", opt_fx.__all__)
        self.assertNotIn("MatchResult", opt_fx.__all__)


if __name__ == "__main__":
    unittest.main()
