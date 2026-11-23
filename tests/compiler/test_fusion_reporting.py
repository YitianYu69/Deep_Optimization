"""Static fusion decisions remain distinct from observed execution evidence."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import FP8Config
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.compiler.ir import ExternKernelNode, TensorValue, register_uses
from deepopt.compiler.linear import DeepOptLinearNode, fuse_activations
from deepopt.compiler.lowering import GraphLowering
from deepopt.opt.policy import prepare_fp8_model


class FusionReportingTests(unittest.TestCase):
    def test_report_serializes_owned_native_plans_without_mutating_policy(self):
        from deepopt.compiler.reporting import GraphReport
        from deepopt.opt.nn.functional import _prepare_linear_plan

        plan = _prepare_linear_plan({"fp8": False,
            "activation_compression": {"enabled": True, "bits": 2, "group_size": 1024}})
        report = CompilationReport(graphs=[GraphReport(operations=1,
            differentiation_rules=({"rule": "attention", "saved_policy": plan},))])
        before = deepcopy(plan)
        payload = report.as_dict()
        restored = payload["graphs"][0]["differentiation_rules"][0]["saved_policy"]
        self.assertEqual(restored, before)
        self.assertTrue(json.loads(json.dumps(payload))["graphs"][0]["differentiation_rules"][0]
                        ["saved_policy"]["activation_compression"]["enabled"])
        self.assertIsNot(restored, plan)
        for candidate in (plan, restored, restored["activation_compression"]):
            with self.assertRaises(TypeError):
                candidate["new_field"] = True
        self.assertEqual(plan, before)

    def graph(self, *, rows=257, activation=True, delayed=False):
        root = nn.Module()
        root.projection = prepare_fp8_model(
            nn.Linear(17, 17),
            FP8Config(enabled=True, scaling_recipe="delayed" if delayed else "current"),
        )[0]
        graph = torch.fx.Graph()
        value = graph.placeholder("x")
        if activation:
            value = graph.call_function(nn.functional.gelu, (value,))
        value = graph.call_module("projection", (value,))
        graph.output(value)
        gm = torch.fx.GraphModule(root, graph)
        sample = torch.randn(rows, 17, requires_grad=True)
        ShapeProp(gm).propagate(sample)
        return gm, sample

    def lowered(self, *, cuda=True, **kwargs):
        gm, sample = self.graph(**kwargs)
        ir = GraphLowering(gm, [sample]).run()
        if cuda:
            # Exercise static legality only. No fake metadata is ever executed.
            for node in ir.nodes:
                for value in node.outputs():
                    if value.meta is not None:
                        value.meta = replace(value.meta, device=torch.device("cuda:0"))
        linear = next(node for node in ir.nodes if isinstance(node, DeepOptLinearNode))
        return ir, linear

    def decide(self, ir, *, enabled=True, unsafe_values=frozenset(), kernel_policy=None):
        diagnostics = []
        patterns = fuse_activations(
            ir, enabled=enabled, unsafe_values=unsafe_values,
            kernel_policy=kernel_policy, diagnostics=diagnostics,
        )
        self.assertEqual(len(diagnostics), 1)
        json.dumps(diagnostics, allow_nan=False)
        return patterns, diagnostics[0]

    def test_positive_reports_qualified_site_signature_and_static_selection(self):
        ir, linear = self.lowered()
        patterns, candidate = self.decide(ir)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(candidate["module_path"], "projection")
        self.assertEqual(candidate["phase"], "train")
        self.assertEqual(candidate["status"], "selected")
        self.assertTrue(candidate["requested"] and candidate["effective_enabled"])
        self.assertTrue(candidate["discovered"] and candidate["eligible"] and candidate["selected"])
        self.assertIsNone(candidate["reason"])
        self.assertEqual(candidate["input_signature"]["shape"], [257, 17])
        self.assertEqual(candidate["input_signature"]["stride"], [17, 1])
        self.assertEqual(candidate["activation"], {"kind": "gelu", "approximate": "none"})
        self.assertEqual(candidate["selected_schedule"], "fused_gelu_operand_preparation")
        self.assertIsNone(candidate["observed_cuda_launches"])
        self.assertEqual(linear.activation, ("gelu", "none"))

    def test_disabled_records_discovery_without_claiming_eligibility(self):
        ir, linear = self.lowered()
        before = list(ir.nodes)
        patterns, candidate = self.decide(ir, enabled=False)
        self.assertFalse(patterns)
        self.assertEqual(candidate["status"], "disabled")
        self.assertEqual(candidate["reason"], "disabled")
        self.assertTrue(candidate["discovered"])
        self.assertIsNone(candidate["eligible"])
        self.assertFalse(candidate["requested"] or candidate["effective_enabled"])
        self.assertEqual(ir.nodes, before)
        self.assertIsNone(linear.activation)

    def test_negative_decisions_have_stable_specific_reasons(self):
        for reason in ("no_activation", "device", "size", "layout", "gradient", "fanout",
                       "unsafe", "effects", "approximation", "codec", "metadata"):
            with self.subTest(reason=reason):
                ir, linear = self.lowered(
                    cuda=reason != "device", rows=3 if reason == "size" else 257,
                    activation=reason != "no_activation",
                )
                value = linear.args[0]
                producer = value.producer
                unsafe = frozenset()
                if reason == "layout":
                    producer.args[0].meta = replace(producer.args[0].meta, stride=(34, 2))
                elif reason == "gradient":
                    value.meta = replace(value.meta, requires_grad=False)
                elif reason == "fanout":
                    output = ir.nodes[-1]
                    output.result = output.result, value
                    register_uses(output)
                elif reason == "unsafe":
                    unsafe = frozenset((value,))
                elif reason == "effects":
                    effect = ExternKernelNode(
                        op="rand_like", target=torch.rand_like, args=producer.args, kwargs={},
                        value=TensorValue("random", value.meta),
                    )
                    ir.add(effect)
                    ir.nodes.remove(effect)
                    ir.nodes.insert(ir.nodes.index(producer) + 1, effect)
                elif reason == "approximation":
                    producer.kwargs["approximate"] = "unsupported"
                elif reason == "codec":
                    # Invalid static metadata is rejected before any codec executes.
                    linear.plan = {
                        **linear.plan,
                        "fp8_training": {
                            **linear.plan["fp8_training"],
                            "activation_compression": {
                                **linear.plan["fp8_training"]["activation_compression"],
                                "enabled": True, "group_size": 32768,
                            },
                        },
                    }
                elif reason == "metadata":
                    producer.args[0].meta = None
                patterns, candidate = self.decide(ir, unsafe_values=unsafe)
                self.assertFalse(patterns)
                self.assertEqual(candidate["status"], "rejected")
                self.assertEqual(candidate["reason"], reason)
                self.assertFalse(candidate["eligible"] or candidate["selected"])
                self.assertTrue(candidate["reason_detail"])
                self.assertIsNone(linear.activation)

    def test_backend_route_and_atomic_preparation_conflicts_are_explicit(self):
        for families in (
            {"compiler_pointwise": "tilelang", "compiler_preparation": "triton"},
            {"compiler_preparation": "triton", "quantization": "tilelang"},
        ):
            with self.subTest(families=families):
                ir, _linear = self.lowered()
                patterns, candidate = self.decide(
                    ir, kernel_policy={"backend": "mixed", "families": families},
                )
                self.assertFalse(patterns)
                self.assertEqual(candidate["reason"], "backend")

    def test_effective_options_and_unique_source_sites_round_trip(self):
        gm, sample = self.graph()
        report = CompilationReport()
        config = BackendConfig(cache_dir=Path("/tmp/deepopt-reporting"))
        compiled = compile_fx(gm, [sample], config=config, report=report)
        compiled(sample).float().sum().backward()
        gm.eval()
        with torch.no_grad():
            compile_fx(gm, [sample], config=config, report=report)
        payload = json.loads(json.dumps(report.as_dict(), allow_nan=False))
        self.assertFalse(payload["effective_compiler_options"][0]["fuse_deepopt_activations"])
        self.assertEqual(payload["effective_compiler_options"][0]["cache_dir"], "/tmp/deepopt-reporting")
        self.assertEqual(payload["effective_compiler_options"][0]["kernel_policy"]["backend"], "triton")
        self.assertEqual(payload["source_module_sites"], {"train": ["projection"], "eval": ["projection"]})
        self.assertEqual(len(payload["fusion_candidates"]), 2)
        self.assertEqual(payload["static_regions"]["linear_launch_programs"], 2)
        self.assertEqual(payload["static_regions"]["fused_preparation_regions"], 0)
        self.assertIsNone(payload["observed_cuda_launches"])
        self.assertIsNone(payload["graphs"][0]["observed_cuda_launches"])

    def test_native_delayed_recipe_retains_explicit_fusion_decision(self):
        gm, sample = self.graph(delayed=True)
        report = CompilationReport()
        compile_fx(gm, [sample], config=BackendConfig(fuse_deepopt_activations=True), report=report)
        candidate, = report.as_dict()["fusion_candidates"]
        self.assertEqual(candidate["module_path"], "projection")
        self.assertEqual(candidate["reason"], "fp8_recipe_requires_separate_preparation")
        self.assertIn("delayed", candidate["reason_detail"])
        self.assertTrue(candidate["discovered"])
        self.assertFalse(candidate["eligible"] or candidate["selected"])

    def test_profile_ranges_are_opt_in_and_label_the_actual_compiled_path(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch.dict(
                os.environ, {"DEEPOPT_PROFILE_RANGES": "1" if enabled else "0"},
            ):
                gm, sample = self.graph(activation=False)
                expected_model = deepcopy(gm)
                report = CompilationReport()
                compiled = compile_fx(gm, [sample], report=report)
                if enabled:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                        actual = compiled(sample)
                        actual.float().sum().backward()
                    labels = {event.key for event in profile.key_averages()}
                    self.assertIn("deepopt.linear::projection::forward", labels)
                    self.assertIn("deepopt.linear::projection::backward", labels)
                else:
                    with patch("torch.profiler.record_function", side_effect=AssertionError("unexpected profiling")):
                        actual = compiled(sample)
                        actual.float().sum().backward()
                expected = expected_model(sample.detach().clone())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(report.as_dict()["graphs"][0]["program_schedules"][0]["profile_ranges"], enabled)


if __name__ == "__main__":
    unittest.main()
