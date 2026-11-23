"""Compiler schedules preserve model-qualified module paths and phase metadata."""

import json
import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import FP8Config
from deepopt.compiler import CompilationReport, compilation_report, compile_fx
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model
from deepopt.opt.policy import prepare_fp8_model


class ModuleCoverageTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        torch.manual_seed(91)

    def test_captured_paths_distinguish_training_and_eval_graphs(self):
        model = prepare_fp8_model(
            nn.Sequential(nn.Linear(7, 11), nn.ELU(), nn.Linear(11, 3)),
            FP8Config(enabled=True),
        )[0]
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True))
        model(torch.randn(2, 7, requires_grad=True)).float().sum().backward()
        model.eval()
        with torch.no_grad():
            model(torch.randn(2, 7))
        payload = compilation_report(model).as_dict()
        schedules = [program for graph in payload["graphs"]
                     for program in graph["program_schedules"]]
        training = [program for program in schedules if program["module_training"]]
        evaluation = [program for program in schedules if not program["module_training"]]
        self.assertEqual({program["module_path"] for program in training}, {"0", "2"})
        self.assertEqual({program["module_path"] for program in evaluation}, {"0", "2"})
        self.assertTrue(all(program["training"] and program["backward"] for program in training))
        self.assertTrue(all(not program["training"] and not program["backward"] for program in evaluation))
        self.assertTrue(all(isinstance(program["node_name"], str) for program in schedules))
        json.dumps(payload)

    def test_root_linear_retains_empty_qualified_path(self):
        model = prepare_fp8_model(nn.Linear(7, 3), FP8Config(enabled=True))[0]
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True))
        model(torch.randn(2, 7)).sum().backward()
        schedule = compilation_report(model).as_dict()["graphs"][0]["program_schedules"][0]
        self.assertEqual(schedule["module_path"], "")
        self.assertTrue(schedule["module_training"])
        self.assertTrue(schedule["training"])

    def test_explicit_fx_call_module_retains_its_path(self):
        prepared = prepare_fp8_model(nn.Linear(7, 3), FP8Config(enabled=True))[0]
        root = nn.Module()
        root.encoder = nn.Module()
        root.encoder.projection = prepared
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(graph.call_module("encoder.projection", (x,)))
        model = torch.fx.GraphModule(root, graph)
        sample = torch.randn(2, 7, requires_grad=True)
        ShapeProp(model).propagate(sample)
        report = CompilationReport()
        compile_fx(model, [sample], report=report)(sample).sum().backward()
        schedule = report.as_dict()["graphs"][0]["program_schedules"][0]
        self.assertEqual(schedule["module_path"], "encoder.projection")
        self.assertTrue(schedule["training"])


if __name__ == "__main__":
    unittest.main()
