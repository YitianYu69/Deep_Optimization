"""Preserve iterable container APIs and precision planning during module conversion."""

from copy import deepcopy
import unittest

import torch
from torch import nn

from deepopt import FP8Config
from deepopt.opt.policy import prepare_fp8_model
from deepopt.opt.precision import precision_report
from deepopt.opt.nn.linear import DOLinear


class IteratedBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16)))

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


class ContainerPrecisionPlanningTests(unittest.TestCase):
    def test_iterated_sequential_preserves_api_and_plans_fp8_edges_in_both_modes(self):
        original = IteratedBlocks()
        parameters = dict(original.named_parameters())
        keys = tuple(original.state_dict())
        prepared, _ = prepare_fp8_model(original, FP8Config(enabled=True, trace_mode="modules"))
        self.assertIs(type(prepared.blocks), nn.Sequential)
        self.assertIs(type(prepared.blocks[0]), nn.Sequential)
        self.assertEqual(len(list(prepared.blocks)), 1)
        self.assertIsInstance(prepared.blocks[0][0], DOLinear)
        self.assertEqual(tuple(prepared.state_dict()), keys)
        for name, parameter in prepared.named_parameters():
            self.assertIs(parameter, parameters[name])
        report = precision_report(prepared)
        self.assertFalse(report["fallbacks"])
        for mode in ("train", "eval"):
            self.assertTrue(any(edge["mode"] == mode and edge["precision"] == "fp8" for edge in report["edges"]))
        for training in (True, False):
            prepared.train(training)
            x = torch.randn(2, 4, 16, requires_grad=True)
            prepared(x).square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_shared_containers_parameters_and_native_root_methods_survive(self):
        source = IteratedBlocks()
        source.alias = source.blocks
        prepared, _ = prepare_fp8_model(source, FP8Config(enabled=True, trace_mode="modules"))
        self.assertIs(prepared.blocks, prepared.alias)
        self.assertIsInstance(prepared.blocks[0][2], DOLinear)
        x = torch.randn(3, 16)
        torch.testing.assert_close(prepared.blocks(x), prepared.alias(x), rtol=0, atol=0)

    def test_noninplace_conversion_preserves_source_container(self):
        from deepopt.opt import transform_model
        from deepopt.opt.policy import fp8_plan

        source = IteratedBlocks()
        before = deepcopy(source.state_dict())
        transformed, _ = transform_model(source, plan=fp8_plan(True), trace_mode="modules", inplace=False)
        self.assertIs(type(transformed.blocks), nn.Sequential)
        self.assertIs(type(source.blocks[0][0]), nn.Linear)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
