"""Keep FP8 primitives eager while compiling surrounding model operations."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt import autocast
from deepopt.opt.nn.autograd.linear import _DOLinearFunction
from deepopt.opt.nn.linear import DOLinear
from deepopt.opt.policy import prepare_fp8_model


def prepared_model():
    torch.manual_seed(419)
    # ELU stays native; sigmoid/tanh now have their own eager DO boundaries.
    model = nn.Sequential(nn.Linear(7, 11), nn.ELU(), nn.Linear(11, 3))
    return prepare_fp8_model(model, True)[0]


def record_compilation(model):
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    model.compile(backend=backend, fullgraph=False)
    return graphs


def contains_fp8_dtype(value):
    if isinstance(value, torch.dtype):
        return value in (torch.float8_e4m3fn, torch.float8_e5m2)
    if isinstance(value, dict):
        return any(contains_fp8_dtype(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(contains_fp8_dtype(item) for item in value)
    return False


class FP8CompileBoundaryBlackBoxTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_prepared_compiled_model_keeps_fp8_gradients_and_updates_equal_to_eager(self):
        model = prepared_model()
        self.assertIsInstance(model[0], DOLinear)
        self.assertIsInstance(model[2], DOLinear)
        baseline = deepcopy(model)
        graphs = record_compilation(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
        baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.03, momentum=0.9)

        for batch_size in (4, 5):
            with self.subTest(batch_size=batch_size):
                optimizer.zero_grad(set_to_none=True)
                baseline_optimizer.zero_grad(set_to_none=True)
                inputs = torch.randn(batch_size, 7, requires_grad=True)
                reference_inputs = inputs.detach().clone().requires_grad_()
                with autocast("cpu", enabled=False, fp8=True):
                    actual = model(inputs)
                    expected = baseline(reference_inputs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                upstream = torch.randn_like(actual)
                # Both backward passes deliberately execute after the FP8 scope.
                actual.backward(upstream)
                expected.backward(upstream)
                torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
                for name, parameter in model.named_parameters():
                    reference = baseline.get_parameter(name)
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
                    torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)
                optimizer.step()
                baseline_optimizer.step()
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter, baseline.get_parameter(name), rtol=0, atol=0)

        nodes = [node for graph in graphs for node in graph.graph.nodes]
        self.assertTrue(any(getattr(node.target, "__name__", node.target) == "elu" for node in nodes),
                        "The non-FP8 part of the model must still be compiled")
        # These operations belong to FP8 scaling/quantization, not this model's
        # ELU layer. Inspect real captured graphs without an instrumentation mock.
        for node in nodes:
            self.assertFalse(contains_fp8_dtype((node.args, node.kwargs)), str(node))
            self.assertNotIn(node.target, (torch.isfinite, torch.amax, torch.clamp, "amax", "clamp"))


class FP8CompileBoundaryWhiteBoxTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_custom_autograd_forward_runs_outside_dynamo_under_active_fp8(self):
        model = prepared_model()
        graphs = record_compilation(model)
        compiling = []
        original_forward = _DOLinearFunction.forward

        def observed_forward(ctx, *args):
            compiling.append(torch.compiler.is_compiling())
            return original_forward(ctx, *args)

        # A plain callable preserves the custom Function signature; no MagicMock
        # is put into the computational path. The black-box test above separately
        # proves the graph boundary without instrumentation affecting capture.
        with patch.object(_DOLinearFunction, "forward", new=staticmethod(observed_forward)):
            with autocast("cpu", enabled=False, fp8=True):
                output = model(torch.randn(4, 7, requires_grad=True))
            output.sum().backward()
        self.assertGreaterEqual(len(compiling), 2)
        self.assertFalse(any(compiling), f"FP8 autograd was traced by Dynamo: {compiling}")
        self.assertTrue(any(getattr(node.target, "__name__", node.target) == "elu"
                            for graph in graphs for node in graph.graph.nodes))


if __name__ == "__main__":
    unittest.main()
