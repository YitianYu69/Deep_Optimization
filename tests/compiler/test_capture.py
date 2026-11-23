"""Prepared operation capture must retain numerical policy and saved state."""

from copy import deepcopy
from contextlib import nullcontext
import unittest

import torch
from torch import nn
from torch.utils._pytree import tree_leaves

from deepopt import autocast
from deepopt.compiler import compilation_report
from deepopt.config import ActivationCompressionConfig, CompileConfig, FP8Config
from deepopt.engine.execution.compilation import compile_model
from deepopt.opt.policy import prepare_fp8_model


class CaptureTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        torch.manual_seed(120)

    def compare(
        self,
        original,
        shape,
        *,
        fp8=True,
        compression=False,
        recipe="current",
        options=None,
        fullgraph=False,
        backward_precision="native",
    ):
        prepared = getattr(original, "_deepopt_fp8_config", None)
        settings = FP8Config(
            enabled=fp8,
            scaling_recipe=recipe,
            backward_precision=backward_precision,
            trace_mode=prepared.trace_mode if prepared is not None else "modules",
        )
        model = prepare_fp8_model(
            original,
            settings,
            activation_compression=ActivationCompressionConfig(
                enabled=compression, bits=4, group_size=1024
            ),
        )[0]
        reference = deepcopy(model)
        identities = [id(parameter) for parameter in model.parameters()]
        compile_model(
            model,
            CompileConfig(
                enabled=True,
                backend="deepopt",
                dynamic=False,
                fullgraph=fullgraph,
                options=options or {},
            ),
        )
        for size in (2, 3):
            x = torch.randn((size, *shape), requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            context = autocast("cpu", enabled=False, fp8=True) if fp8 else nullcontext()
            with context:
                actual, expected = model(x), reference(xr)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual_tensors = [
                value
                for value in tree_leaves(actual)
                if isinstance(value, torch.Tensor) and value.requires_grad
            ]
            expected_tensors = [
                value
                for value in tree_leaves(expected)
                if isinstance(value, torch.Tensor) and value.requires_grad
            ]
            sum(value.sum() for value in actual_tensors).backward()
            sum(value.sum() for value in expected_tensors).backward()
            torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(
                    parameter.grad, reference.get_parameter(name).grad, rtol=0, atol=0
                )
            for name, buffer in model.named_buffers():
                torch.testing.assert_close(
                    buffer, reference.get_buffer(name), rtol=0, atol=0
                )
        self.assertEqual(
            identities, [id(parameter) for parameter in model.parameters()]
        )
        self.assertGreater(compilation_report(model).deepopt_operations, 0)
        self.assertEqual(compilation_report(model).runtime_fallbacks, 0)
        return model, reference

    def test_linear_capture_and_optimizers_match_with_backward_outside_autocast(self):
        model, reference = self.compare(
            nn.Sequential(nn.Linear(7, 11), nn.ELU(), nn.Linear(11, 3)), (7,)
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
        baseline = torch.optim.SGD(reference.parameters(), lr=0.03, momentum=0.9)
        optimizer.step()
        baseline.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(compilation_report(model).deepopt_operations, 4)

    def test_root_linear_and_tied_weights(self):
        self.compare(nn.Linear(7, 3), (7,))

        class Tied(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Linear(7, 7)
                self.second = self.first

            def forward(self, x):
                return self.second(self.first(x))

        self.compare(Tied(), (7,))

    def test_activations_layernorm_and_compression(self):
        for activation in (
            nn.ReLU(),
            nn.ReLU6(),
            nn.Sigmoid(),
            nn.SiLU(),
            nn.GELU(approximate="tanh"),
            nn.Tanh(),
        ):
            with self.subTest(activation=type(activation).__name__):
                torch.compiler.reset()
                self.compare(
                    nn.Sequential(
                        nn.Linear(8, 8), activation, nn.LayerNorm(8), nn.Linear(8, 3)
                    ),
                    (65, 8),
                    compression=True,
                )

    def test_spatial_operations_batchnorm_and_pooling_do_not_advance_state_during_compile(
        self,
    ):
        self.compare(
            nn.Sequential(
                nn.Conv2d(3, 4, 3, padding=1),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(2),
                nn.ConvTranspose2d(4, 3, 2, stride=2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(3, 2),
            ),
            (3, 16, 16),
            compression=True,
        )

    def test_delayed_scale_buffers_match_eager(self):
        self.compare(
            nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.LayerNorm(8), nn.Linear(8, 3)),
            (65, 8),
            recipe="delayed",
            compression=True,
        )

    def test_native_compression_retains_its_own_autograd(self):
        self.compare(
            nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.LayerNorm(8), nn.Linear(8, 3)),
            (65, 8),
            fp8=False,
            compression=True,
        )

    def test_functional_projection_preparation_and_pooling_indices(self):
        class Functional(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(5, 7))

            def forward(self, x):
                return nn.functional.linear(x, self.weight).sigmoid()

        model = prepare_fp8_model(
            Functional(),
            FP8Config(enabled=True, trace_mode="full"),
            activation_compression=ActivationCompressionConfig(
                enabled=False, bits=4, group_size=1024
            ),
        )[0]
        self.compare(model, (7,))
        self.assertTrue(
            any(
                "linear" in name
                for graph in compilation_report(model).graphs
                for name in graph.deepopt_operations
            )
        )
        ordinary = deepcopy(model)
        compile_model(
            ordinary,
            CompileConfig(
                enabled=True, backend="deepopt", options={"capture_deepopt": False}
            ),
        )
        with autocast("cpu", enabled=False, fp8=True):
            ordinary(torch.randn(2, 7))
        self.assertEqual(compilation_report(ordinary).deepopt_operations, 0)
        cloned = deepcopy(model)
        compile_model(cloned, CompileConfig(enabled=True, backend="eager"))
        with autocast("cpu", enabled=False, fp8=True):
            x = torch.randn(2, 7)
            torch.testing.assert_close(cloned(x), model(x), rtol=0, atol=0)
        torch.compiler.reset()
        self.compare(
            nn.Sequential(nn.Conv2d(3, 4, 1), nn.MaxPool2d(2, return_indices=True)),
            (3, 4, 4),
        )

    def test_context_overrides_and_train_eval_transitions_remain_runtime_decisions(
        self,
    ):
        model = prepare_fp8_model(
            nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.SiLU()), True
        )[0]
        reference = deepcopy(model)
        compile_model(model, CompileConfig(enabled=True, backend="deepopt"))
        for training, enabled in (
            (True, True),
            (False, False),
            (True, False),
            (False, True),
        ):
            model.train(training)
            reference.train(training)
            x = torch.randn(3, 8)
            with autocast("cpu", enabled=False, fp8=enabled):
                actual, expected = model(x), reference(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for name, value in model.named_buffers():
                torch.testing.assert_close(
                    value, reference.get_buffer(name), rtol=0, atol=0
                )

    def test_attention_native_decomposition_matches_existing_recipe(self):
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    8, 2, dropout=0, batch_first=True
                )

            def forward(self, x):
                return self.attention(x, x, x, need_weights=False)[0]

        self.compare(Attention(), (65, 8), compression=True)

    def test_planned_fp8_gradient_transport_and_groupnorm(self):
        self.compare(
            nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.LayerNorm(8), nn.Linear(8, 3)),
            (65, 8),
            backward_precision="auto",
            compression=True,
        )
        torch.compiler.reset()
        self.compare(
            nn.Sequential(
                nn.Conv2d(4, 4, 1), nn.GroupNorm(2, 4), nn.AdaptiveAvgPool2d(1)
            ),
            (4, 17, 17),
            compression=True,
        )

    def test_deepcopy_recovers_eager_forward_and_independent_parameters(self):
        model, _ = self.compare(nn.Sequential(nn.Linear(7, 7), nn.Tanh()), (7,))
        cloned = deepcopy(model)
        self.assertIsNone(cloned._compiled_call_impl)
        self.assertEqual(compilation_report(cloned).captured_graphs, 0)
        self.assertFalse(getattr(cloned[0].forward, "_deepopt_compiler_capture", False))
        before = model[0].weight.detach().clone()
        with torch.no_grad():
            cloned[0].weight.add_(1)
        torch.testing.assert_close(model[0].weight, before, rtol=0, atol=0)

    def test_failed_setup_restores_eager_forwards_and_removes_its_report(self):
        class Reject(nn.Sequential):
            def compile(self, **kwargs):
                raise RuntimeError("construction rejected")

        model = prepare_fp8_model(Reject(nn.Linear(7, 3)), True)[0]
        with self.assertRaisesRegex(RuntimeError, "construction rejected"):
            compile_model(model, CompileConfig(enabled=True, backend="deepopt"))
        self.assertFalse(getattr(model[0].forward, "_deepopt_compiler_capture", False))
        self.assertIsNone(compilation_report(model))

    def test_capture_is_optional_and_strict_aot_does_not_misreport_do_autograd(self):
        model = prepare_fp8_model(nn.Sequential(nn.Linear(7, 3), nn.ELU()), True)[0]
        compile_model(
            model,
            CompileConfig(
                enabled=True, backend="deepopt", options={"capture_deepopt": False}
            ),
        )
        self.assertFalse(getattr(model[0].forward, "_deepopt_compiler_capture", False))
        strict = prepare_fp8_model(nn.Sequential(nn.Linear(7, 3), nn.ELU()), True)[0]
        compile_model(
            strict,
            CompileConfig(
                enabled=True, backend="deepopt", options={"require_full_aot": True}
            ),
        )
        with self.assertRaisesRegex(Exception, "autograd"):
            with autocast("cpu", enabled=False, fp8=True):
                strict(torch.randn(2, 7, requires_grad=True))


if __name__ == "__main__":
    unittest.main()
