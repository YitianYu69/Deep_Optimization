"""Hardware gates for generated pointwise kernels and prepared FP8 execution."""

from copy import deepcopy
import importlib.util
import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import CUDAGraphCore, FP8Config, autocast
from deepopt.compiler import CompilationReport, compilation_report, compile_fx
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model
from deepopt.opt.policy import prepare_fp8_model


HAS_CUDA = torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


class Pointwise(nn.Module):
    def forward(self, x):
        return torch.sigmoid(torch.relu(x * 1.5 + 0.25))


@unittest.skipUnless(HAS_CUDA, "CUDA and Triton are required")
class CompilerCUDATests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_fused_pointwise_values_gradients_and_tail_masks(self):
        for dtype in (torch.float32, torch.bfloat16):
            model = compile_model(
                Pointwise().cuda(),
                CompileConfig(enabled=True, backend="deepopt", fullgraph=True),
            )
            x = torch.randn(5, 257, device="cuda", dtype=dtype, requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            actual, expected = model(x), Pointwise()(xr)
            actual.sum().backward()
            expected.sum().backward()
            tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-6
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(x.grad, xr.grad, rtol=tolerance, atol=tolerance)
            self.assertGreater(compilation_report(model).triton_regions, 0)
            self.assertEqual(compilation_report(model).runtime_fallbacks, 0)

    def test_compiled_callable_cuda_graph_replay_does_not_compile_again(self):
        sample = torch.randn(5, 257, device="cuda", requires_grad=True)
        graph = torch.fx.symbolic_trace(Pointwise())
        ShapeProp(graph).propagate(sample)
        report = CompilationReport()
        compiled = compile_fx(graph, [sample], report=report)
        handle = CUDAGraphCore("cuda").capture_callable(
            compiled, (sample,), warmup=3, autograd=True
        )
        try:
            x = torch.randn_like(sample, requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            actual, expected = handle(x), Pointwise()(xr)
            actual.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
            torch.testing.assert_close(x.grad, xr.grad, rtol=2e-6, atol=2e-6)
            self.assertEqual(handle.statistics["forward_replays"], 1)
            self.assertEqual(report.runtime_fallbacks, 0)
        finally:
            handle.close()

    def test_activation_options_nonfinite_values_and_signed_zeros(self):
        sample = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, -1e-8, 1e-8, 0.2],
            device="cuda",
        )
        for activation in (
            nn.ReLU(),
            nn.Tanh(),
            nn.GELU(approximate="tanh"),
            nn.GELU(),
            nn.Hardtanh(0.0, 0.4),
            nn.LeakyReLU(-0.3),
        ):
            torch.compiler.reset()
            reference = deepcopy(activation)
            compile_model(
                activation,
                CompileConfig(enabled=True, backend="deepopt", fullgraph=True),
            )
            actual, expected = activation(sample), reference(sample)
            torch.testing.assert_close(
                actual, expected, rtol=2e-6, atol=2e-6, equal_nan=True
            )
            zeros = expected == 0
            torch.testing.assert_close(
                torch.signbit(actual[zeros]), torch.signbit(expected[zeros])
            )

    def test_prepared_fp8_keeps_existing_executor_and_saved_state(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("FP8 Triton GEMM requires compute capability 9+")
        settings = FP8Config(enabled=True, backend="triton")
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                torch.compiler.reset()
                model = prepare_fp8_model(
                    nn.Sequential(nn.Linear(128, 129), nn.SiLU(), nn.Linear(129, 128)).cuda(),
                    settings,
                    activation_compression={"enabled": compressed, "bits": 2, "group_size": 32},
                )[0]
                reference = deepcopy(model)
                compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True))
                # Three-dimensional token batches and an unaligned output
                # width exercise the capture path used by ViT projections.
                x = torch.randn(2, 17, 128, device="cuda", requires_grad=True)
                xr = x.detach().clone().requires_grad_()
                with autocast("cuda", dtype=torch.bfloat16, fp8=settings, check_finite=False):
                    actual, expected = model(x), reference(xr)
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
                for parameter, baseline in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(parameter.grad, baseline.grad, rtol=0, atol=0)
                self.assertGreater(compilation_report(model).deepopt_operations, 0)
                self.assertEqual(compilation_report(model).runtime_fallbacks, 0)

    def test_experimental_native_gelu_linear_fusion_executes_generated_templates(self):
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("FP8 GEMM requires compute capability 9+")

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(128, 128)

            def forward(self, x):
                y = nn.functional.gelu(x)
                return self.projection(nn.functional.dropout(y, p=0.0, training=self.training))

        settings = FP8Config(enabled=True, backend="triton", intercept_linear=False, out_dtype="bf16", autotune=False)
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                torch.compiler.reset()
                model = prepare_fp8_model(
                    MLP().cuda(), settings,
                    activation_compression={"enabled": compressed, "bits": 2, "group_size": 512},
                )[0]
                reference = deepcopy(model)
                compile_model(model, CompileConfig(
                    enabled=True, backend="deepopt", fullgraph=True,
                    options={"fuse_deepopt_activations": True},
                ))
                x = torch.randn(2, 17, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                xr = x.detach().clone().requires_grad_()
                with autocast("cuda", dtype=torch.bfloat16, fp8=settings, activation_compression=None, check_finite=False):
                    actual, expected = model(x), reference(xr)
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
                torch.testing.assert_close(x.grad, xr.grad, rtol=2e-2, atol=2e-2)
                for a, b in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, rtol=2e-2, atol=2e-2)
                report = compilation_report(model)
                self.assertEqual(report.lowered_deepopt_operations, 1)
                self.assertEqual(report.fused_deepopt_regions, 1)
                self.assertEqual(report.compiled_deepopt_backward_regions, 1)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                    with autocast("cuda", dtype=torch.bfloat16, fp8=settings, activation_compression=None, check_finite=False):
                        model(x)
                    torch.cuda.synchronize()
                self.assertTrue(any("activation_amax" in event.name for event in profile.events()))
                self.assertFalse(any(event.key == "aten::gelu" for event in profile.key_averages()))


if __name__ == "__main__":
    unittest.main()
