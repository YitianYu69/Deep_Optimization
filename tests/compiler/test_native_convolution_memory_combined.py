"""The two opt-in convolution changes compose under native AutoDiff."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.compiler.autodiff import convolution_geometry
from tests.compiler.test_native_convolution_geometry import _TwoConvolutions


def _check(test, device):
    torch.manual_seed(720)
    source = _TwoConvolutions().to(device)
    sample = torch.randn(2, 4, 7, 11, dtype=torch.bfloat16, device=device)
    results = []
    for enabled in (False, True):
        model = torch.fx.symbolic_trace(deepcopy(source))
        report, captures = CompilationReport(), []
        original = convolution_geometry.backward

        def observe(graph, context, **options):
            selected = original(graph, context, **options)
            captures.append(graph)
            return selected

        with patch.object(convolution_geometry, "backward", new=observe), autocast(
                device, dtype=torch.bfloat16, fp8=False,
                activation_compression=dict(enabled=True, bits=2, group_size=1024),
                bf16_policy="bf16_storage" if device == "cuda" else "torch_amp"):
            ShapeProp(model).propagate(sample)
            executable = compile_fx(model, [sample], config=BackendConfig(require_full_aot=True,
                native_conv_backward_order="dw_first" if enabled else "source",
                cuda_conv_dx_geometry=enabled), report=report)
            output = executable(sample)
        seed = torch.linspace(-.7, .9, output.numel(), device=device).to(output.dtype).reshape(output.shape)
        first = torch.autograd.grad(output, tuple(model.parameters()), seed, retain_graph=True)
        again = torch.autograd.grad(output, tuple(model.parameters()), seed)
        for actual, expected in zip(again, first):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        test.assertFalse(sample.requires_grad)
        test.assertIsNone(sample.grad)
        test.assertEqual(report.runtime_fallbacks, 0)
        test.assertFalse(report.as_dict()["legacy_backward_operations"])
        selections = report.as_dict()["fusion_candidates"]
        test.assertEqual(any(row["selected"] for row in selections if row["pattern"] == "native_conv_dw_first"), enabled)
        test.assertEqual(any(row["selected"] for row in selections if row["pattern"] == "native_conv_cuda_geometry"),
                         enabled and device == "cuda")
        calls = [node for node in captures[0].nodes if getattr(node, "op", "").startswith("convolution.native_")]
        dx = next(node for node in calls if node.op.startswith("convolution.native_dx_db"))
        dw = next(node for node in calls if node.op == "convolution.native_dw"
                  and node.target.specification is dx.target.specification)
        test.assertEqual(calls.index(dw) < calls.index(dx), enabled)
        if enabled and device == "cuda":
            test.assertEqual(dx.op, convolution_geometry.OPERATION)
            test.assertTrue(dx.scheduling_constraints["metadata_input_shape_only"])
            test.assertIsNone(dx.values[0].fresh_allocation)
        results.append((output.detach(), *first))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class NativeConvolutionMemoryCombinedTests(unittest.TestCase):
    def test_cpu_native_order_and_device_ineligible_geometry_preserve_parameter_vjp(self):
        _check(self, "cpu")

    def test_unqualified_geometry_and_donation_combination_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "qualified ownership composition"):
            BackendConfig(cuda_conv_dx_geometry=True, donate_fresh_backward_add=True)


@unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA combined native convolution execution")
class NativeConvolutionMemoryCombinedCudaTests(unittest.TestCase):
    def test_actual_cuda_combined_native_vjp_and_retained_backward(self):
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
            _check(self, "cuda")


if __name__ == "__main__":
    unittest.main()
