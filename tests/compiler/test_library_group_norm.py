"""Raw uncompressed GroupNorm preserves PyTorch's native numerical algorithm."""

from copy import deepcopy
import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.compiler.autodiff.library_group_norm import LibraryGroupNorm


class LibraryGroupNormTests(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        torch.manual_seed(719)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def check(self, dtype, parameter_dtype, layout, mask=(True, True, True), affine=True):
        model = nn.Sequential(nn.GroupNorm(2, 4, affine=affine, dtype=parameter_dtype, device=self.device))
        if affine:
            model[0].weight.requires_grad_(mask[1])
            model[0].bias.requires_grad_(mask[2])
        reference = deepcopy(model)
        x = torch.randn(2, 4, 5, 7, device=self.device, dtype=dtype)
        if layout == "channels_last":
            x = x.contiguous(memory_format=torch.channels_last)
        elif layout == "strided":
            x = x.transpose(-1, -2)
        x.requires_grad_(mask[0])
        rx = x.detach().clone().requires_grad_(mask[0])
        graph = torch.fx.symbolic_trace(model)
        report = CompilationReport()
        with autocast(self.device, dtype=torch.bfloat16, fp8=False, activation_compression=False, config={}):
            ShapeProp(graph).propagate(x)
            compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True), report=report)
            actual, expected = compiled(x), reference(rx)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.stride(), expected.stride())
        gradient = torch.randn_like(actual)
        if actual.requires_grad:
            actual.backward(gradient)
            expected.backward(gradient)
            torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
            for parameter, control in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter.grad, control.grad, rtol=0, atol=0)
            region, = compiled.backend_artifact.joint_autograd_regions
            self.assertIsInstance(region.specifications[0], LibraryGroupNorm)
            for source in region.trace.saved_tensors:
                if source.name.endswith(("__native_mean", "__native_rstd")):
                    expected_stats_dtype = actual.dtype if self.device == "cuda" else parameter_dtype if affine else actual.dtype
                    self.assertEqual(source.meta.dtype, expected_stats_dtype)
                    self.assertEqual(source.meta.shape, (2, 2))
            if mask == (False, False, True):
                self.assertEqual(region.trace.saved_tensors, ())
        self.assertFalse(report.graphs[-1].eager_autograd_operations)
        self.assertFalse(report.graphs[-1].legacy_backward_operations)

    def test_native_forward_backward_dtypes_and_layouts_are_exact(self):
        for dtype, parameter_dtype in ((torch.bfloat16, torch.float32), (torch.bfloat16, torch.bfloat16),
                                       (torch.float32, torch.float32), (torch.float64, torch.float64)):
            for layout in ("contiguous", "channels_last", "strided"):
                with self.subTest(dtype=dtype, parameter_dtype=parameter_dtype, layout=layout):
                    self.check(dtype, parameter_dtype, layout)

    def test_selective_gradients_and_affine_free_execution(self):
        for mask in ((True, False, False), (False, True, False), (False, False, True), (False, False, False)):
            for dtype in (torch.bfloat16, torch.float32, torch.float64):
                for layout in ("contiguous", "channels_last", "strided"):
                    with self.subTest(mask=mask, dtype=dtype, layout=layout):
                        self.check(dtype, torch.float32 if dtype == torch.bfloat16 else dtype, layout, mask)
        self.check(torch.bfloat16, torch.bfloat16, "channels_last", affine=False)


@unittest.skipUnless(torch.cuda.is_available(), "Native GroupNorm CUDA AMP qualification requires CUDA")
class LibraryGroupNormCudaTests(LibraryGroupNormTests):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
