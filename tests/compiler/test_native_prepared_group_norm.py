"""Prepared GroupNorm preserves eager's native and compressed saved recipes."""

from copy import deepcopy
from contextlib import nullcontext
import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.opt.nn.normalization.modules import DOGroupNorm, DOFunctionalNormalization


class PreparedGroupNormTests(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        torch.manual_seed(541)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def check(self, *, compressed, bits=2, affine=True, needs=(True, True, True),
              dtype=torch.bfloat16, affine_dtype=torch.bfloat16, channels_last=False, amp=False,
              shape=(2, 4, 16, 32), groups=2):
        compression = dict(enabled=compressed, bits=bits, group_size=1024, norm_xhat_recenter=True)
        module = DOGroupNorm(nn.GroupNorm(groups, shape[1], affine=affine).to(self.device, affine_dtype),
                             plan=dict(fp8=False, activation_compression=compression))
        if affine:
            module.weight.requires_grad_(needs[1])
            module.bias.requires_grad_(needs[2])
        root = nn.Module()
        root.add_module("norm", module)
        graph = torch.fx.Graph()
        value = graph.placeholder("x")
        graph.output(graph.call_module("norm", (value,)))
        model = torch.fx.GraphModule(root, graph)
        reference = deepcopy(model)
        x = torch.randn(shape, device=self.device, dtype=dtype)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x.requires_grad_(needs[0])
        ref = x.detach().clone().requires_grad_(needs[0])
        scope = (autocast(self.device, dtype=torch.bfloat16, fp8=False, activation_compression=compression)
                 if amp else nullcontext())
        report = CompilationReport()
        saved = []
        with scope:
            ShapeProp(model).propagate(x)
            compiled = compile_fx(model, [x], config=BackendConfig(require_full_aot=True,
                fuse_deepopt_activations=False), report=report)
            expected = reference(ref)
            with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t.dtype) or t, lambda t: t):
                actual = compiled(x)
        # Compare the same recipe, never compressed against uncompressed.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if actual.requires_grad:
            gradient = torch.randn_like(actual)
            actual.backward(gradient)
            expected.backward(gradient)
            if needs[0]:
                torch.testing.assert_close(x.grad, ref.grad, rtol=0, atol=0)
            if affine:
                for value, control in zip(model.parameters(), reference.parameters()):
                    if value.requires_grad:
                        torch.testing.assert_close(value.grad, control.grad, rtol=0, atol=0)
        if compressed and (needs[0] or affine and needs[1]):
            self.assertIn(torch.int32, saved, "Positive group1024 fixture must pack normalized values")
        for graph in report.graphs:
            self.assertFalse(graph.eager_autograd_operations)
            self.assertFalse(graph.legacy_backward_operations)

    def test_uncompressed_prepared_matches_native_statistics_and_gradient_masks(self):
        for dtype, affine_dtype in ((torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32),
                                    (torch.float32, torch.float32), (torch.float64, torch.float64)):
            for needs in ((True, True, True), (False, True, True), (True, False, False),
                          (False, False, True), (False, False, False)):
                for channels_last in (False, True):
                    with self.subTest(dtype=dtype, affine_dtype=affine_dtype, needs=needs, channels_last=channels_last):
                        self.check(compressed=False, dtype=dtype, affine_dtype=affine_dtype,
                                   needs=needs, channels_last=channels_last)
        self.check(compressed=False, affine=False)

    def test_compressed_recipe_remains_shared_and_packs_group1024(self):
        for bits in (1, 2, 4, 8):
            for channels_last in (False, True):
                with self.subTest(bits=bits, channels_last=channels_last):
                    self.check(compressed=True, bits=bits, channels_last=channels_last)

    def test_functional_uncompressed_groups_and_eps_remain_per_call(self):
        root = nn.Module()
        root.norm = DOFunctionalNormalization(dict(fp8=False,
            activation_compression=dict(enabled=False, bits=2, group_size=1024)), "group")
        root.weight = nn.Parameter(torch.randn(4, device=self.device, dtype=torch.bfloat16))
        root.bias = nn.Parameter(torch.randn(4, device=self.device, dtype=torch.bfloat16))
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        weight, bias = graph.get_attr("weight"), graph.get_attr("bias")
        outputs = tuple(graph.call_module("norm", (x, groups, weight, bias), dict(eps=eps))
                        for groups, eps in ((2, .001), (1, .0001)))
        graph.output(outputs)
        model = torch.fx.GraphModule(root, graph)
        reference = deepcopy(model)
        value = torch.randn(2, 4, 16, 32, device=self.device, dtype=torch.bfloat16, requires_grad=True)
        ref = value.detach().clone().requires_grad_()
        ShapeProp(model).propagate(value)
        compiled = compile_fx(model, [value], config=BackendConfig(require_full_aot=True))
        actual, expected = compiled(value), reference(ref)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        sum(a.sum() for a in actual).backward()
        sum(a.sum() for a in expected).backward()
        torch.testing.assert_close(value.grad, ref.grad, rtol=0, atol=0)
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        self.assertFalse(hasattr(root.norm, "num_groups"))


@unittest.skipUnless(torch.cuda.is_available(), "Prepared GroupNorm CUDA qualification requires CUDA")
class PreparedGroupNormCUDATests(PreparedGroupNormTests):
    device = "cuda"

    def test_cuda_amp_fp32_output_matches_prepared_eager(self):
        for compressed in (False, True):
            self.check(compressed=compressed, dtype=torch.float32, affine_dtype=torch.float32, amp=True)

    def test_archived_dense_geometry_matches_eager_before_fusion(self):
        # Preserve the representative full dense shape; CPU tests stay tiny.
        for channels_last in (False, True):
            self.check(compressed=False, shape=(32, 768, 16, 16), groups=32,
                       channels_last=channels_last)


if __name__ == "__main__":
    unittest.main()
