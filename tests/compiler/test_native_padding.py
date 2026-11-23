"""Strict native ownership and metadata-only state for padding VJPs."""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F

from deepopt.compiler import BackendConfig, CompilationReport, compile_fx


@contextmanager
def _strict_algorithms():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _zero_native_vjp(test, *, device, dtype, original224):
    # Direct FX -> DeepOpt native AutoDiff tests the shared adjoint, separately
    # from Dynamo's strict-mode nonstrict_trace capture contract.
    class Model(nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode

        def forward(self, x):
            return F.pad(x, (0, 0, 0, 0), mode=self.mode)

    shape = (1, 3, 224, 224) if original224 else (2, 3, 5, 7)
    source = torch.randn(shape, dtype=dtype, device=device).contiguous(memory_format=torch.channels_last)
    source.requires_grad_()
    gradient = torch.randn((*shape[:-1], shape[-1] * 2), dtype=dtype, device=device)[..., ::2]
    for mode in ("reflect", "replicate"):
        with test.subTest(mode=mode, dtype=dtype):
            graph = torch.fx.symbolic_trace(Model(mode))
            ShapeProp(graph).propagate(source)
            report = CompilationReport()
            compiled = compile_fx(graph, [source], config=BackendConfig(require_full_aot=True), report=report)
            region, = compiled.backend_artifact.joint_autograd_regions
            test.assertEqual(region.trace.saved_tensors, ())
            test.assertEqual(region.trace.saved_value_specs[0]["representation"], "metadata_only")
            test.assertTrue(any(getattr(node, "op", None) == "padding.metadata_vjp"
                                for node in region.trace.graph.nodes))
            output = compiled(source)
            actual, = torch.autograd.grad(output, source, gradient)
            torch.testing.assert_close(output, source, rtol=0, atol=0)
            torch.testing.assert_close(actual, gradient, rtol=0, atol=0)
            test.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))
            test.assertNotEqual(actual.data_ptr(), gradient.data_ptr())
            test.assertFalse(report.graphs[0].eager_autograd_operations)
            test.assertFalse(report.graphs[0].legacy_backward_operations)
            test.assertTrue(torch.are_deterministic_algorithms_enabled())
            test.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())


class NativePaddingTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_zero_padding_native_vjp_avoids_native_rejection_under_strict_policy(self):
        rejection = RuntimeError("padding_backward_cuda does not have a deterministic implementation")
        with _strict_algorithms(), \
                patch.object(torch.ops.aten.reflection_pad2d_backward, "default", side_effect=rejection) as reflect, \
                patch.object(torch.ops.aten.replication_pad2d_backward, "default", side_effect=rejection) as replicate:
            for dtype in (torch.float32, torch.bfloat16):
                _zero_native_vjp(self, device="cpu", dtype=dtype, original224=False)
            self.assertEqual(reflect.call_count, 0)
            self.assertEqual(replicate.call_count, 0)

    def test_nonzero_native_padding_keeps_backend_strict_rejection(self):
        class Model(nn.Module):
            def forward(self, x):
                return F.pad(x, (-1, 1, 0, 0), mode="replicate")

        graph = torch.fx.symbolic_trace(Model())
        source = torch.randn(2, 3, 5, 7, requires_grad=True)
        with _strict_algorithms():
            ShapeProp(graph).propagate(source)
            compiled = compile_fx(graph, [source], config=BackendConfig(require_full_aot=True))
            with patch.object(torch.ops.aten.replication_pad2d_backward, "default",
                    side_effect=RuntimeError("fixture does not have a deterministic implementation")) as backward:
                with self.assertRaisesRegex(RuntimeError, "does not have a deterministic implementation"):
                    compiled(source).sum().backward()
                backward.assert_called_once()
                self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_native_padding_modes_have_no_source_valued_residual(self):
        for mode in ("constant", "reflect", "replicate", "circular"):
            with self.subTest(mode=mode):
                class Model(nn.Module):
                    def forward(self, x):
                        return F.pad(x, (-1, 2, 1, -1), mode=mode)

                graph = torch.fx.symbolic_trace(Model())
                x = torch.randn(2, 3, 5, 7, requires_grad=True)
                reference = x.detach().clone().requires_grad_()
                ShapeProp(graph).propagate(x)
                report = CompilationReport()
                compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True), report=report)
                region, = compiled.backend_artifact.joint_autograd_regions
                self.assertEqual(region.trace.saved_tensors, ())
                self.assertEqual(region.trace.saved_value_specs[0]["representation"], "metadata_only")
                saved = []
                with torch.autograd.graph.saved_tensors_hooks(lambda value: saved.append(value) or value, lambda value: value):
                    actual = compiled(x)
                expected = graph(reference)
                self.assertEqual(saved, [])
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
                expected.backward(gradient)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, reference.grad, rtol=1e-6, atol=1e-6)
                self.assertFalse(report.graphs[0].eager_autograd_operations)
                self.assertFalse(report.graphs[0].legacy_backward_operations)

    def test_frozen_convolution_masks_keep_only_exact_weight_when_needed(self):
        from deepopt import autocast

        for need_dx, need_db in ((True, False), (False, True), (True, True)):
            with self.subTest(need_dx=need_dx, need_db=need_db):
                layer = nn.Conv2d(3, 4, 3, padding=1, padding_mode="reflect")
                layer.weight.requires_grad_(False)
                layer.bias.requires_grad_(need_db)
                graph = torch.fx.symbolic_trace(nn.Sequential(layer))
                x = torch.randn(2, 3, 5, 7, requires_grad=need_dx)
                reference = x.detach().clone().requires_grad_(need_dx)
                with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                              activation_compression={"enabled": True, "bits": 2, "group_size": 1024}, config={}):
                    ShapeProp(graph).propagate(x)
                    compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True))
                    actual = compiled(x)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    expected = graph(reference)
                region, = compiled.backend_artifact.joint_autograd_regions
                self.assertEqual(len(region.trace.saved_tensors), int(need_dx))
                self.assertFalse(any(node.op == "codec.save_convolution_source" for node in region.graph.nodes if hasattr(node, "op")))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
                expected_dx = torch.autograd.grad(expected, reference, gradient)[0] if need_dx else None
                if need_dx:
                    torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)

    def test_convolution_source_pack_is_explicit_typed_and_consumer_attributed(self):
        from deepopt import autocast
        from deepopt.compiler.ir import CallNode, TensorMeta

        graph = torch.fx.symbolic_trace(nn.Sequential(nn.Conv2d(3, 4, 3, padding=1)))
        x = torch.randn(2, 3, 17, 19, requires_grad=True)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                      activation_compression={"enabled": True, "bits": 2, "group_size": 1024}, config={}):
            ShapeProp(graph).propagate(x)
            compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True))
            actual = compiled(x)
        region, = compiled.backend_artifact.joint_autograd_regions
        pack, = [node for node in region.graph.nodes if isinstance(node, CallNode) and node.op == "codec.save_convolution_source"]
        self.assertEqual(len(pack.values), 3)
        self.assertTrue(all(isinstance(value.meta, TensorMeta) for value in pack.values))
        self.assertEqual(pack.values[0].meta.dtype, torch.int32)
        self.assertEqual(pack.specification["source_cast"], torch.bfloat16)
        binding, = [record for record in region.trace.saved_value_specs if record["representation"] == "packed_convolution_source"]
        self.assertEqual(binding["components"], tuple(value.name for value in pack.values))
        self.assertEqual(binding["consumers"][0]["role"], "weight_gradient_coefficient")
        actual.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(graph.get_submodule("0").weight.grad)

    def test_aten_reflection_padding_has_native_derivative(self):
        class Model(nn.Module):
            def forward(self, x):
                return torch.ops.aten.reflection_pad2d.default(x, [1, 2, 2, 1])

        graph = torch.fx.symbolic_trace(Model())
        x = torch.randn(2, 3, 5, 7, requires_grad=True)
        reference = x.detach().clone().requires_grad_()
        ShapeProp(graph).propagate(x)
        compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True))
        compiled(x).sum().backward()
        graph(reference).sum().backward()
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)

    def test_padding_modules_keep_native_ownership_and_metadata_only_state(self):
        for layer in (nn.ReflectionPad2d(1), nn.ReplicationPad2d(1), nn.CircularPad2d(1), nn.ConstantPad2d(1, 2)):
            with self.subTest(layer=type(layer).__name__):
                graph = torch.fx.symbolic_trace(nn.Sequential(layer))
                x = torch.randn(2, 3, 5, 7, requires_grad=True)
                reference = x.detach().clone().requires_grad_()
                ShapeProp(graph).propagate(x)
                compiled = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True))
                region, = compiled.backend_artifact.joint_autograd_regions
                self.assertEqual(region.trace.saved_tensors, ())
                compiled(x).sum().backward()
                graph(reference).sum().backward()
                torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "Strict native zero-padding CUDA regression requires a GPU")
class NativePaddingStrictZeroCudaTests(unittest.TestCase):
    def setUp(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_actual_original224_native_zero_padding_vjp_obeys_strict_policy(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 support required")
        with _strict_algorithms():
            for dtype in (torch.float32, torch.bfloat16):
                _zero_native_vjp(self, device="cuda", dtype=dtype, original224=True)
            torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
