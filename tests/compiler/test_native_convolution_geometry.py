"""CUDA geometry selection is strict; CPU tests make no launch claim."""

from contextlib import nullcontext
from copy import copy, deepcopy
from dataclasses import replace
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.compiler.autodiff import convolution_geometry
from deepopt.compiler.autodiff.semantic import recorded
from deepopt.compiler.ir import (CallNode, FreshAllocationContract, InputNode,
                                 IRGraph, OutputNode, TensorMeta, TensorValue)
from deepopt.compiler.trace.metadata_utils import contiguous_stride
from deepopt.opt.backends.native.convolution import ops
from deepopt.opt.nn.convolution.geometry import ConvGeometry


class _TwoConvolutions(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv2d(4, 6, 1, bias=False)
        self.second = nn.Conv2d(6, 3, 1, bias=False)

    def forward(self, x):
        return self.second(self.first(x))


def _compile(source, sample, enabled):
    model = torch.fx.symbolic_trace(deepcopy(source))
    report, graphs = CompilationReport(), []
    original = convolution_geometry.backward

    def observe(graph, context, **options):
        result = original(graph, context, **options)
        graphs.append(graph)
        return result

    scope = (autocast(sample.device.type, dtype=torch.bfloat16, fp8=False,
                     activation_compression=dict(enabled=False, bits=2, group_size=1024),
                     bf16_policy="bf16_storage" if sample.device.type == "cuda" else "torch_amp")
             if sample.dtype == torch.bfloat16 else nullcontext())
    with patch.object(convolution_geometry, "backward", new=observe), scope:
        ShapeProp(model).propagate(sample)
        executable = compile_fx(model, [sample], config=BackendConfig(
            require_full_aot=True, cuda_conv_dx_geometry=enabled), report=report)
        output = executable(sample)
    return model, output, report, graphs


def _decisions(report):
    return [row for row in report.as_dict()["fusion_candidates"]
            if row["pattern"] == convolution_geometry.PATTERN]


def _metadata_candidate():
    """Reuse a real native spec, changing only metadata for host selection tests."""
    _, _, _, graphs = _compile(_TwoConvolutions(), torch.randn(2, 4, 3, 7).bfloat16(), False)
    original = next(node for node in graphs[0].nodes if isinstance(node, CallNode)
                    and node.op == "convolution.native_dx_db")
    spec = copy(original.target.specification)
    spec.metas = tuple(replace(meta, device=torch.device("cuda:0")) if meta is not None else None
                       for meta in spec.metas)
    source, weight = spec.metas[:2]
    dy = TensorValue("dy", TensorMeta((2, 3, 3, 7), torch.bfloat16, "cuda:0", stride=None))
    w = TensorValue("saved_weight", replace(weight, requires_grad=False))
    dx = TensorValue("dx", replace(source, requires_grad=False))
    dx.fresh_allocation = FreshAllocationContract(dx.meta, "convolution.native_dx_db")
    graph = IRGraph()
    graph.add(InputNode(0, dy))
    graph.add(InputNode(1, w))
    node = CallNode("convolution.native_dx_db", recorded(spec, spec.dx_db, "backward"), (dy, w), {}, (dx,))
    node.scheduling_constraints["adjoint_order_control"] = "dw_before_dx"
    graph.add(node)
    graph.add(OutputNode(dx))
    return graph, node, spec


class NativeConvolutionGeometryTests(unittest.TestCase):
    def test_explicit_false_preserves_native_helper_route_and_metadata(self):
        x = torch.randn(2, 4, 3, 7)
        weight = torch.randn(3, 4, 1, 1)
        gradient = torch.randn(2, 3, 3, 7)
        geometry = ConvGeometry.from_tensors(x, weight)
        observations = []
        original = ops.native_backward

        def observe(dy, descriptor, w, g, mask, **kwargs):
            observations.append((tuple(descriptor.shape), tuple(descriptor.stride()),
                                 descriptor.untyped_storage().nbytes(), mask))
            return original(dy, descriptor, w, g, mask, **kwargs)

        with patch.object(ops, "_shape_only_cpu_eligible", return_value=False), \
                patch.object(ops, "native_backward", side_effect=observe):
            first = ops.native_input_bias_gradients(gradient, weight, geometry, (True, False, False))
            second = ops.native_input_bias_gradients(gradient, weight, geometry, (True, False, False),
                                                     cuda_geometry=False)
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
        self.assertEqual(observations[0], observations[1])
        self.assertEqual(observations[0][2], x.numel() * x.element_size())

    def test_cpu_and_unsupported_requests_raise_before_placeholder_allocation(self):
        x = torch.randn(2, 4, 3, 7).bfloat16()
        weight = torch.randn(3, 4, 1, 1).bfloat16()
        gradient = torch.randn(2, 3, 3, 7).bfloat16()
        geometry = ConvGeometry.from_tensors(x, weight)
        cases = [({}, geometry, (True, False, False), "same_device_cuda_bf16"),
                 ({}, replace(geometry, groups=2), (True, False, False), "unsupported_convolution_geometry"),
                 ({}, replace(geometry, stride=(2, 2)), (True, False, False), "unsupported_convolution_geometry"),
                 ({}, replace(geometry, padding=(1, 1)), (True, False, False), "unsupported_convolution_geometry"),
                 ({}, replace(geometry, dilation=(2, 2)), (True, False, False), "unsupported_convolution_geometry"),
                 ({}, replace(geometry, synthetic_padding_right=(0, 1)), (True, False, False),
                  "unsupported_convolution_geometry"),
                 ({}, geometry, (True, False, True), "dx_only_output_mask"),
                 ({}, geometry, (True, 0, False), "dx_only_output_mask"),
                 ({"transposed": True}, geometry, (True, False, False), "ordinary_nchw"),
                 ({"input_memory_format": torch.channels_last}, geometry, (True, False, False), "ordinary_nchw")]
        with patch.object(torch.Tensor, "new_empty", side_effect=AssertionError("descriptor allocated")), \
                patch.object(ops, "native_backward", side_effect=AssertionError("native call executed")):
            for options, current, mask, reason in cases:
                with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                    ops.native_input_bias_gradients(gradient, weight, current, mask,
                                                    cuda_geometry=True, **options)

    def test_cpu_native_vjp_remains_unselected_and_retained_backward_is_repeatable(self):
        torch.manual_seed(418)
        source = _TwoConvolutions()
        sample = torch.randn(2, 4, 3, 7).bfloat16()
        seed = torch.linspace(-1.1, 0.9, 2 * 3 * 3 * 7).reshape(2, 3, 3, 7).bfloat16()
        results = []
        for enabled in (False, True):
            model, output, report, graphs = _compile(source, sample, enabled)
            first = torch.autograd.grad(output, tuple(model.parameters()), seed, retain_graph=True)
            saved = tuple(value.clone() for value in first)
            again = torch.autograd.grad(output, tuple(model.parameters()), seed)
            for value, expected in zip(again, saved):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            self.assertFalse(sample.requires_grad)
            self.assertIsNone(sample.grad)
            self.assertTrue(all(value.dtype == torch.float32 for value in first))
            self.assertEqual(report.runtime_fallbacks, 0)
            self.assertFalse(any(row["selected"] for row in _decisions(report)))
            self.assertTrue(_decisions(report))
            dx_nodes = [node for node in graphs[0].nodes if isinstance(node, CallNode)
                        and node.op.startswith("convolution.native_dx_db")]
            self.assertEqual([node.target.specification.module_path for node in dx_nodes], ["second"])
            if enabled:
                self.assertEqual(_decisions(report)[0]["reason"], "cuda_only_geometry_route")
            targets = report.as_dict()["graphs"][0]["native_autodiff_graphs"][0]["gradient_targets"]
            self.assertEqual(len(targets), 2)
            results.append((output.detach(), saved))
        for value, expected in zip((results[1][0], *results[1][1]), (results[0][0], *results[0][1])):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_mock_backend_contract_rejects_disabled_unavailable_or_non_cudnn(self):
        geometry = ConvGeometry((2, 6, 3, 7), (3, 6, 1, 1), (1, 1), (0, 0), (1, 1), 1)
        descriptor, weight = object(), object()
        with patch.object(torch.Tensor, "new_empty", side_effect=AssertionError("descriptor allocated")), \
                patch.object(torch._C, "_select_conv_backend") as selector:
            with torch.backends.cudnn.flags(enabled=False):
                self.assertEqual(ops._cuda_geometry_backend_reason(),
                                 "enabled_available_cudnn_backend_required")
            with torch.backends.cudnn.flags(enabled=True), \
                    patch.object(torch.backends.cudnn, "is_available", return_value=False):
                self.assertEqual(ops._cuda_geometry_backend_reason(),
                                 "enabled_available_cudnn_backend_required")
            selector.assert_not_called()
        with torch.backends.cudnn.flags(enabled=True), \
                patch.object(torch.backends.cudnn, "is_available", return_value=True), \
                patch.object(torch._C, "_select_conv_backend", return_value=torch._C._ConvBackend.Slow2d) as selector:
            self.assertIn("native_backend_not_cudnn", ops._cuda_geometry_backend_reason(descriptor, weight, geometry))
            selector.assert_called_once_with(descriptor, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        with torch.backends.cudnn.flags(enabled=True), \
                patch.object(torch.backends.cudnn, "is_available", return_value=True), \
                patch.object(torch._C, "_select_conv_backend", return_value=torch._C._ConvBackend.Cudnn):
            self.assertIsNone(ops._cuda_geometry_backend_reason(descriptor, weight, geometry))
        with torch.backends.cudnn.flags(enabled=True), \
                patch.object(torch.backends.cudnn, "is_available", return_value=True), \
                patch.object(torch._C, "_select_conv_backend", None):
            self.assertEqual(ops._cuda_geometry_backend_reason(),
                             "native_convolution_backend_selector_unavailable")

    def test_host_metadata_selection_is_explicit_and_drops_old_ownership_proof(self):
        graph, original, _ = _metadata_candidate()
        decisions = []
        with patch.object(ops, "native_input_bias_gradients", side_effect=AssertionError("CUDA executed")):
            patterns = convolution_geometry.backward(graph, SimpleNamespace(cuda_conv_dx_geometry=True),
                                                      diagnostics=decisions)
        self.assertEqual(patterns, (convolution_geometry.PATTERN,))
        replacement = graph.nodes[2]
        self.assertEqual(replacement.op, convolution_geometry.OPERATION)
        self.assertEqual(replacement.scheduling_constraints["adjoint_order_control"], "dw_before_dx")
        self.assertIsNone(replacement.values[0].fresh_allocation)
        self.assertEqual(replacement.target.numerical_function.keywords, {"cuda_geometry": True})
        self.assertIs(replacement.values[0], original.values[0])
        record = decisions[0]
        self.assertFalse(record["kernel_fusion"])
        self.assertEqual(record["transformation"], "metadata_allocation_elision")
        self.assertFalse(record["replacement_contract"]["materialized_fallback"])
        self.assertEqual(record["replacement_metadata_storage_bytes"], 2)
        json.dumps(record, allow_nan=False)

    def test_disabled_or_unsupported_metadata_retains_original_call(self):
        graph, original, _ = _metadata_candidate()
        rows = []
        self.assertEqual(convolution_geometry.backward(graph, SimpleNamespace(), diagnostics=rows), ())
        self.assertIs(graph.nodes[2], original)
        self.assertFalse(rows[0]["requested"])
        for kind in ("cpu", "bias", "padding", "layout", "float32", "gradient_layout",
                     "weight_metadata", "effect", "stream", "untrusted", "spoofed_wrapper"):
            graph, node, spec = _metadata_candidate()
            if kind == "cpu":
                spec.metas = tuple(replace(meta, device="cpu") if meta is not None else None for meta in spec.metas)
            elif kind == "bias":
                spec.metas = (*spec.metas[:2], replace(spec.metas[1], shape=(3,), stride=(1,)))
            elif kind == "padding":
                spec.explicit_padding = (0, 0, 0, 0)
            elif kind == "layout":
                spec.input_format = torch.channels_last
            elif kind == "float32":
                spec.compute_dtype = torch.float32
            elif kind == "gradient_layout":
                node.args[0].meta = replace(node.args[0].meta, stride=(63, 21, 1, 3))
            elif kind == "weight_metadata":
                node.args[1].meta = replace(node.args[1].meta, dtype=torch.float64)
            elif kind == "effect":
                node.target.mutates = True
            elif kind == "stream":
                node.target.stream_id = 1
            elif kind == "untrusted":
                node.target.numerical_function = lambda *args: None
            else:
                def opaque(*args):
                    return spec.dx_db(*args)
                opaque.numerical_function, opaque.specification = spec.dx_db, spec
                node.target = opaque
            rows = []
            with self.subTest(kind=kind):
                self.assertEqual(convolution_geometry.backward(graph, SimpleNamespace(cuda_conv_dx_geometry=True),
                                                               diagnostics=rows), ())
                self.assertIs(graph.nodes[2], node)
                self.assertFalse(rows[0]["selected"])


def _cuda_helper_check(test, shape, channels_out, *, profile=False):
    torch.manual_seed(813)
    source = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((channels_out, shape[1], 1, 1), device="cuda", dtype=torch.bfloat16)
    dy = torch.linspace(-0.73, 1.21, shape[0] * channels_out * shape[2] * shape[3],
                        device="cuda", dtype=torch.bfloat16).reshape(shape[0], channels_out, *shape[2:])
    geometry = ConvGeometry.from_tensors(source, weight)
    info, descriptors = {}, []
    original = ops.native_backward

    def observe(gradient, descriptor, operand, current, mask, **kwargs):
        descriptors.append((tuple(descriptor.shape), tuple(descriptor.stride()),
                            descriptor.untyped_storage().nbytes(), mask))
        return original(gradient, descriptor, operand, current, mask, **kwargs)

    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True):
        expected = original(dy, source, weight, geometry, (True, False, False))[0]
        scope = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], profile_memory=True, record_shapes=True)
            if profile else nullcontext())
        with scope as profiler, patch.object(ops, "native_backward", side_effect=observe):
            actual = ops.native_input_bias_gradients(dy, weight, geometry, (True, False, False),
                                                     cuda_geometry=True, execution_info=info)[0]
            torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        test.assertEqual(actual.dtype, torch.bfloat16)
        test.assertEqual(tuple(actual.stride()), contiguous_stride(shape))
        test.assertEqual(descriptors, [(shape, (0, 0, 0, 0), 2, (True, False, False))])
        test.assertEqual(info["input_bias_gradient_metadata_storage_bytes"], 2)
        test.assertEqual(info["input_bias_gradient_route"], "cuda_geometry_shape_only")
        if profiler is not None:
            full_input_clones = [event for event in profiler.events()
                if event.name in ("aten::clone", "aten::contiguous")
                and event.input_shapes and event.input_shapes[0] == list(shape)]
            test.assertFalse(full_input_clones, "Native shape-only route materialized its descriptor")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA geometry execution requires a GPU")
class NativeConvolutionGeometryCudaTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 support is required")

    def test_same_source_native_dx_and_no_full_placeholder_or_contiguous_clone(self):
        _cuda_helper_check(self, (2, 6, 7, 11), 3, profile=True)

    def test_disabled_cudnn_rejects_before_descriptor_allocation(self):
        source = torch.randn(2, 6, 7, 11, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(3, 6, 1, 1, device="cuda", dtype=torch.bfloat16)
        gradient = torch.randn(2, 3, 7, 11, device="cuda", dtype=torch.bfloat16)
        geometry = ConvGeometry.from_tensors(source, weight)
        with torch.backends.cudnn.flags(enabled=False), \
                patch.object(torch.Tensor, "new_empty", side_effect=AssertionError("descriptor allocated")), \
                self.assertRaisesRegex(ValueError, "enabled_available_cudnn"):
            ops.native_input_bias_gradients(gradient, weight, geometry, (True, False, False), cuda_geometry=True)

    def test_unsupported_cuda_operands_reject_before_descriptor_allocation(self):
        source = torch.randn(2, 6, 7, 11, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(3, 6, 1, 1, device="cuda", dtype=torch.bfloat16)
        gradient = torch.randn(2, 3, 7, 11, device="cuda", dtype=torch.bfloat16)
        geometry = ConvGeometry.from_tensors(source, weight)
        strided = gradient.transpose(2, 3).contiguous().transpose(2, 3)
        cases = [(gradient.float(), (True, False, False), "cuda_bf16"),
                 (strided, (True, False, False), "contiguous_nchw"),
                 (gradient, (True, True, False), "dx_only_output_mask")]
        with patch.object(torch.Tensor, "new_empty", side_effect=AssertionError("descriptor allocated")), \
                patch.object(ops, "native_backward", side_effect=AssertionError("native call executed")):
            for current, mask, reason in cases:
                with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                    ops.native_input_bias_gradients(current, weight, geometry, mask, cuda_geometry=True)

    def test_native_cuda_retained_backward_and_no_external_image_gradient(self):
        torch.manual_seed(284)
        source = _TwoConvolutions().cuda()
        sample = torch.randn(2, 4, 7, 11, device="cuda", dtype=torch.bfloat16)
        results = []
        seed = torch.randn(2, 3, 7, 11, device="cuda", dtype=torch.bfloat16)
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True):
            for enabled in (False, True):
                model, output, report, graphs = _compile(source, sample, enabled)
                first = torch.autograd.grad(output, tuple(model.parameters()), seed, retain_graph=True)
                snapshots = tuple(value.clone() for value in first)
                again = torch.autograd.grad(output, tuple(model.parameters()), seed)
                for value, expected in zip(again, snapshots):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                self.assertFalse(sample.requires_grad)
                self.assertIsNone(sample.grad)
                self.assertEqual(report.runtime_fallbacks, 0)
                self.assertEqual(any(row["selected"] for row in _decisions(report)), enabled)
                nodes = [node for node in graphs[0].nodes if isinstance(node, CallNode)
                         and node.op.startswith("convolution.native_dx_db")]
                self.assertEqual([node.target.specification.module_path for node in nodes], ["second"])
                self.assertEqual(nodes[0].op, convolution_geometry.OPERATION if enabled
                                 else "convolution.native_dx_db")
                results.append((output.detach(), snapshots))
            torch.cuda.synchronize()
        for value, expected in zip((results[1][0], *results[1][1]), (results[0][0], *results[0][1])):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("DEEPOPT_TEST_LARGE_CUDA_CONV_GEOMETRY") == "1",
                     "Original decoder geometry is an opt-in CUDA capacity test")
class NativeConvolutionGeometryOriginalShapeCudaTests(unittest.TestCase):
    def test_original_decoder_ten_body_eight_dx(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 support is required")
        _cuda_helper_check(self, (128, 768, 224, 224), 112, profile=True)


if __name__ == "__main__":
    unittest.main()
