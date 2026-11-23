"""Strict native capture of PyTorch's deterministic no-op replication path."""

from contextlib import contextmanager
from copy import deepcopy
import operator
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from deepopt import autocast
from deepopt.compiler import BackendConfig, CompilationReport, compilation_report, compile_fx
from deepopt.compiler.nonstrict_padding import normalize_nonstrict_zero_replication_padding
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model
from deepopt.opt.policy import prepare_fp8_model


class _Pad(nn.Module):
    def __init__(self, pads=(0, 0, 0, 0), *, keyword=False, value=None, derived=False):
        super().__init__()
        self.pads, self.keyword, self.value = pads, keyword, value
        self.derived = derived

    def forward(self, x):
        h, w = x.shape[-2:]
        pads = (0, (-w) % 8, 0, (-h) % 8) if self.derived else self.pads
        if self.keyword:
            return F.pad(input=x, pad=pads, mode="replicate", value=self.value)
        return F.pad(x, pads, "replicate", self.value)


@contextmanager
def _strict_control():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.utils.deterministic.fill_uninitialized_memory = False
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = benchmark, deterministic
        torch.utils.deterministic.fill_uninitialized_memory = fill


class _NativePaddingCaptureAssertions:
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def assert_native(self, report):
        self.assertTrue(report.graphs)
        self.assertGreater(sum(graph.native_autodiff_operations for graph in report.graphs), 0)
        self.assertEqual(report.as_dict()["runtime_fallbacks"], 0)
        for graph in report.graphs:
            self.assertFalse(graph.eager_autograd_operations)
            self.assertFalse(graph.legacy_backward_operations)
            self.assertFalse(graph.unsupported_deepopt_operations)
            self.assertTrue(all(region["differentiation_owner"] == "deepopt_native_autodiff"
                                for region in graph.native_autodiff_graphs))


class NativeDeterministicPaddingCaptureTests(_NativePaddingCaptureAssertions, unittest.TestCase):
    def captured(self, function, source):
        captured = []
        def backend(gm, inputs):
            captured.append(gm)
            return gm.forward
        output = torch.compile(function, backend=backend, fullgraph=True)(source)
        self.assertEqual(len(captured), 1)
        return captured[0], output

    def replication_graph(self, pads=(0, 0, 0, 0)):
        from torch._decomp.decompositions import _replication_pad

        replication = torch._dynamo.nonstrict_trace(_replication_pad)
        source = torch.randn(1, 3, 7, 9, requires_grad=True)
        graph, _ = self.captured(lambda x: replication(x, pads), source)
        return graph, source

    def test_actual_cpu_nonstrict_replication_graph_has_native_metadata_vjp(self):
        with _strict_control():
            graph, source = self.replication_graph()
            original = graph.code
            self.assertIn("flat_apply_capture", original)
            normalized = normalize_nonstrict_zero_replication_padding(graph)
            self.assertIsNot(normalized, graph)
            self.assertEqual(graph.code, original)
            self.assertNotIn("flat_apply_capture", normalized.code)
            self.assertTrue(any(node.target is F.pad for node in normalized.graph.nodes))
            report = CompilationReport()
            compiled = compile_fx(graph, [source],
                config=BackendConfig(require_full_aot=True, use_native_autodiff=True), report=report)
            output, = compiled(source)
            gradient = torch.randn_like(output)
            output.backward(gradient)
            torch.testing.assert_close(output, source, rtol=0, atol=0)
            torch.testing.assert_close(source.grad, gradient, rtol=0, atol=0)
            self.assertNotEqual(output.data_ptr(), source.data_ptr())
            self.assertEqual(graph.code, original)
            self.assert_native(report)
            region, = report.graphs[0].native_autodiff_graphs
            self.assertEqual(region["saved_state"], ())
            self.assertIn("padding.metadata_vjp", region["backward_operations"])

    def test_original_shape_derived_cuda_branch_metadata_is_normalized(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from deepopt.compiler.backend import make_backend

        compression = dict(enabled=False, bits=2, group_size=1024)
        config = {"runtime": {"precision": "bf16", "bf16_policy": "bf16_storage",
                              "fp8": {"enabled": False}, "activation_compression": compression}}
        captured = []
        def recording_backend(*args, **kwargs):
            backend = make_backend(*args, **kwargs)
            def run(gm, inputs, *call_args, **call_kwargs):
                normalized = normalize_nonstrict_zero_replication_padding(gm)
                captured.append((gm.code, normalized.code))
                return backend(gm, inputs, *call_args, **call_kwargs)
            return run
        # CPU-only Torch cannot create CUDA autograd accumulators. This checks
        # the real CUDA Python branch as metadata; the CUDA class below owns
        # actual forward/backward validation with requires_grad=True.
        with _strict_control(), torch.no_grad(), FakeTensorMode():
            source = torch.empty(1, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
            model = prepare_fp8_model(_Pad(derived=True), False,
                activation_compression=compression, config=config)[0]
            with patch("deepopt.compiler.backend.make_backend", side_effect=recording_backend):
                compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                    dynamic=False, options={"require_full_aot": True, "use_native_autodiff": True}), config=config)
            output = model(source)
            self.assertEqual(output.shape, source.shape)
            self.assertEqual(output.dtype, source.dtype)
        self.assertEqual(len(captured), 1)
        self.assertIn("flat_apply_capture", captured[0][0])
        self.assertNotIn("flat_apply_capture", captured[0][1])
        self.assertIn("torch.nn.functional.pad", captured[0][1])

    def test_rewritten_copy_keeps_parameter_and_buffer_identity(self):
        graph, _ = self.replication_graph()
        graph.register_parameter("kept_weight", nn.Parameter(torch.ones(())))
        graph.register_buffer("kept_buffer", torch.ones(()))
        source = next(node for node in graph.graph.nodes if node.op == "placeholder")
        flat = next(node for node in graph.graph.nodes if node.op == "call_function"
                    and node.target is not operator.getitem)
        with graph.graph.inserting_before(flat):
            weight = graph.graph.get_attr("kept_weight")
            buffer = graph.graph.get_attr("kept_buffer")
            product = graph.graph.call_function(torch.mul, (source, weight))
            added = graph.graph.call_function(torch.add, (product, buffer))
        flat.args = (*flat.args[:2], added, *flat.args[3:])
        graph.recompile()
        original = graph.code
        result = normalize_nonstrict_zero_replication_padding(graph)
        self.assertIsNot(result, graph)
        self.assertIs(result.kept_weight, graph.kept_weight)
        self.assertIs(result.kept_buffer, graph.kept_buffer)
        self.assertEqual(graph.code, original)
        self.assertIn("flat_apply_capture", original)

    def test_nonzero_replication_leaf_keeps_full_coverage_rejection(self):
        graph, source = self.replication_graph((1, 1, 1, 1))
        self.assertIs(normalize_nonstrict_zero_replication_padding(graph), graph)
        with self.assertRaisesRegex(NotImplementedError, "coverage incomplete.*flat_apply_capture"):
            compile_fx(graph, [source], config=BackendConfig(require_full_aot=True, use_native_autodiff=True))

    def test_other_nonstrict_callable_keeps_full_coverage_rejection(self):
        sine = torch._dynamo.nonstrict_trace(torch.sin)
        source = torch.randn(1, 3, 7, 9, requires_grad=True)
        graph, _ = self.captured(sine, source)
        self.assertIs(normalize_nonstrict_zero_replication_padding(graph), graph)
        with self.assertRaisesRegex(NotImplementedError, "coverage incomplete.*flat_apply_capture"):
            compile_fx(graph, [source], config=BackendConfig(require_full_aot=True, use_native_autodiff=True))

    def test_unknown_leaf_is_never_evaluated_by_normalization(self):
        graph, _ = self.replication_graph()
        leaf = next(getattr(graph, node.target) for node in graph.graph.nodes
                    if node.op == "get_attr" and node.target.endswith("_callable"))
        def forbidden(*args, **kwargs):
            self.fail("Normalizer evaluated an unknown callable")
        leaf._fn = forbidden
        self.assertIs(normalize_nonstrict_zero_replication_padding(graph), graph)

    def test_changed_private_factory_or_input_spec_fails_closed(self):
        from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

        graph, _ = self.replication_graph()
        with patch.object(TorchInGraphFunctionVariable, "_call_nonstrict_traceable_function", None):
            self.assertIs(normalize_nonstrict_zero_replication_padding(graph), graph)
        class UnknownSpec:
            def __eq__(self, other):
                raise AssertionError("Normalizer evaluated an unknown TreeSpec")
        spec_node = next(node for node in graph.graph.nodes if node.op == "get_attr"
                         and node.target.endswith("_input_spec"))
        setattr(graph, spec_node.target, UnknownSpec())
        self.assertIs(normalize_nonstrict_zero_replication_padding(graph), graph)

    def test_nonstandard_output_use_is_not_rewritten(self):
        graph, _ = self.replication_graph()
        output = next(node for node in graph.graph.nodes if node.op == "output")
        flat = next(node for node in graph.graph.nodes if node.op == "call_function"
                    and node.target is not operator.getitem)
        output.args = ((flat,),)
        self.assertIs(normalize_nonstrict_zero_replication_padding(graph), graph)

    def test_zero_replication_positional_and_keyword_calls_preserve_values_and_validation(self):
        original = F.pad
        with _strict_control():
            for keyword in (False, True):
                with self.subTest(keyword=keyword):
                    model = compile_model(_Pad(keyword=keyword), CompileConfig(enabled=True, backend="deepopt",
                        fullgraph=True, options={"require_full_aot": True, "use_native_autodiff": True}))
                    source = torch.randn(1, 3, 7, 9, requires_grad=True)
                    output = model(source)
                    gradient = torch.randn_like(output)
                    output.backward(gradient)
                    torch.testing.assert_close(output, source, rtol=0, atol=0)
                    torch.testing.assert_close(source.grad, gradient, rtol=0, atol=0)
                    self.assert_native(compilation_report(model))
            for shape, pads, value, message in (
                ((1, 3, 7, 9), (0, 0, 0), None, "divisible by 2"),
                ((1, 2, 3, 7, 9), (0, 0, 0, 0), None, "Padding size|Only"),
                ((1, 3, 7, 9), (0, 0, 0, 0), 1., "value"),
            ):
                with self.subTest(shape=shape, pads=pads, value=value):
                    model = compile_model(_Pad(pads, keyword=True, value=value),
                        CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                            options={"require_full_aot": True, "use_native_autodiff": True}))
                    with self.assertRaisesRegex(Exception, message):
                        model(torch.randn(shape, requires_grad=True))
        self.assertIs(F.pad, original)


@unittest.skipUnless(torch.cuda.is_available(), "Strict deterministic padding capture requires actual CUDA")
class NativeDeterministicPaddingCaptureCudaTests(_NativePaddingCaptureAssertions, unittest.TestCase):
    def test_original224_noop_replication_with_precision_and_native_backward(self):
        from deepopt.compiler.backend import make_backend

        compression = dict(enabled=False, bits=2, group_size=1024, norm_xhat_recenter=True)
        config = {"runtime": {"precision": "bf16", "bf16_policy": "bf16_storage",
                              "fp8": {"enabled": False}, "activation_compression": compression}}
        def recording_backend(*args, **kwargs):
            backend = make_backend(*args, **kwargs)
            def run(gm, inputs, *call_args, **call_kwargs):
                try:
                    return backend(gm, inputs, *call_args, **call_kwargs)
                except Exception:
                    print("Strict deterministic padding incoming FX graph:", file=sys.stderr)
                    print(gm.code, file=sys.stderr, flush=True)
                    raise
            return run
        with _strict_control(), patch.dict("os.environ", {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            for dtype in (torch.float32, torch.bfloat16):
                for keyword in (False, True):
                    with self.subTest(dtype=dtype, keyword=keyword):
                        model = prepare_fp8_model(_Pad(keyword=keyword, derived=True).cuda(), False,
                            activation_compression=compression, config=config)[0]
                        reference = deepcopy(model)
                        with patch("deepopt.compiler.backend.make_backend", side_effect=recording_backend):
                            compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                                options={"require_full_aot": True, "use_native_autodiff": True}), config=config)
                        source = torch.randn(1, 3, 224, 224, dtype=dtype, device="cuda", requires_grad=True)
                        control = source.detach().clone().requires_grad_()
                        with autocast("cuda", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                                      activation_compression=compression, config={}):
                            expected, actual = reference(control), model(source)
                        cotangent = torch.randn_like(actual)
                        expected.backward(cotangent)
                        actual.backward(cotangent)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(source.grad, cotangent, rtol=0, atol=0)
                        torch.testing.assert_close(source.grad, control.grad, rtol=0, atol=0)
                        self.assert_native(compilation_report(model))
                        self.assertTrue(torch.are_deterministic_algorithms_enabled())
                        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
                        self.assertFalse(torch.backends.cudnn.benchmark)
                        self.assertTrue(torch.backends.cudnn.deterministic)


if __name__ == "__main__":
    unittest.main()
