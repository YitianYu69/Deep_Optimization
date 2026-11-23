"""Native VJP regressions for indexing, reductions, call options and SDPA."""

from copy import deepcopy
import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn.attention import SDPBackend, sdpa_kernel

from deepopt import autocast
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model
from deepopt.opt.nn.activation.elementwise_modules import DOGELU, DOSiLU
from deepopt.opt.nn.activation.modules import DOReLU
from deepopt.opt.policy import prepare_fp8_model


GROUP = 1024


class _Function(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class NativeSemanticContractTests(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        torch.manual_seed(834)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def compile(self, graph, inputs):
        ShapeProp(graph).propagate(*inputs)
        report = CompilationReport()
        compiled = compile_fx(graph, inputs, config=BackendConfig(require_full_aot=True), report=report)
        self.assertFalse(report.graphs[0].eager_autograd_operations)
        self.assertFalse(report.graphs[0].legacy_backward_operations)
        return compiled

    def test_index_duplicates_and_mixed_advanced_indices_accumulate(self):
        forms = (
            lambda x: x[[0, 0, 3]],
            lambda x: x[[0, 0, 3], :],
            lambda x: x[:, [1, 1, 2]],
            lambda x: x[[0, 0, 3], [1, 1, 2]],
            lambda x: x[(0, 0, 3), :],
            lambda x: x[[True, False, True, False]],
            lambda x: x[[0, 0, 3], None, ...],
            lambda x: x[None, ..., [1, 1, 2]],
            lambda x: x[torch.tensor(True)],
            lambda x: x[torch.tensor([True, False, True, False])],
        )
        for dtype in (torch.float64, torch.float32, torch.bfloat16):
            for compressed in (False, True):
                for function in forms:
                    with self.subTest(dtype=dtype, compressed=compressed, function=function):
                        x = torch.randn(4, 3, dtype=dtype, device=self.device, requires_grad=True)
                        ref = x.detach().clone().requires_grad_()
                        graph = torch.fx.symbolic_trace(_Function(function))
                        with autocast(self.device, dtype=torch.bfloat16, fp8=False,
                                      activation_compression=dict(enabled=compressed, bits=2, group_size=GROUP)):
                            compiled = self.compile(graph, [x])
                            actual, expected = compiled(x), graph(ref)
                        gradient = torch.arange(1, actual.numel() + 1, device=self.device, dtype=dtype).reshape(actual.shape)
                        actual.backward(gradient)
                        expected.backward(gradient)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                        torch.testing.assert_close(x.grad, ref.grad, rtol=0, atol=0)

    def test_mean_empty_dimensions_and_rank_specific_t(self):
        forms = tuple((shape, function) for shape in ((4, 3), (), (0, 3)) for function in (
            lambda x: x.mean(dim=()), lambda x: x.mean(dim=[]),
            lambda x: x.mean(dim=(), keepdim=True), lambda x: x.mean(),
            lambda x: x.mean(dim=-1, keepdim=True))) + tuple(
                (shape, lambda x: x.t().sum()) for shape in ((), (3,), (4, 3)))
        for dtype in (torch.float64, torch.float32, torch.bfloat16):
            for compressed in (False, True):
                for shape, function in forms:
                    with self.subTest(dtype=dtype, compressed=compressed, shape=shape, function=function):
                        x = torch.randn(shape, dtype=dtype, device=self.device, requires_grad=True)
                        ref = x.detach().clone().requires_grad_()
                        graph = torch.fx.symbolic_trace(_Function(function))
                        with autocast(self.device, dtype=torch.bfloat16, fp8=False,
                                      activation_compression=dict(enabled=compressed, bits=2, group_size=GROUP)):
                            compiled = self.compile(graph, [x])
                            actual, expected = compiled(x), graph(ref)
                        actual.sum().backward()
                        expected.sum().backward()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                        torch.testing.assert_close(x.grad, ref.grad, rtol=0, atol=0)

    def test_public_compressed_linear_repeated_rows_preserve_weight_gradient(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(257, 257, bias=False)

            def forward(self, x):
                return self.linear(x)[[0, 0, 3]].sum()

        compression = dict(enabled=True, bits=2, group_size=GROUP)
        model = prepare_fp8_model(Model().to(self.device).bfloat16(), False,
                                  activation_compression=compression, config={})[0]
        reference = deepcopy(model)
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                                           options={"require_full_aot": True}))
        x = torch.randn(4, 257, device=self.device, dtype=torch.bfloat16, requires_grad=True)
        ref = x.detach().clone().requires_grad_()
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t.dtype) or t, lambda t: t):
            actual = model(x)
        expected = reference(ref)
        actual.backward()
        expected.backward()
        self.assertIn(torch.int32, saved, "Positive compression fixture must actually pack")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, ref.grad, rtol=0, atol=0)
        torch.testing.assert_close(model.linear.weight.grad, reference.linear.weight.grad, rtol=0, atol=0)

    def activation_graph(self, module, calls):
        root = nn.Module()
        root.add_module("activation", module)
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        outputs = tuple(graph.call_module("activation", (x,), kwargs) for kwargs in calls)
        graph.output(outputs if len(outputs) > 1 else outputs[0])
        return torch.fx.GraphModule(root, graph)

    def test_shared_gelu_options_are_snapshotted_per_call_and_invocation(self):
        for bits in (1, 2, 4, 8):
            for compressed in (False, True):
                with self.subTest(bits=bits, compressed=compressed):
                    plan = dict(fp8=False, activation_compression=dict(enabled=compressed, bits=bits, group_size=GROUP))
                    module = DOGELU(plan=plan, approximate="none").to(self.device)
                    graph = self.activation_graph(module, ({"approximate": "tanh"}, {"approximate": None}))
                    x = torch.randn(2, GROUP, device=self.device, dtype=torch.bfloat16, requires_grad=True)
                    ref = x.detach().clone().requires_grad_()
                    compiled = self.compile(graph, [x])
                    saved = []
                    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t.dtype) or t, lambda t: t):
                        actual = compiled(x)
                        second = compiled(x * 2)
                    expected, expected_second = graph(ref), graph(ref * 2)
                    # Consume outstanding invocations in reverse order.
                    sum(y.sum() for y in second).backward(retain_graph=True)
                    sum(y.sum() for y in expected_second).backward(retain_graph=True)
                    sum(y.sum() for y in actual).backward()
                    sum(y.sum() for y in expected).backward()
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    torch.testing.assert_close(x.grad, ref.grad, rtol=0, atol=0)
                    self.assertEqual(module.approximate, "none")
                    if compressed:
                        self.assertIn(torch.int32, saved)

    def test_effective_mutation_is_rejected_and_false_override_is_pure(self):
        plan = dict(fp8=False, activation_compression=dict(enabled=True, bits=2, group_size=GROUP))
        for kind in (DOSiLU, DOReLU):
            for default, override in ((False, True), (True, False)):
                with self.subTest(kind=kind, default=default, override=override):
                    module = kind(plan=plan, inplace=default)
                    graph = self.activation_graph(module, ({"inplace": override},))
                    x = torch.randn(2, GROUP, device=self.device, dtype=torch.bfloat16)
                    before = x.clone()
                    # Shape propagation uses a disposable input when mutation is requested.
                    ShapeProp(graph).propagate(x.clone())
                    if override:
                        with self.assertRaises(NotImplementedError):
                            compile_fx(graph, [x], config=BackendConfig(require_full_aot=True))
                    else:
                        compiled = self.compile(graph, [x])
                        actual, expected = compiled(x), graph(x.clone())
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertNotEqual(actual.data_ptr(), x.data_ptr())
                    torch.testing.assert_close(x, before, rtol=0, atol=0)
                    self.assertEqual(module.inplace, default)


@unittest.skipUnless(torch.cuda.is_available(), "Native semantic CUDA qualification requires CUDA")
class NativeSemanticContractCUDATests(NativeSemanticContractTests):
    device = "cuda"


class NativeAttentionBroadcastTests(unittest.TestCase):
    device = "cpu"
    setUp = NativeSemanticContractTests.setUp
    compile = NativeSemanticContractTests.compile
    def test_math_broadcast_and_gqa_reduce_before_cast(self):
        from deepopt.opt.nn.functional import do_scaled_dot_product_attention

        cases = ((False, (2, 1, 1)), (True, (2, 1, 1)),
                 (True, (2, 2, 1)), (False, (1, 2, 2)))
        for dtype in (torch.float64, torch.float32, torch.bfloat16):
            for compressed in (False, True):
                for gqa, batches in cases:
                    with self.subTest(dtype=dtype, compressed=compressed, gqa=gqa, batches=batches):
                        compression = dict(enabled=compressed, bits=2, attention_bits=4, group_size=GROUP)
                        graph = torch.fx.Graph()
                        args = tuple(graph.placeholder(name) for name in ("q", "k", "v", "bias"))
                        graph.output(graph.call_function(do_scaled_dot_product_attention, args,
                            dict(enable_gqa=gqa, scale=-.3, plan=dict(fp8=False, activation_compression=compression))))
                        model = torch.fx.GraphModule(nn.Module(), graph)
                        shapes = ((batches[0], 4, 17, 16), (batches[1], 2 if gqa else 4, 33, 16),
                                  (batches[2], 2 if gqa else 4, 33, 16), (1, 1, 17, 33))
                        inputs = tuple(torch.randn(shape, device=self.device, dtype=dtype, requires_grad=True) for shape in shapes)
                        refs = tuple(value.detach().clone().requires_grad_() for value in inputs)
                        with sdpa_kernel(SDPBackend.MATH):
                            compiled = self.compile(model, inputs)
                            actual, expected = compiled(*inputs), model(*refs)
                            gradient = torch.randn_like(expected)
                            actual.backward(gradient)
                            expected.backward(gradient)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        for value, ref in zip(inputs, refs):
                            torch.testing.assert_close(value.grad, ref.grad, rtol=.02 if compressed and dtype != torch.float64 else None,
                                                       atol=.002 if compressed and dtype != torch.float64 else None)

    def test_math_dropout_masks_and_selective_gradients_survive_scope_exit(self):
        from deepopt.opt.nn.functional import do_scaled_dot_product_attention

        for compressed in (False, True):
            for needs in ((True, False, True, True), (False, True, False, False), (False, False, False, True)):
                with self.subTest(compressed=compressed, needs=needs):
                    compression = dict(enabled=compressed, bits=2, attention_bits=4, group_size=GROUP)
                    graph = torch.fx.Graph()
                    args = tuple(graph.placeholder(name) for name in ("q", "k", "v", "bias"))
                    graph.output(graph.call_function(do_scaled_dot_product_attention, args,
                        dict(enable_gqa=True, dropout_p=.2, plan=dict(fp8=False, activation_compression=compression))))
                    model = torch.fx.GraphModule(nn.Module(), graph)
                    shapes = ((2, 4, 17, 16), (1, 2, 33, 16), (1, 2, 33, 16), (1, 1, 17, 33))
                    inputs = tuple(torch.randn(shape, dtype=torch.bfloat16, device=self.device, requires_grad=need)
                                   for shape, need in zip(shapes, needs))
                    refs = tuple(value.detach().clone().requires_grad_(need) for value, need in zip(inputs, needs))
                    with sdpa_kernel(SDPBackend.MATH), autocast(self.device, dtype=torch.bfloat16, fp8=False,
                                                               activation_compression=compression):
                        compiled = self.compile(model, inputs)
                        torch.manual_seed(911)
                        expected = model(*refs)
                        torch.manual_seed(911)
                        actual = compiled(*inputs)
                    gradient = torch.randn_like(expected)
                    actual.backward(gradient)
                    expected.backward(gradient)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for value, ref, needed in zip(inputs, refs, needs):
                        if needed:
                            torch.testing.assert_close(value.grad, ref.grad)
                        else:
                            self.assertIsNone(value.grad)


@unittest.skipUnless(torch.cuda.is_available(), "Math SDPA CUDA qualification requires CUDA")
class NativeAttentionBroadcastCUDATests(NativeAttentionBroadcastTests):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
