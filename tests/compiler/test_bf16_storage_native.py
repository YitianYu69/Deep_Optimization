"""CPU semantic receipts for explicit storage policy; CUDA qualification separate."""

from copy import deepcopy
import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt._bf16_policy import bf16_policy_scope
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.opt.nn.activation.elementwise_modules import adapter
from deepopt.opt.backends.reference.activation.math import evaluate
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.nn.normalization.modules import DOGroupNorm
from deepopt.opt.nn.primitive_state import model_exp
from deepopt import autocast


class BF16StorageActivationTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)

    def test_exact_input_recompute_and_coefficient_codec(self):
        for kind in ("silu", "gelu", "sigmoid", "tanh"):
            for bits in (1, 2, 4, 8):
                for packed in (False, True):
                    for strided in (False, True):
                        with self.subTest(kind=kind, bits=bits, packed=packed, strided=strided):
                            compression = dict(enabled=packed, bits=bits, group_size=1024)
                            module = adapter(kind, plan=dict(fp8=False, activation_compression=compression))
                            x = torch.randn(4, 1024, dtype=torch.bfloat16)
                            if strided:
                                x = x.t()
                            x.requires_grad_()
                            original = x.detach().clone()
                            saved = []
                            with bf16_policy_scope("bf16_storage", "cpu"), torch.autograd.graph.saved_tensors_hooks(
                                    lambda value: saved.append(value) or value, lambda value: value):
                                y = module(x)
                            self.assertEqual(y.dtype, torch.bfloat16)
                            values, coefficient = evaluate(original.float(), kind, "none")
                            torch.testing.assert_close(y, values.bfloat16(), rtol=0, atol=0)
                            if packed:
                                self.assertTrue(any(value.dtype == torch.int32 for value in saved))
                                coefficient = dequantize_unpack_activation(quantize_pack_activation(
                                    coefficient.reshape(-1, original.shape[-1]), 1024, bits=bits),
                                    dtype=torch.float32).reshape(original.shape)
                            else:
                                self.assertTrue(all(value.dtype == torch.bfloat16 for value in saved))
                                with torch.no_grad():
                                    x.add_(1)
                            gradient = torch.randn_like(y)
                            with bf16_policy_scope("torch_amp", "cpu"):
                                y.backward(gradient)
                            torch.testing.assert_close(x.grad, (gradient.float() * coefficient).bfloat16(), rtol=0, atol=0)

    def test_native_policy_metadata_and_backward_after_scope(self):
        for packed in (False, True):
            plan = dict(fp8=False, activation_compression=dict(enabled=packed, bits=2, group_size=1024))
            module = nn.Sequential(adapter("silu", plan=plan), adapter("gelu", plan=plan))
            control = deepcopy(module)
            x = torch.randn(2, 1024, dtype=torch.bfloat16, requires_grad=True)
            reference = x.detach().clone().requires_grad_()
            fx_graph = torch.fx.Graph()
            input_value = fx_graph.placeholder("x")
            first = fx_graph.call_module("0", (input_value,))
            second = fx_graph.call_module("1", (first,))
            fx_graph.output(second)
            graph = torch.fx.GraphModule(module, fx_graph)
            report = CompilationReport()
            with bf16_policy_scope("bf16_storage", "cpu"):
                ShapeProp(graph).propagate(x)
                compiled = compile_fx(graph, (x,), config=BackendConfig(require_full_aot=True), report=report)
                expected, actual = control(reference), compiled(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            gradient = torch.randn_like(actual)
            with bf16_policy_scope("torch_amp", "cpu"):
                actual.backward(gradient, retain_graph=True)
                expected.backward(gradient, retain_graph=True)
                torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                actual.backward(gradient)
                expected.backward(gradient)
            torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
            self.assertEqual(report.graphs[0].effective_compiler_options["bf16_policies"]["cpu"],
                             {"name": "bf16_storage", "version": 1})
            self.assertTrue(all(item["bf16_policy"]["name"] == "bf16_storage"
                                for item in report.graphs[0].native_autodiff_graphs))
            with bf16_policy_scope("torch_amp", "cpu"):
                with self.assertRaisesRegex(ValueError, "numerical scope changed"):
                    compiled(x)

    def test_group_norm_activation_convolution_dtype_chain(self):
        for policy in ("torch_amp", "bf16_storage"):
            for packed in (False, True):
                with self.subTest(policy=policy, packed=packed):
                    plan = dict(fp8=False, activation_compression=dict(
                        enabled=packed, bits=2, group_size=1024, norm_xhat_recenter=True))
                    module = nn.Sequential(DOGroupNorm(nn.GroupNorm(2, 4), plan=plan),
                                           adapter("silu", plan=plan), nn.Conv2d(4, 4, 3, padding=1))
                    control = deepcopy(module)
                    fx_graph = torch.fx.Graph()
                    current = fx_graph.placeholder("x")
                    for index in range(3):
                        current = fx_graph.call_module(str(index), (current,))
                    fx_graph.output(current)
                    graph = torch.fx.GraphModule(module, fx_graph)
                    x = torch.randn(2, 4, 16, 32, dtype=torch.bfloat16, requires_grad=True)
                    reference = x.detach().clone().requires_grad_()
                    saved = []
                    report = CompilationReport()
                    with bf16_policy_scope(policy, "cpu"), torch.autocast("cpu", dtype=torch.bfloat16):
                        ShapeProp(graph).propagate(x)
                        for node in graph.graph.nodes:
                            if node.op == "call_module":
                                self.assertEqual(node.meta["tensor_meta"].dtype, torch.bfloat16)
                        compiled = compile_fx(graph, (x,), config=BackendConfig(require_full_aot=True), report=report)
                        expected = control(reference)
                        with torch.autograd.graph.saved_tensors_hooks(
                                lambda value: saved.append(value) or value, lambda value: value):
                            actual = compiled(x)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    gradient = torch.randn_like(actual)
                    actual.backward(gradient)
                    expected.backward(gradient)
                    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                    for parameter, oracle in zip(module.parameters(), control.parameters()):
                        self.assertEqual(parameter.dtype, torch.float32)
                        self.assertEqual(parameter.grad.dtype, torch.float32)
                        torch.testing.assert_close(parameter.grad, oracle.grad, rtol=0, atol=0)
                    if packed:
                        self.assertTrue(any(value.dtype == torch.int32 for value in saved))
                    self.assertTrue(all(not item.eager_autograd_operations and not item.legacy_backward_operations
                                        for item in report.graphs))


    def test_model_exp_exact_coefficient_and_explicit_fp32_island(self):
        for packed in (False, True):
            for bits in (1, 2, 4, 8):
                with self.subTest(packed=packed, bits=bits):
                    x = (torch.randn(4, 1024, dtype=torch.bfloat16) * .2).requires_grad_()
                    original = x.detach().clone()
                    compression = dict(enabled=packed, bits=bits, group_size=1024)
                    fx_graph = torch.fx.Graph()
                    input_value = fx_graph.placeholder("x")
                    output = fx_graph.call_function(model_exp, (input_value,))
                    fx_graph.output(output)
                    graph = torch.fx.GraphModule(nn.Module(), fx_graph)
                    report = CompilationReport()
                    with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                                  bf16_policy="bf16_storage", activation_compression=compression):
                        ShapeProp(graph).propagate(x)
                        compiled = compile_fx(graph, (x,), config=BackendConfig(require_full_aot=True), report=report)
                        actual = compiled(x)
                        expected = model_exp(x)
                        island = model_exp(x.float())
                    self.assertEqual(actual.dtype, torch.bfloat16)
                    self.assertEqual(island.dtype, torch.float32)
                    from examples.vae.qualification import saved_state_evidence

                    evidence = saved_state_evidence(report.as_dict(), packed)
                    self.assertNotIn("approved_exact_coefficient", evidence["categories"])
                    if packed:
                        self.assertEqual(evidence["categories"]["codec_metadata"], 2)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    coefficient = original.float().exp()
                    if packed:
                        coefficient = dequantize_unpack_activation(quantize_pack_activation(
                            coefficient.reshape(1, -1), 1024, bits=bits), dtype=torch.float32).reshape(original.shape)
                    gradient = torch.randn_like(actual)
                    actual.backward(gradient)
                    torch.testing.assert_close(x.grad, (gradient.float() * coefficient).bfloat16(), rtol=0, atol=0)
                    x.grad = None
                    expected.backward(gradient)
                    torch.testing.assert_close(x.grad, (gradient.float() * coefficient).bfloat16(), rtol=0, atol=0)

    def test_recompute_snapshot_does_not_retain_view_base(self):
        module = adapter("silu", plan=dict(fp8=False, activation_compression=dict(
            enabled=False, bits=2, group_size=1024)))
        base = torch.randn(1024, 1024, dtype=torch.bfloat16)
        x = base[:2].detach().requires_grad_()
        original = x.detach().clone()
        saved = []
        with bf16_policy_scope("bf16_storage", "cpu"), torch.autograd.graph.saved_tensors_hooks(
                lambda value: saved.append(value) or value, lambda value: value):
            output = module(x)
        self.assertEqual(len(saved), 1)
        self.assertNotEqual(saved[0].untyped_storage().data_ptr(), base.untyped_storage().data_ptr())
        self.assertEqual(saved[0].untyped_storage().nbytes(), x.numel() * x.element_size())
        base.fill_(0)
        output.backward(torch.ones_like(output))
        torch.testing.assert_close(x.grad, evaluate(original.float(), "silu", "none")[1].bfloat16(), rtol=0, atol=0)

    def test_swiglu_fp32_input_native_adjoint_recipe(self):
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.opt.policy import prepare_fp8_model
        from examples.vae.mbconv_vae import SwiGLU

        self.addCleanup(torch.compiler.reset)
        for packed in (False, True):
            torch.compiler.reset()
            torch.manual_seed(5201)
            compression = dict(enabled=packed, bits=2, group_size=1024, norm_xhat_recenter=True)
            config = dict(runtime=dict(precision="bf16", bf16_policy="bf16_storage",
                                       fp8=dict(enabled=False), activation_compression=compression))
            eager = prepare_fp8_model(SwiGLU(8), False, config=config)[0]
            native = deepcopy(eager)
            compile_model(native, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                options=dict(require_full_aot=True)), config=config)
            x = torch.randn(1, 8, 17, 19, requires_grad=True)
            reference = x.detach().clone().requires_grad_()
            with autocast("cpu", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                          activation_compression=compression):
                expected, actual = eager(reference), native(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            gradient = torch.randn_like(actual)
            actual.backward(gradient)
            expected.backward(gradient)
            torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
            self.assertEqual(x.grad.dtype, torch.float32)
            for parameter, oracle in zip(native.parameters(), eager.parameters()):
                torch.testing.assert_close(parameter.grad, oracle.grad, rtol=0, atol=0)
            if not packed:
                contracts = native._deepopt_compile_report.graphs[0].precision_contracts
                self.assertTrue(any(item["operation"] == "to" and any(
                    value["dtype"] == "torch.bfloat16" for value in item["outputs"]) for item in contracts),
                    "Shared eager autocast casts must be explicit before native differentiation")

    def test_mixed_cat_norm_and_promoted_primitive_contributions(self):
        for packed in (False, True):
            torch.manual_seed(5201)
            plan = dict(fp8=False, activation_compression=dict(
                enabled=packed, bits=2, group_size=1024, norm_xhat_recenter=True))
            module = nn.Module()
            module.a = DOGroupNorm(nn.GroupNorm(2, 4), plan=plan)
            module.b = DOGroupNorm(nn.GroupNorm(2, 8), plan=plan)
            graph = torch.fx.Graph()
            x_value = graph.placeholder("x")
            a = graph.call_module("a", (x_value,))
            mixed = graph.call_function(torch.cat, ((x_value, a), 1))
            b = graph.call_module("b", (mixed,))
            graph.output(b)
            module = torch.fx.GraphModule(module, graph)
            control = deepcopy(module)
            x = torch.randn(1, 4, 17, 19, requires_grad=True)
            reference = x.detach().clone().requires_grad_()
            with bf16_policy_scope("bf16_storage", "cpu"), torch.autocast("cpu", dtype=torch.bfloat16):
                ShapeProp(module).propagate(x)
                compiled = compile_fx(module, (x,), config=BackendConfig(require_full_aot=True))
                actual, expected = compiled(x), control(reference)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            gradient = torch.randn_like(actual)
            actual.backward(gradient)
            expected.backward(gradient)
            torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)

            for parameter, oracle in zip(module.parameters(), control.parameters()):
                torch.testing.assert_close(parameter.grad, oracle.grad, rtol=0, atol=0)

    def test_shared_autocast_leaf_cast_rounding_and_scope(self):
        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(17, 19)
                self.b = nn.Linear(17, 19)
                self.b.weight = self.a.weight

            def forward(self, x):
                return self.a(x), self.b(x)

        for policy in ("torch_amp", "bf16_storage"):
            for cache in (False, True):
                with self.subTest(policy=policy, cache=cache):
                    torch.manual_seed(180)
                    model = torch.fx.symbolic_trace(Shared())
                    control = deepcopy(model)
                    x = torch.randn(5, 17, requires_grad=True)
                    reference = x.detach().clone().requires_grad_()
                    with bf16_policy_scope(policy, "cpu"), torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=cache):
                        ShapeProp(model).propagate(x)
                        compiled = compile_fx(model, (x,), config=BackendConfig(require_full_aot=True))
                        actual, expected = compiled(x), control(reference)
                    gradients = tuple(torch.randn_like(value) for value in actual)
                    torch.autograd.backward(actual, gradients)
                    torch.autograd.backward(expected, gradients)
                    torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                    for parameter, oracle in zip(model.parameters(), control.parameters()):
                        torch.testing.assert_close(parameter.grad, oracle.grad, rtol=0, atol=0)
                    if cache:
                        model.b.weight = nn.Parameter(model.b.weight.detach().clone())
                        with bf16_policy_scope(policy, "cpu"), torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=cache):
                            with self.assertRaisesRegex(ValueError, "alias changed"):
                                compiled(x)
    def test_promoted_reduction_and_stack_contributions(self):
        for operation in (lambda x: x.sum(dtype=torch.float32), lambda x: x.mean(dtype=torch.float32),
                          lambda x: torch.stack((x, x.float())).sum()):
            class Primitive(nn.Module):
                def forward(self, x):
                    return operation(x)
            graph = torch.fx.symbolic_trace(Primitive())
            x = torch.randn(4, 1024, dtype=torch.bfloat16, requires_grad=True)
            reference = x.detach().clone().requires_grad_()
            ShapeProp(graph).propagate(x)
            compiled = compile_fx(graph, (x,), config=BackendConfig(require_full_aot=True))
            actual, expected = compiled(x), graph(reference)
            gradient = torch.randn_like(actual)
            actual.backward(gradient)
            expected.backward(gradient)
            torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; no target qualification")
class BF16StorageNativeCUDATests(unittest.TestCase):
    def test_model_exp_exact_input_and_codec_all_bits(self):
        for packed in (False, True):
            for bits in (1, 2, 4, 8):
                for strided in (False, True):
                    with self.subTest(packed=packed, bits=bits, strided=strided):
                        x = torch.randn(4, 1024, dtype=torch.bfloat16, device="cuda") * .2
                        if strided:
                            x = x.t()
                        x.requires_grad_()
                        reference = x.detach().clone().requires_grad_()
                        compression = dict(enabled=packed, bits=bits, group_size=1024)
                        graph = torch.fx.Graph()
                        input_value = graph.placeholder("x")
                        output = graph.call_function(model_exp, (input_value,))
                        graph.output(output)
                        graph = torch.fx.GraphModule(nn.Module(), graph)
                        saved = []
                        with autocast("cuda", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                                      activation_compression=compression):
                            ShapeProp(graph).propagate(x)
                            compiled = compile_fx(graph, (x,), config=BackendConfig(require_full_aot=True))
                            expected = model_exp(reference)
                            with torch.autograd.graph.saved_tensors_hooks(
                                    lambda value: saved.append(value) or value, lambda value: value):
                                actual = compiled(x)
                        self.assertEqual(actual.dtype, torch.bfloat16)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        gradient = torch.randn_like(actual)
                        actual.backward(gradient)
                        expected.backward(gradient)
                        self.assertEqual(x.grad.dtype, torch.bfloat16)
                        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                        if packed:
                            self.assertTrue(any(value.dtype == torch.int32 for value in saved))
                        else:
                            self.assertTrue(all(value.dtype == torch.bfloat16 for value in saved))

    def test_chain_policies_compression_layouts_and_masks(self):
        for policy in ("torch_amp", "bf16_storage"):
            for packed in (False, True):
                for channels_last in (False, True):
                    for input_grad, affine_grad in ((True, True), (True, False), (False, True)):
                        with self.subTest(policy=policy, packed=packed, channels_last=channels_last,
                                          input_grad=input_grad, affine_grad=affine_grad):
                            plan = dict(fp8=False, activation_compression=dict(
                                enabled=packed, bits=2, group_size=1024, norm_xhat_recenter=True))
                            module = nn.Sequential(DOGroupNorm(nn.GroupNorm(2, 4), plan=plan),
                                adapter("silu", plan=plan), nn.Conv2d(4, 4, 3, padding=1)).cuda()
                            for parameter in module[0].parameters():
                                parameter.requires_grad_(affine_grad)
                            if channels_last:
                                module.to(memory_format=torch.channels_last)
                            control = deepcopy(module)
                            graph = torch.fx.Graph()
                            current = graph.placeholder("x")
                            for index in range(3):
                                current = graph.call_module(str(index), (current,))
                            graph.output(current)
                            graph = torch.fx.GraphModule(module, graph)
                            x = torch.randn(2, 4, 16, 32, dtype=torch.bfloat16, device="cuda")
                            if channels_last:
                                x = x.contiguous(memory_format=torch.channels_last)
                            x.requires_grad_(input_grad)
                            reference = x.detach().clone().requires_grad_(input_grad)
                            saved = []
                            with bf16_policy_scope(policy, "cuda"), torch.autocast("cuda", dtype=torch.bfloat16):
                                ShapeProp(graph).propagate(x)
                                expected_dtype = torch.bfloat16 if policy == "bf16_storage" else torch.float32
                                self.assertEqual(list(graph.graph.nodes)[1].meta["tensor_meta"].dtype, expected_dtype)
                                self.assertEqual(list(graph.graph.nodes)[2].meta["tensor_meta"].dtype, expected_dtype)
                                compiled = compile_fx(graph, (x,), config=BackendConfig(require_full_aot=True))
                                expected = control(reference)
                                with torch.autograd.graph.saved_tensors_hooks(
                                        lambda value: saved.append(value) or value, lambda value: value):
                                    actual = compiled(x)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            gradient = torch.randn_like(actual)
                            actual.backward(gradient)
                            expected.backward(gradient)
                            if input_grad:
                                self.assertEqual(x.grad.dtype, torch.bfloat16)
                                torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
                            for parameter, oracle in zip(module.parameters(), control.parameters()):
                                self.assertEqual(parameter.dtype, torch.float32)
                                if parameter.requires_grad:
                                    self.assertEqual(parameter.grad.dtype, torch.float32)
                                    torch.testing.assert_close(parameter.grad, oracle.grad, rtol=0, atol=0)
                            if packed:
                                self.assertTrue(any(value.dtype == torch.int32 for value in saved))


if __name__ == "__main__":
    unittest.main()
