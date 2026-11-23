"""BF16 compiler ownership, AMP rounding, codecs and generated execution."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt.compiler import BackendConfig, compile_fx, compilation_report
from deepopt.compiler.lowering import GraphLowering
from deepopt.compiler.native_linear import NativeLinearNode, NativeLinearProgram, fuse_native
from deepopt.compiler.registry import KernelCodegenContext
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model
from deepopt.kernel_policy import KernelPolicy
from deepopt.opt.nn.functional import do_linear
from deepopt.opt.policy import linear_plan


class BF16CompilerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(714)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_native_linear_values_and_backward_with_bf16_and_amp(self):
        for amp in (False, True):
            for compressed in (False, True):
                with self.subTest(amp=amp, compression=compressed):
                    dtype = torch.float32 if amp else torch.bfloat16
                    model = nn.Sequential(nn.Linear(11, 7)).to(dtype)
                    reference = deepcopy(model)
                    gm = torch.fx.symbolic_trace(model)
                    x = torch.randn(2, 65, 11, dtype=dtype, requires_grad=True)
                    xr = x.detach().clone().requires_grad_()
                    compression = {"enabled": compressed, "bits": 2, "group_size": 1024}
                    with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=compression, config={}):
                        ShapeProp(gm).propagate(x)
                        compiled = compile_fx(gm, [x])
                        with patch("deepopt.opt.nn.autograd.native_linear._NativeLinearFunction.forward",
                                   side_effect=AssertionError("legacy native Function was invoked")):
                            actual = compiled(x)
                        expected = do_linear(xr, reference[0].weight, reference[0].bias,
                                             plan=linear_plan(False, activation_compression=compression, config={}))
                    gradient = torch.randn_like(actual)
                    actual.backward(gradient)
                    expected.backward(gradient)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
                    for a, b in zip(gm.parameters(), reference.parameters()):
                        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                    self.assertEqual(len(compiled.backend_artifact.deepopt_programs), 1)
                    schedule = compiled.compilation_report.program_schedules[0]
                    self.assertEqual(schedule["precision"], "bf16")
                    self.assertEqual(schedule["implementation"], "native")
                    self.assertEqual("x_payload" in schedule["saved_components"], compressed)

    def test_model_helper_captures_native_modules_without_fp8_preparation(self):
        model = nn.Sequential(nn.Linear(9, 13), nn.GELU(), nn.Linear(13, 5))
        reference = deepcopy(model)
        identities = [id(p) for p in model.parameters()]
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True))
        x = torch.randn(3, 9, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual, expected = model(x), reference(xr)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
        self.assertEqual(identities, [id(p) for p in model.parameters()])
        report = compilation_report(model)
        self.assertEqual(report.graphs[0].static_regions["bf16_linear_programs"], 2)
        self.assertEqual({p["module_path"] for p in report.graphs[0].program_schedules}, {"0", "2"})

    def test_frozen_weight_and_eval_do_not_save_codec_or_dense_input(self):
        model = nn.Sequential(nn.Linear(7, 5)).bfloat16()
        model[0].weight.requires_grad_(False)
        gm = torch.fx.symbolic_trace(model)
        x = torch.randn(3, 7, dtype=torch.bfloat16, requires_grad=True)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression={"enabled": True, "group_size": 1024}, config={}):
            ShapeProp(gm).propagate(x)
            compiled = compile_fx(gm, [x])
            program = compiled.backend_artifact.deepopt_programs[0]
            self.assertFalse(program.compressed)
            self.assertNotIn("bf16_input", program.saved_components)
            with patch("deepopt.opt.quantization.activation.quantize_pack_activation",
                       side_effect=AssertionError("unneeded pack")):
                compiled(x).sum().backward()

    def test_compressed_save_is_payload_and_metadata_not_dense_input(self):
        gm = torch.fx.symbolic_trace(nn.Sequential(nn.Linear(11, 7)).bfloat16())
        x = torch.randn(97, 11, dtype=torch.bfloat16, requires_grad=True)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression={"enabled": True, "group_size": 1024}, config={}):
            ShapeProp(gm).propagate(x)
            compiled = compile_fx(gm, [x])
            saved = []
            with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
                output = compiled(x)
        self.assertTrue(any(t.dtype == torch.int32 for t in saved))
        self.assertFalse(any(t is x for t in saved))
        output.sum().backward()

    def test_fusion_selects_bf16_epilogue_and_producer_pack(self):
        for compressed in (False, True):
            gm = torch.fx.symbolic_trace(nn.Sequential(nn.Linear(17, 17), nn.GELU(), nn.Linear(17, 5)).bfloat16())
            x = torch.randn(257, 17, dtype=torch.bfloat16, requires_grad=True)
            with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                          activation_compression={"enabled": compressed, "group_size": 1024}, config={}):
                ShapeProp(gm).propagate(x)
                ir = GraphLowering(gm, [x]).run()
            for node in ir.nodes:
                for value in node.outputs():
                    if value.meta is not None:
                        value.meta = replace(value.meta, device=torch.device("cuda:0"))
            decisions = []
            patterns = fuse_native(ir, enabled=True, unsafe_values=frozenset(),
                                   config=BackendConfig(bf16_gemm_backend="generated"),
                                   kernel_policy=KernelPolicy(), diagnostics=decisions)
            self.assertEqual(len(patterns), 1)
            linears = [n for n in ir.nodes if isinstance(n, NativeLinearNode)]
            self.assertTrue(linears[1].activation if compressed else linears[0].epilogue)
            self.assertEqual(sum(d["selected"] for d in decisions), 1)
            # CPU reference exercises the same fused saved-state contract.
            program = NativeLinearProgram(linears[1] if compressed else linears[0], KernelCodegenContext())
            source = torch.randn(257, 17, dtype=torch.bfloat16, requires_grad=True)
            module = gm.get_submodule("2" if compressed else "0")
            output = program(source, module.weight, module.bias)
            output.sum().backward()
            self.assertTrue(torch.isfinite(source.grad).all())

    def test_prepared_native_activation_norm_and_linear_own_saved_state(self):
        from deepopt.opt.policy import prepare_fp8_model

        model = nn.Sequential(nn.Linear(11, 17), nn.GELU(), nn.LayerNorm(17), nn.Linear(17, 7))
        model = prepare_fp8_model(model, False, activation_compression={"enabled": True, "group_size": 1024}, config={})[0]
        reference = deepcopy(model)
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True))
        x = torch.randn(3, 11, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with patch("deepopt.opt.nn.autograd.activation.elementwise.ActivationFunction.forward",
                       side_effect=AssertionError("legacy activation Function")), patch(
                           "deepopt.opt.nn.autograd.normalization.layer.LayerNormFunction.forward",
                           side_effect=AssertionError("legacy LayerNorm Function")):
                actual = model(x)
            expected = reference(xr)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        report = compilation_report(model).graphs[0]
        self.assertEqual(report.static_regions["bf16_linear_programs"], 2)
        self.assertEqual(report.static_regions["native_saved_state_programs"], 2)

    def test_backward_orders_preserve_all_gradients(self):
        model = nn.Sequential(nn.Linear(11, 7)).bfloat16()
        for order in ("dx_first", "dw_first"):
            gm = torch.fx.symbolic_trace(deepcopy(model))
            reference = deepcopy(model)
            x = torch.randn(97, 11, dtype=torch.bfloat16, requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression={"enabled": True, "group_size": 1024}, config={}):
                ShapeProp(gm).propagate(x)
                compiled = compile_fx(gm, [x], config=BackendConfig(bf16_backward_order=order))
                actual, expected = compiled(x), reference(xr)
            actual.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
            for a, b in zip(gm.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_residual_epilogue_and_aliased_input_gradients(self):
        class Residual(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(17, 17)

            def forward(self, x):
                return self.projection(x) + x

        gm = torch.fx.symbolic_trace(Residual().bfloat16())
        reference = deepcopy(gm)
        x = torch.randn(5, 17, dtype=torch.bfloat16, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        ShapeProp(gm).propagate(x)
        ir = GraphLowering(gm, [x]).run()
        for node in ir.nodes:
            for value in node.outputs():
                if value.meta is not None:
                    value.meta = replace(value.meta, device=torch.device("cuda:0"))
        patterns = fuse_native(ir, enabled=True, unsafe_values=frozenset(),
            config=BackendConfig(bf16_gemm_backend="generated"), kernel_policy=KernelPolicy(), diagnostics=[])
        self.assertEqual(len(patterns), 1)
        node = next(n for n in ir.nodes if isinstance(n, NativeLinearNode))
        program = NativeLinearProgram(node, KernelCodegenContext())
        actual = program(x, gm.projection.weight, gm.projection.bias, x)
        expected = reference(xr)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)

    def test_explicit_native_functional_plan_overrides_fp8_scope(self):
        root = nn.Module()
        root.weight = nn.Parameter(torch.randn(7, 11))
        root.bias = nn.Parameter(torch.randn(7))
        graph = torch.fx.Graph()
        xnode = graph.placeholder("x")
        w, b = graph.get_attr("weight"), graph.get_attr("bias")
        plan = linear_plan(False, activation_compression={"enabled": True, "group_size": 1024}, config={})
        graph.output(graph.call_function(do_linear, (xnode, w, b), {"plan": plan}))
        gm = torch.fx.GraphModule(root, graph)
        x = torch.randn(97, 11, requires_grad=True)
        with autocast("cpu", dtype=torch.bfloat16, fp8=True, activation_compression=False, config={}):
            ShapeProp(gm).propagate(x)
            compiled = compile_fx(gm, [x])
            actual = compiled(x)
            expected = do_linear(x, root.weight, root.bias, plan=plan)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(compiled.backend_artifact.deepopt_programs[0].compressed)

    def test_original_parameter_version_is_checked(self):
        gm = torch.fx.symbolic_trace(nn.Sequential(nn.Linear(11, 7)).bfloat16())
        x = torch.randn(97, 11, dtype=torch.bfloat16, requires_grad=True)
        ShapeProp(gm).propagate(x)
        compiled = compile_fx(gm, [x])
        output = compiled(x)
        with torch.no_grad():
            gm.get_submodule("0").weight.add_(1)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            output.sum().backward()

    def test_joint_prepared_activation_preserves_packed_derivative_and_dw(self):
        from deepopt.opt.policy import prepare_fp8_model
        from deepopt.compiler.operations import is_deepopt_target

        class Leaves(torch.fx.Tracer):
            def is_leaf_module(self, module, name):
                return is_deepopt_target(module) or super().is_leaf_module(module, name)

        for activation in (nn.GELU, nn.SiLU):
            for bits in (1, 2, 4, 8):
                prepared = prepare_fp8_model(nn.Sequential(activation(), nn.Linear(17, 7)).bfloat16(), False,
                    activation_compression={"enabled": True, "group_size": 1024, "bits": bits}, config={})[0]
                reference = deepcopy(prepared)
                gm = torch.fx.GraphModule(prepared, Leaves().trace(prepared))
                x = torch.randn(257, 17, dtype=torch.bfloat16, requires_grad=True)
                xr = x.detach().clone().requires_grad_()
                ShapeProp(gm).propagate(x)
                ir = GraphLowering(gm, [x]).run()
                for node in ir.nodes:
                    for value in node.outputs():
                        if value.meta is not None:
                            value.meta = replace(value.meta, device=torch.device("cuda:0"))
                patterns = fuse_native(ir, enabled=True, unsafe_values=frozenset(), config=BackendConfig(),
                                       kernel_policy=KernelPolicy(), diagnostics=[])
                self.assertEqual(len(patterns), 1)
                node = next(n for n in ir.nodes if isinstance(n, NativeLinearNode))
                program = NativeLinearProgram(node, KernelCodegenContext())
                module = gm.get_submodule("1")
                actual, expected = program(x, module.weight, module.bias), reference(xr)
                gradient = torch.randn_like(actual)
                actual.backward(gradient)
                expected.backward(gradient)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, xr.grad, rtol=0, atol=0)
                torch.testing.assert_close(module.weight.grad, reference[1].weight.grad, rtol=0, atol=0)
                self.assertIn("derivative_payload", program.saved_components)
                self.assertNotIn("activation_input", program.saved_components)

    def test_native_preparation_does_not_change_fp32_linear_compression(self):
        from deepopt.opt.policy import prepare_fp8_model

        original = nn.Sequential(nn.Linear(11, 7))
        reference = deepcopy(original)
        model = prepare_fp8_model(original, False, activation_compression={"enabled": True, "group_size": 1024}, config={})[0]
        x = torch.randn(97, 11, requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        actual, expected = model(x), reference(xr)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(model[0].weight.grad, reference[0].weight.grad, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class BF16GeneratedCUDA(unittest.TestCase):
    def test_generated_products_epilogues_packed_read_and_training(self):
        from deepopt.opt.backends.bf16 import product
        from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation

        if torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("BF16 tensor cores require NVIDIA compute capability >=8")
        for backend in ("triton", "tilelang"):
            if backend == "tilelang":
                import importlib.util
                if importlib.util.find_spec("tilelang") is None:
                    continue
            a = torch.randn(37, 29, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(19, 29, device="cuda", dtype=torch.bfloat16)
            bias = torch.randn(38, device="cuda")[::2]
            y, before = product(a, b, bias=bias, activation=("gelu", "none"), backend=backend,
                                return_preactivation=True)
            ref = nn.functional.linear(a, b, bias.bfloat16())
            torch.testing.assert_close(before, ref, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(y, nn.functional.gelu(ref), rtol=2e-2, atol=2e-2)
            residual = torch.randn_like(ref)
            torch.testing.assert_close(product(a, b, bias=bias, residual=residual, backend=backend),
                                       ref + residual, rtol=2e-2, atol=2e-2)
            packed = quantize_pack_activation(b.T.contiguous(), 1024)
            decoded = dequantize_unpack_activation(packed, dtype=torch.bfloat16)
            actual = product(a, None, backend=backend, packed_b=packed)
            torch.testing.assert_close(actual, a @ decoded, rtol=2e-2, atol=2e-2)
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1, self.fc2 = nn.Linear(17, 33), nn.Linear(33, 7)
                self.activation = nn.GELU()

            def forward(self, x):
                return self.fc2(self.activation(self.fc1(x)))

        from deepopt.opt.policy import prepare_fp8_model
        for compressed in (False, True):
            model = prepare_fp8_model(MLP().cuda(), False, activation_compression={"enabled": compressed, "group_size": 1024}, config={})[0]
            reference = deepcopy(model)
            compile_model(model, CompileConfig(enabled=True, backend="deepopt", options={
                "bf16_gemm_backend": "generated", "bf16_decode": "tiled",
                "fuse_deepopt_activations": True,
            }))
            x = torch.randn(2, 257, 17, device="cuda", requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y, expected = model(x), reference(xr)
            y.sum().backward()
            expected.sum().backward()
            torch.testing.assert_close(y, expected, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(x.grad, xr.grad, rtol=4e-2, atol=4e-2)
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=4e-2, atol=4e-2)
            self.assertGreater(compilation_report(model).graphs[0].static_regions["generated_bf16_programs"], 0)
            self.assertGreater(compilation_report(model).fused_deepopt_regions, 0)


if __name__ == "__main__":
    unittest.main()
