"""Native multiplication coefficient masks, recipes and derivative ownership."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.opt.nn.functional import do_mul


COMPRESSION = {"enabled": True, "bits": 2, "group_size": 1024}


class NativeMultiplicationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1828)
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def compile(self, module, samples):
        graph = torch.fx.symbolic_trace(module)
        report = CompilationReport()
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=COMPRESSION):
            ShapeProp(graph).propagate(*samples)
            compiled = compile_fx(graph, samples, config=BackendConfig(require_full_aot=True), report=report)
        return graph, compiled, report.graphs[0]

    def check(self, shape, other_shape, mask):
        class Mul(nn.Module):
            def forward(self, x, y):
                return x * y

        samples = [torch.randn(dimensions, dtype=torch.bfloat16, requires_grad=need)
                   for dimensions, need in zip((shape, other_shape), mask)]
        references = [value.detach().clone().requires_grad_(value.requires_grad) for value in samples]
        _, compiled, report = self.compile(Mul(), samples)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=COMPRESSION):
            actual = compiled(*samples)
        expected = do_mul(*references, plan={"fp8": False, "activation_compression": COMPRESSION})
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        for value, reference in zip(samples, references):
            torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)
        self.assertEqual(report.legacy_backward_operations, ())
        self.assertEqual(report.eager_autograd_operations, ())
        self.assertTrue(report.native_autodiff_operations)
        region = compiled.backend_artifact.joint_autograd_regions[0]
        bindings = [item for item in region.trace.saved_value_specs if item["representation"] == "packed_saved_operand"]
        expected_packs = int(mask[1] and samples[0].numel() >= 1024) + int(mask[0] and samples[1].numel() >= 1024)
        self.assertEqual(len(bindings), expected_packs)
        self.assertEqual(sum(getattr(node, "op", None) == "codec.save_activation" for node in region.graph.nodes), expected_packs)
        return region

    def test_se_swiglu_and_scale_masks_match_eager(self):
        for shapes in (((2, 16, 8, 8), (2, 16, 1, 1)), ((2, 512), (2, 512)), ((2, 512), ())):
            for mask in ((True, False), (False, True), (True, True)):
                with self.subTest(shapes=shapes, mask=mask):
                    self.check(*shapes, mask)

    def test_same_source_records_each_gradient_consumer_on_one_pack(self):
        class Square(nn.Module):
            def forward(self, x):
                return x * x

        x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        _, compiled, _ = self.compile(Square(), [x])
        region = compiled.backend_artifact.joint_autograd_regions[0]
        bindings = [item for item in region.trace.saved_value_specs if item["representation"] == "packed_saved_operand"]
        self.assertEqual(len(bindings), 1)
        self.assertEqual({item["role"] for item in bindings[0]["consumers"]},
                         {"mul.grad_input_coefficient", "mul.grad_other_coefficient"})
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=COMPRESSION):
            outputs = [compiled(x), compiled(x)]
        reference = x.detach().clone().requires_grad_()
        expected = do_mul(reference, reference, plan={"fp8": False, "activation_compression": COMPRESSION})
        outputs[0].sum().backward(retain_graph=True)
        expected.sum().backward()
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
        x.grad = None
        outputs[0].sum().backward()
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
        x.grad = None
        outputs[1].sum().backward()
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)

    def test_fp32_parameter_scalar_promotion_and_native_owner(self):
        class Scale(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(0.7))

            def forward(self, x):
                return self.scale * x

        model = Scale()
        reference = deepcopy(model)
        x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        rx = x.detach().clone().requires_grad_()
        graph, compiled, _ = self.compile(model, [x])
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=COMPRESSION):
            actual = compiled(x)
        expected = do_mul(reference.scale, rx, plan={"fp8": False, "activation_compression": COMPRESSION})
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
        torch.testing.assert_close(graph.scale.grad, reference.scale.grad, rtol=0, atol=0)
        specs = compiled.backend_artifact.joint_autograd_regions[0].trace.saved_value_specs
        self.assertTrue(any(item.get("retention_reason") == "model_owned_state" for item in specs))

    def test_torch_and_tensor_multiply_aliases_remain_native(self):
        class Multiply(nn.Module):
            def forward(self, x, y):
                return torch.multiply(x, y).multiply(y)

        samples = [torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True) for _ in range(2)]
        _, compiled, report = self.compile(Multiply(), samples)
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=COMPRESSION):
            actual = compiled(*samples)
        self.assertEqual(actual.dtype, torch.bfloat16)
        actual.sum().backward()
        self.assertEqual(report.eager_autograd_operations, ())
        self.assertEqual(report.legacy_backward_operations, ())

    def test_fullgraph_capture_preserves_native_owner_and_scope_snapshot(self):
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model

        class Square(nn.Module):
            def forward(self, x):
                return torch.multiply(x, x)

        model = Square()
        compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                                           options={"require_full_aot": True}))
        x = torch.randn(2, 512, dtype=torch.bfloat16, requires_grad=True)
        reference = x.detach().clone().requires_grad_()
        with autocast("cpu", dtype=torch.bfloat16, fp8=False, activation_compression=COMPRESSION):
            expected = do_mul(reference, reference)
            with patch("deepopt.opt.nn.binary.MultiplicationFunction.forward", side_effect=AssertionError("eager owner")):
                actual = model(x)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, reference.grad, rtol=0, atol=0)
        report = compilation_report(model)
        self.assertTrue(report.graphs)
        self.assertTrue(all(not item.eager_autograd_operations and not item.legacy_backward_operations
                            for item in report.graphs))
        self.assertTrue(any(item.native_autodiff_operations for item in report.graphs))


@unittest.skipUnless(torch.cuda.is_available(), "BF16 multiplication qualification requires CUDA")
class NativeMultiplicationCudaTests(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("BF16 qualification requires compute capability >= 8")
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_bf16_vae_binary_paths_with_fp32_trainable_scalar(self):
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model

        class Binary(nn.Module):
            def __init__(self, scalar):
                super().__init__()
                self.scalar = nn.Parameter(torch.tensor(0.7, device="cuda")) if scalar else None

            def forward(self, x, y):
                return x * (self.scalar if self.scalar is not None else y)

        for kind in ("se", "swiglu", "residual_scale"):
            with self.subTest(kind=kind):
                torch.compiler.reset()
                model = Binary(kind == "residual_scale")
                reference = deepcopy(model)
                x = torch.randn(2, 16, 8, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                y = torch.randn((2, 16, 1, 1) if kind == "se" else x.shape,
                                device="cuda", dtype=torch.bfloat16, requires_grad=True)
                rx, ry = (value.detach().clone().requires_grad_() for value in (x, y))
                compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                                                   options={"require_full_aot": True}))
                with autocast("cuda", dtype=torch.bfloat16, fp8=False, activation_compression=COMPRESSION):
                    expected = reference(rx, ry)
                    with patch("deepopt.opt.nn.binary.MultiplicationFunction.forward", side_effect=AssertionError("eager owner")):
                        actual = model(x, y)
                gradient = torch.randn_like(actual)
                expected.backward(gradient)
                actual.backward(gradient)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
                torch.testing.assert_close(y.grad, ry.grad, rtol=0, atol=0)
                for value, ref in zip(model.parameters(), reference.parameters()):
                    torch.testing.assert_close(value.grad, ref.grad, rtol=0, atol=0)
                reports = compilation_report(model).graphs
                self.assertTrue(reports)
                self.assertTrue(all(not item.eager_autograd_operations and not item.legacy_backward_operations
                                    for item in reports))


if __name__ == "__main__":
    unittest.main()
