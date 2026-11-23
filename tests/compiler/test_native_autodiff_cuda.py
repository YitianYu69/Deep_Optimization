"""Target-GPU qualification for native-derived BF16 and FP8 training graphs."""

from copy import deepcopy
import importlib.util
import unittest
from unittest.mock import patch

import torch
from torch import nn


@unittest.skipUnless(torch.cuda.is_available(), "CUDA qualification requires a GPU")
class NativeAutoDiffCudaTests(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("Generated BF16 requires NVIDIA compute capability >= 8")
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)

    def test_generated_bf16_saved_state_and_backward_fusion(self):
        from deepopt import autocast
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.kernel_policy import kernel_policy_scope
        from deepopt.opt.policy import prepare_fp8_model

        executed = 0
        for backend in ("triton", "tilelang"):
            if importlib.util.find_spec(backend) is None:
                continue
            for compressed in (False, True):
                with self.subTest(backend=backend, compressed=compressed), kernel_policy_scope(backend):
                    torch.compiler.reset()
                    torch.manual_seed(614)
                    compression = {"enabled": compressed, "bits": 4, "group_size": 1024}
                    model = prepare_fp8_model(nn.Sequential(nn.Linear(63, 96), nn.SiLU(), nn.Linear(96, 31)).cuda().bfloat16(),
                                              False, activation_compression=compression, config={})[0]
                    reference = deepcopy(model)
                    compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True, options={
                        "require_full_aot": True, "kernel_backend": backend, "bf16_gemm_backend": "generated",
                        "bf16_decode": "tiled", "fuse_deepopt_activations": True}))
                    x = torch.randn(3, 17, 63, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                    xr = x.detach().clone().requires_grad_()
                    with autocast("cuda", enabled=False, fp8=False, activation_compression=compression, config={}):
                        expected = reference(xr)
                        with patch("deepopt.compiler.native_linear._BF16Function.forward", side_effect=AssertionError("legacy")):
                            actual = model(x)
                    gradient = torch.randn_like(actual)
                    expected.backward(gradient)
                    with patch("deepopt.compiler.native_linear.NativeLinearProgram.backward", side_effect=AssertionError("legacy")):
                        actual.backward(gradient)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                    torch.testing.assert_close(x.grad, xr.grad, rtol=0.02, atol=0.02)
                    for a, b in zip(model.parameters(), reference.parameters()):
                        torch.testing.assert_close(a.grad, b.grad, rtol=0.02, atol=0.02)
                    report = compilation_report(model).graphs[0]
                    self.assertGreater(report.native_autodiff_operations, 0)
                    self.assertFalse(report.legacy_backward_operations or report.eager_autograd_operations)
                    self.assertGreater(report.fused_deepopt_regions, 0)
                    if compressed:
                        self.assertTrue(any("dx_packed_derivative" in pattern for pattern in report.fusion_patterns))
                    executed += 1
        if not executed:
            self.skipTest("Install Triton or TileLang for generated BF16 qualification")

    def test_fp8_native_derived_backward_uses_existing_gpu_kernels(self):
        from deepopt import FP8Config, autocast
        from deepopt.compiler import compilation_report
        from deepopt.config import CompileConfig
        from deepopt.engine.execution.compilation import compile_model
        from deepopt.kernel_policy import kernel_policy_scope
        from deepopt.opt.policy import prepare_fp8_model

        if importlib.util.find_spec("triton") is None:
            self.skipTest("Install Triton for FP8 qualification")
        with kernel_policy_scope("triton"):
            settings = FP8Config(enabled=True, backend="triton", out_dtype="bf16", backward_precision="auto")
            compression = {"enabled": True, "bits": 4, "group_size": 1024}
            model = prepare_fp8_model(nn.Sequential(nn.Linear(64, 1024), nn.SiLU(), nn.LayerNorm(1024), nn.Linear(1024, 32)).cuda(),
                                      settings, activation_compression=compression, config={})[0]
            reference = deepcopy(model)
            compile_model(model, CompileConfig(enabled=True, backend="deepopt", fullgraph=True,
                                               options={"require_full_aot": True}))
            x = torch.randn(4, 17, 64, device="cuda", requires_grad=True)
            xr = x.detach().clone().requires_grad_()
            with autocast("cuda", enabled=True, dtype=torch.bfloat16, fp8=settings,
                          activation_compression=compression, config={}):
                expected, actual = reference(xr), model(x)
            gradient = torch.randn_like(actual)
            expected.backward(gradient)
            with patch("deepopt.opt.nn.autograd.linear._DOLinearFunction._backward", side_effect=AssertionError("legacy")):
                actual.backward(gradient)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
            torch.testing.assert_close(x.grad, xr.grad, rtol=0.02, atol=0.02)
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=0.02, atol=0.02)
            self.assertTrue(all(not graph.eager_autograd_operations and not graph.legacy_backward_operations
                                for graph in compilation_report(model).graphs))

    def test_native_flash_attention_handles_unaligned_head_dimensions(self):
        from torch.fx.passes.shape_prop import ShapeProp
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from deepopt import autocast
        from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
        from deepopt.opt.nn.functional import do_scaled_dot_product_attention

        for dim in (2, 6, 10):
            for compressed in (False, True):
                for scale in (None, .31):
                    with self.subTest(dim=dim, compressed=compressed, scale=scale):
                        torch.manual_seed(1828)
                        compression = {"enabled": compressed, "bits": 4, "group_size": 1024}
                        dropout = .2 if dim == 10 else 0.
                        graph = torch.fx.Graph()
                        args = tuple(graph.placeholder(name) for name in ("q", "k", "v"))
                        graph.output(graph.call_function(do_scaled_dot_product_attention, args,
                            {"dropout_p": dropout, "scale": scale,
                             "plan": {"fp8": False, "activation_compression": compression}}))
                        model = torch.fx.GraphModule(nn.Module(), graph)
                        inputs = tuple(torch.randn(86, 2, length, dim, device="cuda", dtype=torch.bfloat16,
                                                   requires_grad=True) for length in (3, 5, 5))
                        references = tuple(value.detach().clone().requires_grad_() for value in inputs)
                        report = CompilationReport()
                        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                            with autocast("cuda", enabled=False, fp8=False,
                                          activation_compression=compression, config={}):
                                ShapeProp(model).propagate(*inputs)
                                compiled = compile_fx(model, inputs,
                                    config=BackendConfig(require_full_aot=True), report=report)
                                torch.cuda.manual_seed_all(1828)
                                expected = model(*references)
                                torch.cuda.manual_seed_all(1828)
                                actual = compiled(*inputs)
                            gradient = torch.randn_like(expected)
                            expected.backward(gradient)
                            with patch("torch.autograd.grad", side_effect=AssertionError("delegated derivative")):
                                actual.backward(gradient)
                        torch.cuda.synchronize()
                        self.assertEqual(actual.shape, expected.shape)
                        torch.testing.assert_close(actual, expected, rtol=.02, atol=.02)
                        for value, reference in zip(inputs, references):
                            self.assertEqual(value.grad.shape, value.shape)
                            torch.testing.assert_close(value.grad, reference.grad, rtol=.02, atol=.02)
                        self.assertGreater(report.graphs[0].native_autodiff_operations, 0)
                        self.assertFalse(report.graphs[0].legacy_backward_operations)
                        self.assertFalse(report.graphs[0].eager_autograd_operations)


if __name__ == "__main__":
    unittest.main()
