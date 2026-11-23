"""Target selection, generated formulas and import boundaries without a GPU.

These tests validate compiler contracts and scalar formulas; real TileLang
parsing, CUDA compilation, streams and performance require the GPU suite.
"""

from dataclasses import replace
import importlib.util
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt.compiler import BackendConfig, compile_fx
from deepopt.compiler.autodiff_runtime import build_autograd_region
from deepopt.compiler.ir import PointwiseNode
from deepopt.compiler.lowering import GraphLowering
from deepopt.compiler.registry import BackendRegistry
from deepopt.compiler.scheduler import FusionGroup, Scheduler, SinglePointwiseOp
from deepopt.kernel_policy import kernel_policy_scope, resolve_kernel_policy


class TorchScalarLanguage:
    exp = staticmethod(torch.exp)
    log = staticmethod(torch.log)
    abs = staticmethod(torch.abs)
    tanh = staticmethod(torch.tanh)
    erf = staticmethod(torch.erf)

    @staticmethod
    def cast(value, dtype):
        return torch.as_tensor(value).to(getattr(torch, dtype))

    @staticmethod
    def if_then_else(condition, yes, no):
        return torch.where(condition, torch.as_tensor(yes), torch.as_tensor(no))

    @staticmethod
    def min(left, right):
        return torch.minimum(torch.as_tensor(left), torch.as_tensor(right))

    @staticmethod
    def max(left, right):
        return torch.maximum(torch.as_tensor(left), torch.as_tensor(right))


def lowered(module, *, requires_grad=True, cuda_metadata=False, dtype=torch.float32):
    sample = torch.randn(3, 17, dtype=dtype, requires_grad=requires_grad)
    graph = torch.fx.symbolic_trace(module)
    ShapeProp(graph).propagate(sample)
    ir = GraphLowering(graph, [sample]).run()
    if cuda_metadata:
        for node in ir.nodes:
            for value in node.outputs():
                if value.meta is not None:
                    value.meta = replace(value.meta, device=torch.device("cuda:0"))
    return graph, sample, ir


class KernelBackendCompilerTests(unittest.TestCase):
    def test_configuration_validates_targets_and_shared_policy(self):
        for backend in ("triton", "tilelang", "mixed"):
            self.assertEqual(BackendConfig.from_options({"kernel_backend": backend}).kernel_backend, backend)
        for options in ({"kernel_backend": "automatic"},
                        {"kernel_backend": "tilelang", "kernel_policy": {"backend": "triton"}}):
            with self.assertRaises(ValueError):
                BackendConfig.from_options(options)

    def test_compiler_import_and_cpu_training_need_neither_optional_compiler(self):
        code = '''
import sys
import torch
sys.modules['triton'] = None
sys.modules['tilelang'] = None
from deepopt.compiler import BackendConfig, compile_fx
from torch.fx.passes.shape_prop import ShapeProp
for backend in ('triton', 'tilelang', 'mixed'):
    x = torch.randn(3, 7, requires_grad=True)
    gm = torch.fx.symbolic_trace(torch.nn.Sigmoid())
    ShapeProp(gm).propagate(x)
    f = compile_fx(gm, [x], config=BackendConfig(kernel_backend=backend))
    f(x).sum().backward()
    assert torch.isfinite(x.grad).all()
assert not any(n.startswith('deepopt.compiler.codegen.triton') for n in sys.modules)
assert 'tilelang' in sys.modules and sys.modules['tilelang'] is None
assert 'triton' in sys.modules and sys.modules['triton'] is None
'''
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root / "benchmarks"))
        from compiler.installed_runtime import child_environment

        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root,
                                env=child_environment(root),
                                text=True, capture_output=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mixed_scheduler_splits_target_boundaries(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.relu(x * 2.0 + 1.0)

        _, _, ir = lowered(M(), cuda_metadata=True)
        policy = resolve_kernel_policy({"backend": "mixed", "roles": {"compiler.relu": "tilelang"}})
        scheduler = Scheduler(ir, kernel_policy=policy)
        partitions = [item for item in scheduler.run().items if isinstance(item, (FusionGroup, SinglePointwiseOp))]
        self.assertEqual(len(partitions), 2)
        nodes = lambda p: p.nodes if isinstance(p, FusionGroup) else [p.node]
        self.assertEqual([scheduler.backend_for(nodes(p)[0]) for p in partitions], ["triton", "tilelang"])
        self.assertEqual([n.op for n in nodes(partitions[0])], ["mul", "add"])

    def test_mixed_extension_fallback_uses_emitter_metadata_and_preserves_hard_pins(self):
        from deepopt.compiler.registry import OperatorExtension

        class ExtensionNode(PointwiseNode):
            pass

        _, _, ir = lowered(nn.Sigmoid(), requires_grad=False, cuda_metadata=True)
        old = next(node for node in ir.nodes if isinstance(node, PointwiseNode))
        old.__class__ = ExtensionNode
        registry = BackendRegistry()
        registry.register(OperatorExtension(
            name="tilelang_sigmoid", node_type=ExtensionNode,
            matcher=lambda node, gm: False, lowering=lambda context, node: None,
            tilelang_codegen=lambda node, args, kwargs: f"(1.0 / (1.0 + T.exp(-{args[0]})))",
        ))
        policy = resolve_kernel_policy({"backend": "mixed", "fallback": "other"})
        scheduler = Scheduler(ir, registry=registry, kernel_policy=policy)
        partition = next(item for item in scheduler.run().items if isinstance(item, SinglePointwiseOp))
        self.assertEqual(partition.kernel_backend, "tilelang")
        for options in ({"backend": "mixed"},
                        {"backend": "mixed", "fallback": "other", "roles": {"compiler.sigmoid": "triton"}},
                        {"backend": "mixed", "fallback": "other", "families": {"compiler_pointwise": "triton"}}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "expression emitter"):
                Scheduler(ir, registry=registry, kernel_policy=options).run()

    def test_tilelang_source_rounds_each_node_and_is_lazy(self):
        from deepopt.compiler.codegen.tilelang.pointwise import TileLangCodegen, generate_kernel_source

        class M(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x * 1.5 + 0.25)

        _, _, ir = lowered(M(), cuda_metadata=True, dtype=torch.bfloat16)
        policy = resolve_kernel_policy("tilelang")
        partitions = [item for item in Scheduler(ir, kernel_policy=policy).run().items
                      if isinstance(item, (FusionGroup, SinglePointwiseOp))]
        source, _ = generate_kernel_source(partitions[0])
        compile(source, "<tilelang-source-contract>", "exec")
        self.assertEqual(sum(' = T.cast(' in line and line.endswith(', "bfloat16")')
                             for line in source.splitlines()), 3)
        self.assertIn('T.cast(block, "int64")', source)
        self.assertNotIn("triton", source)
        with tempfile.TemporaryDirectory() as directory:
            before = sys.modules.get("tilelang")
            first = TileLangCodegen(cache_dir=directory, kernel_policy=policy).compile_partitions(partitions)
            again = TileLangCodegen(cache_dir=directory, kernel_policy=policy).compile_partitions(partitions)
            self.assertEqual(first[0].path, again[0].path)
            self.assertEqual(first[0].backend, "tilelang")
            self.assertIs(sys.modules.get("tilelang"), before)
            self.assertFalse(first[0].kernel.executables)

    def test_aot_backward_keeps_selected_target_after_scope_exit(self):
        graph, _, ir = lowered(nn.Sigmoid(), cuda_metadata=True)
        node = next(node for node in ir.nodes if isinstance(node, PointwiseNode))
        policy = resolve_kernel_policy("tilelang")
        with tempfile.TemporaryDirectory() as directory:
            region = build_autograd_region(
                SinglePointwiseOp(node), None, graph, registry=BackendRegistry(),
                cache_dir=directory, block_size=128, kernel_policy=policy,
            )
            self.assertEqual(region.kernel_policy, policy)
            artifact = region.backward_fn.backend_artifact
            self.assertTrue(artifact.tilelang_artifacts)
            self.assertFalse(artifact.triton_artifacts)
            self.assertTrue(all('"float32"' in a.source for a in artifact.tilelang_artifacts))

    def test_scalar_activation_formulas_match_native_values_and_gradients(self):
        from deepopt.compiler.codegen.tilelang.expressions import expression

        x = torch.tensor([-8.0, -3.0, -1.0, -0.0, 0.0, 0.3, 1.0, 3.0, 8.0])
        g = torch.linspace(0.1, 1.0, x.numel())
        cases = (
            ("relu", nn.ReLU(), "threshold_backward", {}),
            ("sigmoid", nn.Sigmoid(), "sigmoid_backward", {}),
            ("tanh", nn.Tanh(), "tanh_backward", {}),
            ("gelu", nn.GELU(), "gelu_backward", {}),
            ("gelu", nn.GELU(approximate="tanh"), "gelu_backward", {"approximate": repr("tanh")}),
            ("silu", nn.SiLU(), "silu_backward", {}),
            ("leaky_relu", nn.LeakyReLU(-0.3), "leaky_relu_backward", {"negative_slope": "-0.3"}),
            ("hardtanh", nn.Hardtanh(-0.7, 0.8), "hardtanh_backward", {"min_val": "-0.7", "max_val": "0.8"}),
            ("hardswish", nn.Hardswish(), "hardswish_backward", {}),
            ("hardsigmoid", nn.Hardsigmoid(), "hardsigmoid_backward", {}),
            ("softplus", nn.Softplus(beta=1.7, threshold=7), "softplus_backward", {"beta": "1.7", "threshold": "7"}),
            ("mish", nn.Mish(), "mish_backward", {}),
        )
        for op, module, backward, options in cases:
            with self.subTest(op=op, options=options):
                sample = x.clone().requires_grad_()
                expected = module(sample)
                expected.backward(g)
                actual = eval(expression(op, ["x"], options), {"T": TorchScalarLanguage, "x": x})
                backward_input = actual if op in ("sigmoid", "tanh") else x
                gradient = eval(expression(backward, ["g", "x"], options),
                                {"T": TorchScalarLanguage, "x": backward_input, "g": g})
                torch.testing.assert_close(actual, expected.detach(), rtol=2e-6, atol=2e-6)
                torch.testing.assert_close(gradient, sample.grad, rtol=3e-6, atol=3e-6)

    def test_fused_preparation_source_covers_pair_and_codec_without_dense_activation(self):
        from deepopt.compiler.codegen.tilelang.activation_quantize import generate_preparation_source

        for activation in (("gelu", "none"), ("gelu", "tanh"), ("silu", "none")):
            for paired, group, bits in ((False, 0, 2), (True, 0, 2), (False, 17, 1), (False, 33, 4)):
                source = generate_preparation_source(7, 19, "bfloat16", "float8_e4m3fn", activation,
                                                     paired=paired, group=group, bits=bits)
                compile(source, "<tilelang-preparation-contract>", "exec")
                self.assertIn("def statistics(", source)
                self.assertIn("def cast(", source)
                self.assertNotIn("triton", source)
                self.assertNotIn("torch.", source)
                self.assertNotIn("T.empty", source)

    def test_atomic_preparation_rejects_opposite_pins_in_both_directions(self):
        from deepopt.compiler.activation_quantize import preparation_backend

        for target, other in (("triton", "tilelang"), ("tilelang", "triton")):
            for family in ("quantization", "scaling", "activation_codec"):
                policy = {"backend": "mixed", "preferred": target,
                          "families": {"compiler_preparation": target, family: other}}
                compression = {"enabled": True, "group_size": 17, "bits": 2}
                with self.subTest(target=target, family=family), self.assertRaisesRegex(ValueError, "unfused schedule"):
                    preparation_backend(compression=compression, policy=policy)
            unused_codec = {"backend": "mixed", "preferred": target,
                            "families": {"compiler_preparation": target, "activation_codec": other}}
            self.assertEqual(preparation_backend(policy=unused_codec), target)

    def test_generated_pointwise_source_executes_tail_and_rounding_contract(self):
        from deepopt.compiler.codegen.tilelang.pointwise import generate_kernel_source
        from deepopt.opt.backends.tilelang import _runtime
        from tests.opt.test_tilelang_operations import _Language

        class Language(_Language):
            log = staticmethod(lambda value: torch.log(torch.as_tensor(value)))

        class M(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x * 1.5 + 0.25)

        language = Language()
        module = M()
        for dtype in (torch.float32, torch.bfloat16):
            _, sample, ir = lowered(module, dtype=dtype, cuda_metadata=True)
            partition = next(item for item in Scheduler(ir, kernel_policy="tilelang").run().items
                             if isinstance(item, FusionGroup))
            source, _ = generate_kernel_source(partition, block_size=32)
            namespace = {}
            exec(compile(source, "<interpreted-tilelang-pointwise>", "exec"), namespace)
            with patch.object(_runtime, "runtime", return_value=(None, language)), \
                    patch.object(_runtime, "compile_kernel", language.compile):
                kernel = namespace["build_pointwise"]("cpu")
                output = torch.full((sample.numel() + 1,), -123.0, dtype=dtype)
                kernel(sample.detach().flatten(), output)
            self.assertEqual(output[-1].item(), -123.0)
            torch.testing.assert_close(output[:-1].reshape(sample.shape), module(sample).detach(),
                                       atol=2e-6 if dtype == torch.float32 else 0,
                                       rtol=2e-6 if dtype == torch.float32 else 0)

    def test_fused_preparation_kernels_match_independent_cpu_codec(self):
        from deepopt.compiler.codegen.tilelang.activation_quantize import generate_preparation_source
        from deepopt.opt.backends.tilelang import _runtime
        from deepopt.opt.quantization.activation import quantize_pack_activation
        from deepopt.opt.quantization.quantizer import quantize_fp8
        from tests.opt.test_tilelang_operations import _Language

        language = _Language()
        source = generate_preparation_source(3, 19, "bfloat16", "float8_e4m3fn", ("gelu", "none"),
                                             group=17, bits=2)
        namespace = {}
        exec(compile(source, "<interpreted-tilelang-preparation>", "exec"), namespace)
        with patch.object(_runtime, "runtime", return_value=(None, language)), \
                patch.object(_runtime, "compile_kernel", language.compile):
            statistics, cast = namespace["build"]("cpu")
        x = torch.randn(3, 19, dtype=torch.bfloat16)
        activated = nn.functional.gelu(x)
        q = quantize_fp8(activated)
        expected = quantize_pack_activation(q.data, 17, bits=2)
        partial = torch.empty(1)
        statistics(x.flatten(), partial)
        torch.testing.assert_close(partial[0], activated.float().abs().max(), rtol=0, atol=0)
        scale = q.scale_inv.reciprocal().flatten()
        normal = torch.full((3 * 32,), -1.0, dtype=torch.float8_e4m3fn)
        payload, steps, minima = torch.empty(4, 2, dtype=torch.int32), torch.empty(4), torch.empty(4)
        cast(x.flatten(), scale, normal, payload.flatten(), steps, minima)
        torch.testing.assert_close(normal.reshape(3, 32)[:, :19].float(), q.data.float(), rtol=0, atol=0)
        self.assertTrue((normal.reshape(3, 32)[:, 19:].float() == 0).all())
        for actual, wanted in ((payload, expected.data), (steps, expected.step), (minima, expected.minimum)):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


_CUDA_REASON = ("Real NVIDIA CUDA hardware is required for TileLang compiler qualification"
                if not torch.cuda.is_available() or torch.version.hip is not None else
                "TileLang is not installed" if importlib.util.find_spec("tilelang") is None else "")
if _CUDA_REASON and os.environ.get("DEEPOPT_REQUIRE_TILELANG") == "1":
    raise RuntimeError(_CUDA_REASON)


@unittest.skipIf(_CUDA_REASON, _CUDA_REASON)
class TileLangCompilerCUDATests(unittest.TestCase):
    def test_forward_aot_backward_and_capture_never_enter_triton(self):
        from deepopt import CUDAGraphCore

        class M(nn.Module):
            def forward(self, x):
                return nn.functional.gelu(torch.sigmoid(x * 1.5 + 0.25))

        prefixes = ("deepopt.compiler.codegen.triton", "deepopt.opt.backends.triton")
        blocked = {name: None for name in tuple(sys.modules) if name.startswith(prefixes)}
        blocked.update({name: None for name in prefixes})
        model = M().cuda()
        sample = torch.randn(5, 257, device="cuda", requires_grad=True)
        gm = torch.fx.symbolic_trace(model)
        ShapeProp(gm).propagate(sample)
        with patch.dict(sys.modules, blocked), kernel_policy_scope("tilelang"):
            compiled = compile_fx(gm, [sample], config=BackendConfig(kernel_backend="tilelang"))
            output = compiled(sample)
            output.sum().backward()
            self.assertTrue(compiled.backend_artifact.tilelang_artifacts)
            self.assertFalse(compiled.backend_artifact.triton_artifacts)
            sample.grad = None
            handle = CUDAGraphCore("cuda").capture_callable(compiled, (sample,), warmup=3, autograd=True)
            try:
                actual_input = torch.randn_like(sample, requires_grad=True)
                reference_input = actual_input.detach().clone().requires_grad_()
                actual, expected = handle(actual_input), model(reference_input)
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
                torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=3e-6, atol=3e-6)
            finally:
                handle.close()

    def test_fused_preparation_both_formats_paired_and_packed(self):
        from deepopt.compiler.activation_quantize import prepare
        from deepopt.opt.quantization.activation import quantize_pack_activation
        from deepopt.opt.quantization.quantizer import quantize_fp8

        x = torch.linspace(-3, 3, 7 * 19, device="cuda", dtype=torch.bfloat16).reshape(7, 19)
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for paired, compression in ((False, None), (True, None),
                                        (False, {"group_size": 17, "bits": 2})):
                with self.subTest(dtype=dtype, paired=paired, compression=compression), kernel_policy_scope("tilelang"):
                    normal, transposed, packed = prepare(x, activation=("gelu", "none"), fp8_dtype=dtype,
                                                        paired=paired, compression=compression)
                    expected = quantize_fp8(nn.functional.gelu(x).cpu(), fp8_dtype=dtype)
                    torch.testing.assert_close(normal.data.float().cpu(), expected.data.float(), rtol=0, atol=0)
                    torch.testing.assert_close(normal.scale_inv.cpu(), expected.scale_inv, rtol=0, atol=0)
                    if paired:
                        torch.testing.assert_close(transposed.data.float().cpu(), expected.data.T.float(), rtol=0, atol=0)
                    if compression is not None:
                        codec = quantize_pack_activation(expected.data, 17, bits=2)
                        for a, b in ((packed.data, codec.data), (packed.step, codec.step), (packed.minimum, codec.minimum)):
                            torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
