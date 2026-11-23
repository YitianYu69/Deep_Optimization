"""Native-derived SwiGLU preserves ten saved outputs and virtual BF16 casts."""

from dataclasses import replace
from copy import deepcopy
from math import prod
from types import SimpleNamespace
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt._bf16_policy import resolve_bf16_policy
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.compiler.autodiff import semantic, swiglu_fusion
from deepopt.compiler.autodiff.activation_saving import SavedActivationPolicy
from deepopt.compiler.autodiff.aot_auto_diff import AOTAutoDiff
from deepopt.compiler.autodiff.optimization import rebuild
from deepopt.compiler.codegen.wrapper import WrapperCodegen
from deepopt.compiler.ir import IRGraph, InputNode, TensorMeta, TensorValue, register_uses
from deepopt.compiler.registry import BackendRegistry, KernelCodegenContext
from deepopt.compiler.scheduler import Scheduler
from deepopt.compiler.trace.builder import GraphBuilder
from deepopt.compiler.trace.metadata_utils import contiguous_stride
from deepopt.kernel_policy import KernelPolicy
from deepopt.opt.nn.activation.elementwise_modules import adapter


class _ProjectionSwiGLU(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Conv2d(4, 8, 1, bias=False), nn.Conv2d(4, 8, 1, bias=False)
        self.activation = adapter("silu", plan=dict(fp8=False, bf16_policy=resolve_bf16_policy(policy).as_dict(),
            activation_compression=dict(enabled=True, bits=2, group_size=1024)))

    def forward(self, x):
        gate = self.activation(self.gate_proj(x))
        return gate * self.up_proj(x)


class _PreparedTracer(torch.fx.Tracer):
    def is_leaf_module(self, module, name):
        return hasattr(module, "plan") or super().is_leaf_module(module, name)


def compile_check(test, device):
    torch.manual_seed(291)
    source = _ProjectionSwiGLU("bf16_storage" if device == "cuda" else "torch_amp").to(device)
    sample = torch.randn(2, 4, 13, 17, device=device)
    results = []
    for enabled in (False, True):
        current = deepcopy(source)
        model = torch.fx.GraphModule(current, _PreparedTracer().trace(current))
        report = CompilationReport()
        with autocast(device, dtype=torch.bfloat16, fp8=False,
                bf16_policy="bf16_storage" if device == "cuda" else "torch_amp",
                activation_compression=dict(enabled=True, bits=2, group_size=1024)):
            ShapeProp(model).propagate(sample)
            executable = compile_fx(model, [sample], config=BackendConfig(require_full_aot=True,
                fuse_swiglu_residuals=enabled, swiglu_fusion_stage="forward_backward" if enabled else "off"), report=report)
            output = executable(sample)
        seed = torch.tensor(0.137, device=device, dtype=torch.bfloat16).expand(output.shape)
        gradients = torch.autograd.grad(output, tuple(model.parameters()), seed, retain_graph=True)
        again = torch.autograd.grad(output, tuple(model.parameters()), seed)
        for actual, expected in zip(gradients, again):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        test.assertTrue(all(value.dtype == torch.float32 and torch.isfinite(value).all().item() for value in gradients))
        test.assertIsNone(sample.grad)
        summary = report.as_dict()
        test.assertEqual(summary["runtime_fallbacks"], 0)
        test.assertFalse(summary["legacy_backward_operations"])
        test.assertEqual(summary["native_autodiff_regions"], 1)
        selected = [row["pattern"] for row in summary["fusion_candidates"] if row["selected"]]
        if device == "cuda" and enabled:
            test.assertIn(swiglu_fusion.FORWARD, selected)
            test.assertIn(swiglu_fusion.BACKWARD, selected)
        else:
            test.assertNotIn(swiglu_fusion.FORWARD, selected)
            test.assertNotIn(swiglu_fusion.BACKWARD, selected)
        results.append((output.detach(), gradients))
    for actual, expected in zip((results[1][0], *results[1][1]), (results[0][0], *results[0][1])):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def fixture(shape=(2, 8, 13, 17), *, device="cpu", need_x=True, need_up=True, fanout=False):
    graph, registry = IRGraph(), BackendRegistry()
    builder = GraphBuilder(graph)
    metadata = TensorMeta(shape, torch.bfloat16, torch.device(device), contiguous_stride(shape), need_x)
    x = builder.input(0, metadata, name="preactivation")
    up = builder.input(1, replace(metadata, requires_grad=need_up), name="raw_up")
    gate = TensorValue("rounded_gate", metadata)
    policy = resolve_bf16_policy("bf16_storage")
    plan = dict(fp8=False, bf16_policy=policy.as_dict(), activation_compression=dict(enabled=True, bits=2, group_size=1024))
    module = adapter("silu", plan=plan)
    node = SimpleNamespace(module=module, plan=module.plan, args=(x.value, None, None), kwargs={}, value=gate,
        output=lambda: gate, bf16_policy=policy, module_path="mlp.silu", node_name="silu", op="native_activation")
    spec = semantic.NativeActivation(node, KernelCodegenContext(kernel_policy=KernelPolicy()))
    graph.add(spec.lower(node))
    result = builder.wrap(gate) * up
    builder.output((result, builder.wrap(gate)) if fanout else result)
    semantic.register_rules(registry)
    trace = AOTAutoDiff(graph, registry=registry, saved_activation_policy=SavedActivationPolicy(True, 2, 1024)).diff()
    return graph, trace, registry, spec


def context(stage, *, enabled=True):
    return SimpleNamespace(fuse_swiglu_residuals=enabled, swiglu_fusion_stage=stage,
                           fusion_phases=("forward", "backward"), kernel_policy=KernelPolicy())


def select(graph, trace, stage):
    records, patterns = swiglu_fusion.forward(graph, context(stage))
    backward = swiglu_fusion.backward(trace.graph, context(stage), records)
    graph.validate()
    trace.graph.validate()
    return records, (*patterns, *backward)


def executables(graph, trace, registry):
    output = graph.nodes[-1]
    output.result = (output.result, trace.saved_tensors)
    register_uses(output)
    for node in trace.graph.nodes:
        if isinstance(node, InputNode) and node.index == 0:
            node.value.meta = replace(node.value.meta, stride=None)
    fw = WrapperCodegen(torch.nn.Module(), Scheduler(graph, registry=registry).run(), registry=registry,
                        enable_autodiff=False).run().fn
    bw = WrapperCodegen(torch.nn.Module(), Scheduler(trace.graph, registry=registry).run(), registry=registry,
                        enable_autodiff=False, diagnostic_phase="backward").run().fn
    return fw, bw


def cuda_metadata(graph, trace):
    for current in (graph, trace.graph):
        for node in current.nodes:
            for value in node.outputs():
                value.meta = replace(value.meta, device=torch.device("cuda:0"))


class NativeSwiGLUFusionTests(unittest.TestCase):
    def test_actual_compile_fx_cpu_native_projection_vjp_retains_legacy_route(self):
        compile_check(self, "cpu")

    def test_actual_compile_graph_native_projection_boundaries_admit_static_cuda_metadata(self):
        # Selection only: execute the real CPU control first, then inspect
        # its native compiler graphs under CUDA metadata without running them.
        forwards, backwards = [], []
        original_forward, original_backward = swiglu_fusion.forward, swiglu_fusion.backward

        def capture_forward(graph, options):
            forwards.append(graph)
            return original_forward(graph, options)

        def capture_backward(graph, options, diagnostics=None):
            backwards.append(graph)
            return original_backward(graph, options, diagnostics)

        with patch.object(swiglu_fusion, "forward", side_effect=capture_forward), \
                patch.object(swiglu_fusion, "backward", side_effect=capture_backward):
            compile_check(self, "cpu")
        graph, backward = forwards[-1], backwards[-1]
        graph.nodes[-1].result = graph.nodes[-1].result[0]
        rebuild(graph)
        trace = SimpleNamespace(graph=backward)
        cuda_metadata(graph, trace)
        for node in graph.nodes:
            for value in node.outputs():
                if value.meta.dtype == torch.bfloat16:
                    value.meta = replace(value.meta, stride=contiguous_stride(value.meta.shape))
        rows, patterns = select(graph, trace, "forward_backward")
        self.assertEqual(patterns, (swiglu_fusion.FORWARD, swiglu_fusion.BACKWARD),
                         [(row["pattern"], row["reason"]) for row in rows])

    def test_default_and_cpu_rejection_execute_real_native_saved_vjp(self):
        torch.manual_seed(811)
        graph, trace, registry, _ = fixture()
        before = tuple(id(node) for node in graph.nodes)
        rows, patterns = swiglu_fusion.forward(graph, context("off", enabled=False))
        self.assertEqual(patterns, ())
        self.assertEqual(tuple(id(node) for node in graph.nodes), before)
        self.assertTrue(rows)
        rows, patterns = swiglu_fusion.forward(graph, context("forward_backward"))
        self.assertEqual(patterns, ())
        self.assertEqual(rows[0]["reason"], "equal_shape_contiguous_cuda_bf16_required")
        fw, bw = executables(graph, trace, registry)
        x, up = torch.randn(2, 8, 13, 17).bfloat16(), torch.randn(2, 8, 13, 17).bfloat16()
        output, saved = fw(x, up)
        self.assertEqual(len(saved), 9)
        dy = torch.linspace(-1.17, 0.83, output.numel()).reshape(output.shape).bfloat16()
        first, again = bw(dy, *saved), bw(dy, *saved)
        for a, b in zip(first, again):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_stages_preserve_original_saved_identities_and_remove_actual_nodes(self):
        for stage, output_count, expected in (("activation_pack", 7, (swiglu_fusion.ACTIVATION_PACK,)),
                ("forward", 10, (swiglu_fusion.FORWARD,)),
                ("forward_backward", 10, (swiglu_fusion.FORWARD, swiglu_fusion.BACKWARD))):
            graph, trace, _, _ = fixture()
            saved = tuple(trace.saved_tensors)
            cuda_metadata(graph, trace)
            records, patterns = select(graph, trace, stage)
            self.assertEqual(patterns, expected)
            self.assertEqual(tuple(trace.saved_tensors), saved)
            composite = next(node for node in graph.nodes if getattr(node, "op", "").startswith("swiglu."))
            self.assertEqual(len(composite.values), output_count)
            self.assertEqual(composite.saved_tensors_for_bwd, composite.values[1:])
            self.assertTrue(all(any(value is result for result in composite.values) for value in
                                saved if value.name.startswith("rounded_gate")))
            if stage == "forward_backward":
                backward = next(node for node in trace.graph.nodes if getattr(node, "op", "") == "swiglu.backward")
                self.assertEqual(len(backward.values), 2)
                self.assertFalse(any(getattr(node, "op", "") in ("codec.restore_activation", "activation.coefficient_vjp")
                                     for node in trace.graph.nodes))
            json.dumps(records, allow_nan=False)

    def test_executor_spoofs_effects_policies_and_unknown_stages_reject(self):
        for kind in ("activation", "codec", "effect", "stream", "policy"):
            graph, trace, _, _ = fixture()
            cuda_metadata(graph, trace)
            activation = next(node for node in graph.nodes if getattr(node, "op", "") == "autodiff_native_activation")
            pack = next(node for node in graph.nodes if getattr(node, "op", "") == "codec.save_activation")
            options = context("forward_backward")
            if kind == "activation":
                activation.target.numerical_function = lambda *args: None
            elif kind == "codec":
                pack.target = lambda *args: None
            elif kind == "effect":
                activation.scheduling_constraints["mutation"] = "external"
            elif kind == "stream":
                activation.target.stream_id = 1
            else:
                options.kernel_policy = KernelPolicy(backend="tilelang")
            before = tuple(id(node) for node in graph.nodes)
            rows, patterns = swiglu_fusion.forward(graph, options)
            self.assertEqual(patterns, (), kind)
            self.assertEqual(tuple(id(node) for node in graph.nodes), before)
            self.assertFalse(rows[0]["selected"])
        graph, _, _, _ = fixture()
        with self.assertRaisesRegex(ValueError, "stage"):
            swiglu_fusion.forward(graph, context("unknown"))

    def test_fingerprints_are_stable_and_backward_admits_zero_stride_seed(self):
        fingerprints = []
        for _ in range(2):
            graph, trace, _, _ = fixture()
            cuda_metadata(graph, trace)
            seed = next(node.value for node in trace.graph.nodes if isinstance(node, InputNode) and node.index == 0)
            seed.meta = replace(seed.meta, stride=(0, 0, 0, 0))
            rows, patterns = select(graph, trace, "forward_backward")
            self.assertIn(swiglu_fusion.BACKWARD, patterns)
            fingerprints.append(tuple(row["implementation_fingerprint"] for row in rows))
            self.assertEqual(rows[-1]["training_fusion_abi"]["iteration_layout"], "logical_strided_reads")
        self.assertEqual(*fingerprints)

    def test_backward_fanout_retains_original_vjp(self):
        graph, trace, _, _ = fixture()
        cuda_metadata(graph, trace)
        rows, _ = swiglu_fusion.forward(graph, context("forward_backward"))
        activation = next(node for node in trace.graph.nodes if getattr(node, "op", "") == "activation.coefficient_vjp")
        intermediate = activation.args[0]
        output = trace.graph.nodes[-1]
        output.result = (output.result, intermediate)
        register_uses(output)
        before = tuple(id(node) for node in trace.graph.nodes)
        self.assertEqual(swiglu_fusion.backward(trace.graph, context("forward_backward"), rows), ())
        self.assertEqual(tuple(id(node) for node in trace.graph.nodes), before)
        self.assertEqual(rows[-1]["reason"], "activation_adjoint_fanout_or_operation")

    def test_full_forward_rejects_fanout_and_selective_saved_abi(self):
        for options in (dict(fanout=True), dict(need_x=False), dict(need_up=False)):
            graph, trace, _, _ = fixture(**options)
            cuda_metadata(graph, trace)
            before = tuple(id(node) for node in graph.nodes)
            rows, patterns = swiglu_fusion.forward(graph, context("forward_backward"))
            self.assertEqual(patterns, ())
            self.assertEqual(before, tuple(id(node) for node in graph.nodes))
            self.assertTrue(rows and not rows[0]["selected"])

    def test_activation_pack_retains_dense_fanout_and_runtime_rejects_cpu(self):
        graph, trace, _, _ = fixture(fanout=True)
        cuda_metadata(graph, trace)
        rows, patterns = swiglu_fusion.forward(graph, context("activation_pack"))
        self.assertEqual(patterns, (swiglu_fusion.ACTIVATION_PACK,))
        node = next(node for node in graph.nodes if getattr(node, "op", "") == "swiglu.activation_pack")
        with self.assertRaisesRegex(ValueError, "CUDA BF16"):
            node.target(torch.zeros((2, 8, 13, 17), dtype=torch.bfloat16))
        self.assertTrue(rows[0]["replacement_contract"]["dense_gate_materialized"])

    def test_original_geometry_metadata_and_kernel_offsets_exceed_int32(self):
        shape = (128, 384, 224, 224)
        self.assertGreater(prod(shape), 2**31)
        graph, trace, _, _ = fixture(shape)
        cuda_metadata(graph, trace)
        records, patterns = select(graph, trace, "forward_backward")
        self.assertEqual(patterns, (swiglu_fusion.FORWARD, swiglu_fusion.BACKWARD))
        self.assertEqual(records[0]["replacement_contract"]["output_count"], 10)
        source = Path(__file__).resolve().parents[2] / "src/deepopt/opt/backends/triton/activation/swiglu.py"
        text = source.read_text()
        self.assertIn("tl.program_id(0).to(tl.int64) * BLOCK", text)
        self.assertIn("group = tl.program_id(0).to(tl.int64)", text)


def _cuda_check(test, shape, *, stage="forward_backward", gradient_layout="contiguous"):
    torch.manual_seed(181)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    up = torch.randn_like(x)
    dy = torch.linspace(-1.31, 0.79, x.numel(), dtype=torch.bfloat16, device="cuda").reshape(shape)
    if gradient_layout == "zero":
        dy = torch.tensor(0.137, dtype=torch.bfloat16, device="cuda").expand(shape)
    elif gradient_layout == "transposed":
        dy = dy.transpose(-1, -2).contiguous().transpose(-1, -2)
    results = []
    for active in (False, True):
        graph, trace, registry, _ = fixture(shape, device="cuda")
        if active:
            records, patterns = select(graph, trace, stage)
            test.assertTrue(patterns)
            test.assertTrue(all(row["selected"] for row in records if row["requested"]))
        fw, bw = executables(graph, trace, registry)
        output, saved = fw(x, up)
        gradients = bw(dy, *saved)
        again = bw(dy, *saved)
        for a, b in zip(gradients, again):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        results.append((output, saved, gradients))
    for actual, expected in zip((results[1][0], *results[1][1], *results[1][2]),
                                (results[0][0], *results[0][1], *results[0][2])):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.cuda.synchronize()


@unittest.skipUnless(torch.cuda.is_available(), "SwiGLU fused CUDA kernels require a GPU")
class NativeSwiGLUFusionCudaTests(unittest.TestCase):
    def test_actual_compile_fx_native_ad_retained_backward_and_expanded_cotangent(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 required")
        compile_check(self, "cuda")

    def test_all_training_stages_tails_and_bf16_rounding(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 required")
        for stage in ("activation_pack", "forward", "forward_backward"):
            _cuda_check(self, (2, 8, 13, 17), stage=stage)

    def test_strided_and_expanded_cotangents_preserve_adjoint_rounding(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 required")
        for layout in ("zero", "transposed"):
            _cuda_check(self, (2, 8, 13, 17), gradient_layout=layout)

    def test_concurrent_invocations_keep_saved_payloads_private(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 required")
        shape = (2, 8, 13, 17)
        graph, trace, registry, _ = fixture(shape, device="cuda")
        select(graph, trace, "forward_backward")
        fw, bw = executables(graph, trace, registry)
        inputs = [(torch.randn(shape, device="cuda", dtype=torch.bfloat16),
                   torch.randn(shape, device="cuda", dtype=torch.bfloat16)) for _ in range(2)]
        expected = []
        for x, up in inputs:
            output, saved = fw(x, up)
            expected.append((output, saved, bw(torch.ones_like(output), *saved)))
        streams, actual = [torch.cuda.Stream(), torch.cuda.Stream()], []
        for stream, (x, up) in zip(streams, inputs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                output, saved = fw(x, up)
                actual.append((output, saved, bw(torch.ones_like(output), *saved)))
        for stream in streams:
            stream.synchronize()
        self.assertNotEqual(actual[0][1][0].data_ptr(), actual[1][1][0].data_ptr())
        for a, b in zip(actual, expected):
            for actual_value, expected_value in zip((a[0], *a[1], *a[2]), (b[0], *b[1], *b[2])):
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)

    def test_native_derived_vjp_returns_finite_fp32_master_parameter_gradients(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 required")
        shape = (2, 8, 13, 17)
        masters = (torch.randn(shape, device="cuda"), torch.randn(shape, device="cuda"))
        seed = torch.linspace(-1.3, 0.7, prod(shape), device="cuda", dtype=torch.bfloat16).reshape(shape)
        results = []
        for enabled in (False, True):
            graph, trace, registry, _ = fixture(shape, device="cuda")
            if enabled:
                select(graph, trace, "forward_backward")
            fw, bw = executables(graph, trace, registry)

            class NativeDerivedSwiGLU(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, up):
                    output, saved = fw(x, up)
                    ctx.save_for_backward(*saved)
                    return output

                @staticmethod
                def backward(ctx, gradient):
                    return bw(gradient, *ctx.saved_tensors)

            parameters = tuple(torch.nn.Parameter(value.clone()) for value in masters)
            output = NativeDerivedSwiGLU.apply(*(value.bfloat16() for value in parameters))
            gradients = torch.autograd.grad(output, parameters, seed)
            self.assertTrue(all(value.dtype == torch.float32 and torch.isfinite(value).all().item()
                                for value in gradients))
            results.append(gradients)
        for actual, expected in zip(results[1], results[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_bf16_rounded_output_pack_emitter(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 required")
        from deepopt.opt.backends.triton.activation import elementwise
        from deepopt.opt.nn.activation.geometry import ReLUGeometry
        from deepopt.opt.quantization.activation import PackedActivation, quantize_pack_activation

        shape = (2, 8, 13, 17)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        geometry = ReLUGeometry(shape)
        groups = (x.numel() + 1023) // 1024
        packed = PackedActivation(torch.empty((groups, 64), dtype=torch.int32, device="cuda"),
            torch.empty((groups,), dtype=torch.float32, device="cuda"),
            torch.empty((groups,), dtype=torch.float32, device="cuda"), geometry.matrix_shape, 1024, bits=2)
        output, _, _, _, _, _ = elementwise.forward(x, geometry, kind="silu", approximate="none",
            dtype=torch.bfloat16, block_shape=None, packed=True, bits=2, group_size=1024,
            save_mask=True, inplace=False, output_pack=packed)
        control = quantize_pack_activation(output, 1024, bits=2)
        for actual, expected in zip((packed.data, packed.step, packed.minimum),
                                    (control.data, control.step, control.minimum)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("DEEPOPT_TEST_LARGE_CUDA_SWIGLU") == "1",
                     "Original SwiGLU geometry requires explicit CUDA capacity opt-in")
class NativeSwiGLUFusionOriginalShapeCudaTests(unittest.TestCase):
    def test_original_128_384_224_224_native_saved_vjp(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("BF16 required")
        _cuda_check(self, (128, 384, 224, 224))


if __name__ == "__main__":
    unittest.main()
