"""Native training residual fusion preserves saves and removes actual stages."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt._bf16_policy import BF16Policy
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.compiler.autodiff import AOTAutoDiff, norm_activation_residuals as fusion, semantic, spatial
from deepopt.compiler.ir import InputNode, IRGraph, OutputNode, TensorMeta, TensorValue, register_uses
from deepopt.compiler.registry import BackendRegistry, KernelCodegenContext
from deepopt.compiler.trace.metadata_utils import contiguous_stride
from deepopt.kernel_policy import KernelPolicy
from deepopt.opt.nn.activation.elementwise_modules import adapter
from deepopt.opt.nn.normalization.modules import DOGroupNorm
from tests.compiler.test_normalization_fusion import execute


def fixture(shape=(2, 64, 32, 32), *, needs=(True, True, True), stage="forward_backward", cuda=True):
    """Only allocate affine vectors, including for original B128 metadata."""
    graph = IRGraph()
    context = KernelCodegenContext(kernel_policy=KernelPolicy(), fuse_norm_activation_residuals=stage != "off",
        norm_activation_fusion_stage=stage)
    module = nn.GroupNorm(8 if shape[1] >= 8 else 2, shape[1])
    module.plan = {"activation_compression": dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True)}
    source_meta = TensorMeta(shape, torch.bfloat16, "cpu", contiguous_stride(shape), needs[0])
    source = TensorValue("input", source_meta)
    weight = TensorValue("weight", TensorMeta((shape[1],), torch.float32, "cpu", (1,), needs[1]))
    bias = TensorValue("bias", replace(weight.meta, requires_grad=needs[2]))
    for index, value in enumerate((source, weight, bias)):
        graph.add(InputNode(index, value))
    normalized = TensorValue("normalized", replace(source_meta, requires_grad=any(needs)))
    raw = SimpleNamespace(module=module, module_path="norm", node_name="norm", args=(source, weight, bias, None, None, None),
        value=normalized, values=(normalized,), output=lambda: normalized, kwargs={}, op="deepopt_normalization",
        options={"kind": "group", "groups": module.num_groups, "eps": module.eps}, plan=module.plan,
        root=module, bf16_policy=BF16Policy("bf16_storage"))
    norm = spatial.Normalization(raw, context)
    norm_node = norm.lower(raw)
    graph.add(norm_node)
    act_module = nn.SiLU()
    act_module.kind, act_module.plan = "silu", module.plan
    activated = TensorValue("activated", replace(normalized.meta))
    raw = SimpleNamespace(module=act_module, module_path="silu", node_name="silu", args=(normalized, None, None),
        value=activated, values=(activated,), output=lambda: activated, kwargs={}, op="native_activation",
        bf16_policy=BF16Policy("bf16_storage"))
    activation = semantic.NativeActivation(raw, context)
    activation_node = activation.lower(raw)
    graph.add(activation_node)
    graph.add(OutputNode(activated))
    registry = BackendRegistry()
    semantic.register_rules(registry)
    spatial.register_rules(registry)
    trace = AOTAutoDiff(graph, registry=registry).diff()
    output = graph.nodes[-1]
    output.result = (output.result, trace.saved_tensors)
    register_uses(output)
    if cuda:
        for current in (graph, trace.graph):
            for node in current.nodes:
                for value in node.outputs():
                    if value.meta is not None and not (value.meta.dtype == torch.int64 and value.meta.shape == ()):
                        value.meta = replace(value.meta, device="cuda:0")
        norm.metas = tuple(replace(meta, device="cuda:0") if meta is not None else None for meta in norm.metas)
        norm.backend = "triton"
        activation.input_meta = replace(activation.input_meta, device="cuda:0")
    return graph, trace, norm, activation, context, norm_node, activation_node


class _NormSiLU(nn.Module):
    def __init__(self, channels=16, policy="torch_amp"):
        super().__init__()
        plan = dict(fp8=False, bf16_policy=BF16Policy(policy).as_dict(),
            activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True))
        self.norm = DOGroupNorm(nn.GroupNorm(8, channels), plan=plan)
        self.activation = adapter("silu", plan=plan)

    def forward(self, x):
        return self.activation(self.norm(x))


class _PreparedTracer(torch.fx.Tracer):
    def is_leaf_module(self, module, name):
        return hasattr(module, "plan") or super().is_leaf_module(module, name)


def prepared_trace(source):
    return torch.fx.GraphModule(source, _PreparedTracer().trace(source))


class NativeNormActivationResidualTests(unittest.TestCase):
    def test_cotangent_layout_preparation_preserves_values_and_accounts_for_real_copies(self):
        from deepopt.opt.backends._shared.normalization_activation import canonicalize_group_silu_cotangent

        source = torch.linspace(-1, 1, 2 * 16 * 64 * 64).bfloat16().reshape(2, 16, 64, 64)
        expanded = torch.tensor(.173, dtype=torch.bfloat16).expand_as(source)
        storage = torch.empty(2, 16, 64, 128, dtype=torch.bfloat16)
        storage[..., ::2].copy_(source)
        views = (source, expanded, source.transpose(-1, -2).contiguous().transpose(-1, -2), storage[..., ::2])
        for value in views:
            with self.subTest(stride=tuple(value.stride())):
                prepared, info = canonicalize_group_silu_cotangent(value)
                torch.testing.assert_close(prepared, value, rtol=0, atol=0)
                self.assertTrue(prepared.is_contiguous())
                self.assertEqual(info["upstream_cotangent_stride"], tuple(value.stride()))
                self.assertEqual(info["reduction_cotangent_stride"], tuple(source.stride()))
                self.assertEqual(info["raw_cotangent_copy_bytes"],
                    0 if value.is_contiguous() else value.numel() * value.element_size())
                self.assertEqual(prepared is value, value.is_contiguous())
        with self.assertRaisesRegex(ValueError, "dense BF16"):
            canonicalize_group_silu_cotangent(source.float())

    def test_typed_controls_and_native_owner_fail_closed(self):
        self.assertFalse(BackendConfig().fuse_norm_activation_residuals)
        for values in ({"fuse_norm_activation_residuals": True}, {"norm_activation_fusion_stage": "forward"},
                {"fuse_norm_activation_residuals": 1}, {"norm_activation_fusion_stage": None},
                {"fuse_norm_activation_residuals": True, "norm_activation_fusion_stage": "activation_pack"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BackendConfig(**values)
        with self.assertRaisesRegex(ValueError, "native AutoDiff"):
            BackendConfig(fuse_norm_activation_residuals=True, norm_activation_fusion_stage="forward",
                          use_native_autodiff=False)
        with self.assertRaisesRegex(ValueError, "fusion_phases"):
            BackendConfig(fuse_norm_activation_residuals=True, norm_activation_fusion_stage="forward_backward",
                          fusion_phases="forward")

    def test_local_stages_do_not_override_global_phase_controls(self):
        graph, trace, _, _, context, _, _ = fixture()
        context = replace(context, fusion_phases=("forward",))
        records, patterns = fusion.forward(graph, context)
        self.assertTrue(patterns)
        self.assertFalse(fusion.backward(trace.graph, context, records))
        graph, _, _, _, context, _, _ = fixture()
        self.assertFalse(fusion.forward(graph, replace(context, fusion_phases=("backward",)))[1])

    def test_backward_does_not_move_coefficient_work_across_opaque_effects(self):
        from deepopt.compiler.ir import CallNode

        graph, trace, _, _, context, _, _ = fixture()
        records, _ = fusion.forward(graph, context)
        derivative = next(node for node in trace.graph.nodes if getattr(node, "op", None) == "activation.coefficient_vjp")
        effect = CallNode("opaque_mutates_upstream", lambda gradient: gradient.zero_(), (derivative.args[0],), {}, ())
        effect.stateful = True
        trace.graph.nodes.insert(trace.graph.nodes.index(derivative) + 1, effect)
        self.assertFalse(fusion.backward(trace.graph, context, records))
        self.assertEqual(records[-1]["reason"], "intervening_backward_operation")

    def test_forward_and_backward_preserve_twelve_outputs_and_saved_bindings(self):
        for stage in ("off", "forward", "forward_backward"):
            graph, trace, _, _, context, norm, act = fixture(stage=stage)
            saves, norm_slots, activation_slots = trace.saved_tensors, norm.values[1:], act.values[1:]
            outputs = trace.graph.nodes[-1].result
            decisions, patterns = fusion.forward(graph, context)
            backward = fusion.backward(trace.graph, context, decisions)
            self.assertEqual(patterns, () if stage == "off" else (fusion.FORWARD,))
            self.assertEqual(backward, (fusion.BACKWARD,) if stage == "forward_backward" else ())
            self.assertEqual(trace.saved_tensors, saves)
            self.assertIs(trace.graph.nodes[-1].result, outputs)
            if stage != "off":
                composite = next(node for node in graph.nodes if getattr(node, "op", None) == "normalization.silu_residual_forward")
                self.assertEqual(composite.values, (act.values[0], *norm_slots, *activation_slots))
                self.assertEqual(len(composite.values), 12)
                self.assertEqual(composite.saved_tensors_for_bwd, (*norm_slots, *activation_slots))
                self.assertNotIn(act, graph.nodes)
                self.assertEqual(len(decisions[0]["saved_output_abi"]["outputs"]), 12)
            if stage == "forward_backward":
                self.assertFalse(any(getattr(node, "op", None) == "activation.coefficient_vjp" for node in trace.graph.nodes))
                self.assertEqual(decisions[-1]["materialized_activation_gradient_bytes"], 0)
            graph.validate()
            trace.graph.validate()
            json.dumps(decisions, allow_nan=False)

    def test_original_geometry_selection_without_payload_and_unaligned_rejection(self):
        for shape, accepted in (((128, 384, 224, 224), True), ((128, 1728, 56, 56), False)):
            graph, _, _, _, context, _, _ = fixture(shape)
            decisions, patterns = fusion.forward(graph, context)
            self.assertEqual(bool(patterns), accepted)
            if accepted:
                self.assertEqual(decisions[0]["removed_normalization_output_bytes"], 4932501504)
                self.assertEqual(decisions[0]["saved_output_abi"]["index_bits"], 64)
            else:
                self.assertEqual(decisions[0]["reason"], "normalization_and_flat_activation_groups_misaligned")

    def test_cpu_fanout_effects_and_wrong_codec_do_not_select(self):
        for failure in ("cpu", "fanout", "intervening", "codec", "untrusted", "variant", "saved_metadata", "stream"):
            graph, trace, norm, activation, context, norm_node, act = fixture(cuda=failure != "cpu")
            if failure == "fanout":
                graph.nodes[-1].result = (graph.nodes[-1].result, norm_node.values[0])
                register_uses(graph.nodes[-1])
            elif failure == "intervening":
                from deepopt.compiler.ir import CallNode
                graph.nodes.insert(graph.nodes.index(act), CallNode("effect", lambda: None, (), {}, ()))
            elif failure == "codec":
                activation.compression = dict(activation.compression, bits=4)
            elif failure == "untrusted":
                norm_node.target = lambda *args: None
            elif failure == "variant":
                norm.normalization_variant = "candidate"
            elif failure == "saved_metadata":
                act.values[1].meta = replace(act.values[1].meta, stride=None)
            elif failure == "stream":
                act.target.stream_id = 1
            decisions, patterns = fusion.forward(graph, context)
            self.assertFalse(patterns)
            self.assertFalse(fusion.backward(trace.graph, context, decisions))
            self.assertTrue(all(not row["selected"] for row in decisions))

    def test_cpu_native_execution_remains_unchanged_with_enabled_device_ineligible_option(self):
        torch.manual_seed(21)
        source, sample = _NormSiLU(), torch.randn(2, 16, 32, 32).bfloat16()
        results = []
        for enabled in (False, True):
            model, report = prepared_trace(deepcopy(source)), CompilationReport()
            x = sample.clone().requires_grad_()
            with autocast("cpu", dtype=torch.bfloat16, fp8=False,
                    activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True)):
                ShapeProp(model).propagate(x)
                executable = compile_fx(model, [x], config=BackendConfig(require_full_aot=True,
                    fuse_norm_activation_residuals=enabled,
                    norm_activation_fusion_stage="forward_backward" if enabled else "off"), report=report)
                output = executable(x)
            seed = torch.linspace(-1, 1, output.numel()).to(output.dtype).reshape(output.shape)
            targets = (x, *model.parameters())
            first = torch.autograd.grad(output, targets, seed, retain_graph=True)
            again = torch.autograd.grad(output, targets, seed)
            for actual, expected in zip(first, again):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            results.append((output.detach(), *first))
            self.assertEqual(report.runtime_fallbacks, 0)
            self.assertFalse(any(row["selected"] for row in report.as_dict()["fusion_candidates"]
                                 if row["pattern"] in (fusion.FORWARD, fusion.BACKWARD)))
        for actual, expected in zip(results[0], results[1]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_runtime_rejects_cpu_before_kernel_launch(self):
        from deepopt.opt.backends._shared.normalization_activation import validate_group_silu_residuals
        _, _, norm, _, _, _, _ = fixture(cuda=False)
        source = torch.empty(norm.geometry.shape, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "CUDA BF16"):
            validate_group_silu_residuals(source, (), norm.geometry, norm.compression, ("silu", "none"))


def cuda_compare(test, shape, *, needs=(True, True, True), profile=False, gradient_layout="contiguous"):
    graph, trace, spec, activation, context, _, _ = fixture(shape, needs=needs)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight, bias = torch.randn(shape[1], device="cuda"), torch.randn(shape[1], device="cuda")
    with torch.no_grad():
        expected, saved = execute(graph, x, weight, bias)
        upstream = torch.randn_like(expected)
        if gradient_layout == "zero":
            upstream = torch.tensor(.173, device="cuda", dtype=torch.bfloat16).expand(shape)
        elif gradient_layout == "transposed":
            upstream = upstream.transpose(-1, -2).contiguous().transpose(-1, -2)
        expected_gradients = execute(trace.graph, upstream, *saved)
        control_repeat = execute(trace.graph, upstream, *saved)
        gradient_roles = [role for role, need in zip(("gradient/input", "gradient/weight", "gradient/bias"), needs) if need]
        for role, first, again in zip(gradient_roles, expected_gradients, control_repeat):
            torch.testing.assert_close(first, again, rtol=0, atol=0,
                msg=f"unfused backward repeatability: layout={gradient_layout}, role={role}")
        decisions, patterns = fusion.forward(graph, context)
        backward = fusion.backward(trace.graph, context, decisions)
        test.assertEqual(patterns, (fusion.FORWARD,))
        test.assertEqual(backward, (fusion.BACKWARD,))
        if profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                  torch.profiler.ProfilerActivity.CUDA], profile_memory=True) as capture:
                actual, fused_saved = execute(graph, x, weight, bias)
                actual_gradients = execute(trace.graph, upstream, *fused_saved)
                torch.cuda.synchronize()
            names = [event.name for event in capture.events()]
            test.assertFalse(any("_activation_forward" in name or "_activation_backward" in name for name in names))
        else:
            actual, fused_saved = execute(graph, x, weight, bias)
            actual_gradients = execute(trace.graph, upstream, *fused_saved)
        roles = ("output", *(f"saved/{value.name}" for value in trace.saved_tensors), *gradient_roles)
        for role, actual_value, expected_value in zip(roles, (actual, *fused_saved, *actual_gradients),
                                                        (expected, *saved, *expected_gradients)):
            for start in range(0, actual_value.numel(), 1 << 20):
                torch.testing.assert_close(actual_value.reshape(-1)[start:start + (1 << 20)],
                    expected_value.reshape(-1)[start:start + (1 << 20)], rtol=0, atol=0,
                    msg=f"native normalization fusion: layout={gradient_layout}, role={role}, chunk_start={start}")
        repeated = execute(trace.graph, upstream, *fused_saved)
        for actual_value, expected_value in zip(repeated, actual_gradients):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        test.assertTrue(spec.module.last_execution["activation_residual_fused"])
        test.assertEqual(spec.module.last_execution["materialized_activation_gradient_bytes"], 0)
        test.assertEqual(spec.module.last_execution["raw_cotangent_copy_bytes"],
            0 if upstream.is_contiguous() else upstream.numel() * upstream.element_size())
        test.assertTrue(activation.module.last_execution["compression"])


@unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA saved-residual fusion execution")
class NativeNormActivationResidualCudaTests(unittest.TestCase):
    def test_strided_and_expanded_cotangents_preserve_native_normalization_vjp(self):
        for layout in ("zero", "transposed"):
            with self.subTest(gradient_layout=layout):
                cuda_compare(self, (2, 16, 64, 64), gradient_layout=layout)

    def test_concurrent_invocations_keep_residual_storage_private(self):
        graph, trace, _, _, context, _, _ = fixture((2, 16, 64, 64))
        records, _ = fusion.forward(graph, context)
        self.assertTrue(fusion.backward(trace.graph, context, records))
        weight, bias = torch.randn(16, device="cuda"), torch.randn(16, device="cuda")
        inputs = [torch.randn(2, 16, 64, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2)]

        def run(x):
            output, saved = execute(graph, x, weight, bias)
            return output, saved, execute(trace.graph, torch.ones_like(output), *saved)

        expected = [run(x) for x in inputs]
        streams, actual = [torch.cuda.Stream(), torch.cuda.Stream()], []
        for stream, x in zip(streams, inputs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                actual.append(run(x))
        for stream in streams:
            stream.synchronize()
        # Weight is intentionally a saved alias; inverse, xhat and derivative
        # states belong to each invocation and may never share their storage.
        private = [(a, b) for a, b in zip(actual[0][1], actual[1][1])
                   if a.numel() and a.device.type == "cuda" and a.data_ptr() != weight.data_ptr()]
        self.assertTrue(private)
        self.assertTrue(all(a.data_ptr() != b.data_ptr() for a, b in private))
        for a, b in zip(actual, expected):
            for actual_value, expected_value in zip((a[0], *a[1], *a[2]), (b[0], *b[1], *b[2])):
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)

    def test_actual_compiled_native_forward_and_retained_backward_select_composites(self):
        torch.manual_seed(22)
        source, sample = _NormSiLU(64, "bf16_storage").cuda(), torch.randn(2, 64, 32, 32, device="cuda", dtype=torch.bfloat16)
        results = []
        for enabled in (False, True):
            model, report = prepared_trace(deepcopy(source)), CompilationReport()
            x = sample.clone().requires_grad_()
            with autocast("cuda", dtype=torch.bfloat16, fp8=False, bf16_policy="bf16_storage",
                    activation_compression=dict(enabled=True, bits=2, group_size=1024, norm_xhat_recenter=True)):
                ShapeProp(model).propagate(x)
                executable = compile_fx(model, [x], config=BackendConfig(require_full_aot=True,
                    share_compatible_packs=True, fuse_norm_activation_residuals=enabled,
                    norm_activation_fusion_stage="forward_backward" if enabled else "off"), report=report)
                output = executable(x)
            seed = torch.linspace(-1, 1, output.numel(), device="cuda").to(output.dtype).reshape(output.shape)
            first = torch.autograd.grad(output, (x, *model.parameters()), seed, retain_graph=True)
            repeated = torch.autograd.grad(output, (x, *model.parameters()), seed)
            for actual, expected in zip(first, repeated):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            selected = {row["pattern"] for row in report.as_dict()["fusion_candidates"] if row["selected"]}
            self.assertEqual(fusion.FORWARD in selected, enabled)
            self.assertEqual(fusion.BACKWARD in selected, enabled)
            self.assertEqual(report.runtime_fallbacks, 0)
            self.assertFalse(report.as_dict()["legacy_backward_operations"])
            results.append((output.detach(), *first))
        for actual, expected in zip(results[0], results[1]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_saved_outputs_and_vjp_match_standalone_native_recipe(self):
        for needs in ((True, True, True), (True, False, False), (False, True, True)):
            with self.subTest(needs=needs):
                cuda_compare(self, (2, 16, 64, 64), needs=needs, profile=True)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("DEEPOPT_RUN_H200_NORM_ACTIVATION") == "1",
                     "opt in to original B128 CUDA normalization/SiLU geometry")
class NativeNormActivationResidualOriginalShapeCudaTests(unittest.TestCase):
    def test_original_b128_saved_abi_vjp_and_64bit_addressing(self):
        cuda_compare(self, (128, 384, 224, 224))


if __name__ == "__main__":
    unittest.main()
