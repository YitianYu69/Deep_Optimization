"""Native-derived normalization fusion legality and materialized codec oracles."""

from dataclasses import replace
import json
from math import prod
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from deepopt._bf16_policy import BF16Policy

from deepopt.compiler.autodiff import AOTAutoDiff
from deepopt.compiler.autodiff import normalization_fusion, semantic, spatial
from deepopt.compiler.autodiff.joint import map_tree
from deepopt.compiler.ir import InputNode, IRGraph, OutputNode, TensorMeta, TensorValue, register_uses
from deepopt.compiler.registry import BackendRegistry, KernelCodegenContext
from deepopt.compiler.trace.builder import GraphBuilder
from deepopt.kernel_policy import KernelPolicy


def fixture(kind="layer", *, shape=(3, 17), normalized_shape=None, bits=2, compressed=False,
            needs=(True, True, True), dtype=torch.bfloat16, activation="silu", approximate="none",
            policy=None, groups=2, dtype_out=None, channels_last=False, affine=True, strided=False):
    graph = IRGraph()
    builder = GraphBuilder(graph)
    policy = policy or KernelPolicy()
    context = KernelCodegenContext(kernel_policy=policy, fuse_deepopt_activations=True)
    x = (torch.randn((*shape[:-1], shape[-1] * 2), dtype=dtype)[..., ::2] if strided else
         torch.randn(shape, dtype=dtype))
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    x.requires_grad_(needs[0])
    module = (nn.LayerNorm(normalized_shape or shape[-1], elementwise_affine=affine) if kind == "layer" else
              nn.GroupNorm(groups, shape[1], affine=affine))
    if affine:
        module.weight.requires_grad_(needs[1])
        module.bias.requires_grad_(needs[2])
    module.plan = {"activation_compression": {"enabled": compressed, "bits": bits, "group_size": 1024,
                                             "norm_xhat_recenter": True}}
    tensors = tuple(value for value in (x, module.weight, module.bias) if value is not None)
    supplied = iter(enumerate(tensors))
    args = []
    for value in (x, module.weight, module.bias):
        if value is None:
            args.append(None)
        else:
            index, value = next(supplied)
            args.append(builder.input(index, TensorMeta(tuple(value.shape), value.dtype, value.device,
                tuple(value.stride()), value.requires_grad)).value)
    args = tuple(args)
    output = TensorValue("normalized", TensorMeta(shape, dtype_out or dtype, x.device, None, any(needs)))
    raw = SimpleNamespace(module=module, module_path="norm", node_name="norm_call", args=args,
        value=output, values=(output,), output=lambda: output, kwargs={}, op="native_layer_norm" if kind == "layer" else "deepopt_normalization",
        options={"kind": "group", "groups": groups, "eps": module.eps} if kind == "group" else None,
        plan=module.plan, root=module, bf16_policy=BF16Policy())
    if kind == "group":
        raw.args = (*args, None, None, None)
    spec = semantic.NativeLayerNorm(raw, context) if kind == "layer" else spatial.Normalization(raw, context)
    norm = spec.lower(raw)
    graph.add(norm)
    activated = builder.emit(activation, output, **({"approximate": approximate} if activation == "gelu" else {}))
    builder.output(activated)
    registry = BackendRegistry()
    semantic.register_rules(registry)
    spatial.register_rules(registry)
    trace = AOTAutoDiff(graph, registry=registry).diff()
    output_node = graph.nodes[-1]
    output_node.result = (output_node.result, trace.saved_tensors)
    register_uses(output_node)
    return graph, trace, spec, norm, context, tensors


def mark_cuda(*graphs):
    for graph in graphs:
        for node in graph.nodes:
            for value in node.outputs():
                if value.meta is not None:
                    value.meta = replace(value.meta, device="cuda:0")


def execute(graph, *inputs):
    values = {}
    for node in graph.nodes:
        if isinstance(node, InputNode):
            values[node.value] = inputs[node.index]
        elif isinstance(node, OutputNode):
            return map_tree(node.result, values)
        else:
            actual = node.target(*map_tree(node.args, values), **map_tree(node.kwargs, values))
            outputs = node.outputs()
            actual = (actual,) if len(outputs) == 1 else actual
            values.update(zip(outputs, actual))


class NormalizationFusionTests(unittest.TestCase):
    def test_policy_descriptor_is_part_of_candidate_execution_identity(self):
        records = []
        for policy in ('torch_amp', 'bf16_storage'):
            graph, _, spec, _, context, _ = fixture('group', shape=(2, 4, 3, 5), compressed=True)
            spec.bf16_policy = BF16Policy(policy)
            mark_cuda(graph)
            candidates, _ = normalization_fusion.forward(graph, context)
            records.append(candidates[0])
        self.assertNotEqual(records[0]['candidate_id'], records[1]['candidate_id'])
        self.assertNotEqual(records[0]['implementation_fingerprint'], records[1]['implementation_fingerprint'])

    def test_experimental_native_group_preserves_spatial_residual_abi_and_rounding(self):
        for needs in ((True, True, True), (True, False, False), (False, True, False), (False, False, True)):
            for activation, approximate in (("silu", "none"), ("gelu", "none"), ("gelu", "tanh")):
                with self.subTest(needs=needs, activation=activation, approximate=approximate):
                    graph, trace, spec, norm, context, tensors = fixture("group", shape=(2, 4, 3, 17),
                        needs=needs, activation=activation, approximate=approximate, channels_last=True)
                    self.assertIsNotNone(spec.native_group)
                    expected, saved = execute(graph, *tensors)
                    upstream = torch.randn_like(expected)
                    expected_grads = execute(trace.graph, upstream, *saved)
                    mark_cuda(graph, trace.graph)
                    abi = tuple(value.meta for value in norm.values[1:])
                    self.assertEqual(len(abi), 8)
                    context = replace(context, qualify_native_group_fusion=True)
                    candidates, patterns = normalization_fusion.forward(graph, context)
                    patterns += normalization_fusion.backward(trace.graph, context, candidates)
                    self.assertEqual(len(patterns), 2)
                    self.assertEqual(tuple(value.meta for value in norm.values[-8:]), abi)
                    actual, actual_saved = execute(graph, *tensors)
                    actual_grads = execute(trace.graph, upstream, *actual_saved)
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
                    for a, e in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(a, e, rtol=0.025, atol=0.04)
                    for row in candidates:
                        self.assertEqual(row['qualification'], 'experimental_cuda_unqualified')
                        self.assertFalse(row['native_statistics_contract']['cuda_numerically_qualified'])
                        self.assertTrue(row['kernel_evidence']['required_kernel_roles'])

    def test_native_vjp_is_derived_before_replacement_and_has_candidate_specific_evidence(self):
        graph, trace, spec, norm, context, tensors = fixture()
        self.assertTrue(any(getattr(node, "op", "") == "silu_backward" for node in trace.graph.nodes))
        self.assertTrue(any(getattr(node, "op", "") == "normalization.layer_norm_vjp" for node in trace.graph.nodes))
        mark_cuda(graph, trace.graph)
        diagnostics, patterns = normalization_fusion.forward(graph, context)
        patterns += normalization_fusion.backward(trace.graph, context, diagnostics)
        self.assertEqual(patterns, ("layer_norm_silu_epilogue", "layer_norm_silu_vjp"))
        graph.validate()
        trace.graph.validate()
        for candidate in diagnostics:
            self.assertTrue(candidate["selected"])
            self.assertEqual(candidate["profile_range"], "deepopt.fusion::" + candidate["candidate_id"])
            self.assertEqual(len(candidate["implementation_fingerprint"]), 64)
            self.assertTrue(candidate["kernel_evidence"]["required_kernel_roles"])
            self.assertEqual(candidate["qualification"], "unqualified")
        json.dumps(diagnostics, allow_nan=False)

    def test_reference_recipe_oracle_preserves_casts_codec_state_and_selective_gradients(self):
        for kind in ("layer", "group"):
            for bits in (1, 2, 4, 8):
                for needs in ((True, True, True), (True, False, False), (False, True, False), (False, False, True)):
                    with self.subTest(kind=kind, bits=bits, needs=needs):
                        graph, trace, spec, norm, context, tensors = fixture(kind,
                            shape=(2, 4, 3, 5) if kind == "group" else (3, 17), bits=bits, compressed=True,
                            needs=needs, dtype=torch.float32 if kind == "group" else torch.bfloat16,
                            channels_last=kind == "group")
                        reference, saved = execute(graph, *tensors)
                        upstream = torch.randn_like(reference)
                        expected = execute(trace.graph, upstream, *saved)
                        mark_cuda(graph, trace.graph)
                        diagnostics, _ = normalization_fusion.forward(graph, context)
                        normalization_fusion.backward(trace.graph, context, diagnostics)
                        actual, saved_actual = execute(graph, *tensors)
                        actual_grads = execute(trace.graph, upstream, *saved_actual)
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                        for a, b in zip(actual_grads, expected):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        self.assertTrue(all(candidate["selected"] for candidate in diagnostics))
                        for a, b in zip(saved_actual, saved):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_group_norm_captured_fp32_output_policy_survives_bf16_input(self):
        from deepopt.opt.backends.reference import normalization as oracle

        graph, trace, spec, norm, context, tensors = fixture("group", shape=(2, 4, 3, 5),
            dtype=torch.bfloat16, dtype_out=torch.float32, compressed=True, channels_last=True)
        mark_cuda(graph, trace.graph)
        candidates, _ = normalization_fusion.forward(graph, context)
        actual, _ = execute(graph, *tensors)
        expected_norm, _, _, _ = oracle.forward(*tensors, spec.geometry, spec.compression, out_dtype=torch.float32)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, nn.functional.silu(expected_norm), rtol=0, atol=0)
        self.assertTrue(candidates[0]["selected"])

    def test_phase_controls_qualify_forward_and_backward_independently(self):
        for phases in (("forward",), ("backward",), ("forward", "backward"), ()):
            graph, trace, _, _, context, _ = fixture()
            context = replace(context, fusion_phases=phases)
            mark_cuda(graph, trace.graph)
            diagnostics, fwd = normalization_fusion.forward(graph, context)
            bwd = normalization_fusion.backward(trace.graph, context, diagnostics)
            self.assertEqual(bool(fwd), "forward" in phases)
            self.assertEqual(bool(bwd), "backward" in phases)
            self.assertEqual(sum(candidate["selected"] for candidate in diagnostics), len(phases))

    def test_noncontiguous_input_and_absent_affine_have_matching_oracles(self):
        for kind in ("layer", "group"):
            graph, trace, _, _, context, tensors = fixture(kind,
                shape=(2, 4, 3, 5) if kind == "group" else (3, 17),
                needs=(True, False, False), affine=False, strided=True)
            expected, saved = execute(graph, *tensors)
            upstream = torch.randn_like(expected)
            expected_gradients = execute(trace.graph, upstream, *saved)
            mark_cuda(graph, trace.graph)
            diagnostics, _ = normalization_fusion.forward(graph, context)
            normalization_fusion.backward(trace.graph, context, diagnostics)
            actual, fused_saved = execute(graph, *tensors)
            actual_gradients = execute(trace.graph, upstream, *fused_saved)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=0, atol=0)

    def test_native_group_statistics_contract_never_selects_shared_epilogues(self):
        graph, trace, spec, _, context, _ = fixture("group", shape=(2, 4, 3, 5))
        spec.native_group = object()
        mark_cuda(graph, trace.graph)
        before = tuple(graph.nodes), tuple(trace.graph.nodes)
        candidates, patterns = normalization_fusion.forward(graph, context)
        backward = normalization_fusion.backward(trace.graph, context, candidates)
        self.assertFalse(patterns or backward)
        self.assertTrue(candidates)
        self.assertTrue(all(row["reason"] == "native_group_norm_statistics_contract" for row in candidates))
        self.assertEqual(before, (tuple(graph.nodes), tuple(trace.graph.nodes)))

    def test_cpu_and_hard_mixed_policy_pins_reject_combined_kernel_visibly(self):
        graph, trace, _, _, context, _ = fixture()
        candidates, patterns = normalization_fusion.forward(graph, context)
        self.assertEqual(patterns, ())
        self.assertEqual(candidates[0]["reason"], "device")
        for family in ("activation", "activation_codec", "compiler_pointwise"):
            policy = KernelPolicy("mixed", families=((family, "tilelang"),))
            graph, trace, _, _, context, _ = fixture(policy=policy, compressed=True)
            mark_cuda(graph, trace.graph)
            candidates, patterns = normalization_fusion.forward(graph, context)
            self.assertFalse(patterns)
            self.assertEqual(candidates[0]["reason"], "consumed_role_backend_conflict")

    def test_forward_keeps_escaped_exact_preactivation_and_no_inference_residual(self):
        graph, trace, _, norm, context, _ = fixture()
        original = norm.values[0]
        mark_cuda(graph, trace.graph)
        candidates, _ = normalization_fusion.forward(graph, context)
        self.assertIn(original, norm.values)
        self.assertEqual(candidates[0]["retained_preactivation_bytes"], prod(original.meta.shape) * 2)
        graph, trace, _, norm, context, tensors = fixture(needs=(False, False, False))
        mark_cuda(graph, trace.graph)
        candidates, _ = normalization_fusion.forward(graph, context)
        self.assertEqual(len(norm.values), 1)
        self.assertEqual(candidates[0]["retained_preactivation_bytes"], 0)
        _, saved = execute(graph, *tensors)
        self.assertEqual(saved, ())

    def test_backward_fanout_rejects_without_removing_shared_gradient(self):
        graph, trace, _, _, context, _ = fixture()
        derivative = next(node for node in trace.graph.nodes if getattr(node, "op", "") == "silu_backward")
        trace.graph.nodes[-1].result = (*trace.graph.nodes[-1].result, derivative.value)
        register_uses(trace.graph.nodes[-1])
        mark_cuda(graph, trace.graph)
        diagnostics = []
        self.assertFalse(normalization_fusion.backward(trace.graph, context, diagnostics))
        self.assertEqual(diagnostics[0]["reason"], "activation_gradient_fanout")
        self.assertIn(derivative, trace.graph.nodes)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA normalization kernels required")
class NormalizationFusionCudaTests(unittest.TestCase):
    def test_experimental_native_group_launches_replacements(self):
        graph, trace, spec, _, context, tensors = fixture("group", shape=(2, 8, 3, 17))
        mark_cuda(graph, trace.graph)
        context = replace(context, qualify_native_group_fusion=True)
        tensors = tuple(t.detach().cuda().requires_grad_(t.requires_grad) for t in tensors)
        expected, saved = execute(graph, *tensors)
        dy = torch.randn_like(expected)
        expected_grads = execute(trace.graph, dy, *saved)
        candidates, forward_patterns = normalization_fusion.forward(graph, context)
        backward_patterns = normalization_fusion.backward(trace.graph, context, candidates)
        self.assertTrue(forward_patterns and backward_patterns)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profile:
            actual, fused_saved = execute(graph, *tensors)
            actual_grads = execute(trace.graph, dy, *fused_saved)
            torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        for a, e in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, e, rtol=0.02, atol=0.04)
        kernels = [event.name for event in profile.events()
                   if event.device_type == torch.autograd.DeviceType.CUDA]
        for symbol in ('_native_group_apply', '_native_ds_db', '_native_group_dx', '_native_affine_finish'):
            self.assertTrue(any(symbol in name for name in kernels), (symbol, kernels))
        self.assertFalse(any('_activation_forward' in name or '_activation_backward' in name for name in kernels))
        self.assertEqual(spec.module.last_execution['materialized_activation_gradient_bytes'], 0)
        self.assertFalse(spec.module.last_execution['forward_global_fp32_staging_output'])

    def compare(self, kind, **options):
        graph, trace, spec, norm, context, tensors = fixture(kind, **options)
        mark_cuda(graph, trace.graph)
        if kind == "group":
            spec.backend = "triton"
        tensors = tuple(value.detach().cuda().requires_grad_(value.requires_grad) for value in tensors)
        with torch.no_grad():
            expected, saved = execute(graph, *tensors)
            upstream = torch.randn_like(expected)
            expected_gradients = execute(trace.graph, upstream, *saved)
            candidates, _ = normalization_fusion.forward(graph, context)
            normalization_fusion.backward(trace.graph, context, candidates)
            actual, fused_saved = execute(graph, *tensors)
            actual_gradients = execute(trace.graph, upstream, *fused_saved)
        tolerance = {"rtol": 2e-2, "atol": 2e-2} if expected.dtype == torch.bfloat16 else {"rtol": 3e-5, "atol": 3e-5}
        torch.testing.assert_close(actual, expected, **tolerance)
        for a, b in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(a, b, **tolerance)
        for a, b in zip(fused_saved, saved):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        if getattr(spec, "native_group", None) is not None:
            self.assertTrue(all(not candidate["selected"] and candidate["reason"] ==
                                "native_group_norm_statistics_contract" for candidate in candidates))
        else:
            self.assertTrue(all(candidate["selected"] for candidate in candidates))

    def test_layer_norm_widths_multi_axis_and_gelu_modes(self):
        for width in (17, 64, 768, 1024, 3072):
            for kind, approximate in (("silu", "none"), ("gelu", "none"), ("gelu", "tanh")):
                with self.subTest(width=width, activation=kind, approximate=approximate):
                    self.compare("layer", shape=(3, width), dtype=torch.bfloat16,
                                 activation=kind, approximate=approximate)
        self.compare("layer", shape=(2, 3, 4, 5), normalized_shape=(4, 5), dtype=torch.float32)
        self.compare("layer", shape=(3, 17), strided=True, affine=False, needs=(True, False, False))

    def test_compressed_norm_tail_groups_layouts_and_selective_masks(self):
        for kind in ("layer", "group"):
            for bits in (1, 2, 4, 8):
                for needs in ((True, True, True), (True, False, False), (False, True, False), (False, False, True)):
                    with self.subTest(kind=kind, bits=bits, needs=needs):
                        self.compare(kind, shape=(2, 8, 3, 5) if kind == "group" else (129, 17),
                                     compressed=True, bits=bits, needs=needs,
                                     channels_last=kind == "group", dtype=torch.float32)
        self.compare("group", shape=(2, 8, 3, 5), strided=True, affine=False, needs=(True, False, False))


if __name__ == "__main__":
    unittest.main()
