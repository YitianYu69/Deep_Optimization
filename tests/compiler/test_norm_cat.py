"""Body-only GroupNorm destination writes, saved ABI and native VJP contracts."""

from dataclasses import replace
import json
from math import prod
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from deepopt._bf16_policy import BF16Policy
from deepopt.compiler.autodiff import AOTAutoDiff, norm_cat, spatial
from deepopt.compiler.autodiff.joint import map_tree
from deepopt.compiler.autodiff.primitives import register_rules
from deepopt.compiler.ir import CallNode, InputNode, IRGraph, OutputNode, TensorMeta, TensorValue, register_uses
from deepopt.compiler.registry import BackendRegistry, KernelCodegenContext
from deepopt.kernel_policy import KernelPolicy
from deepopt.opt.backends._shared.normalization_output import validate_normalization_output
from deepopt.opt.nn.primitive_state import model_cat


def fixture(*, shape=(2, 4, 17, 37), groups=2, compressed=True, device="cpu",
            needs=(True, True, True)):
    graph = IRGraph()
    policy = KernelPolicy("triton")
    context = KernelCodegenContext(kernel_policy=policy)
    compression = dict(enabled=compressed, bits=2, group_size=1024, norm_xhat_recenter=True)
    module = nn.GroupNorm(groups, shape[1], device=device).train()
    module.weight.requires_grad_(needs[1])
    module.bias.requires_grad_(needs[2])
    module.plan = {"activation_compression": compression}
    x = torch.randn(shape, dtype=torch.bfloat16, device=device, requires_grad=needs[0])
    skip = torch.randn_like(x, requires_grad=True)
    tensors = (x, module.weight, module.bias, skip)
    inputs = []
    for index, tensor in enumerate(tensors):
        value = TensorValue(f"arg{index}", TensorMeta(tuple(tensor.shape), tensor.dtype, tensor.device,
            tuple(tensor.stride()), tensor.requires_grad))
        graph.add(InputNode(index, value))
        inputs.append(value)
    output = TensorValue("body", TensorMeta(shape, torch.bfloat16, x.device, tuple(x.stride()), any(needs)))
    raw = SimpleNamespace(module=module, module_path="block.body.9", node_name="body_norm", args=(*inputs[:3], None, None, None),
        values=(output,), kwargs={}, op="deepopt_normalization", options={"kind": "group", "groups": groups, "eps": module.eps},
        plan=module.plan, root=module, bf16_policy=BF16Policy("bf16_storage"))
    spec = spatial.Normalization(raw, context)
    norm = spec.lower(raw)
    norm.bf16_policy = raw.bf16_policy
    graph.add(norm)
    cat_shape = (shape[0], shape[1] * 2, *shape[2:])
    stride = tuple(prod(cat_shape[axis + 1:]) for axis in range(len(shape)))
    concatenated = TensorValue("cat", TensorMeta(cat_shape, torch.bfloat16, x.device, stride, True))
    cat = CallNode("autodiff_storage_cat", model_cat, ((inputs[3], output), 1), {}, (concatenated,))
    cat.bf16_policy = raw.bf16_policy
    graph.add(cat)
    graph.add(OutputNode((concatenated, inputs[3])))
    registry = BackendRegistry()
    spatial.register_rules(registry)
    register_rules(registry)
    trace = AOTAutoDiff(graph, registry=registry).diff()
    graph.nodes[-1].result = (graph.nodes[-1].result, trace.saved_tensors)
    register_uses(graph.nodes[-1])
    graph.validate()
    return graph, trace, spec, norm, cat, context, tensors


def execute(graph, *inputs):
    values = {}
    for node in graph.nodes:
        if isinstance(node, InputNode):
            values[node.value] = inputs[node.index]
        elif isinstance(node, OutputNode):
            return map_tree(node.result, values)
        else:
            actual = node.target(*map_tree(node.args, values), **map_tree(node.kwargs, values))
            if len(node.outputs()) == 1:
                actual = (actual,)
            values.update(zip(node.outputs(), actual))


def mark_cuda(graph, spec):
    # Structural metadata only; no CUDA execution or device claim.
    for node in graph.nodes:
        for value in node.outputs():
            if value.meta is not None:
                value.meta = replace(value.meta, device="cuda:0")
    spec.backend = "triton"


def controls(enabled=True, phases=("forward", "backward")):
    return SimpleNamespace(fuse_norm_cat_body=enabled, fusion_phases=phases)


class NormCatOutputTests(unittest.TestCase):
    def test_output_destination_accepts_channel_slice_and_rejects_unsafe_metadata(self):
        _, _, spec, _, _, _, tensors = fixture()
        x, weight, bias, _ = tensors
        shape = tuple(x.shape)
        allocation = torch.empty(shape[0], shape[1] * 2, *shape[2:], dtype=x.dtype)
        destination = allocation.narrow(1, shape[1], shape[1])
        self.assertIs(validate_normalization_output(destination, x, weight, bias, spec.geometry, dtype=x.dtype), destination)
        for invalid in (destination.float(), destination[..., :-1], destination.requires_grad_()):
            with self.subTest(shape=tuple(invalid.shape), dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    validate_normalization_output(invalid, x, weight, bias, spec.geometry, dtype=x.dtype)
        destination.requires_grad_(False)
        for invalid in (x.detach(), torch.empty(1, shape[1], *shape[2:], dtype=x.dtype).expand(shape)):
            with self.assertRaises(ValueError):
                validate_normalization_output(invalid, x, weight, bias, spec.geometry, dtype=x.dtype)
        with self.assertRaisesRegex(ValueError, "unactivated"):
            validate_normalization_output(destination, x, weight, bias, spec.geometry, dtype=x.dtype, activation=("silu", "none"))

    def test_cpu_oracle_writes_destination_and_keeps_saved_leaves_and_native_vjp_exact(self):
        from deepopt.opt.backends.reference import normalization as oracle

        for compressed in (False, True):
            for needs in ((True, True, True), (True, False, False), (False, True, True)):
                with self.subTest(compressed=compressed, needs=needs):
                    graph, trace, spec, norm, cat, _, tensors = fixture(compressed=compressed, needs=needs)
                    expected, saved = execute(graph, *tensors)
                    seed = (torch.randn_like(expected[0]), torch.randn_like(expected[1]))
                    expected_grads = execute(trace.graph, *seed, *saved)
                    wrapper = norm_cat._NormCatBody(spec, graph.nodes[3].value.meta, cat.values[0].meta)
                    skip = tensors[-1]
                    version, storage = skip._version, skip.untyped_storage()._cdata
                    original_forward = oracle.forward
                    calls = []

                    def into(*args, **kwargs):
                        self.assertIsNotNone(kwargs.get("output"))
                        destination = kwargs["output"]
                        result = original_forward(*args, **kwargs)
                        self.assertIs(result[0], destination)
                        calls.append(tuple(destination.stride()))
                        return result

                    with patch.object(oracle, "forward", side_effect=into), \
                            patch("torch.empty_strided", side_effect=AssertionError("standalone normalized output")), \
                            patch("torch.cat", side_effect=AssertionError("ordinary cat")):
                        actual = wrapper.forward(*tensors[:3], None, None, None, skip)
                    self.assertEqual(len(calls), 1)
                    self.assertGreater(calls[0][0], prod(tensors[0].shape[1:]))
                    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                    self.assertEqual(len(actual) - 1, len(norm.values) - 1)
                    self.assertEqual(len(saved), len(actual) - 1)
                    for a, e in zip(actual[1:], saved):
                        torch.testing.assert_close(a, e, rtol=0, atol=0)
                    grads = execute(trace.graph, *seed, *actual[1:])
                    for a, e in zip(grads, expected_grads):
                        torch.testing.assert_close(a, e, rtol=0, atol=0)
                    self.assertEqual(skip._version, version)
                    self.assertEqual(skip.untyped_storage()._cdata, storage)
                    self.assertNotEqual(actual[0].untyped_storage()._cdata, storage)
                    torch.testing.assert_close(actual[0].narrow(1, 0, skip.shape[1]), skip, rtol=0, atol=0)

    def test_cpu_is_not_selected_and_eager_apply_argument_counts_stay_unchanged(self):
        from deepopt.opt.nn.autograd.normalization.function import NormalizationFunction
        from deepopt._bf16_policy import bf16_policy_scope

        graph, _, _, _, _, _, _ = fixture()
        decisions, patterns = norm_cat.forward(graph, controls())
        self.assertFalse(patterns)
        self.assertEqual(decisions[0]["reason"], "device")
        # The destination is keyword-only on the manual compiler forward;
        # ordinary eager Function.apply still receives its original arguments.
        import inspect
        self.assertEqual(inspect.signature(NormalizationFunction.forward).parameters["output"].kind,
                         inspect.Parameter.KEYWORD_ONLY)
        for explicit_policy in (False, True):
            _, _, spec, _, _, _, tensors = fixture()
            x, weight, bias = tensors[:3]
            arguments = (x, weight, bias, None, None, spec.geometry, spec.plan, None, None, True, .1, True, {})
            with bf16_policy_scope("bf16_storage", "cpu"):
                result = NormalizationFunction.apply(*arguments, *((spec.bf16_policy,) if explicit_policy else ()))
            result.sum().backward()
            self.assertEqual(x.grad.dtype, torch.bfloat16)
            self.assertEqual(weight.grad.dtype, torch.float32)

    def test_failed_executor_does_not_retry_with_materialized_output(self):
        graph, _, spec, _, cat, _, tensors = fixture()
        wrapper = norm_cat._NormCatBody(spec, graph.nodes[3].value.meta, cat.values[0].meta)
        with patch.object(spec, "forward", side_effect=RuntimeError("destination kernel failed")) as call:
            with self.assertRaisesRegex(RuntimeError, "destination kernel failed"):
                wrapper.forward(*tensors[:3], None, None, None, tensors[-1])
        self.assertEqual(call.call_count, 1)


class NormCatSelectionTests(unittest.TestCase):
    def test_real_cpu_joint_program_propagates_option_and_preserves_backward(self):
        from deepopt.compiler.autodiff.joint import JointProgram
        from deepopt.compiler.autodiff.optimization import rebuild

        graph, trace, spec, _, _, context, tensors = fixture()
        outputs, saved = execute(graph, *tensors)
        seeds = tuple(torch.randn_like(value) for value in outputs)
        expected = execute(trace.graph, *seeds, *saved)
        # Rebuild the ordinary public-output graph; JointProgram owns its own
        # native differentiation and saved boundary, rather than reusing this
        # test's oracle trace.
        graph.nodes[-1].result = graph.nodes[-1].result[0]
        rebuild(graph)
        registry = BackendRegistry()
        spatial.register_rules(registry)
        register_rules(registry)
        context = replace(context, fuse_norm_cat_body=True)
        inputs = tuple(node.value for node in graph.nodes if isinstance(node, InputNode))
        program = JointProgram(graph, inputs, tuple(graph.nodes[-1].result), [spec], registry, context)
        decisions = [row for row in program.fusion_candidates if row["pattern"] == norm_cat.PATTERN]
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["reason"], "device")
        actual = program(*tensors)
        torch.autograd.backward(actual, seeds)
        for tensor, gradient in zip(tensors, expected):
            torch.testing.assert_close(tensor.grad, gradient, rtol=0, atol=0)

    def test_original_b128_first_block_geometry_selects_without_tensor_allocation(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode(allow_fallback_kernels=False):
            graph, _, spec, norm, cat, _, _ = fixture(shape=(128, 112, 224, 224), groups=8)
            mark_cuda(graph, spec)
            decisions, patterns = norm_cat.forward(graph, controls())
        self.assertEqual(patterns, (norm_cat.PATTERN,))
        self.assertEqual(decisions[0]["replacement_contract"]["skip_copy_bytes"], 1438646272)
        replacement = next(node for node in graph.nodes if getattr(node, "op", None) == "normalization.cat_body_forward")
        self.assertEqual(replacement.values[0].meta.shape, (128, 224, 224, 224))
        self.assertEqual(replacement.values[1:], norm.values[1:])

    def test_default_off_phase_controls_and_serializable_selected_contract(self):
        graph, trace, spec, norm, cat, context, _ = fixture()
        mark_cuda(graph, spec)
        original_nodes = tuple(graph.nodes)
        for flags in (context, controls(False), controls(True, ("backward",))):
            decisions, patterns = norm_cat.forward(graph, flags)
            self.assertFalse(patterns)
            self.assertFalse(decisions[0]["selected"])
            self.assertEqual(tuple(graph.nodes), original_nodes)
        saved = tuple(norm.values[1:])
        backward = tuple(trace.graph.nodes)
        decisions, patterns = norm_cat.forward(graph, controls())
        self.assertEqual(patterns, (norm_cat.PATTERN,))
        replacement = next(node for node in graph.nodes if getattr(node, "op", None) == "normalization.cat_body_forward")
        self.assertIs(replacement.values[0], cat.values[0])
        self.assertEqual(replacement.values[1:], saved)
        self.assertEqual(tuple(trace.graph.nodes), backward)
        self.assertNotIn(norm, graph.nodes)
        self.assertNotIn(cat, graph.nodes)
        graph.validate()
        trace.graph.validate()
        row = decisions[0]
        self.assertTrue(row["selected"])
        self.assertFalse(row["kernel_fusion"])
        self.assertEqual(row["qualification"], "experimental_cuda_unqualified")
        self.assertEqual(row["replacement_contract"]["skip_storage"], "original_separate")
        self.assertIsNone(row["replacement_contract"]["peak_saving_bytes"])
        self.assertEqual(row["replacement_contract"]["saved_slot_names"], tuple(value.name for value in saved))
        json.dumps(decisions, allow_nan=False)

    def test_fanout_variants_backends_layouts_and_intervening_effects_reject(self):
        for mutation, expected in (
                ("fanout", "body_output_fanout_or_saved_lease"),
                ("variant", "normalization_variant"),
                ("native_group", "unsupported_normalization_executor"),
                ("fp8_output", "unsupported_normalization_executor"),
                ("backend", "normalization_backend_has_no_destination"),
                ("layout", "contiguous_layout_required"),
                ("effect", "intervening_operation")):
            with self.subTest(mutation=mutation):
                graph, _, spec, norm, cat, _, _ = fixture()
                mark_cuda(graph, spec)
                if mutation == "fanout":
                    graph.nodes[-1].result = (graph.nodes[-1].result, norm.values[0])
                    register_uses(graph.nodes[-1])
                elif mutation == "variant":
                    spec.normalization_variant = "small_forward"
                elif mutation == "native_group":
                    spec.native_group = object()
                elif mutation == "fp8_output":
                    spec.output_layout = "channels"
                elif mutation == "backend":
                    spec.kernel_policy = KernelPolicy("tilelang")
                elif mutation == "layout":
                    norm.values[0].meta = replace(norm.values[0].meta, stride=None)
                else:
                    value = TensorValue("effect", replace(cat.values[0].meta, shape=(1,), stride=(1,)))
                    graph.nodes.insert(graph.nodes.index(cat), CallNode("external_effect", lambda: None, (), {}, (value,)))
                decisions, patterns = norm_cat.forward(graph, controls())
                self.assertFalse(patterns)
                self.assertEqual(decisions[0]["reason"], expected)
                self.assertIn(norm, graph.nodes)
                self.assertIn(cat, graph.nodes)


@unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA execution")
class NormCatBodyCUDATests(unittest.TestCase):
    def check_shape(self, shape, groups):
        graph, trace, spec, norm, _, _, tensors = fixture(shape=shape, groups=groups, device="cuda")
        expected, saved = execute(graph, *tensors)
        seeds = tuple(torch.randn_like(value) for value in expected)
        expected_gradients = execute(trace.graph, *seeds, *saved)
        decisions, patterns = norm_cat.forward(graph, controls())
        self.assertEqual(patterns, (norm_cat.PATTERN,))
        actual, actual_saved = execute(graph, *tensors)
        actual_gradients = execute(trace.graph, *seeds, *actual_saved)
        torch.cuda.synchronize()
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        for a, e in zip(actual_saved, saved):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        for a, e in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        self.assertEqual(tuple(value.meta for value in norm.values[1:]), tuple(value.meta for value in trace.saved_tensors))
        self.assertTrue(decisions[0]["selected"])

    def test_actual_cuda_groupnorm_cat_tail_and_native_backward(self):
        self.check_shape((2, 4, 17, 37), 2)

    @unittest.skipUnless(os.environ.get("DEEPOPT_RUN_H200_NORM_CAT") == "1", "opt-in original B128 block geometry")
    def test_original_b128_encoder1_groupnorm_cat_geometry(self):
        self.check_shape((128, 112, 224, 224), 8)


if __name__ == "__main__":
    unittest.main()
