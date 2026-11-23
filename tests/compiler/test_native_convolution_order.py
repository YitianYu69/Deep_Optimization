"""Adjacent native convolution adjoints keep their VJP and saved-owner ABI.

CPU execution and static CUDA metadata are diagnostics. The CUDA classes run
real kernels; the original decoder geometry is an explicit capacity opt-in.
"""

from contextlib import nullcontext
from copy import copy, deepcopy
from dataclasses import replace
from functools import wraps
import gc
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn
from torch.fx.passes.shape_prop import ShapeProp

from deepopt import autocast
from deepopt._bf16_policy import BF16Policy
from deepopt.compiler import BackendConfig, CompilationReport, compile_fx
from deepopt.compiler.autodiff import convolution_order
from deepopt.compiler.autodiff.optimization import rebuild
from deepopt.compiler.autodiff.semantic import packed_metas, recorded
from deepopt.compiler.autodiff.spatial import Convolution
from deepopt.compiler.ir import CallNode, ExternKernelNode, InputNode, IRGraph, OutputNode, TensorMeta, TensorValue
from deepopt.compiler.registry import KernelCodegenContext
from deepopt.compiler.runtime.memory import MemoryManager
from deepopt.compiler.scheduler import Scheduler
from deepopt.compiler.trace.metadata_utils import contiguous_stride
from deepopt.kernel_policy import KernelPolicy


DX = "convolution.native_dx_db"
DW = "convolution.native_dw"


class _Projection(nn.Module):
    def __init__(self, channels=4, outputs=6, *, bias=True, kernel=1, padding=0, groups=1):
        super().__init__()
        self.projection = nn.Conv2d(channels, outputs, kernel, padding=padding, groups=groups, bias=bias)

    def forward(self, x):
        return self.projection(x)


class _TiedProjections(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Conv2d(4, 4, 1, bias=False)
        self.right = nn.Conv2d(4, 4, 1, bias=False)
        self.right.weight = self.left.weight

    def forward(self, x):
        return self.left(x) + self.right(x)


def _calls(graph):
    return [node for node in graph.nodes if getattr(node, "op", None) in (DX, DW)]


def _rows(report):
    return [row for row in report.as_dict()["fusion_candidates"] if row["pattern"] == convolution_order.PATTERN]


def _compile(source, x, order, *, compressed=False, shared=False, modifier=None, observer=None):
    graph = torch.fx.symbolic_trace(deepcopy(source))
    report, captures = CompilationReport(), []
    original = convolution_order.backward

    def transform(backward_graph, context, **options):
        if modifier is not None:
            modifier(backward_graph)
            rebuild(backward_graph)
        calls = _calls(backward_graph)
        before = tuple((node, tuple(node.inputs()), tuple(node.outputs())) for node in calls)
        output = backward_graph.nodes[-1]
        result = original(backward_graph, context, **options)
        captures.append((backward_graph, before, output.result))
        if observer is not None:
            observer(backward_graph)
        return result

    scope = (autocast(x.device.type, dtype=torch.bfloat16, fp8=False,
        activation_compression=dict(enabled=compressed, bits=2, group_size=1024),
        bf16_policy="bf16_storage" if x.device.type == "cuda" else "torch_amp")
        if x.dtype == torch.bfloat16 else nullcontext())
    with patch.object(convolution_order, "backward", new=transform), scope:
        ShapeProp(graph).propagate(x)
        executable = compile_fx(graph, [x], config=BackendConfig(require_full_aot=True,
            native_conv_backward_order=order, share_compatible_packs=shared), report=report)
        output = executable(x)
    return graph, executable, output, report, captures


def _metadata_fixture(*, shape=(2, 4, 5, 7), outputs=6, kernel=1, padding=0,
                      groups=1, compressed=True, dtype=torch.bfloat16):
    """No source/output payload allocation, including original B128 geometry."""
    context = KernelCodegenContext(kernel_policy=KernelPolicy("triton"), native_conv_backward_order="dw_first")
    module = nn.Conv2d(shape[1], outputs, kernel, padding=padding, groups=groups, bias=False)
    module.plan = {"activation_compression": dict(enabled=compressed, bits=2, group_size=1024)}
    source_meta = TensorMeta(shape, dtype, "cuda:0", contiguous_stride(shape), True)
    weight_shape = (outputs, shape[1] // groups, kernel, kernel)
    weight_meta = TensorMeta(weight_shape, torch.float32, "cuda:0", contiguous_stride(weight_shape), True)
    output_shape = (shape[0], outputs, shape[2] + 2 * padding - kernel + 1, shape[3] + 2 * padding - kernel + 1)
    output_meta = TensorMeta(output_shape, dtype, "cuda:0", contiguous_stride(output_shape), True)
    raw = SimpleNamespace(module=module, module_path="block.body.projection", node_name="projection",
        op="deepopt_conv", args=(TensorValue("source", source_meta), TensorValue("weight", weight_meta), None),
        values=(TensorValue("output", output_meta),), options={}, kwargs={}, plan=module.plan,
        bf16_policy=BF16Policy("bf16_storage"))
    spec = Convolution(raw, context)
    graph = IRGraph()
    gradient = TensorValue("seed", replace(output_meta, requires_grad=False, stride=None))
    weight = TensorValue("saved_weight", weight_meta)
    if compressed:
        descriptions = packed_metas(spec.geometry.input_matrix_shape, spec.compression, "cuda:0")
        saved = tuple(TensorValue(f"saved_{index}", meta) for index, meta in enumerate(descriptions))
    else:
        saved = (TensorValue("saved_source", replace(source_meta, requires_grad=False)), None, None)
    for index, value in enumerate((gradient, weight, *(value for value in saved if value is not None))):
        graph.add(InputNode(index, value))
    dx = TensorValue("input_gradient", replace(source_meta, requires_grad=False, stride=None))
    dw = TensorValue("weight_gradient", replace(weight_meta, requires_grad=False))
    dx_node = CallNode(DX, recorded(spec, spec.dx_db, "backward"), (gradient, weight), {}, (dx,))
    dw_node = ExternKernelNode(DW, recorded(spec, spec.dw, "backward"), (gradient, *saved), {}, dw)
    graph.add(dx_node)
    graph.add(dw_node)
    graph.add(OutputNode((dx, dw)))
    return graph, context, spec, dx_node, dw_node


class NativeConvolutionOrderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(601)

    def test_default_and_invalid_controls_fail_closed(self):
        self.assertEqual(BackendConfig().native_conv_backward_order, "source")
        self.assertEqual(KernelCodegenContext().native_conv_backward_order, "source")
        for invalid in (True, None, 1, "dx_first", "DW_FIRST"):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "native_conv_backward_order"):
                BackendConfig(native_conv_backward_order=invalid)
        with self.assertRaisesRegex(ValueError, "native AutoDiff"):
            BackendConfig(native_conv_backward_order="dw_first", use_native_autodiff=False)
        graph, _, _, _, _ = _metadata_fixture()
        with self.assertRaisesRegex(ValueError, "native_conv_backward_order"):
            convolution_order.backward(graph, SimpleNamespace(native_conv_backward_order="invalid"))

    def test_actual_cpu_native_vjp_preserves_calls_saved_leaves_and_output_order(self):
        source, sample = _Projection(), torch.randn(2, 4, 5, 7)
        for dtype, compressed in ((torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)):
            results = []
            for order in ("source", "dw_first"):
                x = sample.to(dtype).clone().requires_grad_()
                model, _, output, report, captures = _compile(source, x, order, compressed=compressed)
                seed = torch.linspace(-.8, 1.1, output.numel(), dtype=output.dtype).reshape(output.shape)
                version = x._version
                gradients = torch.autograd.grad(output, (x, *model.parameters()), seed)
                self.assertEqual(x._version, version)
                results.append((output.detach(), gradients))
                backward, before, outputs = captures[0]
                self.assertEqual(tuple(_calls(backward)), tuple(row[0] for row in (before if order == "source" else reversed(before))))
                self.assertIs(backward.nodes[-1].result, outputs)
                for node, inputs, values in before:
                    self.assertEqual(tuple(node.inputs()), inputs)
                    self.assertEqual(tuple(node.outputs()), values)
                rows = _rows(report)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["selected"], order == "dw_first")
                self.assertFalse(rows[0]["kernel_fusion"])
                self.assertIsNone(rows[0]["global_peak_saving_bytes"])
                if order == "source":
                    self.assertEqual(rows[0]["reason"], "disabled")
                    self.assertTrue(all(not node.scheduling_constraints for node in _calls(backward)))
                else:
                    self.assertEqual(rows[0]["transformation"], "adjoint_scheduling")
                    self.assertEqual(tuple(rows[0]["selected_order"]), (DW, DX))
                    self.assertEqual(rows[0]["expected_local_overlap_removed_bytes"], x.numel() * x.element_size())
                    self.assertTrue(rows[0]["pair_proof"]["gradient_accumulation_order_unchanged"])
                    final_calls = [row["operation"] for row in rows[0]["actual_order"] if row["operation"] in (DX, DW)]
                    self.assertEqual(final_calls, [DW, DX])
                self.assertEqual(report.runtime_fallbacks, 0)
                self.assertFalse(report.as_dict()["legacy_backward_operations"])
                json.dumps(rows, allow_nan=False)
            for actual, expected in zip((results[1][0], *results[1][1]), (results[0][0], *results[0][1])):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_selective_gradient_masks_keep_required_internal_adjoints(self):
        for needs in ((False, True, True), (True, False, True), (True, False, False), (False, False, True)):
            source = _Projection()
            source.projection.weight.requires_grad_(needs[1])
            source.projection.bias.requires_grad_(needs[2])
            results = []
            for order in ("source", "dw_first"):
                x = torch.randn(2, 4, 3, 5, requires_grad=needs[0])
                if order == "source":
                    sample = x.detach().clone()
                else:
                    x = sample.clone().requires_grad_(needs[0])
                model, _, output, report, captures = _compile(source, x, order)
                targets = tuple(value for value in (x, *model.parameters()) if value.requires_grad)
                results.append(torch.autograd.grad(output, targets, torch.ones_like(output)))
                self.assertFalse(any(row["selected"] for row in _rows(report)))
                self.assertEqual(any(node.op == DX and node.target.specification.needs[0] for node in _calls(captures[0][0])), needs[0])
            for actual, expected in zip(results[1], results[0]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_tied_weights_shared_source_and_retained_backward_preserve_accumulation(self):
        source, sample = _TiedProjections(), torch.randn(2, 4, 7, 9, dtype=torch.bfloat16)
        results = []
        for order in ("source", "dw_first"):
            x = sample.clone().requires_grad_()
            model, _, output, report, captures = _compile(source, x, order, compressed=True, shared=True)
            targets = (x, *model.parameters())
            self.assertEqual(len(targets), 2)
            seed = torch.linspace(-1, 1, output.numel(), dtype=output.dtype).reshape(output.shape)
            first = torch.autograd.grad(output, targets, seed, retain_graph=True)
            second = torch.autograd.grad(output, targets, seed)
            for actual, expected in zip(second, first):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            results.append(first)
            selected = [row for row in _rows(report) if row["selected"]]
            self.assertEqual(len(selected), 2 if order == "dw_first" else 0)
            graph, before, _ = captures[0]
            for node, inputs, outputs in before:
                self.assertEqual(tuple(node.inputs()), inputs)
                self.assertEqual(tuple(node.outputs()), outputs)
            if selected:
                dw = [node for node in _calls(graph) if node.op == DW]
                self.assertEqual(dw[0].args[1:], dw[1].args[1:])
                self.assertEqual(selected[0]["pair_proof"]["saved_leaves_unchanged"], selected[1]["pair_proof"]["saved_leaves_unchanged"])
        for actual, expected in zip(results[1], results[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_runtime_last_use_releases_packed_leaves_but_retained_graph_keeps_owners(self):
        observations, references = [], []

        def observe(graph):
            for node in _calls(graph):
                function = node.target
                if node.op == DW:
                    @wraps(function)
                    def dw(*args, _function=function, **kwargs):
                        references[:] = [weakref.ref(value) for value in args[1:]]
                        return _function(*args, **kwargs)
                    node.target = dw
                else:
                    @wraps(function)
                    def dx(*args, _function=function, **kwargs):
                        observations.append(tuple(reference() is not None for reference in references))
                        return _function(*args, **kwargs)
                    node.target = dx

        x = torch.randn(2, 4, 7, 9, dtype=torch.bfloat16, requires_grad=True)
        model, _, output, report, _ = _compile(_Projection(bias=False), x, "dw_first", compressed=True, observer=observe)
        targets, seed = (x, *model.parameters()), torch.ones_like(output)
        first = torch.autograd.grad(output, targets, seed, retain_graph=True)
        second = torch.autograd.grad(output, targets, seed)
        self.assertEqual(observations, [(True, True, True), (False, False, False)])
        for actual, expected in zip(second, first):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(_rows(report)[0]["selected"])

    def test_weight_failure_propagates_without_retry_or_input_adjoint(self):
        from deepopt.opt.backends.native import convolution as kernels

        x = torch.randn(2, 4, 5, 7, requires_grad=True)
        model, executable, output, _, _ = _compile(_Projection(), x, "dw_first")
        before = (x.detach().clone(), *(value.detach().clone() for value in model.parameters()))
        with patch.object(kernels, "native_weight_gradient", side_effect=RuntimeError("weight sentinel")) as dw, \
                patch.object(kernels, "native_input_bias_gradients", wraps=kernels.native_input_bias_gradients) as dx:
            with self.assertRaisesRegex(RuntimeError, "weight sentinel"):
                output.sum().backward()
            self.assertEqual(dw.call_count, 1)
            self.assertEqual(dx.call_count, 0)
        for actual, expected in zip((x, *model.parameters()), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gc.collect()
        again = executable(x)
        gradients = torch.autograd.grad(again, (x, *model.parameters()), torch.ones_like(again))
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))


class NativeConvolutionOrderProofTests(unittest.TestCase):
    def test_original_b128_metadata_selects_without_payload_and_final_order_has_earlier_source_release(self):
        graph, context, _, dx, dw = _metadata_fixture(shape=(128, 768, 224, 224), outputs=112)
        leaves, output = dw.args[1:], graph.nodes[-1].result
        diagnostics = []
        self.assertEqual(convolution_order.backward(graph, context, diagnostics=diagnostics), (convolution_order.PATTERN,))
        self.assertEqual(_calls(graph), [dw, dx])
        self.assertEqual(dw.args[1:], leaves)
        self.assertIs(graph.nodes[-1].result, output)
        schedule = Scheduler(graph, kernel_policy=context.kernel_policy, backward=True).run()
        self.assertLess(schedule.items.index(dw), schedule.items.index(dx))
        self.assertIn(id(dw), schedule.effect_barriers)
        self.assertIn(id(dx), schedule.effect_barriers)
        memory = MemoryManager(schedule)
        memory.collect_schedule_level_remaining_users()
        lifetimes = {row["name"]: row for row in memory.lifetime_plan()}
        for leaf in leaves:
            self.assertLess(lifetimes[leaf.name]["local_release_after"], lifetimes[dx.values[0].name]["birth"])
        row = diagnostics[0]
        self.assertEqual(row["expected_local_overlap_removed_bytes"], 9_865_003_008)
        self.assertIsNone(row["global_peak_saving_bytes"])
        self.assertEqual(row["replacement_contract"]["estimated_scratch_bytes"], None)
        self.assertFalse(row["kernel_fusion"])
        json.dumps(row, allow_nan=False)

    def test_source_default_does_not_wrap_reorder_or_change_any_callable(self):
        graph, context, _, dx, dw = _metadata_fixture()
        functions, nodes = (dx.target, dw.target), tuple(graph.nodes)
        rows = []
        self.assertEqual(convolution_order.backward(graph, replace(context, native_conv_backward_order="source"), diagnostics=rows), ())
        self.assertEqual(tuple(graph.nodes), nodes)
        self.assertEqual((dx.target, dw.target), functions)
        self.assertEqual(rows[0]["status"], "disabled")
        self.assertTrue(rows[0]["discovered"])

    def test_unsupported_geometry_layout_and_precision_reject_truthfully(self):
        for kind in ("kernel", "groups", "empty", "source_stride", "channels_last", "fp8", "padding"):
            with self.subTest(kind=kind):
                options = ({"kernel": 3, "padding": 1} if kind == "kernel" else
                    {"groups": 2} if kind == "groups" else {"shape": (0, 4, 5, 7)} if kind == "empty" else {})
                graph, context, spec, dx, dw = _metadata_fixture(**options)
                if kind == "source_stride":
                    spec.metas = (replace(spec.metas[0], stride=None), *spec.metas[1:])
                elif kind == "channels_last":
                    spec.input_format = torch.channels_last
                elif kind == "fp8":
                    spec.precision = "fp8_spatial"
                elif kind == "padding":
                    spec.explicit_padding = (0, 1, 0, 1)
                rows = []
                self.assertEqual(convolution_order.backward(graph, context, diagnostics=rows), ())
                self.assertEqual(_calls(graph), [dx, dw])
                self.assertEqual(rows[0]["status"], "rejected")
                self.assertIn(rows[0]["reason"], ("unsupported_convolution_adjoint_geometry", "convolution_operand_metadata"))

    def test_untrusted_nonadjacent_effect_stream_alias_and_data_dependencies_reject(self):
        cases = {
            "untrusted": "untrusted_input_bias_adjoint",
            "different_spec": "untrusted_or_different_weight_adjoint",
            "nonadjacent": "adjacent_weight_adjoint_required",
            "effect": "effect_stream_or_alias_constraint",
            "stream": "effect_stream_or_alias_constraint",
            "constraints": "effect_stream_or_alias_constraint",
            "alias": "adjoint_output_alias",
            "dependency": "adjoint_data_dependency",
            "upstream": "adjoint_arguments_or_upstream",
        }
        for kind, reason in cases.items():
            with self.subTest(kind=kind):
                graph, context, spec, dx, dw = _metadata_fixture()
                if kind == "untrusted":
                    original = dx.target
                    @wraps(original)
                    def opaque(*args, **kwargs):
                        return original(*args, **kwargs)
                    dx.target = opaque
                elif kind == "different_spec":
                    other = copy(spec)
                    dw.target = recorded(other, other.dw, "backward")
                elif kind == "nonadjacent":
                    effect = CallNode("opaque_effect", lambda: None, (), {}, ())
                    graph.nodes.insert(graph.nodes.index(dw), effect)
                elif kind == "effect":
                    dx.stateful = True
                elif kind == "stream":
                    dw.target.stream = 0
                elif kind == "constraints":
                    dx.scheduling_constraints["mutation"] = "unknown"
                elif kind == "alias":
                    dx.values[0].storage_source = "parameter"
                elif kind == "dependency":
                    dw.args = (dw.args[0], dx.values[0], *dw.args[2:])
                elif kind == "upstream":
                    dw.args = (dx.args[1], *dw.args[1:])
                rebuild(graph)
                nodes, functions = tuple(graph.nodes), (dx.target, dw.target)
                rows = []
                self.assertEqual(convolution_order.backward(graph, context, diagnostics=rows), ())
                self.assertEqual(tuple(graph.nodes), nodes)
                self.assertEqual((dx.target, dw.target), functions)
                self.assertEqual(rows[0]["reason"], reason)


def _check_cuda(test, shape, outputs, *, retained=False):
    torch.manual_seed(811)
    source = _Projection(shape[1], outputs, bias=False).cuda()
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    control = None
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
        for order in ("source", "dw_first"):
            model, _, output, report, _ = _compile(source, x, order, compressed=True)
            seed = torch.ones_like(output)
            gradients = torch.autograd.grad(output, (x, *model.parameters()), seed, retain_graph=retained)
            if retained:
                again = torch.autograd.grad(output, (x, *model.parameters()), seed)
                for actual, expected in zip(again, gradients):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            test.assertEqual(any(row["selected"] for row in _rows(report)), order == "dw_first")
            test.assertEqual(report.runtime_fallbacks, 0)
            test.assertFalse(report.as_dict()["legacy_backward_operations"])
            if control is None:
                control = tuple(value.detach() for value in gradients)
            else:
                for actual, expected in zip(gradients, control):
                    # Bound comparison scratch even for the 9.1875-GiB dX.
                    for offset in range(0, actual.numel(), 1 << 20):
                        torch.testing.assert_close(actual.reshape(-1)[offset:offset + (1 << 20)],
                            expected.reshape(-1)[offset:offset + (1 << 20)], rtol=0, atol=0)
            del gradients, output, seed, model
        torch.cuda.synchronize()


@unittest.skipUnless(torch.cuda.is_available(), "requires actual CUDA convolution execution")
class NativeConvolutionOrderCudaTests(unittest.TestCase):
    def test_cuda_same_native_vjp_and_retained_saved_owner(self):
        _check_cuda(self, (2, 8, 7, 11), 6, retained=True)

    @unittest.skipUnless(os.environ.get("DEEPOPT_RUN_H200_CONV_ORDER") == "1", "opt-in original B128 decoder convolution")
    def test_original_b128_decoder_body8_geometry(self):
        _check_cuda(self, (128, 768, 224, 224), 112)


if __name__ == "__main__":
    unittest.main()
