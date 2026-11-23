"""Default-off ordering of an adjacent, built-in convolution adjoint pair.

The two numerical calls and every saved leaf remain unchanged. Only a proven
weight adjoint moves before its adjacent input/bias adjoint. This removes the
input-adjoint result from that weight call's live interval; it is not a claim
about the model's global peak or permission to release retained autograd state.
"""

from math import prod

import torch

from ..ir import CallNode, ExternKernelNode, TensorValue, model_storage_source
from ..trace.metadata_utils import contiguous_stride
from .fusion_contracts import candidate, select
from .semantic import recorded
from .spatial import Convolution


PATTERN = "native_conv_dw_first"
_DX = "convolution.native_dx_db"
_DW = "convolution.native_dw"
_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _trusted(node, spec, method):
    """Admit the original recorded bound method, not an opaque lookalike."""
    function = getattr(node.target, "numerical_function", None)
    expected = getattr(Convolution, method)
    node_type = CallNode if method == "dx_db" else ExternKernelNode
    return (type(node) is node_type and type(spec) is Convolution
            and getattr(node.target, "specification", None) is spec
            and getattr(function, "__self__", None) is spec
            and getattr(function, "__func__", None) is expected
            and getattr(node.target, "__code__", None)
            is recorded(spec, getattr(spec, method), "backward").__code__)


def _annotated(node):
    # Built-in calls use the current stream and read-only operands. An explicit
    # alternate stream/effect/alias contract needs a different commutation proof.
    if node.scheduling_constraints:
        return True
    if isinstance(node, ExternKernelNode) and (node.autograd_inputs or node.autograd_input_names
            or node.autograd_buffers or node.autograd_buffer_names or node.autograd_owner != "pytorch"):
        return True
    for owner in (node, node.target):
        for name in ("stateful", "effects", "side_effects", "mutates", "mutation", "alias_writes"):
            if getattr(owner, name, False):
                return True
        if any(getattr(owner, name, None) is not None for name in ("stream", "stream_id")):
            return True
    return False


def _dense(metadata, shape, dtype, device, *, contiguous=False):
    return (metadata is not None and metadata.device is not None
            and metadata.shape == tuple(shape) and metadata.dtype == dtype
            and torch.device(metadata.device) == device
            and metadata.representation == "dense" and metadata.storage_dtype is None
            and (not contiguous or metadata.stride == contiguous_stride(metadata.shape)))


def _reason(dx, dw, spec):
    if not _trusted(dx, spec, "dx_db"):
        return "untrusted_input_bias_adjoint"
    if dw is None or getattr(dw, "op", None) != _DW:
        return "adjacent_weight_adjoint_required"
    if not _trusted(dw, spec, "dw"):
        return "untrusted_or_different_weight_adjoint"
    if _annotated(dx) or _annotated(dw):
        return "effect_stream_or_alias_constraint"
    geometry = spec.geometry
    if (spec.precision != "native" or "fp8_training" in spec.plan
            or not spec.needs[0] or not spec.needs[1]
            or spec.rank != 2 or geometry.rank != 2
            or spec.transposed or spec.unbatched or spec.explicit_padding
            or any(spec.synthetic_right) or spec.pad_mode != "zeros"
            or geometry.groups != 1 or geometry.kernel_size != (1, 1)
            or geometry.stride != (1, 1) or geometry.dilation != (1, 1)
            or geometry.padding != (0, 0)
            or spec.input_format is not torch.contiguous_format
            or spec.weight_format is not torch.contiguous_format
            or any(size <= 0 for size in (*geometry.input_shape, *geometry.weight_shape))):
        return "unsupported_convolution_adjoint_geometry"
    if (dx.kwargs or dw.kwargs or len(dx.args) != 2 or len(dw.args) != 4
            or not all(isinstance(value, TensorValue) for value in dx.args)
            or dw.args[0] is not dx.args[0]):
        return "adjoint_arguments_or_upstream"
    left, right = set(dx.outputs()), set(dw.outputs())
    if (left & right or left.intersection(dw.inputs()) or right.intersection(dx.inputs())):
        return "adjoint_data_dependency"
    # The exact built-in executors allocate their results and never overwrite
    # their operands. Reject IR annotations that contradict that producer ABI.
    if any(value.storage_source is not None or model_storage_source(value) is not None
           or value.model_state_copy or value.model_state_origin is not None
           for value in (*dx.outputs(), *dw.outputs())):
        return "adjoint_output_alias"
    source, weight = spec.metas[:2]
    if (source.dtype not in _DTYPES or weight.dtype not in _DTYPES
            or spec.compute_dtype not in _DTYPES or source.device is None):
        return "convolution_operand_metadata"
    device = torch.device(source.device)
    if (device.type not in ("cpu", "cuda")
            or not _dense(source, geometry.input_shape, source.dtype, device, contiguous=True)
            or not _dense(weight, geometry.weight_shape, weight.dtype, device, contiguous=True)
            or not _dense(dx.args[1].meta, geometry.weight_shape, weight.dtype, device, contiguous=True)):
        return "convolution_operand_metadata"
    upstream = dx.args[0].meta
    if (upstream is None or upstream.dtype not in _DTYPES
            or not _dense(upstream, geometry.output_shape, upstream.dtype, device)
            or upstream.stride not in (None, contiguous_stride(upstream.shape))):
        return "upstream_metadata"
    if len(dx.outputs()) != 1 + bool(spec.needs[2]) or len(dw.outputs()) != 1:
        return "adjoint_output_arity"
    expected = (source, *((spec.metas[2],) if spec.needs[2] else ()))
    if any(not _dense(value.meta, metadata.shape, metadata.dtype, device) or value.meta.requires_grad
           for value, metadata in zip(dx.outputs(), expected)):
        return "input_bias_adjoint_metadata"
    if (not _dense(dw.outputs()[0].meta, weight.shape, weight.dtype, device)
            or dw.outputs()[0].meta.requires_grad):
        return "weight_adjoint_metadata"
    saved = dw.args[1:]
    if (not isinstance(saved[0], TensorValue)
            or spec.compressed and not all(isinstance(value, TensorValue) for value in saved)
            or not spec.compressed and saved[1:] != (None, None)):
        return "saved_source_abi"
    if any(value.meta is None or value.meta.device is None
           or torch.device(value.meta.device) != device
           for value in saved if isinstance(value, TensorValue)):
        return "saved_source_metadata"
    return None


def backward(graph, context, *, diagnostics=None):
    """Return selected patterns and retain discovery/default-off decisions."""
    graph.validate()
    order = getattr(context, "native_conv_backward_order", "source")
    if order not in ("source", "dw_first"):
        raise ValueError("native_conv_backward_order must be 'source' or 'dw_first'")
    diagnostics = diagnostics if diagnostics is not None else []
    enabled, patterns = order == "dw_first", []
    # Nodes are never cloned or replaced. Saved inputs, gradient contribution
    # identities and the graph's output/accumulation order therefore stay fixed.
    for dx in tuple(graph.nodes):
        if not isinstance(dx, CallNode) or dx.op != _DX:
            continue
        spec = getattr(dx.target, "specification", None)
        record = candidate(graph, dx, spec, "backward", PATTERN, requested=enabled)
        record.update(transformation="adjoint_scheduling", kernel_fusion=False,
                      requested_order=order, selected_order=None,
                      global_peak_saving_bytes=None, measured_default_promoted=False)
        diagnostics.append(record)
        if not enabled:
            continue
        position = graph.nodes.index(dx)
        dw = graph.nodes[position + 1] if position + 1 < len(graph.nodes) else None
        reason = _reason(dx, dw, spec)
        if reason is not None:
            record.update(eligible=False, reason=reason)
            continue
        before = (dx.op, dw.op)
        after = (dw.op, dx.op)
        outputs = tuple(value.name for node in (dx, dw) for value in node.outputs())
        inputs = tuple(dict.fromkeys(value.name for node in (dx, dw) for value in node.inputs()))
        select(dw, record, roles={"native_weight_gradient": ["aten.convolution_backward"]},
               consumed=before, source_files=("compiler/autodiff/convolution_order.py",
                   "compiler/autodiff/spatial.py", "opt/backends/native/convolution/ops.py"))
        saved = tuple(value for value in dw.args[1:] if isinstance(value, TensorValue))
        record.update(selected_order=after,
            pair_proof={"same_builtin_specification": True, "adjacent_source_calls": True,
                "read_only_operands_fresh_results": True, "no_pair_data_dependency": True,
                "no_effect_or_stream_override": True, "gradient_accumulation_order_unchanged": True,
                "original_order": before, "selected_order": after,
                "shared_upstream": dx.args[0].name, "outputs_unchanged": outputs,
                "saved_leaves_unchanged": tuple(value.name for value in saved)},
            expected_local_overlap_removed_bytes=prod(dx.outputs()[0].meta.shape) * dx.outputs()[0].meta.dtype.itemsize,
            saving_scope="input adjoint absent during this weight call; global peak and workspace are unmeasured",
            source_release_authority="existing final-use manager and autograd saved-owner retention")
        record["replacement_contract"].update(consumed_values=inputs, dependencies=inputs,
            outputs=outputs, effects="read_only_operands_fresh_results", original_order=before,
            selected_order=after, estimated_scratch_bytes=None, saved_leaves_unchanged=True)
        record["legality_signature"].update(selection_control="native_conv_backward_order=dw_first",
                                            pair_operation_order=after)
        for node in (dx, dw):
            node.scheduling_constraints.update(adjoint_scheduling=PATTERN,
                selected_order=after, shared_upstream=dx.args[0].name,
                preserve_gradient_accumulation_order=True)
        graph.nodes[position:position + 2] = (dw, dx)
        graph.validate()
        patterns.append(PATTERN)
    return tuple(patterns)
