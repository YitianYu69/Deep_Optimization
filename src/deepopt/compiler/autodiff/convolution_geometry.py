"""Opt-in elimination of CUDA convolution dX's full-size metadata input.

The numerical call remains a native convolution adjoint. This pass neither
fuses kernels nor grants ownership of the returned gradient to another pass.
"""

from functools import partial, update_wrapper
from math import prod

import torch

from ..ir import CallNode, TensorValue
from ..trace.metadata_utils import contiguous_stride
from .fusion_contracts import candidate, select
from .semantic import recorded


PATTERN = "native_conv_cuda_geometry"
OPERATION = "convolution.native_dx_db_geometry"


def _reason(node, spec):
    from deepopt.opt.backends.native.convolution.ops import _cuda_geometry_reason
    from .spatial import Convolution

    function = getattr(node.target, "numerical_function", None)
    if (type(node) is not CallNode or type(spec) is not Convolution
            or getattr(function, "__self__", None) is not spec
            or getattr(function, "__func__", None) is not Convolution.dx_db
            or getattr(node.target, "__code__", None)
            is not recorded(spec, spec.dx_db, "backward").__code__):
        return "untrusted_convolution_dx_executor"
    if spec.precision != "native" or "fp8_training" in spec.plan:
        return "native_convolution_required"
    for owner in (node, node.target):
        if any(getattr(owner, name, False) for name in
               ("stateful", "effects", "side_effects", "mutates", "mutation", "alias_writes")):
            return "effect_or_alias_override"
        if any(getattr(owner, name, None) is not None for name in ("stream", "stream_id")):
            return "stream_override"
    if node.saved_tensors_for_bwd:
        return "retained_backward_node"
    reason = _cuda_geometry_reason(spec.geometry)
    if reason is not None:
        return reason
    if (spec.transposed or spec.unbatched or spec.explicit_padding or any(spec.synthetic_right)
            or spec.pad_mode != "zeros" or not spec.needs[0] or spec.needs[2]
            or spec.metas[2] is not None or spec.input_format is not torch.contiguous_format
            or spec.weight_format is not torch.contiguous_format):
        return "ordinary_nchw_dx_only_convolution_required"
    if spec.compute_dtype != torch.bfloat16 or spec.metas[0].dtype != torch.bfloat16:
        return "bf16_compute_and_input_gradient_required"
    if (len(node.args) != 2 or node.kwargs or len(node.values) != 1
            or any(not isinstance(value, TensorValue) for value in node.args)):
        return "convolution_dx_call_schema"
    source, weight, gradient, output = spec.metas[0], spec.metas[1], node.args[0].meta, node.values[0].meta
    operand = node.args[1].meta
    if any(meta is None or meta.device is None or meta.representation != "dense"
           or meta.storage_dtype is not None for meta in (source, weight, gradient, output, operand)):
        return "dense_tensor_metadata_required"
    if any(torch.device(meta.device).type != "cuda" for meta in (source, weight, gradient, output, operand)):
        return "cuda_only_geometry_route"
    if any(torch.device(meta.device) != torch.device(source.device) for meta in (weight, gradient, output, operand)):
        return "same_device_metadata_required"
    if (source.shape != spec.geometry.input_shape or weight.shape != spec.geometry.weight_shape
            or weight.dtype not in (torch.float32, torch.bfloat16)
            or operand.shape != weight.shape or operand.dtype != weight.dtype
            or operand.stride != weight.stride
            or source.stride != contiguous_stride(source.shape)
            or weight.stride != contiguous_stride(weight.shape)):
        return "contiguous_source_and_weight_metadata_required"
    expected_gradient = (spec.geometry.batch_size, spec.geometry.out_channels, *spec.geometry.output_spatial)
    if (gradient.shape != expected_gradient or gradient.dtype != torch.bfloat16
            or gradient.stride is not None and gradient.stride != contiguous_stride(gradient.shape)
            or output.shape != source.shape or output.dtype != source.dtype
            or gradient.requires_grad or output.requires_grad
            or output.stride is not None and output.stride != contiguous_stride(output.shape)):
        return "contiguous_bf16_gradient_metadata_required"
    return None


def backward(graph, context, *, diagnostics=None):
    """Select a narrow CUDA geometry route, leaving rejected nodes untouched."""
    from .optimization import rebuild

    graph.validate()
    diagnostics = diagnostics if diagnostics is not None else []
    enabled = bool(getattr(context, "cuda_conv_dx_geometry", False))
    patterns = []
    for node in list(graph.nodes):
        if not isinstance(node, CallNode) or node.op != "convolution.native_dx_db":
            continue
        spec = getattr(node.target, "specification", None)
        record = candidate(graph, node, spec, "backward", PATTERN, requested=enabled)
        record.update(transformation="metadata_allocation_elision", kernel_fusion=False,
                      qualification="experimental_cuda_route",
                      numerical_executor="aten.convolution_backward(dx_only, expanded_one_element_input)")
        diagnostics.append(record)
        if not enabled:
            continue
        reason = _reason(node, spec)
        if reason is not None:
            record.update(eligible=False, reason=reason)
            continue
        function = update_wrapper(partial(spec.dx_db, cuda_geometry=True), spec.dx_db)
        replacement = CallNode(OPERATION, recorded(spec, function, "backward"), node.args,
                               dict(node.kwargs), node.values, spec)
        replacement.bf16_policy = node.bf16_policy
        replacement.scheduling_constraints.update(node.scheduling_constraints)
        replacement.scheduling_constraints.update(metadata_input_shape_only=True,
                                                  native_output_mask=(True, False, False))
        for name in ("module_path", "source_node"):
            if hasattr(node, name):
                setattr(replacement, name, getattr(node, name))
        select(replacement, record, roles={"native_dx_geometry": ["aten.convolution_backward"]},
               consumed=(node.op,), source_files=("compiler/autodiff/convolution_geometry.py",
                   "compiler/autodiff/spatial.py", "opt/backends/native/convolution/ops.py"))
        # The former producer's ownership proof names a different executor.
        # This route has no permission to participate in fresh-add donation.
        for value in replacement.values:
            value.fresh_allocation = None
        placeholder_bytes = prod(spec.geometry.input_shape) * torch.bfloat16.itemsize
        record.update(estimated_removed_full_shape_placeholder_bytes=placeholder_bytes,
                      replacement_metadata_storage_bytes=torch.bfloat16.itemsize,
                      estimated_net_allocation_bytes_removed=placeholder_bytes - torch.bfloat16.itemsize,
                      saving_scope="one static metadata input allocation; not an observed peak reduction")
        record["replacement_contract"].update(native_output_mask=(True, False, False),
            input_descriptor="one_element_expanded_view", output_layout="contiguous_nchw",
            runtime_rejects_unsupported=True, materialized_fallback=False,
            estimated_scratch_bytes=torch.bfloat16.itemsize)
        graph.nodes[graph.nodes.index(node)] = replacement
        patterns.append(PATTERN)
        rebuild(graph)
    return tuple(patterns)
