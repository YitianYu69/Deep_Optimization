"""Default-off private decoded-workspace reuse after an unchanged full dW."""

from math import prod

import torch

from ..ir import CallNode, ExternKernelNode, TensorValue, model_storage_source
from ..trace.metadata_utils import contiguous_stride
from .convolution_order import _annotated, _trusted
from .fusion_contracts import candidate, select
from .semantic import recorded
from .spatial import Convolution


PATTERN = "native_conv_private_workspace_tiled_dx"
OPERATION = "convolution.native_private_workspace_tiled_dx"


def is_private_workspace_consumer(node):
    """Prove this exact internally mutating executor leaves every input intact.

    Offload planning can consume the proof without treating arbitrary effect
    annotations or a matching operation name as an alias-lifetime guarantee.
    """
    if type(node) is not CallNode or node.op != OPERATION or node.kwargs:
        return False
    spec = node.specification
    if type(spec) is not Convolution or len(node.args) != 5 or not node.values:
        return False
    target = node.target
    trusted = False
    for _ in range(4):
        if target is None:
            break
        numerical = getattr(target, "numerical_function", None)
        if (getattr(target, "specification", None) is spec and getattr(numerical, "__self__", None) is spec
                and getattr(numerical, "__func__", None) is Convolution.reuse_gradients
                and getattr(target, "__code__", None) is recorded(spec, spec.reuse_gradients, "backward").__code__):
            trusted = True
            break
        target = getattr(target, "__wrapped__", None)
    constraints = node.scheduling_constraints
    expected_alias = {"temporary": "decoded_source", "output": node.values[0].name, "scope": "backward_invocation"}
    if (not trusted or constraints.get("effects") != "private_decoded_workspace_overwrite"
            or constraints.get("internal_workspace_alias") != expected_alias
            or constraints.get("no_input_mutation") is not True or constraints.get("dW_before_overwrite") is not True
            or constraints.get("stream_dependency") != "current_stream_program_order"
            or node.saved_tensors_for_bwd
            or any(constraints.get(key) for key in ("mutation", "mutates", "alias_writes", "owned_output_alias",
                                                   "output_alias", "stream", "stream_id"))):
        return False
    for owner in (node, node.target):
        if (any(getattr(owner, key, False) for key in ("stateful", "side_effects", "mutates", "mutation", "alias_writes"))
                or any(getattr(owner, key, None) is not None for key in ("stream", "stream_id"))):
            return False
    batch_size = getattr(spec.context, "native_conv_dx_batch_size", 1)
    if (getattr(spec.context, "native_conv_workspace_reuse", "off") != "tiled_dx"
            or type(batch_size) is not int or batch_size < 1 or constraints.get("dx_batch_size") != batch_size):
        return False
    # Reconstruct only the original pair's metadata schema, never its numeric
    # execution or graph links, to share the complete conservative ABI checks.
    dx = CallNode("convolution.native_dx_db", recorded(spec, spec.dx_db, "backward"),
                  node.args[:2], {}, node.values[:-1])
    dw = ExternKernelNode("convolution.native_dw", recorded(spec, spec.dw, "backward"),
                         (node.args[0], *node.args[2:]), {}, node.values[-1])
    return _reason(dx, dw, spec, batch_size) is None


def _reason(dx, dw, spec, batch_size):
    from deepopt.opt.backends.native.convolution.ops import _workspace_geometry_reason

    if not _trusted(dx, spec, "dx_db"):
        return "untrusted_input_bias_adjoint"
    if dw is None or getattr(dw, "op", None) != "convolution.native_dw" or not _trusted(dw, spec, "dw"):
        return "adjacent_same_native_weight_adjoint_required"
    if _annotated(dx) or _annotated(dw) or dx.saved_tensors_for_bwd or dw.saved_tensors_for_bwd:
        return "effect_alias_stream_or_retained_override"
    if (type(spec) is not Convolution or spec.precision != "native" or "fp8_training" in spec.plan
            or not spec.needs[0] or not spec.needs[1] or not spec.compressed
            or spec.compression["bits"] != 2 or spec.compression["group_size"] != 1024
            or spec.codec == "materialized"):
        return "native_bf16_group1024_packed_source_required"
    reason = _workspace_geometry_reason(spec.geometry)
    if reason is not None:
        return reason
    if (spec.transposed or spec.unbatched or spec.explicit_padding or any(spec.synthetic_right)
            or spec.pad_mode != "zeros" or spec.input_format is not torch.contiguous_format
            or spec.weight_format is not torch.contiguous_format):
        return "uncropped_nchw_convolution_required"
    if not 0 < batch_size < spec.geometry.batch_size:
        return "strict_partial_batch_bound_required"
    if (dx.kwargs or dw.kwargs or len(dx.args) != 2 or len(dw.args) != 4
            or dw.args[0] is not dx.args[0]
            or any(not isinstance(value, TensorValue) for value in (*dx.args, *dw.args))):
        return "adjoint_arguments_or_upstream"
    source, weight = spec.metas[:2]
    gradient = dx.args[0].meta
    operand = dx.args[1].meta
    if (source.dtype != torch.bfloat16 or weight.dtype != torch.float32 or spec.compute_dtype != torch.bfloat16
            or any(meta is None or meta.device is None or meta.representation != "dense"
                   or meta.storage_dtype is not None for meta in (source, weight, gradient, operand))):
        return "bf16_source_fp32_master_metadata_required"
    device = torch.device(source.device)
    # CPU's channel-row decoder can return a view for singleton channel or
    # spatial dimensions. Do not clone a full source merely to obtain the
    # private, zero-offset allocation required by this optimization. CUDA's
    # qualified spatial decoder allocates its NCHW result directly.
    if device.type == "cpu" and (source.shape[1] == 1 or prod(source.shape[2:]) == 1):
        return "cpu_singleton_decode_layout_not_qualified"
    if (device.type not in ("cpu", "cuda") or any(torch.device(meta.device) != device for meta in (weight, gradient, operand))
            or source.shape != spec.geometry.input_shape or weight.shape != spec.geometry.weight_shape
            or source.stride != contiguous_stride(source.shape) or weight.stride != contiguous_stride(weight.shape)
            or operand.shape != weight.shape or operand.dtype != weight.dtype or operand.stride != weight.stride
            or gradient.shape != spec.geometry.output_shape or gradient.dtype != torch.bfloat16):
        return "convolution_operand_metadata"
    groups = (prod(source.shape) + 1023) // 1024
    expected = (((groups, 64), torch.int32), ((groups,), torch.float32), ((groups,), torch.float32))
    for value, (shape, dtype) in zip(dw.args[1:], expected):
        meta = value.meta
        if (meta is None or meta.device is None or torch.device(meta.device) != device
                or meta.shape != shape or meta.dtype != dtype or meta.requires_grad
                or meta.stride not in (None, contiguous_stride(shape))):
            return "packed_source_metadata"
    if len(dx.values) != 1 + bool(spec.needs[2]) or len(dw.outputs()) != 1:
        return "adjoint_output_arity"
    outputs = (*dx.values, *dw.outputs())
    expected_outputs = (source, *((spec.metas[2],) if spec.needs[2] else ()), weight)
    if any(value.meta is None or value.meta.shape != meta.shape or value.meta.dtype != meta.dtype
           or value.meta.device != meta.device or value.meta.requires_grad
           or value.meta.storage_dtype is not None or value.meta.representation != "dense"
           or value.storage_source is not None or model_storage_source(value) is not None
           or value.model_state_copy or value.model_state_origin is not None
           for value, meta in zip(outputs, expected_outputs)):
        return "adjoint_output_metadata_or_alias"
    if set(dx.outputs()).intersection(dw.inputs()) or set(dw.outputs()).intersection(dx.inputs()):
        return "adjoint_data_dependency"
    return None


def backward(graph, context, *, diagnostics=None):
    from .optimization import rebuild

    graph.validate()
    mode = getattr(context, "native_conv_workspace_reuse", "off")
    batch_size = getattr(context, "native_conv_dx_batch_size", 1)
    if type(mode) is not str or mode not in ("off", "tiled_dx", "whole_batch"):
        raise ValueError("native_conv_workspace_reuse must be off, tiled_dx, or whole_batch")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("native_conv_dx_batch_size must be a positive integer")
    diagnostics = diagnostics if diagnostics is not None else []
    patterns = []
    for dx in tuple(graph.nodes):
        if type(dx) is not CallNode or dx.op != "convolution.native_dx_db":
            continue
        spec = getattr(dx.target, "specification", None)
        record = candidate(graph, dx, spec, "backward", PATTERN, requested=mode != "off")
        record.update(transformation="private_convolution_workspace", kernel_fusion=False,
            requested_mode=mode, dx_batch_size=batch_size,
            whole_batch_destination_supported=False,
            whole_batch_rejection_reason="no_qualified_true_destination_kernel",
            global_peak_saving_bytes=None, numerical_scope="unchanged_full_batch_dw_db_then_tiled_native_dx")
        diagnostics.append(record)
        if mode == "off":
            continue
        if mode == "whole_batch":
            record.update(eligible=False, reason="whole_batch_destination_unqualified")
            continue
        position = graph.nodes.index(dx)
        dw = graph.nodes[position + 1] if position + 1 < len(graph.nodes) else None
        reason = _reason(dx, dw, spec, batch_size)
        if reason is not None:
            record.update(eligible=False, reason=reason)
            continue
        outputs = (*dx.values, *dw.outputs())
        replacement = CallNode(OPERATION, recorded(spec, spec.reuse_gradients, "backward"),
            (*dx.args, *dw.args[1:]), {}, outputs, spec)
        replacement.bf16_policy = dx.bf16_policy
        replacement.module_path = spec.module_path
        # Bind the native arithmetic shape into candidate/cache evidence as
        # well as the final schedule token; source code alone cannot identify
        # a different bound held inside the immutable codegen context.
        record["native_statistics_contract"] = dict(dx_batch_size=batch_size,
            full_batch_dw_shape=spec.geometry.output_shape, full_batch_db=bool(spec.needs[2]))
        select(replacement, record, roles={"full_batch_weight_gradient": ["aten.convolution_backward"],
            "bounded_native_input_gradient": ["aten.convolution_backward"],
            "partial_copy_into_private_workspace": ["aten.copy_"]}, consumed=(dx.op, dw.op),
            source_files=("compiler/autodiff/convolution_reuse.py", "compiler/autodiff/spatial.py",
                          "opt/backends/native/convolution/ops.py"))
        alias = {"temporary": "decoded_source", "output": outputs[0].name, "scope": "backward_invocation"}
        effects = "private_decoded_workspace_overwrite"
        replacement.scheduling_constraints.update(effects=effects, internal_workspace_alias=alias,
            no_input_mutation=True, dW_before_overwrite=True, stream_dependency="current_stream_program_order",
            preserve_gradient_accumulation_order=True, dx_batch_size=batch_size)
        workspace = prod(spec.geometry.input_shape) * torch.bfloat16.itemsize
        partial = batch_size * prod(spec.geometry.input_shape[1:]) * torch.bfloat16.itemsize
        partial_layout = batch_size * prod(spec.geometry.output_shape[1:]) * torch.bfloat16.itemsize
        record["replacement_contract"].update(effects=effects, internal_workspace_alias=alias,
            no_input_mutation=True, full_batch_dw=True, full_batch_db=bool(spec.needs[2]),
            dx_batch_size=batch_size, dx_slices=(spec.geometry.batch_size + batch_size - 1) // batch_size,
            decoded_workspace_bytes=workspace, max_partial_dx_bytes=partial,
            max_partial_upstream_layout_bytes=partial_layout,
            max_partial_dx_and_layout_bytes=partial + partial_layout,
            metadata_placeholder_bytes=0, vendor_workspace_bytes=None,
            layout_conversion_scope="native_library_partial_gradient_may_copy; full_dw_recipe_unchanged",
            estimated_scratch_bytes=partial + partial_layout, original_gradient_values=tuple(value.name for value in outputs),
            saved_leaves_unchanged=True, fresh_output_producer=OPERATION,
            rounding="same_full_batch_BF16_dw_store_then_FP32_master_projection; changed_native_dx_batch_shape")
        record["legality_signature"].update(selection_control="native_conv_workspace_reuse=tiled_dx",
            dx_batch_size=batch_size, needs=tuple(bool(value) for value in spec.needs[:3]))
        # This is a different producer. It may not inherit the previous dX's
        # permission for another alias-writing transformation.
        for value in outputs:
            value.fresh_allocation = None
        graph.nodes[position:position + 2] = (replacement,)
        rebuild(graph)
        patterns.append(PATTERN)
    return tuple(patterns)
