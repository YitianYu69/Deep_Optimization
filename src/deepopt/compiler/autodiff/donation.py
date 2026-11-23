"""Donate only proven private convolution input gradients to backward adds.

This pass changes an allocating add into an explicit invocation-owned write.
It does not grant ownership to graph inputs, saved values, views, or arbitrary
operations, and it never donates forward state or uses a reusable arena.
"""

from functools import partial
from math import prod

import torch

from ..ir import (CallNode, FreshAllocationContract, PointwiseNode, TensorValue,
                  find_tensor_values, model_storage_source)
from ..trace.metadata_utils import contiguous_stride
from .fusion_contracts import candidate, select


_PATTERN = "fresh_backward_add_donation"
_PRODUCER = "convolution.native_dx_db"
_OWNER = "backward_invocation"
_PROTECTED = frozenset(("user_input", "upstream_gradient", "saved_state", "parameter",
                        "persistent_buffer", "stochastic_depth_mask"))


def _add_out(left, right, *, donor_index, metadata):
    """Keep operand order and fail before writing if the private ABI changed."""
    operands = (left, right)
    if donor_index not in (0, 1):
        raise ValueError("Backward add donation has an invalid destination")
    expected_device = torch.device(metadata.device)
    for value in operands:
        if type(value) is not torch.Tensor:
            raise TypeError("Backward add donation requires ordinary dense tensors")
        if (tuple(value.shape) != metadata.shape or value.dtype != metadata.dtype
                or value.device.type != expected_device.type
                or expected_device.index is not None and value.device.index != expected_device.index
                or tuple(value.stride()) != metadata.stride):
            raise ValueError("Backward add donation operand violates compiled tensor metadata")
        if value.requires_grad:
            raise RuntimeError("Backward add donation supports first-order gradients only")
    donor, other = operands[donor_index], operands[1 - donor_index]
    if (donor._base is not None or donor.storage_offset() != 0
            or not donor.is_contiguous() or donor.numel() == 0
            or donor.untyped_storage().nbytes() != donor.numel() * donor.element_size()):
        raise ValueError("Backward add donation destination is not private full contiguous storage")
    if donor.untyped_storage().data_ptr() == other.untyped_storage().data_ptr():
        raise ValueError("Backward add donation operands share storage")
    # add_(other) would reverse the original operands when the right operand
    # is donated. The out ABI preserves their order and the original dtype.
    return torch.add(left, right, out=donor)


def _geometry_reason(producer):
    from .spatial import Convolution

    spec = getattr(producer.target, "specification", None)
    function = getattr(producer.target, "numerical_function", None)
    if (type(spec) is not Convolution or getattr(function, "__self__", None) is not spec
            or getattr(function, "__func__", None) is not Convolution.dx_db):
        return "untrusted_fresh_producer"
    geometry = spec.geometry
    if (geometry.rank != 2 or tuple(geometry.kernel_size) != (1, 1)
            or geometry.groups != 1 or tuple(geometry.stride) != (1, 1)
            or tuple(geometry.dilation) != (1, 1) or any(geometry.padding)
            or spec.transposed or spec.unbatched or spec.explicit_padding
            or any(spec.synthetic_right) or not spec.needs[0]
            or spec.input_format is not torch.contiguous_format):
        return "unsupported_convolution_gradient_geometry"
    return None


def _donor_reason(graph, node, value, unsafe_values):
    proof = getattr(value, "fresh_allocation", None)
    if not isinstance(proof, FreshAllocationContract):
        return "missing_fresh_allocation_proof"
    if proof.owner != _OWNER or proof.producer_op != _PRODUCER:
        return "fresh_allocation_owner_or_producer"
    producer = value.producer
    if (not isinstance(producer, CallNode) or producer.op != _PRODUCER
            or not producer.values or producer.values[0] is not value):
        return "fresh_allocation_producer_mismatch"
    reason = _geometry_reason(producer)
    if reason:
        return reason
    if value.meta != proof.meta:
        return "stale_fresh_allocation_metadata"
    if value in unsafe_values or node.value in unsafe_values:
        return "unsafe_value"
    if (value.provenance in _PROTECTED or value.storage_source is not None
            or value.model_state_copy or value.model_state_origin is not None
            or model_storage_source(value) is not None):
        return "protected_storage"
    # Inspect the graph, rather than interpreting a possibly duplicated use
    # list as storage ownership. A view or saved export is another consumer.
    consumers = [item for item in graph.nodes if any(source is value for source in item.inputs())]
    if consumers != [node] or any(user is not node for user in value.users):
        return "donor_fanout_or_escape"
    for item in graph.nodes:
        saved = find_tensor_values(getattr(item, "saved_tensors_for_bwd", ()))
        if any(source is value for source in saved):
            return "retained_backward_value"
    return None


def _add_reason(node):
    if len(node.args) != 2 or any(not isinstance(value, TensorValue) for value in node.args):
        return "add_tensor_arguments"
    # Native AD's accumulation add has no options. Do not normalize another
    # callable's alpha/out argument validation while introducing the write.
    if node.kwargs:
        return "add_options"
    if node.args[0] is node.args[1]:
        return "operand_alias"
    if node.scheduling_constraints:
        return "existing_scheduling_constraints"
    metadata = node.value.meta
    if (metadata is None or metadata.device is None or len(metadata.shape) != 4 or not prod(metadata.shape)
            or metadata.dtype not in (torch.bfloat16, torch.float32)
            or metadata.representation != "dense" or metadata.storage_dtype is not None
            or metadata.requires_grad or metadata.stride != contiguous_stride(metadata.shape)):
        return "add_output_metadata"
    try:
        device = torch.device(metadata.device)
    except (TypeError, ValueError, RuntimeError):
        return "add_output_metadata"
    for value in node.args:
        current = value.meta
        if (current is None or current.device is None or current.shape != metadata.shape or current.dtype != metadata.dtype
                or torch.device(current.device) != device
                or current.stride != metadata.stride or current.requires_grad
                or current.representation != "dense" or current.storage_dtype is not None):
            return "add_operand_metadata"
    return None


def backward(graph, context, *, unsafe_values=frozenset(), diagnostics=None):
    """Return selected patterns, recording default-off decisions as metadata."""
    from .optimization import rebuild

    graph.validate()
    diagnostics = diagnostics if diagnostics is not None else []
    enabled = bool(getattr(context, "donate_fresh_backward_add", False))
    patterns = []
    for node in list(graph.nodes):
        if not isinstance(node, PointwiseNode) or node.op != "add":
            continue
        record = candidate(graph, node, None, "backward", _PATTERN, requested=enabled)
        record.update(transformation="buffer_donation", kernel_fusion=False,
                      donation_scope=_OWNER, numerical_executor="torch.add(out=owned_operand)")
        diagnostics.append(record)
        if not enabled:
            continue
        reason = _add_reason(node)
        if reason is None and any(value in unsafe_values for value in (*node.args, node.value)):
            reason = "unsafe_value"
        if reason:
            record.update(eligible=False, reason=reason)
            continue
        reasons = [_donor_reason(graph, node, value, unsafe_values) for value in node.args]
        donor_index = next((index for index, rejection in enumerate(reasons) if rejection is None), None)
        if donor_index is None:
            record.update(eligible=False, reason="no_private_fresh_operand", operand_rejections=reasons)
            continue
        donor = node.args[donor_index]
        proof = donor.fresh_allocation
        metadata = node.value.meta
        producer_spec = donor.producer.target.specification
        function = partial(_add_out, donor_index=donor_index, metadata=metadata)
        replacement = CallNode("memory.donated_add_out", function, node.args, {}, (node.value,))
        replacement.bf16_policy = node.bf16_policy
        for name in ("module_path", "source_node"):
            if hasattr(node, name):
                setattr(replacement, name, getattr(node, name))
        select(replacement, record, roles={"owned_add_out": ["aten.add.out"]}, consumed=(node.op,),
               source_files=("compiler/autodiff/donation.py", "compiler/ir.py",
                             "compiler/autodiff/spatial.py", "opt/backends/native/convolution/ops.py"))
        alias = dict(donor=donor.name, result=node.value.name, scope=_OWNER,
                     producer_op=proof.producer_op)
        replacement.scheduling_constraints.update(mutation="invocation_owned_add_out",
            owned_output_alias=alias, preserve_operand_order=True)
        # The result aliases this allocation. It is deliberately not a new
        # proven convolution allocation eligible for chained donation.
        node.value.fresh_allocation = None
        record.update(donor_index=donor_index, donor=donor.name,
                      donor_producer=dict(operation=proof.producer_op,
                          module_path=producer_spec.module_path, node_name=producer_spec.node_name),
                      first_order_only=True,
                      operand_order=tuple(value.name for value in node.args),
                      estimated_removed_output_allocation_bytes=prod(metadata.shape) * metadata.dtype.itemsize,
                      saving_scope="one static output allocation; not an observed peak reduction")
        record["replacement_contract"].update(effects="invocation_owned_add_out",
            mutates_input=donor.name, owned_output_alias=alias, preserve_operand_order=True,
            estimated_scratch_bytes=0)
        graph.nodes[graph.nodes.index(node)] = replacement
        patterns.append(_PATTERN)
        rebuild(graph)
    return tuple(patterns)
