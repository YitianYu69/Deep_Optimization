"""Fuse only permitted saved affine coefficients in a native-derived mul VJP."""

from functools import partial
from math import prod

from deepopt.opt.backends._shared.packed_multiply import (
    DEFAULT_SCRATCH_BYTES, multiply as _execute, packed_multiply_rejection, plan_packed_multiply,
)
from ..ir import CallNode, PointwiseNode, TensorValue
from .fusion_contracts import candidate, select


def _multiply(gradient, *args, **kwargs):
    # Static CPU graph tests retain the explicit codec oracle. A selected CUDA
    # replacement must never silently materialize an activation-sized fallback.
    return _execute(gradient, *args, **kwargs, allow_materialized=gradient.device.type != "cuda")


def backward(graph, context, diagnostics=None):
    from .optimization import rebuild

    diagnostics = diagnostics if diagnostics is not None else []
    patterns = []
    enabled = context.fuse_packed_multiply and "backward" in context.fusion_phases
    for node in list(graph.nodes):
        if not isinstance(node, PointwiseNode) or node.op != "mul" or len(node.args) != 2:
            continue
        restored = [(index, value.producer) for index, value in enumerate(node.args)
                    if isinstance(value, TensorValue) and getattr(value.producer, "op", None) == "codec.restore_activation"]
        if len(restored) != 1:
            continue
        index, restore = restored[0]
        gradient = node.args[1 - index]
        if not isinstance(gradient, TensorValue):
            continue
        finish = node
        if len(node.value.users) == 1 and getattr(node.value.users[0], "op", None) == "sum_to_size":
            finish = node.value.users[0]
        target = finish.outputs()[0]
        shape = tuple(node.value.meta.shape)
        target_shape = tuple(target.meta.shape)
        options = dict(restore.specification)
        record = candidate(graph, node, None, "backward", "packed_multiply_reduce" if finish is not node else "packed_multiply",
                           requested=enabled)
        record["transformation"] = "kernel_fusion"
        diagnostics.append(record)
        reason = None
        if not enabled:
            continue
        policy = context.kernel_policy
        reason = packed_multiply_rejection(gradient_shape=gradient.meta.shape,
            gradient_dtype=gradient.meta.dtype, gradient_device=gradient.meta.device,
            options=options, shape=shape, product_dtype=node.value.meta.dtype,
            target_shape=target_shape, target_dtype=target.meta.dtype, kernel_policy=policy)
        if reason is None and (gradient.meta.representation != "dense"
                or getattr(gradient.meta, "storage_dtype", None) is not None):
            reason = "prepared_gradient_transport"
        elif reason is None and not prod(shape):
            reason = "broadcast_geometry"
        elif reason is None and target.meta.dtype != node.value.meta.dtype:
            reason = "broadcast_or_sum_to_size_contract"
        memory = getattr(context, "activation_memory", None)
        scratch_cap = getattr(memory, "reduction_scratch_bytes", DEFAULT_SCRATCH_BYTES)
        try:
            launch = plan_packed_multiply(shape, target_shape, scratch_cap)
        except ValueError as error:
            reason = reason or str(error)
        if reason:
            record.update(reason=reason, eligible=False)
            continue
        function = partial(_multiply, options=options, shape=shape, product_dtype=node.value.meta.dtype,
                           target_shape=target_shape, target_dtype=target.meta.dtype,
                           scratch_bytes=scratch_cap, kernel_policy=policy,
                           execution_info=record.setdefault("runtime_execution", {})
                           if context.collect_saved_state_metadata else None)
        replacement = CallNode("codec.packed_multiply_reduce" if finish is not node else "codec.packed_multiply",
                               function, (gradient, *restore.args), {}, (target,), options)
        select(replacement, record, roles={"decode_product": ["_packed_multiply_partial"] if finish is not node else ["_packed_multiply"]} |
               ({"bounded_finish": ["_packed_multiply_finish"]} if finish is not node else {}),
               consumed=(restore.op, node.op, *((finish.op,) if finish is not node else ())),
               source_files=("opt/backends/_shared/packed_multiply.py", "opt/backends/triton/packed_multiply.py"))
        record["replacement_contract"].update(estimated_scratch_bytes=launch.scratch_bytes,
            launch_plan=launch.as_dict(),
            rounding="decode_to_operand_dtype_then_product_to_original_dtype_before_sum",
            decoded_full_temporary_bytes=0, product_full_temporary_bytes=0,
            broadcast_full_temporary_bytes=0 if finish is not node else None)
        graph.nodes[graph.nodes.index(finish)] = replacement
        if finish is not node:
            graph.nodes.remove(node)
        rebuild(graph)
        if not restore.values[0].users:
            graph.nodes.remove(restore)
            rebuild(graph)
        record["replacement_contract"]["coefficient_restore_retained"] = bool(restore.values[0].users)
        if restore.values[0].users:
            record["replacement_contract"]["decoded_full_temporary_bytes"] = (
                prod(restore.values[0].meta.shape) * restore.values[0].meta.dtype.itemsize)
        patterns.append(record["pattern"])
    return tuple(patterns)
