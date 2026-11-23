from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import torch

from ..ir import ExternKernelNode, PointwiseNode, TensorMeta
from .metadata_utils import (
    infer_binary_meta,
    infer_first_tensor_meta,
    infer_sum_to_size_meta,
    infer_unary_meta,
)

MetaInfer = Callable[[Tuple[Any, ...], Dict[str, Any]], TensorMeta]


@dataclass(frozen=True)
class TraceOpNodeDefinition:
    name: str
    target: Any
    node_type: type
    infer_meta: MetaInfer


TRACE_OP_NODE_REGISTRY: dict[str, TraceOpNodeDefinition] = {}


def register_trace_op_node(name: str):
    def decorator(func: Callable[[], TraceOpNodeDefinition]):
        if name in TRACE_OP_NODE_REGISTRY:
            raise RuntimeError(f"Trace op {name!r} has already been registered")
        definition = func()
        if not isinstance(definition, TraceOpNodeDefinition):
            raise TypeError(
                f"Registration function for {name!r} must return "
                f"TraceOpNodeDefinition, got {type(definition).__name__}"
            )
        if definition.name != name:
            raise ValueError(
                f"Registry key {name!r} does not match definition name {definition.name}"
            )
        TRACE_OP_NODE_REGISTRY[name] = definition
        return func

    return decorator


def get_trace_op_node(name: str) -> TraceOpNodeDefinition:
    try:
        return TRACE_OP_NODE_REGISTRY[name]
    except KeyError:
        raise NotImplementedError(f"No trace op node registered for {name!r}")


def _pointwise(name: str, target: Any, infer_meta: MetaInfer):
    return TraceOpNodeDefinition(name, target, PointwiseNode, infer_meta)


def _extern(name: str, target: Any, infer_meta: MetaInfer):
    return TraceOpNodeDefinition(name, target, ExternKernelNode, infer_meta)


@register_trace_op_node("add")
def _register_add_op():
    return _pointwise("add", torch.add, infer_binary_meta)


@register_trace_op_node("sub")
def _register_sub_op():
    return _pointwise("sub", torch.sub, infer_binary_meta)


@register_trace_op_node("mul")
def _register_mul_op():
    return _pointwise("mul", torch.mul, infer_binary_meta)


@register_trace_op_node("div")
def _register_div_op():
    return _pointwise("div", torch.div, infer_binary_meta)


@register_trace_op_node("neg")
def _register_neg_op():
    return _pointwise("neg", torch.neg, infer_unary_meta)


@register_trace_op_node("relu")
def _register_relu_op():
    return _pointwise("relu", torch.relu, infer_unary_meta)


@register_trace_op_node("sigmoid")
def _register_sigmoid_op():
    return _pointwise("sigmoid", torch.sigmoid, infer_unary_meta)


@register_trace_op_node("tanh")
def _register_tanh_op():
    return _pointwise("tanh", torch.tanh, infer_unary_meta)


@register_trace_op_node("gelu")
def _register_gelu_op():
    return _pointwise("gelu", torch.nn.functional.gelu, infer_unary_meta)


@register_trace_op_node("silu")
def _register_silu_op():
    return _pointwise("silu", torch.nn.functional.silu, infer_unary_meta)


@register_trace_op_node("leaky_relu")
def _register_leaky_relu_op():
    return _pointwise("leaky_relu", torch.nn.functional.leaky_relu, infer_unary_meta)


@register_trace_op_node("hardtanh")
def _register_hardtanh_op():
    return _pointwise("hardtanh", torch.nn.functional.hardtanh, infer_unary_meta)


@register_trace_op_node("hardswish")
def _register_hardswish_op():
    return _pointwise("hardswish", torch.nn.functional.hardswish, infer_unary_meta)


@register_trace_op_node("hardsigmoid")
def _register_hardsigmoid_op():
    return _pointwise("hardsigmoid", torch.nn.functional.hardsigmoid, infer_unary_meta)


@register_trace_op_node("softplus")
def _register_softplus_op():
    return _pointwise("softplus", torch.nn.functional.softplus, infer_unary_meta)


@register_trace_op_node("mish")
def _register_mish_op():
    return _pointwise("mish", torch.nn.functional.mish, infer_unary_meta)


@register_trace_op_node("sum_to_size")
def _register_sum_to_size_op():
    return _extern(
        "sum_to_size", torch.ops.aten.sum_to_size.default, infer_sum_to_size_meta
    )


@register_trace_op_node("threshold_backward")
def _register_threshold_backward_op():
    return _extern(
        "threshold_backward",
        torch.ops.aten.threshold_backward.default,
        infer_first_tensor_meta,
    )


@register_trace_op_node("sigmoid_backward")
def _register_sigmoid_backward_op():
    return _extern(
        "sigmoid_backward",
        torch.ops.aten.sigmoid_backward.default,
        infer_first_tensor_meta,
    )


@register_trace_op_node("tanh_backward")
def _register_tanh_backward_op():
    return _extern(
        "tanh_backward", torch.ops.aten.tanh_backward.default, infer_first_tensor_meta
    )


@register_trace_op_node("gelu_backward")
def _register_gelu_backward_op():
    return _extern(
        "gelu_backward", torch.ops.aten.gelu_backward.default, infer_first_tensor_meta
    )


@register_trace_op_node("silu_backward")
def _register_silu_backward_op():
    return _extern(
        "silu_backward", torch.ops.aten.silu_backward.default, infer_first_tensor_meta
    )


@register_trace_op_node("leaky_relu_backward")
def _register_leaky_relu_backward_op():
    return _extern(
        "leaky_relu_backward",
        torch.ops.aten.leaky_relu_backward.default,
        infer_first_tensor_meta,
    )


@register_trace_op_node("hardtanh_backward")
def _register_hardtanh_backward_op():
    return _extern(
        "hardtanh_backward",
        torch.ops.aten.hardtanh_backward.default,
        infer_first_tensor_meta,
    )


@register_trace_op_node("hardswish_backward")
def _register_hardswish_backward_op():
    return _extern(
        "hardswish_backward",
        torch.ops.aten.hardswish_backward.default,
        infer_first_tensor_meta,
    )


@register_trace_op_node("hardsigmoid_backward")
def _register_hardsigmoid_backward_op():
    return _extern(
        "hardsigmoid_backward",
        torch.ops.aten.hardsigmoid_backward.default,
        infer_first_tensor_meta,
    )


@register_trace_op_node("softplus_backward")
def _register_softplus_backward_op():
    return _extern(
        "softplus_backward",
        torch.ops.aten.softplus_backward.default,
        infer_first_tensor_meta,
    )


@register_trace_op_node("mish_backward")
def _register_mish_backward_op():
    return _extern(
        "mish_backward", torch.ops.aten.mish_backward.default, infer_first_tensor_meta
    )
