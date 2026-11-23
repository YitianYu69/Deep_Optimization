"""Explicit branch state and metadata-only VJPs for VAE tensor primitives."""

from dataclasses import dataclass, replace
from functools import partial
from math import prod
from numbers import Real

import torch

from ..ir import CallNode, ExternKernelNode, TensorMeta, TensorValue
from ..lowering import call_method_helper
from ..trace.metadata_utils import contiguous_stride
from deepopt.opt.nn.primitive_state import (
    Nearest2dGeometry, ScalarClampState, clamp_endpoint_policy, nearest2d_precision,
)
from .activation_saving import SavedUse
from .backward_rules import GradWrapper
from .semantic import emit


@dataclass(eq=False)
class NativeNearest2dNode(ExternKernelNode):
    pass


def nearest2d_extension():
    """Unpack the compiler facade while retaining native primitive ownership."""
    from ..registry import OperatorExtension
    from deepopt.opt.nn.primitive_state import NEAREST2D_FUNCTIONS

    def match(node, gm):
        return (node.op == "call_function" and getattr(node.target, "_deepopt_compiler_operation", False)
                and getattr(node.target, "original", None) in NEAREST2D_FUNCTIONS)

    def lower_capture(context, node):
        args, kwargs, _, _ = context.map_arg(node.args)
        result = NativeNearest2dNode("nearest2d", node.target.original, args, kwargs,
                                    context.tensor_value(node), autograd_owner="pytorch")
        context.add_node(node, result)
        return result

    return OperatorExtension(name="native_nearest2d", node_type=NativeNearest2dNode,
        matcher=match, lowering=lower_capture, kernel_codegen=lambda node, context: node.target)


def argument(node, index, name, default=None):
    return node.args[index] if len(node.args) > index else node.kwargs.get(name, default)


def _input(node):
    return argument(node, 0, "input", node.kwargs.get("self"))


def _clamp_kind(node):
    target = node.target
    if isinstance(target, partial) and target.func is call_method_helper:
        name = target.args[0]
    elif isinstance(target, torch._ops.OpOverload):
        if target._schema.overload_name not in ("", "default"):
            return None
        name = target._schema.name.removeprefix("aten::")
    else:
        name = next((name for name in ("clamp", "clip", "clamp_min", "clamp_max")
                     if target is getattr(torch, name, None)), None)
    return "clamp" if name == "clip" else name if name in ("clamp", "clamp_min", "clamp_max") else None


def _bounds(node, name):
    if name == "clamp_min":
        return argument(node, 1, "min"), None
    if name == "clamp_max":
        return None, argument(node, 1, "max")
    return argument(node, 1, "min"), argument(node, 2, "max")


def _resize_kind(node):
    target = getattr(node.target, "original", node.target)
    if type(target) in (torch.nn.Upsample, torch.nn.UpsamplingNearest2d):
        return "nearest2d" if target.mode == "nearest" else None
    if target is torch.nn.functional.interpolate:
        return "nearest2d" if argument(node, 3, "mode", "nearest") == "nearest" else None
    if target is torch._C._nn.upsample_nearest2d:
        return "nearest2d"
    if isinstance(target, torch._ops.OpOverload):
        return ("nearest2d" if target._schema.name == "aten::upsample_nearest2d"
                and target._schema.overload_name in ("", "default", "vec") else None)
    return None


def kind(node):
    if not isinstance(node, ExternKernelNode) or node.autograd_owner != "pytorch":
        return None
    x = _input(node)
    if (not isinstance(x, TensorValue) or x.meta is None or x.meta.representation != "dense"
            or not x.meta.dtype.is_floating_point or node.kwargs.get("out") is not None):
        return None
    name = _clamp_kind(node)
    if name is not None:
        bounds = _bounds(node, name)
        return "clamp" if any(value is not None for value in bounds) and all(
            value is None or isinstance(value, Real) for value in bounds) else None
    name = _resize_kind(node)
    return name if name is not None and len(x.meta.shape) == 4 else None


def _memory_format(metadata):
    shape, stride = metadata.shape, metadata.stride
    if stride is not None and len(shape) == 4 and all(
            size <= 1 or actual == expected for size, actual, expected in zip(
                shape, stride, (shape[1] * shape[2] * shape[3], 1, shape[3] * shape[1], shape[1]))):
        return torch.channels_last
    if stride is not None and len(shape) == 5 and all(
            size <= 1 or actual == expected for size, actual, expected in zip(shape, stride,
                (shape[1] * shape[2] * shape[3] * shape[4], 1, shape[3] * shape[4] * shape[1],
                 shape[4] * shape[1], shape[1]))):
        return torch.channels_last_3d
    return torch.contiguous_format


def _scales(node):
    target = getattr(node.target, "original", node.target)
    recompute = None
    if type(target) in (torch.nn.Upsample, torch.nn.UpsamplingNearest2d):
        factors = target.scale_factor
        recompute = getattr(target, "recompute_scale_factor", None)
    elif target is torch.nn.functional.interpolate:
        factors = argument(node, 2, "scale_factor")
        recompute = argument(node, 5, "recompute_scale_factor")
    elif target is torch._C._nn.upsample_nearest2d:
        factors = argument(node, 2, "scale_factors")
    elif target._schema.overload_name == "vec":
        factors = argument(node, 2, "scale_factors")
    else:
        return (argument(node, 2, "scales_h"), argument(node, 3, "scales_w")), None
    if recompute or factors is None:
        return (None, None), recompute
    factors = (factors, factors) if isinstance(factors, Real) else tuple(factors)
    if len(factors) != 2 or any(not isinstance(value, Real) or value <= 0 for value in factors):
        raise ValueError("Nearest 2D native AutoDiff requires two positive static scale factors")
    return tuple(float(value) for value in factors), recompute


def _gradient_meta(metadata, memory_format):
    shape = metadata.shape
    if memory_format is torch.channels_last:
        stride = (shape[1] * shape[2] * shape[3], 1, shape[3] * shape[1], shape[1])
    elif memory_format is torch.channels_last_3d:
        stride = (shape[1] * shape[2] * shape[3] * shape[4], 1,
                  shape[3] * shape[4] * shape[1], shape[4] * shape[1], shape[1])
    else:
        stride = contiguous_stride(shape)
    return replace(metadata, stride=stride)


def lower(node):
    x = _input(node)
    if _clamp_kind(node) is not None:
        name = _clamp_kind(node)
        minimum, maximum = _bounds(node, name)
        include_minimum, include_maximum = clamp_endpoint_policy(name)
        spec = ScalarClampState(x.meta.shape, x.meta.dtype, minimum, maximum,
                                include_minimum, include_maximum, _memory_format(x.meta))
        words = (prod(x.meta.shape) + 31) // 32
        mask = TensorValue(f"{node.value.name}__clamp_mask", TensorMeta(
            (words,), torch.int32, x.meta.device, (1,), False, representation="packed_mask"))
        return CallNode("autodiff_scalar_clamp", spec.forward, (x,), {}, (node.value, mask), spec)
    scales, recompute = _scales(node)
    precision = nearest2d_precision(x.meta.dtype, x.meta.device, node.bf16_policy)
    spec = Nearest2dGeometry(x.meta.shape, tuple(node.value.meta.shape[-2:]), scales,
                             x.meta.dtype, precision, _memory_format(x.meta), recompute, x.meta.stride)
    if precision.disable_cuda_autocast:
        node.value.meta = replace(node.value.meta, dtype=x.meta.dtype)
    return CallNode("autodiff_nearest2d", spec.forward, (x,), {}, (node.value,), spec)


def clamp_vjp(ctx, node, gradients):
    x = node.args[0]
    if not ctx.needs_grad(x):
        return ()
    mask = node.values[1]
    saved = ctx.saved(mask)
    spec = node.specification
    ctx.record_residual(x, (mask,), representation="lossless_clamp_mask",
        component_roles=("mask",),
        use=SavedUse(consumer=node.op, role="input_gradient_branch"),
        recipe={"minimum": spec.minimum, "maximum": spec.maximum,
                "include_minimum": spec.include_minimum, "include_maximum": spec.include_maximum},
        retention_reason="Original scalar-clamp branch decisions are lossless")
    metadata = _gradient_meta(x.meta, spec.memory_format)
    return (GradWrapper(x, emit(ctx, "clamp.mask_vjp", spec.backward, (gradients[0], saved), metadata)),)


def resize_vjp(ctx, node, gradient):
    x = _input(node)
    if not ctx.needs_grad(x):
        return ()
    spec = node.specification
    ctx.record_residual(x, (), representation="metadata_only",
        use=SavedUse(consumer=node.op, role="input_gradient"),
        recipe={"input_shape": spec.input_shape, "output_size": spec.output_size, "scales": spec.scales,
                "recompute_scale_factor": spec.recompute_scale_factor, "dtype": str(spec.dtype),
                "memory_format": str(spec.memory_format), "input_stride": spec.input_stride,
                "bf16_policy": spec.precision.bf16_policy.as_dict(),
                "disable_cuda_autocast": spec.precision.disable_cuda_autocast},
        retention_reason="Nearest resize adjoint requires geometry only")
    metadata = _gradient_meta(x.meta, spec.memory_format)
    return (GradWrapper(x, emit(ctx, "resize.nearest2d_metadata_vjp", spec.backward, (gradient,), metadata)),)


def register_rules(registry):
    if not registry.has_backward_rule("autodiff_scalar_clamp"):
        registry.register_backward_rule("autodiff_scalar_clamp", clamp_vjp, multi_output=True)
    if not registry.has_backward_rule("autodiff_nearest2d"):
        registry.register_backward_rule("autodiff_nearest2d", resize_vjp)
