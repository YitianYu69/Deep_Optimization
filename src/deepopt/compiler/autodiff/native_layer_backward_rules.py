"""Registered rules that emit PyTorch's native ATen layer backward ops."""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Any, Callable

import torch

from ..ir import ExternKernelNode, TensorValue, find_tensor_values
from ..trace.tracer import SymbolicTensor
from .backward_rules import GradWrapper


def _aten(name: str, overload: str = "default"):
    packet = getattr(torch.ops.aten, name, None)
    if packet is None:
        raise NotImplementedError(f"This PyTorch build has no aten.{name} operator")
    target = getattr(packet, overload, None)
    if target is None:
        raise NotImplementedError(
            f"This PyTorch build has no aten.{name}.{overload} overload"
        )
    return target


def _module(node: ExternKernelNode) -> torch.nn.Module | None:
    return node.target if isinstance(node.target, torch.nn.Module) else None


def _target_name(node: ExternKernelNode) -> str:
    target = node.target
    name = getattr(target, "__name__", "")
    return f"{node.op} {name} {target}".lower()


def _arg(node: ExternKernelNode, index: int, name: str, default=None):
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name, default)


def _input(node: ExternKernelNode) -> TensorValue:
    values = tuple(find_tensor_values((node.args, node.kwargs)))
    if not values:
        raise NotImplementedError(
            f"Layer node {node.op!r} has no explicit tensor input"
        )
    return values[0]


def _module_parameter(node: ExternKernelNode, name: str):
    for input_name, value in zip(node.autograd_input_names, node.autograd_inputs):
        if input_name == name:
            return value
    return None


def _module_buffer(node: ExternKernelNode, name: str):
    for input_name, value in zip(node.autograd_buffer_names, node.autograd_buffers):
        if input_name == name:
            return value
    return None


def _parameter(node: ExternKernelNode, name: str, index: int):
    module = _module(node)
    if module is not None:
        return _module_parameter(node, name)
    return _arg(node, index, name)


def _buffer(node: ExternKernelNode, name: str, index: int):
    module = _module(node)
    if module is not None:
        return _module_buffer(node, name)
    return _arg(node, index, name)


def _emit_tuple_item(
    builder,
    tuple_value: SymbolicTensor,
    index: int,
    *,
    op_name: str,
    meta=None,
) -> SymbolicTensor:
    return builder.emit_native(
        f"{op_name}_getitem",
        lambda values, item: values[item],
        (tuple_value, index),
        {},
        meta,
    )


def _emit_backward_tuple(
    builder,
    aten_name: str,
    aten_target,
    args: tuple[Any, ...],
    targets: tuple[tuple[int, TensorValue | None], ...],
    ctx,
) -> tuple[GradWrapper, ...]:
    output_mask = [
        target is not None and ctx.needs_grad(target) for _, target in targets
    ]
    if not any(output_mask):
        return ()

    result = builder.emit_native(aten_name, aten_target, args, {}, None)
    contributions: list[GradWrapper] = []
    for index, (_, target) in enumerate(targets):
        if target is None or not ctx.needs_grad(target):
            continue
        grad = _emit_tuple_item(
            builder,
            result,
            index,
            op_name=aten_name,
            meta=target.meta,
        )
        contributions.append(GradWrapper(target, grad))
    return tuple(contributions)


def _is_module(target, name: str) -> bool:
    cls = getattr(torch.nn, name, None)
    return isinstance(cls, type) and isinstance(target, cls)


def _native_layer_kind(node: ExternKernelNode) -> str | None:
    """Match trusted native implementations by identity, never by their name."""
    if node.autograd_owner != "pytorch":
        return None
    target = node.target
    families = {
        "linear": ("Linear",),
        "convolution": tuple(
            f"Conv{prefix}{rank}d" for prefix in ("", "Transpose") for rank in (1, 2, 3)
        ),
        "batch_norm": tuple(f"BatchNorm{rank}d" for rank in (1, 2, 3)),
        "layer_norm": ("LayerNorm",),
        "group_norm": ("GroupNorm",),
        "instance_norm": tuple(f"InstanceNorm{rank}d" for rank in (1, 2, 3)),
        "rms_norm": ("RMSNorm",),
        "adaptive_pool": tuple(
            f"Adaptive{kind}Pool{rank}d"
            for kind in ("Max", "Avg")
            for rank in (1, 2, 3)
        ),
        "max_pool": tuple(f"MaxPool{rank}d" for rank in (1, 2, 3)),
        "avg_pool": tuple(f"AvgPool{rank}d" for rank in (1, 2, 3)),
        "reshape_view": ("Flatten",),
    }
    if isinstance(target, torch.nn.Module):
        return next(
            (
                kind
                for kind, names in families.items()
                if any(type(target) is getattr(torch.nn, name, None) for name in names)
            ),
            None,
        )
    functions = {
        "linear": ("linear",),
        "convolution": (
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
        ),
        "batch_norm": ("batch_norm",),
        "layer_norm": ("layer_norm",),
        "group_norm": ("group_norm",),
        "instance_norm": ("instance_norm",),
        "rms_norm": ("rms_norm",),
        "adaptive_pool": tuple(
            f"adaptive_{kind}_pool{rank}d"
            for kind in ("max", "avg")
            for rank in (1, 2, 3)
        ),
        "max_pool": tuple(f"max_pool{rank}d" for rank in (1, 2, 3)),
        "avg_pool": tuple(f"avg_pool{rank}d" for rank in (1, 2, 3)),
        "reshape_view": ("flatten", "reshape"),
    }
    for kind, names in functions.items():
        if any(
            target is getattr(namespace, name, None)
            for name in names
            for namespace in (torch, torch.nn.functional)
        ):
            return kind
    if isinstance(target, torch._ops.OpOverload):
        name = target._schema.name.removeprefix("aten::")
        if target._schema.overload_name not in ("", "default", "input"):
            return None
        if name in ("convolution", "_convolution"):
            return "convolution"
        if name in ("view", "_unsafe_view"):
            return "reshape_view"
        for kind, names in functions.items():
            if name in names:
                return kind
    return None


def is_registered_native_layer_node(node) -> bool:
    if not isinstance(node, ExternKernelNode):
        return False
    if node.value is None or node.value.meta is None or not callable(node.target):
        return False
    if node.value.meta.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        return False
    if any(
        value.meta is None or value.meta.representation != "dense"
        for value in node.inputs()
    ):
        return False
    # Native AMP can cast operands inside the original operation. Replaying
    # a derivative against pre-cast parameters would change its dtype/math.
    # Keep that original autograd graph instead of guessing the cast contract.
    if any(
        value.meta.dtype.is_floating_point and value.meta.dtype != node.value.meta.dtype
        for value in node.inputs()
    ):
        return False
    if not any(
        value.meta is not None and value.meta.requires_grad for value in node.inputs()
    ):
        return False
    kind = _native_layer_kind(node)
    if kind == "rms_norm":
        device = str(node.value.meta.device).split(":", 1)[0]
        # The fused backward ATen operator is registered for CUDA and XPU.
        return device in {"cuda", "xpu"}
    module = _module(node)
    if kind == "convolution":
        name = _target_name(node)
        if module is not None:
            # These module variants lower to extra padding/cropping around the
            # convolution and need a distinct backward rule.
            if module.padding_mode != "zeros" or isinstance(module.padding, str):
                return False
        elif "aten.convolution" not in name and "_convolution" not in name:
            padding = _arg(node, 4, "padding", (0, 0))
            if isinstance(padding, str):
                return False
    return kind is not None


def get_native_layer_backward_rule(node: ExternKernelNode) -> Callable | None:
    kind = _native_layer_kind(node)
    return {
        "linear": linear_backward_rule,
        "convolution": convolution_backward_rule,
        "batch_norm": batch_norm_backward_rule,
        "layer_norm": layer_norm_backward_rule,
        "group_norm": group_norm_backward_rule,
        "instance_norm": instance_norm_backward_rule,
        "rms_norm": rms_norm_backward_rule,
        "max_pool": max_pool_backward_rule,
        "avg_pool": avg_pool_backward_rule,
        "adaptive_pool": adaptive_pool_backward_rule,
        "reshape_view": reshape_view_backward_rule,
    }.get(kind)


def reshape_view_backward_rule(ctx, node: ExternKernelNode, grad_out: SymbolicTensor):
    """A pure shape view maps its cotangent to the source shape."""
    x = _input(node)
    if not ctx.needs_grad(x):
        return ()
    if x.meta is None:
        raise NotImplementedError("View backward requires a static source shape")
    grad_x = grad_out.builder.emit_native(
        "aten.reshape",
        _aten("reshape"),
        (grad_out, tuple(x.meta.shape)),
        {},
        x.meta,
    )
    return (GradWrapper(x, grad_x),)


def linear_backward_rule(ctx, node: ExternKernelNode, grad_out: SymbolicTensor):
    x = _input(node)
    weight = _parameter(node, "weight", 1)
    bias = _parameter(node, "bias", 2)
    if weight is None:
        raise NotImplementedError("Linear AOT backward requires an explicit weight")
    if x.meta is None or weight.meta is None or len(weight.meta.shape) != 2:
        raise NotImplementedError(
            "Linear backward requires static input and 2-D weight metadata"
        )

    device = str(x.meta.device).split(":", 1)[0]
    if device == "mps":
        saved_x = ctx.saved(x)
        saved_weight = ctx.saved(weight)
        mask = [
            ctx.needs_grad(x),
            ctx.needs_grad(weight),
            bias is not None and ctx.needs_grad(bias),
        ]
        return _emit_backward_tuple(
            grad_out.builder,
            "aten.linear_backward",
            _aten("linear_backward"),
            (saved_x, grad_out, saved_weight, mask),
            ((0, x), (1, weight), (2, bias)),
            ctx,
        )

    # aten.linear_backward has backend-specific registrations rather than a
    # general dense CPU/CUDA kernel. Build the dense derivative from ATen GEMM
    # and reduction operators, as PyTorch's linear derivative does. This stays
    # entirely in ATen and does not invoke autograd.grad.
    input_features = int(weight.meta.shape[1])
    output_features = int(weight.meta.shape[0])
    x_shape = tuple(x.meta.shape)
    if not x_shape or x_shape[-1] != input_features:
        raise NotImplementedError(
            "Linear backward needs a known final input feature dimension"
        )

    grad_x = None
    grad_weight = None
    grad_bias = None
    need_x = ctx.needs_grad(x)
    need_weight = ctx.needs_grad(weight)
    need_bias = bias is not None and ctx.needs_grad(bias)
    grad_out_2d = (
        grad_out.builder.emit_native(
            "aten.reshape",
            _aten("reshape"),
            (grad_out, (-1, output_features)),
            {},
            None,
        )
        if need_x or need_weight or need_bias
        else None
    )
    if need_x:
        saved_weight = ctx.saved(weight)
        grad_x_2d = grad_out.builder.emit_native(
            "aten.mm",
            _aten("mm"),
            (grad_out_2d, saved_weight),
            {},
            None,
        )
        grad_x = grad_out.builder.emit_native(
            "aten.reshape",
            _aten("reshape"),
            (grad_x_2d, x_shape),
            {},
            x.meta,
        )
    if need_weight:
        saved_x = ctx.saved(x)
        x_2d = grad_out.builder.emit_native(
            "aten.reshape",
            _aten("reshape"),
            (saved_x, (-1, input_features)),
            {},
            None,
        )
        grad_out_t = grad_out.builder.emit_native(
            "aten.transpose",
            _aten("transpose", "int"),
            (grad_out_2d, 0, 1),
            {},
            None,
        )
        grad_weight = grad_out.builder.emit_native(
            "aten.mm",
            _aten("mm"),
            (grad_out_t, x_2d),
            {},
            weight.meta,
        )
    if need_bias:
        grad_bias = grad_out.builder.emit_native(
            "aten.sum",
            _aten("sum", "dim_IntList"),
            (grad_out_2d, [0], False),
            {},
            bias.meta,
        )

    return tuple(
        GradWrapper(target, gradient)
        for target, gradient in ((x, grad_x), (weight, grad_weight), (bias, grad_bias))
        if target is not None
    )


def _convolution_options(node: ExternKernelNode):
    module = _module(node)
    x = _input(node)
    rank = len(x.meta.shape) - 2 if x.meta is not None else 2
    if module is not None:
        if module.padding_mode != "zeros":
            raise NotImplementedError(
                "Direct aten.convolution_backward currently supports Conv modules "
                "with padding_mode='zeros'"
            )
        if isinstance(module.padding, str):
            raise NotImplementedError(
                "Direct aten.convolution_backward currently requires numeric Conv padding"
            )
        return (
            module.stride,
            module.padding,
            module.dilation,
            bool(module.transposed),
            module.output_padding,
            int(module.groups),
        )

    name = _target_name(node)
    if "aten.convolution" in name or "_convolution" in name:
        return tuple(
            _arg(node, index, key, default)
            for index, (key, default) in enumerate(
                (
                    ("stride", (1,) * rank),
                    ("padding", (0,) * rank),
                    ("dilation", (1,) * rank),
                    ("transposed", False),
                    ("output_padding", (0,) * rank),
                    ("groups", 1),
                ),
                start=3,
            )
        )
    if "conv_transpose" in name:
        padding = _arg(node, 4, "padding", (0,) * rank)
        if isinstance(padding, str):
            raise NotImplementedError("Transposed convolution padding must be numeric")
        return (
            _arg(node, 3, "stride", (1,) * rank),
            padding,
            _arg(node, 7, "dilation", (1,) * rank),
            True,
            _arg(node, 5, "output_padding", (0,) * rank),
            _arg(node, 6, "groups", 1),
        )
    padding = _arg(node, 4, "padding", (0,) * rank)
    if isinstance(padding, str):
        raise NotImplementedError("Convolution padding must be numeric")
    return (
        _arg(node, 3, "stride", (1,) * rank),
        padding,
        _arg(node, 5, "dilation", (1,) * rank),
        False,
        (0,) * rank,
        _arg(node, 6, "groups", 1),
    )


def convolution_backward_rule(ctx, node: ExternKernelNode, grad_out: SymbolicTensor):
    x = _input(node)
    weight = _parameter(node, "weight", 1)
    bias = _parameter(node, "bias", 2)
    if weight is None:
        raise NotImplementedError(
            "ATen convolution_backward requires an explicit weight"
        )
    stride, padding, dilation, transposed, output_padding, groups = (
        _convolution_options(node)
    )
    saved_x = ctx.saved(x)
    saved_weight = ctx.saved(weight)
    mask = [
        ctx.needs_grad(x),
        ctx.needs_grad(weight),
        bias is not None and ctx.needs_grad(bias),
    ]
    bias_sizes = (
        tuple(bias.meta.shape) if bias is not None and bias.meta is not None else None
    )
    return _emit_backward_tuple(
        grad_out.builder,
        "aten.convolution_backward",
        _aten("convolution_backward"),
        (
            grad_out,
            saved_x,
            saved_weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            mask,
        ),
        ((0, x), (1, weight), (2, bias)),
        ctx,
    )


def batch_norm_backward_rule(ctx, node: ExternKernelNode, grad_out: SymbolicTensor):
    x = _input(node)
    module = _module(node)
    if module is not None:
        weight = _module_parameter(node, "weight")
        bias = _module_parameter(node, "bias")
        running_mean = _module_buffer(node, "running_mean")
        running_var = _module_buffer(node, "running_var")
        training = bool(module.training or not module.track_running_stats)
        eps = float(module.eps)
        momentum = float(module.momentum if module.momentum is not None else 0.0)
    else:
        # torch.nn.functional.batch_norm(input, running_mean, running_var,
        # weight, bias, training, momentum, eps)
        running_mean = _arg(node, 1, "running_mean")
        running_var = _arg(node, 2, "running_var")
        weight = _arg(node, 3, "weight")
        bias = _arg(node, 4, "bias")
        training = bool(_arg(node, 5, "training", True))
        momentum = float(_arg(node, 6, "momentum", 0.1))
        eps = float(_arg(node, 7, "eps", 1e-5))
    saved_x = ctx.saved(x)
    saved_weight = ctx.saved(weight) if weight is not None else None
    saved_bias = ctx.saved(bias) if bias is not None else None
    saved_running_mean = ctx.saved(running_mean) if running_mean is not None else None
    saved_running_var = ctx.saved(running_var) if running_var is not None else None
    stats_mean_arg = None if training else saved_running_mean
    stats_var_arg = None if training else saved_running_var
    stats = grad_out.builder.emit_native(
        "aten.native_batch_norm",
        _aten("native_batch_norm"),
        (
            saved_x,
            saved_weight,
            saved_bias,
            stats_mean_arg,
            stats_var_arg,
            training,
            0.0 if training else momentum,
            eps,
        ),
        {},
        None,
    )
    save_mean = _emit_tuple_item(
        grad_out.builder, stats, 1, op_name="aten.native_batch_norm_stats"
    )
    save_invstd = _emit_tuple_item(
        grad_out.builder, stats, 2, op_name="aten.native_batch_norm_stats"
    )
    mask = [
        ctx.needs_grad(x),
        weight is not None and ctx.needs_grad(weight),
        bias is not None and ctx.needs_grad(bias),
    ]
    return _emit_backward_tuple(
        grad_out.builder,
        "aten.native_batch_norm_backward",
        _aten("native_batch_norm_backward"),
        (
            grad_out,
            saved_x,
            saved_weight,
            stats_mean_arg,
            stats_var_arg,
            save_mean,
            save_invstd,
            training,
            eps,
            mask,
        ),
        ((0, x), (1, weight), (2, bias)),
        ctx,
    )


def layer_norm_backward_rule(ctx, node: ExternKernelNode, grad_out: SymbolicTensor):
    x = _input(node)
    normalized_shape = (
        tuple(node.target.normalized_shape)
        if _module(node) is not None
        else _arg(node, 1, "normalized_shape")
    )
    weight = _parameter(node, "weight", 2)
    bias = _parameter(node, "bias", 3)
    eps = (
        float(node.target.eps)
        if _module(node) is not None
        else float(_arg(node, 4, "eps", 1e-5))
    )
    saved_x = ctx.saved(x)
    saved_weight = ctx.saved(weight) if weight is not None else None
    saved_bias = ctx.saved(bias) if bias is not None else None
    stats = grad_out.builder.emit_native(
        "aten.native_layer_norm",
        _aten("native_layer_norm"),
        (saved_x, normalized_shape, saved_weight, saved_bias, eps),
        {},
        None,
    )
    mean = _emit_tuple_item(
        grad_out.builder, stats, 1, op_name="aten.native_layer_norm_stats"
    )
    rstd = _emit_tuple_item(
        grad_out.builder, stats, 2, op_name="aten.native_layer_norm_stats"
    )
    mask = [
        ctx.needs_grad(x),
        weight is not None and ctx.needs_grad(weight),
        bias is not None and ctx.needs_grad(bias),
    ]
    return _emit_backward_tuple(
        grad_out.builder,
        "aten.native_layer_norm_backward",
        _aten("native_layer_norm_backward"),
        (
            grad_out,
            saved_x,
            normalized_shape,
            mean,
            rstd,
            saved_weight,
            saved_bias,
            mask,
        ),
        ((0, x), (1, weight), (2, bias)),
        ctx,
    )


def group_norm_backward_rule(ctx, node: ExternKernelNode, grad_out: SymbolicTensor):
    x = _input(node)
    module = _module(node)
    groups = (
        int(module.num_groups)
        if module is not None
        else int(_arg(node, 1, "num_groups"))
    )
    weight = _parameter(node, "weight", 2)
    bias = _parameter(node, "bias", 3)
    eps = float(module.eps) if module is not None else float(_arg(node, 4, "eps", 1e-5))
    if x.meta is None or len(x.meta.shape) < 2:
        raise NotImplementedError("GroupNorm backward needs static N and C dimensions")
    n, channels, *spatial = x.meta.shape
    spatial_size = reduce(mul, spatial, 1)
    saved_x = ctx.saved(x)
    saved_weight = ctx.saved(weight) if weight is not None else None
    saved_bias = ctx.saved(bias) if bias is not None else None
    stats = grad_out.builder.emit_native(
        "aten.native_group_norm",
        _aten("native_group_norm"),
        (saved_x, saved_weight, saved_bias, n, channels, spatial_size, groups, eps),
        {},
        None,
    )
    mean = _emit_tuple_item(
        grad_out.builder, stats, 1, op_name="aten.native_group_norm_stats"
    )
    rstd = _emit_tuple_item(
        grad_out.builder, stats, 2, op_name="aten.native_group_norm_stats"
    )
    mask = [
        ctx.needs_grad(x),
        weight is not None and ctx.needs_grad(weight),
        bias is not None and ctx.needs_grad(bias),
    ]
    return _emit_backward_tuple(
        grad_out.builder,
        "aten.native_group_norm_backward",
        _aten("native_group_norm_backward"),
        (
            grad_out,
            saved_x,
            mean,
            rstd,
            saved_weight,
            n,
            channels,
            spatial_size,
            groups,
            mask,
        ),
        ((0, x), (1, weight), (2, bias)),
        ctx,
    )


def rms_norm_backward_rule(ctx, node: ExternKernelNode, grad_out: SymbolicTensor):
    x = _input(node)
    module = _module(node)
    normalized_shape = (
        tuple(module.normalized_shape)
        if module is not None
        else _arg(node, 1, "normalized_shape")
    )
    weight = _parameter(node, "weight", 2)
    eps = module.eps if module is not None else _arg(node, 3, "eps")
    saved_x = ctx.saved(x)
    saved_weight = ctx.saved(weight) if weight is not None else None
    stats = grad_out.builder.emit_native(
        "aten._fused_rms_norm",
        _aten("_fused_rms_norm"),
        (saved_x, normalized_shape, saved_weight, eps),
        {},
        None,
    )
    rstd = _emit_tuple_item(
        grad_out.builder, stats, 1, op_name="aten._fused_rms_norm_stats"
    )
    mask = [ctx.needs_grad(x), weight is not None and ctx.needs_grad(weight)]
    return _emit_backward_tuple(
        grad_out.builder,
        "aten._fused_rms_norm_backward",
        _aten("_fused_rms_norm_backward"),
        (grad_out, saved_x, normalized_shape, rstd, saved_weight, mask),
        ((0, x), (1, weight)),
        ctx,
    )


def instance_norm_backward_rule(ctx, node: ExternKernelNode, grad_out: SymbolicTensor):
    module = _module(node)
    x = _input(node)
    if x.meta is None or len(x.meta.shape) < 3:
        raise NotImplementedError(
            "InstanceNorm backward needs static N, C, and spatial dimensions"
        )
    n, channels, *spatial = x.meta.shape
    if module is not None:
        weight = _module_parameter(node, "weight")
        bias = _module_parameter(node, "bias")
        running_mean = _module_buffer(node, "running_mean")
        running_var = _module_buffer(node, "running_var")
        use_input_stats = bool(module.training or not module.track_running_stats)
        eps = float(module.eps)
    else:
        # F.instance_norm(input, running_mean, running_var, weight, bias,
        # use_input_stats, momentum, eps, cudnn_enabled)
        running_mean = _arg(node, 1, "running_mean")
        running_var = _arg(node, 2, "running_var")
        weight = _arg(node, 3, "weight")
        bias = _arg(node, 4, "bias")
        use_input_stats = bool(_arg(node, 5, "use_input_stats", True))
        eps = float(_arg(node, 7, "eps", 1e-5))
    saved_x = ctx.saved(x)
    saved_weight = ctx.saved(weight) if weight is not None else None
    saved_bias = ctx.saved(bias) if bias is not None else None
    saved_mean = ctx.saved(running_mean) if running_mean is not None else None
    saved_var = ctx.saved(running_var) if running_var is not None else None
    flattened_shape = (1, n * channels, *spatial)
    flat_x = grad_out.builder.emit_native(
        "aten.reshape",
        _aten("reshape"),
        (saved_x, flattened_shape),
        {},
        None,
    )
    flat_weight = (
        grad_out.builder.emit_native(
            "aten.repeat", _aten("repeat"), (saved_weight, (n,)), {}, None
        )
        if saved_weight is not None
        else None
    )
    flat_bias = (
        grad_out.builder.emit_native(
            "aten.repeat", _aten("repeat"), (saved_bias, (n,)), {}, None
        )
        if saved_bias is not None
        else None
    )
    flat_mean = (
        grad_out.builder.emit_native(
            "aten.repeat", _aten("repeat"), (saved_mean, (n,)), {}, None
        )
        if saved_mean is not None and not use_input_stats
        else None
    )
    flat_var = (
        grad_out.builder.emit_native(
            "aten.repeat", _aten("repeat"), (saved_var, (n,)), {}, None
        )
        if saved_var is not None and not use_input_stats
        else None
    )
    stats = grad_out.builder.emit_native(
        "aten.native_batch_norm",
        _aten("native_batch_norm"),
        (
            flat_x,
            flat_weight,
            flat_bias,
            flat_mean,
            flat_var,
            use_input_stats,
            0.0,
            eps,
        ),
        {},
        None,
    )
    save_mean = _emit_tuple_item(
        grad_out.builder, stats, 1, op_name="aten.native_instance_norm_stats"
    )
    save_invstd = _emit_tuple_item(
        grad_out.builder, stats, 2, op_name="aten.native_instance_norm_stats"
    )
    flat_grad_out = grad_out.builder.emit_native(
        "aten.reshape",
        _aten("reshape"),
        (grad_out, flattened_shape),
        {},
        None,
    )
    mask = [
        ctx.needs_grad(x),
        weight is not None and ctx.needs_grad(weight),
        bias is not None and ctx.needs_grad(bias),
    ]
    result = grad_out.builder.emit_native(
        "aten.native_batch_norm_backward",
        _aten("native_batch_norm_backward"),
        (
            flat_grad_out,
            flat_x,
            flat_weight,
            flat_mean,
            flat_var,
            save_mean,
            save_invstd,
            use_input_stats,
            eps,
            mask,
        ),
        {},
        None,
    )
    contributions: list[GradWrapper] = []
    if mask[0]:
        grad_x_flat = _emit_tuple_item(
            grad_out.builder, result, 0, op_name="aten.native_batch_norm_backward"
        )
        grad_x = grad_out.builder.emit_native(
            "aten.reshape",
            _aten("reshape"),
            (grad_x_flat, tuple(x.meta.shape)),
            {},
            x.meta,
        )
        contributions.append(GradWrapper(x, grad_x))
    for index, target, enabled in ((1, weight, mask[1]), (2, bias, mask[2])):
        if not enabled or target is None:
            continue
        grad_repeated = _emit_tuple_item(
            grad_out.builder, result, index, op_name="aten.native_batch_norm_backward"
        )
        grad_matrix = grad_out.builder.emit_native(
            "aten.reshape", _aten("reshape"), (grad_repeated, (n, channels)), {}, None
        )
        grad_param = grad_out.builder.emit_native(
            "aten.sum",
            _aten("sum", "dim_IntList"),
            (grad_matrix, [0], False),
            {},
            target.meta,
        )
        contributions.append(GradWrapper(target, grad_param))
    return tuple(contributions)


def _pool_dimensions(node: ExternKernelNode) -> int:
    module = _module(node)
    if module is not None:
        if hasattr(module, "kernel_size"):
            return len(tuple(module.kernel_size))
        class_name = type(module).__name__
        for dims in (1, 2, 3):
            if class_name.endswith(f"Pool{dims}d"):
                return dims
        output_size = module.output_size
        if isinstance(output_size, int):
            return 1
        return len(tuple(output_size))
    name = _target_name(node)
    for dims in (1, 2, 3):
        if f"{dims}d" in name:
            return dims
    x = _input(node)
    return max(1, len(x.meta.shape) - 2) if x.meta is not None else 2


def _pool_arg(node: ExternKernelNode, name: str, index: int, default):
    module = _module(node)
    if module is not None:
        return getattr(module, name, default)
    return _arg(node, index, name, default)


def _pool_tuple(value, dims: int, *, default: int) -> tuple[int, ...]:
    if value is None:
        value = default
    if isinstance(value, int):
        return (value,) * dims
    return tuple(value)


def max_pool_backward_rule(ctx, node: ExternKernelNode, grad_out: SymbolicTensor):
    x = _input(node)
    dims = _pool_dimensions(node)
    aten_dims = 2 if dims == 1 else dims
    suffix = f"{aten_dims}d"
    kernel_value = _pool_arg(node, "kernel_size", 1, None)
    if kernel_value is None:
        raise NotImplementedError("MaxPool backward requires kernel_size")
    kernel = _pool_tuple(kernel_value, dims, default=1)
    stride_value = _pool_arg(node, "stride", 2, None)
    stride = (
        _pool_tuple(stride_value, dims, default=kernel[0])
        if stride_value is not None
        else kernel
    )
    padding = _pool_tuple(_pool_arg(node, "padding", 3, 0), dims, default=0)
    dilation = _pool_tuple(_pool_arg(node, "dilation", 4, 1), dims, default=1)
    ceil_mode = bool(_pool_arg(node, "ceil_mode", 5, False))
    saved_x = ctx.saved(x)
    saved_grad = grad_out
    if dims == 1:
        kernel = (1, *tuple(kernel))
        stride = (1, *tuple(stride))
        padding = (0, *tuple(padding))
        dilation = (1, *tuple(dilation))
        saved_x = grad_out.builder.emit_native(
            "aten.unsqueeze", _aten("unsqueeze"), (saved_x, -2), {}, None
        )
        saved_grad = grad_out.builder.emit_native(
            "aten.unsqueeze", _aten("unsqueeze"), (grad_out, -2), {}, None
        )
    forward = grad_out.builder.emit_native(
        f"aten.max_pool{suffix}_with_indices",
        _aten(f"max_pool{suffix}_with_indices"),
        (saved_x, kernel, stride, padding, dilation, ceil_mode),
        {},
        None,
    )
    indices = _emit_tuple_item(
        grad_out.builder, forward, 1, op_name=f"aten.max_pool{suffix}_with_indices"
    )
    grad_x = grad_out.builder.emit_native(
        f"aten.max_pool{suffix}_with_indices_backward",
        _aten(f"max_pool{suffix}_with_indices_backward"),
        (saved_grad, saved_x, kernel, stride, padding, dilation, ceil_mode, indices),
        {},
        None if dims == 1 else x.meta,
    )
    if dims == 1:
        grad_x = grad_out.builder.emit_native(
            "aten.squeeze", _aten("squeeze", "dim"), (grad_x, -2), {}, x.meta
        )
    return (GradWrapper(x, grad_x),) if ctx.needs_grad(x) else ()


def avg_pool_backward_rule(ctx, node: ExternKernelNode, grad_out: SymbolicTensor):
    x = _input(node)
    dims = _pool_dimensions(node)
    aten_dims = 2 if dims == 1 else dims
    suffix = f"{aten_dims}d"
    kernel_value = _pool_arg(node, "kernel_size", 1, None)
    if kernel_value is None:
        raise NotImplementedError("AvgPool backward requires kernel_size")
    kernel = _pool_tuple(kernel_value, dims, default=1)
    stride_value = _pool_arg(node, "stride", 2, None)
    stride = (
        _pool_tuple(stride_value, dims, default=kernel[0])
        if stride_value is not None
        else kernel
    )
    padding = _pool_tuple(_pool_arg(node, "padding", 3, 0), dims, default=0)
    ceil_mode = bool(_pool_arg(node, "ceil_mode", 4, False))
    count_include_pad = bool(_pool_arg(node, "count_include_pad", 5, True))
    saved_x = ctx.saved(x)
    saved_grad = grad_out
    divisor_override = _pool_arg(node, "divisor_override", 6, None)
    if dims == 1:
        kernel = (1, *tuple(kernel))
        stride = (1, *tuple(stride))
        padding = (0, *tuple(padding))
        saved_x = grad_out.builder.emit_native(
            "aten.unsqueeze", _aten("unsqueeze"), (saved_x, -2), {}, None
        )
        saved_grad = grad_out.builder.emit_native(
            "aten.unsqueeze", _aten("unsqueeze"), (grad_out, -2), {}, None
        )
        divisor_override = None
    args = (
        saved_grad,
        saved_x,
        kernel,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    grad_x = grad_out.builder.emit_native(
        f"aten.avg_pool{suffix}_backward",
        _aten(f"avg_pool{suffix}_backward"),
        args,
        {},
        None if dims == 1 else x.meta,
    )
    if dims == 1:
        grad_x = grad_out.builder.emit_native(
            "aten.squeeze", _aten("squeeze", "dim"), (grad_x, -2), {}, x.meta
        )
    return (GradWrapper(x, grad_x),) if ctx.needs_grad(x) else ()


def adaptive_pool_backward_rule(ctx, node: ExternKernelNode, grad_out: SymbolicTensor):
    x = _input(node)
    dims = _pool_dimensions(node)
    aten_dims = 2 if dims == 1 else dims
    suffix = f"{aten_dims}d"
    target_name = _target_name(node)
    is_max = (
        isinstance(_module(node), torch.nn.modules.pooling._AdaptiveMaxPoolNd)
        or "adaptive_max_pool" in target_name
    )
    output_size = _pool_arg(node, "output_size", 1, None)
    if output_size is None:
        raise NotImplementedError("AdaptivePool backward requires output_size")
    if isinstance(output_size, int):
        output_size = (output_size,) * dims
    else:
        output_size = tuple(output_size)
    if not is_max and output_size == (1,) * dims:
        # Public adaptive average pooling specializes global pooling to mean.
        # Its derivative divides by the complete spatial extent once. The
        # generic adaptive-pool kernel divides by individual extents, changing
        # BF16 rounding for odd sizes (for example 3 by 3).
        from math import prod
        from .activation_saving import SavedUse

        shape = tuple(x.meta.shape)
        elements = prod(shape[-dims:])
        ctx.record_residual(x, (), representation="metadata_only",
            use=SavedUse(role="input_gradient", consumer="global_average_pool"),
            recipe={"shape": shape, "input_elements": elements},
            retention_reason="Public global average adjoint requires geometry only")
        gradient = grad_out.builder.emit_native("pooling.global_average_vjp",
            lambda value: (value / elements).expand(shape), (grad_out,), {}, x.meta)
        return (GradWrapper(x, gradient),) if ctx.needs_grad(x) else ()
    saved_x = ctx.saved(x)
    saved_grad = grad_out
    if dims == 1:
        output_size = (1, *tuple(output_size))
        saved_x = grad_out.builder.emit_native(
            "aten.unsqueeze", _aten("unsqueeze"), (saved_x, -2), {}, None
        )
        saved_grad = grad_out.builder.emit_native(
            "aten.unsqueeze", _aten("unsqueeze"), (grad_out, -2), {}, None
        )
    if is_max:
        forward = grad_out.builder.emit_native(
            f"aten.adaptive_max_pool{suffix}",
            _aten(f"adaptive_max_pool{suffix}"),
            (saved_x, output_size),
            {},
            None,
        )
        indices = _emit_tuple_item(
            grad_out.builder, forward, 1, op_name=f"aten.adaptive_max_pool{suffix}"
        )
        grad_x = grad_out.builder.emit_native(
            f"aten.adaptive_max_pool{suffix}_backward",
            _aten(f"adaptive_max_pool{suffix}_backward"),
            (saved_grad, saved_x, indices),
            {},
            None if dims == 1 else x.meta,
        )
    else:
        grad_x = grad_out.builder.emit_native(
            f"aten._adaptive_avg_pool{suffix}_backward",
            _aten(f"_adaptive_avg_pool{suffix}_backward"),
            (saved_grad, saved_x),
            {},
            None if dims == 1 else x.meta,
        )
    if dims == 1:
        grad_x = grad_out.builder.emit_native(
            "aten.squeeze", _aten("squeeze", "dim"), (grad_x, -2), {}, x.meta
        )
    return (GradWrapper(x, grad_x),) if ctx.needs_grad(x) else ()
