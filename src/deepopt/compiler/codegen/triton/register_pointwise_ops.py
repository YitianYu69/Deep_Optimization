from __future__ import annotations

import ast
from typing import Callable, Dict

POINTWISE_LOWERING_REGISTRY: Dict[str, Callable] = {}


def register_pointwise_lowering(name: str):
    def decorator(func: Callable):
        if name in POINTWISE_LOWERING_REGISTRY:
            raise RuntimeError(
                f"The pointwise op lowering: {name}had already been registered"
            )

        POINTWISE_LOWERING_REGISTRY[name] = func

        return func

    return decorator


def get_pointwise_lowering(name: str):
    try:
        return POINTWISE_LOWERING_REGISTRY[name]
    except KeyError:
        raise NotImplementedError(f"No pointwise lowering registered for {name!r}")


def _option(args, kwargs, name: str, positional_index: int, default):
    value = (
        args[positional_index]
        if len(args) > positional_index
        else kwargs.get(name, repr(default))
    )
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    return value


def _f32(value: str) -> str:
    return f"({value}).to(tl.float32)"


def _tanh_expr(value: str) -> str:
    return f"libdevice.tanh({value})"


# ==================
# Basic Computation
# ==================


@register_pointwise_lowering("add")
def lower_add(args, kwargs=None):
    x, y = args
    return f"({x} + {y})"


@register_pointwise_lowering("sub")
def lower_sub(args, kwargs=None):
    x, y = args
    return f"({x} - {y})"


@register_pointwise_lowering("mul")
def lower_mul(args, kwargs=None):
    x, y = args
    return f"({x} * {y})"


@register_pointwise_lowering("div")
def lower_div(args, kwargs=None):
    x, y = args
    return f"({x} / {y})"


@register_pointwise_lowering("neg")
def lower_neg(args, kwargs=None):
    x = args[0]
    return f"(-{x})"


# ============================================
# Non-Linear Functions / Activation Functions
# ============================================


@register_pointwise_lowering("relu")
def lower_relu(args, kwargs=None):
    x = args[0]
    return f"tl.where({x} < 0.0, 0.0, {x})"


@register_pointwise_lowering("sigmoid")
def lower_sigmoid(args, kwargs=None):
    x = args[0]
    return f"(1.0 / (1.0 + tl.exp(-({x}).to(tl.float32))))"


@register_pointwise_lowering("tanh")
def lower_tanh(args, kwargs=None):
    return _tanh_expr(_f32(args[0]))


@register_pointwise_lowering("gelu")
def lower_gelu(args, kwargs=None):
    x = _f32(args[0])
    approximate = _option(args, kwargs or {}, "approximate", 1, "none")
    if approximate == "tanh":
        inner = f"(0.7978845608028654 * ({x} + 0.044715 * {x} * {x} * {x}))"
        return f"(0.5 * {x} * (1.0 + {_tanh_expr(inner)}))"
    if approximate != "none":
        raise NotImplementedError(f"Unsupported GELU approximation {approximate!r}")
    return f"(0.5 * {x} * (1.0 + libdevice.erf({x} * 0.7071067811865476)))"


@register_pointwise_lowering("silu")
def lower_silu(args, kwargs=None):
    x = _f32(args[0])
    return f"({x} / (1.0 + tl.exp(-({x}))))"


@register_pointwise_lowering("leaky_relu")
def lower_leaky_relu(args, kwargs=None):
    x = _f32(args[0])
    slope = _option(args, kwargs or {}, "negative_slope", 1, 0.01)
    return f"tl.where({x} > 0.0, {x}, {x} * {slope!r})"


@register_pointwise_lowering("hardtanh")
def lower_hardtanh(args, kwargs=None):
    options = kwargs or {}
    x = _f32(args[0])
    min_val = _option(args, options, "min_val", 1, -1.0)
    max_val = _option(args, options, "max_val", 2, 1.0)
    return f"tl.where({x} < {min_val!r}, {min_val!r}, tl.where({x} > {max_val!r}, {max_val!r}, {x}))"


@register_pointwise_lowering("hardswish")
def lower_hardswish(args, kwargs=None):
    x = _f32(args[0])
    gate = f"tl.minimum(tl.maximum(({x} + 3.0), 0.0), 6.0)"
    return f"({x} * {gate} / 6.0)"


@register_pointwise_lowering("hardsigmoid")
def lower_hardsigmoid(args, kwargs=None):
    x = _f32(args[0])
    return f"(tl.minimum(tl.maximum(({x} + 3.0), 0.0), 6.0) / 6.0)"


@register_pointwise_lowering("softplus")
def lower_softplus(args, kwargs=None):
    options = kwargs or {}
    x = _f32(args[0])
    beta = _option(args, options, "beta", 1, 1.0)
    threshold = _option(args, options, "threshold", 2, 20.0)
    scaled = f"({x} * {beta!r})"
    stable = f"(tl.maximum({scaled}, 0.0) + tl.log(1.0 + tl.exp(-tl.abs({scaled}))))"
    return f"tl.where({scaled} > {threshold!r}, {x}, {stable} / {beta!r})"


@register_pointwise_lowering("mish")
def lower_mish(args, kwargs=None):
    x = _f32(args[0])
    softplus = (
        f"tl.where({x} > 20.0, {x}, tl.maximum({x}, 0.0) + "
        f"tl.log(1.0 + tl.exp(-tl.abs({x}))))"
    )
    return f"({x} * {_tanh_expr(softplus)})"
