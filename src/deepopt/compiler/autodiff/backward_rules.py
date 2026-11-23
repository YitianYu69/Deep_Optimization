from __future__ import annotations

from ..ir import IRNode, TensorValue
from ..trace.tracer import SymbolicTensor

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Dict, Any
from types import MappingProxyType


@dataclass(frozen=True)
class GradWrapper:
    target: Any
    grad: SymbolicTensor | TensorValue | None


class BackwardContext(Protocol):
    def needs_grad(self, value: Any) -> bool: ...
    def saved(self, value: Any, *, use=None) -> Any: ...


BackwardRule = Callable[
    [BackwardContext, IRNode, SymbolicTensor],
    Iterable[GradWrapper],
]

BWD_RULES_REGISTRY: Dict[str, BackwardRule] = {}


def register_backward_rule(name: str, *, registry=None):
    def decorator(func: Callable):
        if registry is not None:
            registry.register_backward_rule(name, func)
            return func
        if isinstance(BWD_RULES_REGISTRY, MappingProxyType):
            raise ValueError(
                "Extension backward rules require a per-compilation BackendRegistry"
            )
        if name in BWD_RULES_REGISTRY:
            raise KeyError(f"The backward rule: {name} had already been registered")
        BWD_RULES_REGISTRY[name] = func
        return func

    return decorator


def get_backward_rule(name: str):
    try:
        return BWD_RULES_REGISTRY[name]
    except KeyError:
        raise NotImplementedError(f"No backward rule register for {name}")


@register_backward_rule("add")
def register_backward_add(ctx, node, grad_out):
    x, y = node.args[:2]

    return (
        GradWrapper(x, grad_out if ctx.needs_grad(x) else None),
        GradWrapper(y, grad_out if ctx.needs_grad(y) else None),
    )


@register_backward_rule("sub")
def register_backward_sub(ctx, node, grad_out):
    x, y = node.args[:2]

    return (
        GradWrapper(x, grad_out if ctx.needs_grad(x) else None),
        GradWrapper(y, -grad_out if ctx.needs_grad(y) else None),
    )


@register_backward_rule("mul")
def register_backward_mul(ctx, node, grad_out):
    from .activation_saving import SavedUse

    x, y = node.args[:2]

    grad_x = grad_out * ctx.saved(y, use=SavedUse(role="mul.grad_input_coefficient",
        allow_approximation=True, consumer="mul", rounding_domain="operand_dtype")) if ctx.needs_grad(x) else None
    grad_y = grad_out * ctx.saved(x, use=SavedUse(role="mul.grad_other_coefficient",
        allow_approximation=True, consumer="mul", rounding_domain="operand_dtype")) if ctx.needs_grad(y) else None
    return (GradWrapper(x, grad_x), GradWrapper(y, grad_y))


@register_backward_rule("div")
def register_backward_div(ctx, node, grad_out):
    x, y = node.args[:2]

    grad_x = None
    grad_y = None
    if ctx.needs_grad(x):
        grad_x = grad_out / ctx.saved(y)
    if ctx.needs_grad(y):
        saved_x = ctx.saved(x)
        saved_y = ctx.saved(y)
        grad_y = -(grad_out * saved_x) / (saved_y * saved_y)
    return (GradWrapper(x, grad_x), GradWrapper(y, grad_y))


@register_backward_rule("neg")
def register_backward_neg(ctx, node, grad_out):
    (x,) = node.args
    return (GradWrapper(x, -grad_out if ctx.needs_grad(x) else None),)


@register_backward_rule("relu")
def register_backward_relu(ctx, node, grad_out):
    (x,) = node.args
    grad_x = None
    if ctx.needs_grad(x):
        grad_x = grad_out.builder.emit(
            "threshold_backward", grad_out, ctx.saved(x), 0.0
        )
    return (GradWrapper(x, grad_x),)


@register_backward_rule("sigmoid")
def register_backward_sigmoid(ctx, node, grad_out):
    (x,) = node.args
    grad_x = None
    if ctx.needs_grad(x):
        grad_x = grad_out.builder.emit(
            "sigmoid_backward", grad_out, ctx.saved(node.output())
        )
    return (GradWrapper(x, grad_x),)


def _parameter(node, name: str, positional_index: int, default):
    """Read a scalar option from FX args/kwargs or a module target."""
    args = getattr(node, "args", ())
    if len(args) > positional_index:
        return args[positional_index]
    kwargs = getattr(node, "kwargs", {})
    if name in kwargs:
        return kwargs[name]
    return getattr(getattr(node, "target", None), name, default)


@register_backward_rule("tanh")
def register_backward_tanh(ctx, node, grad_out):
    (x,) = node.args[:1]
    grad_x = None
    if ctx.needs_grad(x):
        grad_x = grad_out.builder.emit(
            "tanh_backward", grad_out, ctx.saved(node.output())
        )
    return (GradWrapper(x, grad_x),)


@register_backward_rule("gelu")
def register_backward_gelu(ctx, node, grad_out):
    (x,) = node.args[:1]
    grad_x = None
    if ctx.needs_grad(x):
        approximate = _parameter(node, "approximate", 1, "none")
        grad_x = grad_out.builder.emit(
            "gelu_backward", grad_out, ctx.saved(x), approximate=approximate
        )
    return (GradWrapper(x, grad_x),)


@register_backward_rule("silu")
def register_backward_silu(ctx, node, grad_out):
    (x,) = node.args[:1]
    grad_x = None
    if ctx.needs_grad(x):
        grad_x = grad_out.builder.emit("silu_backward", grad_out, ctx.saved(x))
    return (GradWrapper(x, grad_x),)


@register_backward_rule("leaky_relu")
def register_backward_leaky_relu(ctx, node, grad_out):
    (x,) = node.args[:1]
    grad_x = None
    if ctx.needs_grad(x):
        negative_slope = _parameter(node, "negative_slope", 1, 0.01)
        grad_x = grad_out.builder.emit(
            "leaky_relu_backward",
            grad_out,
            ctx.saved(x),
            negative_slope,
            False,
        )
    return (GradWrapper(x, grad_x),)


@register_backward_rule("hardtanh")
def register_backward_hardtanh(ctx, node, grad_out):
    (x,) = node.args[:1]
    grad_x = None
    if ctx.needs_grad(x):
        min_val = _parameter(node, "min_val", 1, -1.0)
        max_val = _parameter(node, "max_val", 2, 1.0)
        grad_x = grad_out.builder.emit(
            "hardtanh_backward", grad_out, ctx.saved(x), min_val, max_val
        )
    return (GradWrapper(x, grad_x),)


@register_backward_rule("hardswish")
def register_backward_hardswish(ctx, node, grad_out):
    (x,) = node.args[:1]
    grad_x = None
    if ctx.needs_grad(x):
        grad_x = grad_out.builder.emit("hardswish_backward", grad_out, ctx.saved(x))
    return (GradWrapper(x, grad_x),)


@register_backward_rule("hardsigmoid")
def register_backward_hardsigmoid(ctx, node, grad_out):
    (x,) = node.args[:1]
    grad_x = None
    if ctx.needs_grad(x):
        grad_x = grad_out.builder.emit("hardsigmoid_backward", grad_out, ctx.saved(x))
    return (GradWrapper(x, grad_x),)


@register_backward_rule("softplus")
def register_backward_softplus(ctx, node, grad_out):
    (x,) = node.args[:1]
    grad_x = None
    if ctx.needs_grad(x):
        beta = _parameter(node, "beta", 1, 1.0)
        threshold = _parameter(node, "threshold", 2, 20.0)
        grad_x = grad_out.builder.emit(
            "softplus_backward", grad_out, ctx.saved(x), beta, threshold
        )
    return (GradWrapper(x, grad_x),)


@register_backward_rule("mish")
def register_backward_mish(ctx, node, grad_out):
    (x,) = node.args[:1]
    grad_x = None
    if ctx.needs_grad(x):
        grad_x = grad_out.builder.emit("mish_backward", grad_out, ctx.saved(x))
    return (GradWrapper(x, grad_x),)


# Built-ins are immutable after import. Applications extend BackendRegistry,
# so compiling one model cannot change another model's differentiation rules.
BWD_RULES_REGISTRY = MappingProxyType(BWD_RULES_REGISTRY)
