"""Scalar TileLang expressions, independent of the optional TileLang package."""

import ast


FORWARD_OPS = frozenset((
    "add", "sub", "mul", "div", "neg", "relu", "sigmoid", "tanh", "gelu", "silu",
    "leaky_relu", "hardtanh", "hardswish", "hardsigmoid", "softplus", "mish",
))
BACKWARD_OPS = frozenset((
    "threshold_backward", "sigmoid_backward", "tanh_backward", "gelu_backward",
    "silu_backward", "leaky_relu_backward", "hardtanh_backward", "hardswish_backward",
    "hardsigmoid_backward", "softplus_backward", "mish_backward",
))
SUPPORTED_OPS = FORWARD_OPS | BACKWARD_OPS


def _option(args, kwargs, name, index, default):
    value = args[index] if len(args) > index else kwargs.get(name, repr(default))
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _f32(x):
    return f'T.cast({x}, "float32")'


def _where(condition, yes, no):
    return f"T.if_then_else({condition}, {yes}, {no})"


def _sigmoid(x):
    return f"(1.0 / (1.0 + T.exp(-({x}))))"


def _softplus(x):
    return _where(f"{x} > 20.0", x, f"(T.max({x}, 0.0) + T.log(1.0 + T.exp(-T.abs({x}))))")


def expression(op, args, kwargs=None):
    """Render directly from IR; never translate or execute Triton source."""
    kwargs = kwargs or {}
    if op in ("add", "sub", "mul", "div"):
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
        return f"({_f32(args[0])} {symbol} {_f32(args[1])})"
    x = _f32(args[0])
    if op == "neg":
        return f"(-{x})"
    if op == "relu":
        return _where(f"{x} < 0.0", "0.0", x)
    if op == "sigmoid":
        return _sigmoid(x)
    if op == "tanh":
        return f"T.tanh({x})"
    if op == "silu":
        return f"({x} * {_sigmoid(x)})"
    if op == "gelu":
        approximate = _option(args, kwargs, "approximate", 1, "none")
        if approximate == "tanh":
            inner = f"(0.7978845608028654 * ({x} + 0.044715 * {x} * {x} * {x}))"
            return f"(0.5 * {x} * (1.0 + T.tanh({inner})))"
        if approximate != "none":
            raise NotImplementedError(f"Unsupported GELU approximation {approximate!r}")
        return f"(0.5 * {x} * (1.0 + T.erf({x} * 0.7071067811865476)))"
    if op == "leaky_relu":
        slope = _option(args, kwargs, "negative_slope", 1, 0.01)
        return _where(f"{x} > 0.0", x, f"({x} * {slope!r})")
    if op == "hardtanh":
        low = _option(args, kwargs, "min_val", 1, -1.0)
        high = _option(args, kwargs, "max_val", 2, 1.0)
        return _where(f"{x} < {low!r}", repr(low), _where(f"{x} > {high!r}", repr(high), x))
    if op in ("hardswish", "hardsigmoid"):
        gate = f"(T.min(T.max({x} + 3.0, 0.0), 6.0) / 6.0)"
        return f"({x} * {gate})" if op == "hardswish" else _where(f"{x} != {x}", x, gate)
    if op == "softplus":
        beta = _option(args, kwargs, "beta", 1, 1.0)
        threshold = _option(args, kwargs, "threshold", 2, 20.0)
        scaled = f"({x} * {beta!r})"
        stable = f"((T.max({scaled}, 0.0) + T.log(1.0 + T.exp(-T.abs({scaled})))) / {beta!r})"
        return _where(f"{scaled} > {threshold!r}", x, stable)
    if op == "mish":
        return f"({x} * T.tanh({_softplus(x)}))"
    if op in BACKWARD_OPS:
        g, value = x, _f32(args[1])
        if op == "threshold_backward":
            threshold = _option(args, kwargs, "threshold", 2, 0.0)
            return _where(f"{value} <= {threshold!r}", "0.0", g)
        if op == "sigmoid_backward":
            return f"({g} * {value} * (1.0 - {value}))"
        if op == "tanh_backward":
            return f"({g} * (1.0 - {value} * {value}))"
        if op == "silu_backward":
            s = _sigmoid(value)
            return f"({g} * {s} * (1.0 + {value} * (1.0 - {s})))"
        if op == "gelu_backward":
            approximate = _option(args, kwargs, "approximate", 2, "none")
            if approximate == "tanh":
                inner = f"(0.7978845608028654 * ({value} + 0.044715 * {value} * {value} * {value}))"
                t = f"T.tanh({inner})"
                return f"({g} * (0.5 * (1.0 + {t}) + 0.5 * {value} * (1.0 - {t} * {t}) * 0.7978845608028654 * (1.0 + 0.134145 * {value} * {value})))"
            if approximate != "none":
                raise NotImplementedError(f"Unsupported GELU approximation {approximate!r}")
            return f"({g} * (0.5 * (1.0 + T.erf({value} * 0.7071067811865476)) + {value} * T.exp(-0.5 * {value} * {value}) * 0.3989422804014327))"
        if op == "leaky_relu_backward":
            slope = _option(args, kwargs, "negative_slope", 2, 0.01)
            return _where(f"{value} > 0.0", g, f"({g} * {slope!r})")
        if op == "hardtanh_backward":
            low = _option(args, kwargs, "min_val", 2, -1.0)
            high = _option(args, kwargs, "max_val", 3, 1.0)
            return _where(f"({value} <= {low!r}) | ({value} >= {high!r})", "0.0", g)
        if op == "hardswish_backward":
            return _where(f"{value} <= -3.0", "0.0", _where(f"{value} < 3.0", f"({g} * ({value} / 3.0 + 0.5))", g))
        if op == "hardsigmoid_backward":
            return _where(f"({value} > -3.0) & ({value} < 3.0)", f"({g} / 6.0)", "0.0")
        if op == "softplus_backward":
            beta = _option(args, kwargs, "beta", 2, 1.0)
            threshold = _option(args, kwargs, "threshold", 3, 20.0)
            return _where(f"({value} * {beta!r}) > {threshold!r}", g, f"({g} * {_sigmoid(f'({value} * {beta!r})')})")
        if op == "mish_backward":
            s, t = _sigmoid(value), f"T.tanh({_softplus(value)})"
            return f"({g} * ({t} + {value} * {s} * (1.0 - {t} * {t})))"
    raise NotImplementedError(f"No TileLang pointwise expression for {op!r}")
