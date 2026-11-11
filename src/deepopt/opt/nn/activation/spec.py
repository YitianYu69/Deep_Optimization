"""Shared native activation-call matching without importing dispatch/autograd."""

import torch
from torch.nn import functional as F


KINDS = ("sigmoid", "silu", "gelu", "tanh", "relu6")
FUNCTIONS = {
    F.sigmoid: "sigmoid", torch.sigmoid: "sigmoid", torch.Tensor.sigmoid: "sigmoid",
    torch.sigmoid_: "sigmoid", torch.Tensor.sigmoid_: "sigmoid", F.silu: "silu", F.gelu: "gelu",
    F.tanh: "tanh", torch.tanh: "tanh", torch.tanh_: "tanh", torch.Tensor.tanh: "tanh", torch.Tensor.tanh_: "tanh",
    F.relu6: "relu6",
}
METHODS = {"sigmoid": "sigmoid", "sigmoid_": "sigmoid", "tanh": "tanh", "tanh_": "tanh"}


def arguments(target, args, kwargs):
    """Return (kind, input, options), leaving invalid/unsupported native calls alone."""
    kind = FUNCTIONS.get(target) if not isinstance(target, str) else METHODS.get(target)
    if kind is None:
        return None
    positional = ("input", "inplace") if target in (F.silu, F.relu6) else ("input",)
    allowed = {*positional, "approximate"} if kind == "gelu" else set(positional)
    if (len(args) > len(positional) or set(kwargs) - allowed
            or any(key in kwargs for key in positional[:len(args)])):
        return None
    x = args[0] if args else kwargs.get("input")
    if x is None:
        return None
    inplace = (target in (torch.sigmoid_, torch.tanh_, torch.Tensor.sigmoid_, torch.Tensor.tanh_, "sigmoid_", "tanh_")
               or kwargs.get("inplace", args[1] if len(args) > 1 else False))
    return kind, x, {"inplace": inplace, "approximate": kwargs.get("approximate", "none")}
