"""Native-precision output and pre-output-cast derivative oracle."""

import torch
from torch.nn import functional as F


def evaluate(x, kind, approximate="none"):
    if kind == "sigmoid":
        y = torch.sigmoid(x)
        return y, y * (1 - y)
    if kind == "tanh":
        y = torch.tanh(x)
        return y, 1 - y * y
    if kind == "silu":
        return F.silu(x), torch.ops.aten.silu_backward(torch.ones_like(x), x)
    if kind == "gelu":
        return F.gelu(x, approximate=approximate), torch.ops.aten.gelu_backward(
            torch.ones_like(x), x, approximate=approximate)
    if kind == "relu6":
        return F.relu6(x), ~((x <= 0) | (x >= 6))
    raise ValueError(f"Unsupported activation: {kind}")
