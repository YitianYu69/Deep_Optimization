"""Materialized numerical oracle for normalization activation boundaries."""

import torch
from torch.nn import functional as F


def apply(value, activation, preactivation=None):
    if activation is None:
        return value
    if preactivation is not None:
        preactivation.copy_(value.reshape(preactivation.shape))
    kind, approximate = activation
    output = F.silu(value.float()) if kind == "silu" else F.gelu(value.float(), approximate=approximate)
    return output.to(value.dtype)


def adjust(gradient, preactivation, activation):
    if activation is None:
        return gradient
    kind, approximate = activation
    value = preactivation.float()
    adjusted = (torch.ops.aten.silu_backward.default(gradient.float(), value) if kind == "silu" else
                torch.ops.aten.gelu_backward.default(gradient.float(), value, approximate=approximate))
    return adjusted.to(preactivation.dtype)
