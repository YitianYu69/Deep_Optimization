"""Exact shape-preserving boolean bit packing."""

from math import prod
from types import SimpleNamespace

import torch

from ..._runtime import ceildiv
from .indices import pack, unpack


def pack_boolean_mask(mask):
    output = torch.empty(ceildiv(mask.numel(), 32), device=mask.device, dtype=torch.int32)
    pack(mask, SimpleNamespace(bits=1, data=output))
    return output


def unpack_boolean_mask(packed, shape):
    output = torch.empty(tuple(shape), device=packed.device, dtype=torch.bool)
    if prod(shape):
        unpack(SimpleNamespace(bits=1, data=packed), output)
    return output
