"""Spatial pooling coordinates and scale-aware logical tensor loads."""

import triton
import triton.language as tl

from ...activation.common import _load, _address


@triton.jit
def _logical(index, SHAPE: tl.constexpr, CHANNELS: tl.constexpr):
    if CHANNELS:
        spatial = 1
        for size in tl.static_range(2, len(SHAPE)):
            spatial *= SHAPE[size]
        return (index // SHAPE[1] // spatial * SHAPE[1] + index % SHAPE[1]) * spatial + index // SHAPE[1] % spatial
    return index


@triton.jit
def _matrix_index(logical, SHAPE: tl.constexpr, CHANNELS: tl.constexpr):
    if CHANNELS:
        spatial = 1
        for axis in tl.static_range(2, len(SHAPE)):
            spatial *= SHAPE[axis]
        return (logical // (SHAPE[1] * spatial) * spatial + logical % spatial) * SHAPE[1] + logical // spatial % SHAPE[1]
    return logical


@triton.jit
def _load_value(data, inverse, logical, valid, SHAPE: tl.constexpr, META: tl.constexpr):
    _, value = _load(data, inverse, logical // SHAPE[-1], logical % SHAPE[-1], valid, SHAPE[-1], SHAPE, False, META)
    return value


@triton.jit
def _load_integer(data, logical, valid, SHAPE: tl.constexpr, META: tl.constexpr):
    address, _, _ = _address(logical // SHAPE[-1], logical % SHAPE[-1], SHAPE[-1], SHAPE, False, META)
    return tl.load(data + address, valid, other=0).to(tl.int64)


@triton.jit
def _coordinates(position, SPATIAL: tl.constexpr):
    return position // (SPATIAL[1] * SPATIAL[2]), position // SPATIAL[2] % SPATIAL[1], position % SPATIAL[2]


@triton.jit
def _flatten(z, y, x, SPATIAL: tl.constexpr):
    return (z * SPATIAL[1] + y) * SPATIAL[2] + x
