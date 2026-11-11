"""Addressing, local FP8 decoding and inline boolean packing for activations."""

import triton
import triton.language as tl


@triton.jit
def _address(row, col, COLS: tl.constexpr, SHAPE: tl.constexpr, CHANNELS: tl.constexpr, META: tl.constexpr):
    if CHANNELS:
        spatial: tl.constexpr = META[4]
        logical = (row // spatial * META[3] + col) * spatial + row % spatial
    else:
        logical = row * COLS + col
    if META[0] == 1:
        sr = logical // (META[3] * META[4]) * META[4] + logical % META[4]
        sc = logical // META[4] % META[3]
        address = sr * META[6] + sc * META[7]
    elif META[0] == 2:
        sr, sc = logical // META[5], logical % META[5]
        address = sr * META[6] + sc * META[7]
    else:
        sr, sc = row, col
        address = logical
        if not META[2]:
            address = tl.zeros(logical.shape, tl.int64)
            remaining = logical
            for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
                address += remaining % SHAPE[axis] * META[1][axis]
                remaining = remaining // SHAPE[axis]
    return address, sr, sc


@triton.jit
def _load(data, inverse, row, col, valid, COLS: tl.constexpr, SHAPE: tl.constexpr,
          CHANNELS: tl.constexpr, META: tl.constexpr):
    address, sr, sc = _address(row, col, COLS, SHAPE, CHANNELS, META)
    raw = tl.load(data + address, valid, other=0).to(tl.float32)
    if META[0] > 0:
        scale = tl.load(inverse + sr // META[8] * META[10] + sc // META[9] * META[11], valid, other=0)
        return raw, raw * scale
    return raw, raw


@triton.jit
def _pack_bits(predicate, BLOCK: tl.constexpr):
    codes = tl.reshape(predicate.to(tl.int32), (BLOCK // 32, 32))
    return tl.sum(codes << tl.arange(0, 32)[None, :], axis=1)
