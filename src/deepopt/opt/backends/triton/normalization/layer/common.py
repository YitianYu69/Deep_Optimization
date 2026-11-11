"""Logical suffix addressing, local FP8 decoding and packed xhat extraction."""

import triton
import triton.language as tl

from ..recenter import _correct_xhat, _record_error

from ...activation.common import _load
from ...quantization.activation.affine import _affine_step, _affine_quantize_pack_with_metadata


@triton.jit
def _row_statistics(x, x_inverse, row, K: tl.constexpr, SHAPE: tl.constexpr,
                     META: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    position = tl.arange(0, BLOCK)
    valid = position < K
    _, value = _load(x, x_inverse, row, position, valid, K, SHAPE, False, META)
    mean = tl.sum(tl.where(valid, value / K, 0.0), 0)
    delta = tl.where(valid, value - mean, 0.0)
    inverse = tl.rsqrt(tl.sum(delta * delta, 0) / K + EPS)
    return mean, inverse


@triton.jit
def _logical(row, col, COLS: tl.constexpr, CHANNELS: tl.constexpr, C: tl.constexpr, S: tl.constexpr):
    if CHANNELS:
        return (row // S * C + col) * S + row % S
    return row * COLS + col


@triton.jit
def _destination(logical, CHANNELS: tl.constexpr, C: tl.constexpr, S: tl.constexpr):
    if CHANNELS:
        return (logical // (C * S) * S + logical % S) * C + logical // S % C
    return logical


@triton.jit
def _parameter(pointer, position, valid, NORM: tl.constexpr, STRIDES: tl.constexpr):
    remaining = position
    address = tl.full(position.shape, 0, tl.int64)
    for axis in tl.static_range(len(NORM) - 1, -1, -1):
        address += remaining % NORM[axis] * STRIDES[axis]
        remaining = remaining // NORM[axis]
    return tl.load(pointer + address, valid, other=0).to(tl.float32)


@triton.jit
def _xhat(data, step, minimum, x, x_inverse, mean, inverse, row, position, valid,
           K: tl.constexpr, SHAPE: tl.constexpr, X_META: tl.constexpr,
           COMPRESSED: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
               center_shift=None, RECENTER: tl.constexpr = False):
    if COMPRESSED:
        group = row * triton.cdiv(K, GROUP) + position // GROUP
        offset = position % GROUP
        word = tl.load(data + group * WORDS + offset // (32 // BITS), valid, other=0)
        code = ((word >> (offset % (32 // BITS) * BITS)) & ((1 << BITS) - 1)).to(tl.float32)
        scale, low = tl.load(step + group, valid, other=0), tl.load(minimum + group, valid, other=0)
        result = code * scale + low
        if BITS == 1:
            result = tl.where(scale < 0, (low * 0.5 + code * -scale) * 2, result)
        if RECENTER:
            result, _ = _correct_xhat(result, inverse, center_shift, row, valid, True)
        return result
    else:
        _, value = _load(x, x_inverse, row, position, valid, K, SHAPE, False, X_META)
        index = row + valid.to(tl.int64) * 0
        return (value - tl.load(mean + index, valid, other=0)) * tl.load(inverse + index, valid, other=0)


@triton.jit
def _quantize_pack_xhat_rows(values, valid, data, steps, minima, row,
                    GROUPS: tl.constexpr, PAD_GROUPS: tl.constexpr, PAD: tl.constexpr,
                    WORDS: tl.constexpr, BITS: tl.constexpr,
                    center_shift=None, RECENTER: tl.constexpr = False):
    low = tl.min(tl.where(valid, values, float("inf")), 1)
    high = tl.max(tl.where(valid, values, float("-inf")), 1)
    step = _affine_step(low, high, BITS)
    flat_step = tl.reshape(tl.broadcast_to(step[:, None], (PAD_GROUPS, PAD)), (PAD_GROUPS * PAD,))
    flat_low = tl.reshape(tl.broadcast_to(low[:, None], (PAD_GROUPS, PAD)), (PAD_GROUPS * PAD,))
    words = _affine_quantize_pack_with_metadata(tl.reshape(values, (PAD_GROUPS * PAD,)),
                                               tl.reshape(valid, (PAD_GROUPS * PAD,)), flat_step, flat_low,
                                               BITS, PAD_GROUPS * PAD)
    if RECENTER:
        _record_error(words, flat_step, flat_low, tl.reshape(values, (PAD_GROUPS * PAD,)),
                      tl.reshape(valid, (PAD_GROUPS * PAD,)), center_shift, center_shift,
                      row, 0, 1, 1, BITS, PAD_GROUPS * PAD)
    words = tl.reshape(words, (PAD_GROUPS, PAD // (32 // BITS)))
    group = tl.arange(0, PAD_GROUPS)
    offsets = tl.arange(0, PAD // (32 // BITS))
    tl.store(data + (row * GROUPS + group[:, None]) * WORDS + offsets[None, :], words,
             (group[:, None] < GROUPS) & (offsets[None, :] < WORDS))
    tl.store(steps + row * GROUPS + group, step, group < GROUPS)
    tl.store(minima + row * GROUPS + group, low, group < GROUPS)


@triton.jit
def _provided_scale(scales, logical, valid, COLS: tl.constexpr, CHANNELS: tl.constexpr,
                    C: tl.constexpr, S: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                    SCALE_COLS: tl.constexpr):
    destination = _destination(logical, CHANNELS, C, S)
    row, col = destination // COLS, destination % COLS
    index = row // BR * SCALE_COLS + col // BC
    return tl.load(scales + index, valid, other=1.0), index, (row % BR == 0) & (col % BC == 0)


@triton.jit
def _xhat_rstd(data, step, minimum, x, x_inverse, mean, inverse, row, position, valid,
                K: tl.constexpr, SHAPE: tl.constexpr, X_META: tl.constexpr,
                COMPRESSED: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                center_shift=None, RECENTER: tl.constexpr = False):
    h = _xhat(data, step, minimum, x, x_inverse, mean, inverse, row, position, valid,
              K, SHAPE, X_META, COMPRESSED, GROUP, WORDS, BITS)
    return _correct_xhat(h, inverse, center_shift, row, valid, RECENTER)
