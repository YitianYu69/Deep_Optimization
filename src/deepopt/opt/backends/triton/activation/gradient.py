"""Retiled ReLU backward with inline mask extraction and fresh FP8 output."""

import triton
import triton.language as tl

from .common import _load


@triton.jit
def _relu_gradient(data, inverse, saved, output, scale, out_inverse, observations,
                   ROWS: tl.constexpr, COLS: tl.constexpr, SHAPE: tl.constexpr, META: tl.constexpr,
                   CHANNELS: tl.constexpr, MASK_CHANNELS: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
                   PACKED: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, CURRENT: tl.constexpr,
                   OBSERVE: tl.constexpr, MAXIMUM: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offset = tl.arange(0, BLOCK)
    if CURRENT:
        row = pid // triton.cdiv(COLS, BC) * BR + offset // BC
        col = pid % triton.cdiv(COLS, BC) * BC + offset % BC
        valid = (row < ROWS) & (col < COLS) & (offset < BR * BC)
    else:
        index = pid * BLOCK + offset
        row, col = index // COLS, index % COLS
        valid = index < ROWS * COLS
    if CHANNELS:
        logical = (row // S * C + col) * S + row % S
    else:
        logical = row * COLS + col
    if MASK_CHANNELS:
        mask_index = (logical // (C * S) * S + logical % S) * C + logical // S % C
    else:
        mask_index = logical
    if PACKED:
        word = tl.load(saved + mask_index // 32, valid, other=0)
        predicate = ((word >> (mask_index % 32)) & 1) != 0
    else:
        predicate = tl.load(saved + mask_index, valid, other=0) != 0
    _, value = _load(data, inverse, row, col, valid, COLS, SHAPE, CHANNELS, META)
    value = tl.where(predicate & valid, value, 0.0)
    if CURRENT:
        amax = tl.max(tl.abs(value), 0)
        multiplier = tl.where(amax == 0, 1.0, tl.div_rn(MAXIMUM, tl.maximum(amax, 1e-12)))
        tl.store(out_inverse + pid, tl.div_rn(1.0, multiplier))
    else:
        multiplier = tl.load(scale)
        if pid == 0:
            tl.store(out_inverse, tl.div_rn(1.0, multiplier))
        if OBSERVE:
            tl.store(observations + pid, tl.max(tl.abs(value), 0))
    value = tl.minimum(tl.maximum(value * multiplier, -MAXIMUM), MAXIMUM)
    tl.store(output + row * COLS + col, value.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
