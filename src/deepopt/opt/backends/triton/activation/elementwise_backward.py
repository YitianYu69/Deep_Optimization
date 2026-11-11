"""Inline derivative decoding, FP8 upstream loading and native/FP8 dx epilogues."""

import triton
import triton.language as tl

from .common import _load
from ..quantization.fp8 import _scale


@triton.jit
def _coefficient(saved, steps, minima, index, valid, COLS: tl.constexpr, GROUP: tl.constexpr,
                 BR: tl.constexpr, BC: tl.constexpr, WORDS: tl.constexpr,
                 BITS: tl.constexpr, MASK: tl.constexpr, PACKED: tl.constexpr):
    if MASK:
        if PACKED:
            word = tl.load(saved + index // 32, valid, other=0)
            return ((word >> (index % 32)) & 1).to(tl.float32)
        return tl.load(saved + index, valid, other=0).to(tl.float32)
    elif PACKED:
        if BR > 0:
            row, col = index // COLS, index % COLS
            group = row // BR * triton.cdiv(COLS, BC) + col // BC
            offset = row % BR * BC + col % BC
        else:
            group, offset = index // GROUP, index % GROUP
        word = tl.load(saved + group * WORDS + offset // (32 // BITS), valid, other=0)
        code = ((word >> (offset % (32 // BITS) * BITS)) & ((1 << BITS) - 1)).to(tl.float32)
        step = tl.load(steps + group, valid, other=0)
        minimum = tl.load(minima + group, valid, other=0)
        return code * step + minimum
    else:
        return tl.load(saved + index, valid, other=0).to(tl.float32)


@triton.jit
def _activation_backward(data, inverse, saved, steps, minima, output, scale, out_inverse, observations,
                         ROWS: tl.constexpr, COLS: tl.constexpr, SHAPE: tl.constexpr, META: tl.constexpr,
                         CHANNELS: tl.constexpr, SAVED_CHANNELS: tl.constexpr,
                         C: tl.constexpr, S: tl.constexpr, SAVED_COLS: tl.constexpr,
                         GROUP: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, WORDS: tl.constexpr,
                         BITS: tl.constexpr, MASK: tl.constexpr, PACKED: tl.constexpr,
                         FP8: tl.constexpr, CURRENT: tl.constexpr, OBR: tl.constexpr, OBC: tl.constexpr,
                         OBSERVE: tl.constexpr, MAXIMUM: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offset = tl.arange(0, BLOCK)
    if FP8 and CURRENT:
        row = pid // triton.cdiv(COLS, OBC) * OBR + offset // OBC
        col = pid % triton.cdiv(COLS, OBC) * OBC + offset % OBC
        valid = (row < ROWS) & (col < COLS) & (offset < OBR * OBC)
    else:
        index = pid * BLOCK + offset
        row, col = index // COLS, index % COLS
        valid = index < ROWS * COLS
    logical = (row // S * C + col) * S + row % S if CHANNELS else row * COLS + col
    saved_index = ((logical // (C * S) * S + logical % S) * C + logical // S % C
                   if SAVED_CHANNELS else logical)
    coefficient = _coefficient(saved, steps, minima, saved_index, valid, SAVED_COLS,
                               GROUP, BR, BC, WORDS, BITS, MASK, PACKED)
    _, dy = _load(data, inverse, row, col, valid, COLS, SHAPE, CHANNELS, META)
    dx = tl.where(coefficient != 0, dy, 0.0) if MASK else dy * coefficient
    dx = tl.where(valid, dx, 0.0)
    if FP8:
        if CURRENT:
            multiplier = _scale(tl.max(tl.abs(dx), 0), MAXIMUM)
            tl.store(out_inverse + pid, tl.div_rn(1.0, multiplier))
        else:
            multiplier = tl.load(scale)
            if pid == 0:
                tl.store(out_inverse, tl.div_rn(1.0, multiplier))
        if OBSERVE:
            tl.store(observations + pid, tl.max(tl.abs(dx), 0))
        dx = tl.minimum(tl.maximum(dx * multiplier, -MAXIMUM), MAXIMUM)
        tl.store(output + row * COLS + col, dx.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
    else:
        tl.store(output + row * COLS + col, dx.to(output.dtype.element_ty), valid)
