"""Compute and quantize-pack LP input-local derivative factors without raw shadows."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ..common.addressing import _logical, _load_value
from ...quantization.activation.affine import _affine_quantize_pack, _affine_quantize_pack_with_metadata


@triton.jit
def _coefficient(x, x_inverse, data, step, minimum, lows, highs,
                  SHAPE: tl.constexpr, META: tl.constexpr, CHANNELS: tl.constexpr,
                  ROWS: tl.constexpr, COLS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, GROUP: tl.constexpr,
                  POWER: tl.constexpr, PACK: tl.constexpr, BITS: tl.constexpr, WORDS: tl.constexpr,
                  PARTS: tl.constexpr, PREPASS: tl.constexpr, PRECOMPUTED: tl.constexpr, BLOCK: tl.constexpr):
    group, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    offset = part * BLOCK + tl.arange(0, BLOCK)
    if BR:
        row, col = group // triton.cdiv(COLS, BC) * BR + offset // BC, group % triton.cdiv(COLS, BC) * BC + offset % BC
    else:
        row, col = (group * GROUP + offset) // COLS, (group * GROUP + offset) % COLS
    valid = (offset < GROUP) & (row < ROWS) & (col < COLS)
    logical = _logical(row * COLS + col, SHAPE, CHANNELS)
    value = _load_value(x, x_inverse, logical, valid, SHAPE, META)
    coefficient = POWER * libdevice.pow(value, POWER - 1.0)
    if PREPASS:
        tl.store(lows + group * PARTS + part, tl.min(tl.where(valid, coefficient, float("inf")), 0))
        tl.store(highs + group * PARTS + part, tl.max(tl.where(valid, coefficient, -float("inf")), 0))
    elif PACK:
        if PRECOMPUTED:
            scale, low = tl.load(step + group), tl.load(minimum + group)
            words = _affine_quantize_pack_with_metadata(coefficient, valid, scale, low, BITS, BLOCK)
        else:
            words, scale, low = _affine_quantize_pack(coefficient, valid, BITS, BLOCK)
            tl.store(step + group, scale)
            tl.store(minimum + group, low)
        word = part * (BLOCK // (32 // BITS)) + tl.arange(0, BLOCK // (32 // BITS))
        tl.store(data + group * WORDS + word, words, word < WORDS)
    else:
        tl.store(data + row * COLS + col, coefficient, valid)
