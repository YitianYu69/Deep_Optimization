"""Fused nonlinear forward, output casting and derivative quantize-pack."""

import triton
import triton.language as tl

from .common import _load, _address, _pack_bits
from .math import _evaluate
from ..quantization.activation.affine import _coordinates, _affine_quantize_pack, _affine_quantize_pack_with_metadata
from ..quantization.fp8 import _scale


@triton.jit
def _statistics(data, inverse, partial_min, partial_max,
                ROWS: tl.constexpr, COLS: tl.constexpr, SHAPE: tl.constexpr,
                CHANNELS: tl.constexpr, META: tl.constexpr, KIND: tl.constexpr, APPROXIMATE: tl.constexpr,
                GROUP: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, PARTS: tl.constexpr,
                DERIVATIVE: tl.constexpr, BLOCK: tl.constexpr):
    group, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    offset = part * BLOCK + tl.arange(0, BLOCK)
    row, col, valid = _coordinates(group, offset, ROWS, COLS, GROUP, BR, BC)
    _, x = _load(data, inverse, row, col, valid, COLS, SHAPE, CHANNELS, META)
    y, derivative = _evaluate(x, KIND, APPROXIMATE)
    if DERIVATIVE:
        tl.store(partial_min + group * PARTS + part, tl.min(tl.where(valid, derivative, float("inf")), 0))
        tl.store(partial_max + group * PARTS + part, tl.max(tl.where(valid, derivative, float("-inf")), 0))
    else:
        tl.store(partial_max + group * PARTS + part, tl.max(tl.where(valid, tl.abs(y), 0.0), 0))


@triton.jit
def _reduce_ranges(minimum, maximum, out_min, out_max, PARTS: tl.constexpr,
                   NEXT: tl.constexpr, BLOCK: tl.constexpr):
    group, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    offset = part * BLOCK + tl.arange(0, BLOCK)
    lo = tl.load(minimum + group * PARTS + offset, offset < PARTS, other=float("inf"))
    hi = tl.load(maximum + group * PARTS + offset, offset < PARTS, other=float("-inf"))
    tl.store(out_min + group * NEXT + part, tl.min(lo, 0))
    tl.store(out_max + group * NEXT + part, tl.max(hi, 0))


@triton.jit
def _activation_forward(data, inverse, output, saved, steps, minima, scales, out_inverse,
                        observations, snapshot, errors,
                        ROWS: tl.constexpr, COLS: tl.constexpr, SHAPE: tl.constexpr,
                        CHANNELS: tl.constexpr, META: tl.constexpr, KIND: tl.constexpr, APPROXIMATE: tl.constexpr,
                        GROUP: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, WORDS: tl.constexpr,
                        SAVE: tl.constexpr, PACKED: tl.constexpr, BITS: tl.constexpr, PRECOMPUTED: tl.constexpr,
                        FP8: tl.constexpr, REUSE: tl.constexpr, CURRENT: tl.constexpr, OBR: tl.constexpr, OBC: tl.constexpr,
                        SCALE_COLS: tl.constexpr, MAXIMUM: tl.constexpr, INPLACE: tl.constexpr,
                        OBSERVE: tl.constexpr, SNAPSHOT: tl.constexpr, CHECK: tl.constexpr,
                        PARTS: tl.constexpr, BLOCK: tl.constexpr):
    group, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    offset = part * BLOCK + tl.arange(0, BLOCK)
    row, col, valid = _coordinates(group, offset, ROWS, COLS, GROUP, BR, BC)
    _, x = _load(data, inverse, row, col, valid, COLS, SHAPE, CHANNELS, META)
    y, derivative = _evaluate(x, KIND, APPROXIMATE)
    if FP8:
        if REUSE:
            _, sr, sc = _address(row, col, COLS, SHAPE, CHANNELS, META)
            input_inverse = tl.load(inverse + sr // META[8] * META[10] + sc // META[9] * META[11], valid, other=1.0)
            multiplier = tl.div_rn(1.0, input_inverse)
        elif CURRENT:
            multiplier = _scale(tl.max(tl.where(valid, tl.abs(y), 0.0), 0), MAXIMUM)
            tl.store(out_inverse + group, tl.div_rn(1.0, multiplier))
        else:
            scale_index = row // OBR * SCALE_COLS + col // OBC
            multiplier = tl.load(scales + scale_index, valid, other=1.0)
            tl.store(out_inverse + scale_index, tl.div_rn(1.0, multiplier),
                     valid & (row % OBR == 0) & (col % OBC == 0))
        result = tl.minimum(tl.maximum(y * multiplier, -MAXIMUM), MAXIMUM)
    else:
        result = y
    destination = row * COLS + col
    if INPLACE:
        destination, _, _ = _address(row, col, COLS, SHAPE, CHANNELS, META)
    if FP8:
        tl.store(output + destination, result.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
    else:
        tl.store(output + destination, result.to(output.dtype.element_ty), valid)
    if SAVE:
        if KIND == "relu6":
            if PACKED:
                words = _pack_bits((derivative != 0) & valid, BLOCK)
                word_offset = group * (GROUP // 32) + part * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
                tl.store(saved + word_offset, words, word_offset < triton.cdiv(ROWS * COLS, 32))
            else:
                tl.store(saved + row * COLS + col, derivative != 0, valid)
        elif PACKED:
            if PRECOMPUTED:
                step, minimum = tl.load(steps + group), tl.load(minima + group)
                words = _affine_quantize_pack_with_metadata(derivative, valid, step, minimum, BITS, BLOCK)
            else:
                words, step, minimum = _affine_quantize_pack(derivative, valid, BITS, BLOCK)
                tl.store(steps + group, step)
                tl.store(minima + group, minimum)
            word_offset = part * (BLOCK // (32 // BITS)) + tl.arange(0, BLOCK // (32 // BITS))
            tl.store(saved + group * WORDS + word_offset, words, word_offset < WORDS)
        else:
            tl.store(saved + row * COLS + col, derivative, valid)
    if OBSERVE:
        tl.store(observations + group * PARTS + part, tl.max(tl.where(valid, tl.abs(y), 0.0), 0))
    if SNAPSHOT:
        if group == 0 and part == 0:
            s = tl.load(scales)
            tl.store(snapshot, s)
            tl.store(snapshot + 1, tl.div_rn(1.0, s))
    if CHECK:
        bad = valid & ~(tl.abs(y) < float("inf")) if FP8 else tl.full((BLOCK,), False, tl.int1)
        if SAVE and PACKED and KIND != "relu6":
            bad = bad | (valid & ~(tl.abs(derivative) < float("inf")))
        if FP8 and not CURRENT:
            bad = bad | (valid & ((multiplier <= 0) | ~(tl.abs(multiplier) < float("inf"))))
        tl.store(errors + group * PARTS + part, tl.max(bad.to(tl.int32), 0))
