"""Separate zero ownership from unpool scatter; pack destinations at source."""

import triton
import triton.language as tl

from ..common.addressing import _logical, _matrix_index, _load_value, _load_integer
from ...quantization.activation.indices import _pack_codes


@triton.jit
def _zero(output, inverse, scales, N: tl.constexpr, SCALES: tl.constexpr, FP8: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(output + index, 0.0, index < N)
    if FP8:
        multiplier = tl.load(scales + index, index < SCALES, other=1.0)
        tl.store(inverse + index, tl.div_rn(1.0, multiplier), index < SCALES)


@triton.jit
def _check_indices(errors, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    bad = tl.load(errors + index, index < N, other=0)
    tl.device_assert(tl.sum(bad, 0) == 0, "unpool indices outside output bounds")


@triton.jit
def _scatter(x, x_inverse, indices, output, routing, scales, observations, snapshot, errors,
              SHAPE: tl.constexpr, OUT_SHAPE: tl.constexpr, X_META: tl.constexpr, INDEX_META: tl.constexpr,
              CHANNELS: tl.constexpr, INPUT_S: tl.constexpr, OUTPUT_S: tl.constexpr, TOTAL: tl.constexpr,
              SAVE: tl.constexpr, PACK: tl.constexpr, BITS: tl.constexpr, WORD_BLOCK: tl.constexpr,
              COLS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, FP8: tl.constexpr,
              OBSERVE: tl.constexpr, SNAPSHOT: tl.constexpr, MAXIMUM: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    index = pid * 32 + tl.arange(0, 32)
    valid = index < TOTAL
    logical = _logical(index, SHAPE, CHANNELS)
    destination = _load_integer(indices, logical, valid, SHAPE, INDEX_META)
    within = (destination >= 0) & (destination < OUTPUT_S)
    tl.store(errors + pid, tl.max((valid & ~within).to(tl.int32), 0))
    value = _load_value(x, x_inverse, logical, valid, SHAPE, X_META)
    out_index = _matrix_index(logical // INPUT_S * OUTPUT_S + destination, OUT_SHAPE, CHANNELS)
    if FP8:
        row, col = out_index // COLS, out_index % COLS
        multiplier = tl.load(scales + row // BR * triton.cdiv(COLS, BC) + col // BC, valid & within, other=1)
        result = tl.minimum(tl.maximum(value * multiplier, -MAXIMUM), MAXIMUM)
        tl.store(output + out_index, result.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid & within)
    else:
        tl.store(output + out_index, value.to(output.dtype.element_ty), valid & within)
    if SAVE:
        codes = tl.where(valid & within, destination, 0)
        if PACK:
            _pack_codes(codes, routing, pid, BITS, WORD_BLOCK, TOTAL)
        else:
            tl.store(routing + index, codes, valid)
    if OBSERVE:
        tl.store(observations + pid, tl.max(tl.where(valid, tl.abs(value), 0.0), 0))
    if SNAPSHOT:
        if pid == 0:
            scale = tl.load(scales)
            tl.store(snapshot, scale)
            tl.store(snapshot + 1, tl.div_rn(1.0, scale))
