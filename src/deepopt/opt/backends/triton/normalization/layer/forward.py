"""LayerNorm row fast paths and chunked normalized-output/packing epilogues."""

import triton
import triton.language as tl

from .common import _parameter, _logical, _destination, _quantize_pack_xhat_rows, _provided_scale, _row_statistics
from ..recenter import _record_error
from ...activation.common import _load
from ...quantization.fp8 import _scale
from ...quantization.activation.affine import _affine_quantize_pack, _affine_quantize_pack_with_metadata


@triton.jit
def _row_forward(x, x_inverse, weight, bias, output, mean, inverse, data, steps, minima,
                 scales, out_inverse, observations, snapshot, errors,
                 M: tl.constexpr, K: tl.constexpr, SHAPE: tl.constexpr, NORM: tl.constexpr,
                 META: tl.constexpr, W_STRIDES: tl.constexpr, B_STRIDES: tl.constexpr,
                 HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr, EPS: tl.constexpr,
                 GROUP: tl.constexpr, GROUPS: tl.constexpr, PAD_GROUPS: tl.constexpr, PAD: tl.constexpr,
                 WORDS: tl.constexpr, BITS: tl.constexpr, COMPRESSED: tl.constexpr,
                 FP8: tl.constexpr, PREPASS: tl.constexpr, CURRENT: tl.constexpr, TENSORWISE: tl.constexpr,
                 COLS: tl.constexpr, CHANNELS: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
                 BR: tl.constexpr, BC: tl.constexpr, SCALE_COLS: tl.constexpr, MAXIMUM: tl.constexpr,
                 OBSERVE: tl.constexpr, SNAPSHOT: tl.constexpr, CHECK: tl.constexpr, STATS_BLOCK: tl.constexpr,
                 center_shift=None, RECENTER: tl.constexpr = False):
    row = tl.program_id(0).to(tl.int64)
    # Reduction order must not depend on codec groups/bits: compression changes
    # only saved backward state, including when group padding needs a fallback.
    mu, rstd = _row_statistics(x, x_inverse, row, K, SHAPE, META, EPS, STATS_BLOCK)
    group = tl.arange(0, PAD_GROUPS)
    offset = tl.arange(0, PAD)
    position = group[:, None] * GROUP + offset[None, :]
    valid = (position < K) & (offset[None, :] < GROUP)
    _, value = _load(x, x_inverse, row, position, valid, K, SHAPE, False, META)
    tl.store(mean + row, mu)
    tl.store(inverse + row, rstd)
    h = (value - mu) * rstd
    if COMPRESSED:
        _quantize_pack_xhat_rows(h, valid, data, steps, minima, row, GROUPS, PAD_GROUPS, PAD, WORDS, BITS, center_shift, RECENTER)
    y = h
    if HAS_WEIGHT:
        y *= _parameter(weight, position, valid, NORM, W_STRIDES)
    if HAS_BIAS:
        y += _parameter(bias, position, valid, NORM, B_STRIDES)
    if PREPASS or OBSERVE:
        tl.store(observations + row, tl.max(tl.max(tl.where(valid, tl.abs(y), 0.0), 1), 0))
    if not PREPASS:
        logical = row * K + position
        if FP8:
            if CURRENT:
                if TENSORWISE:
                    multiplier = _scale(tl.max(tl.max(tl.where(valid, tl.abs(y), 0.0), 1), 0), MAXIMUM)
                    tl.store(out_inverse, tl.div_rn(1.0, multiplier))
                else:
                    grouped = tl.reshape(tl.where(valid, tl.abs(y), 0.0), (PAD_GROUPS, PAD // BC, BC))
                    multipliers = _scale(tl.max(grouped, 2), MAXIMUM)
                    multiplier = tl.reshape(tl.broadcast_to(multipliers[:, :, None], (PAD_GROUPS, PAD // BC, BC)), (PAD_GROUPS, PAD))
                    tile = group[:, None] * (PAD // BC) + tl.arange(0, PAD // BC)[None, :]
                    tl.store(out_inverse + row * SCALE_COLS + tile, tl.div_rn(1.0, multipliers), tile < SCALE_COLS)
            else:
                multiplier, scale_index, origin = _provided_scale(scales, logical, valid, COLS, CHANNELS, C, S, BR, BC, SCALE_COLS)
                tl.store(out_inverse + scale_index, tl.div_rn(1.0, multiplier), valid & origin)
            result = tl.minimum(tl.maximum(y * multiplier, -MAXIMUM), MAXIMUM)
        else:
            result = y
        destination = _destination(logical, CHANNELS, C, S)
        if FP8:
            tl.store(output + destination, result.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
        else:
            tl.store(output + destination, result.to(output.dtype.element_ty), valid)
    if SNAPSHOT:
        if row == 0:
            scale = tl.load(scales)
            tl.store(snapshot, scale)
            tl.store(snapshot + 1, tl.div_rn(1.0, scale))
    if CHECK:
        bad = valid & ~(tl.abs(h) < float("inf")) if COMPRESSED else valid & False
        if RECENTER:
            bad = bad | (valid & ((rstd <= 0) | ~(tl.abs(rstd) < float("inf"))))
        if FP8:
            bad = bad | (valid & ~(tl.abs(y) < float("inf")))
            if not PREPASS and not CURRENT:
                bad = bad | (valid & ((multiplier <= 0) | ~(tl.abs(multiplier) < float("inf"))))
        tl.store(errors + row, tl.max(tl.max(bad.to(tl.int32), 1), 0))


@triton.jit
def _epilogue(x, x_inverse, weight, bias, mean, inverse, output, data, steps, minima,
              scales, out_inverse, observations, snapshot, errors,
              K: tl.constexpr, SHAPE: tl.constexpr, NORM: tl.constexpr, META: tl.constexpr,
              W_STRIDES: tl.constexpr, B_STRIDES: tl.constexpr, HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr,
              GROUP: tl.constexpr, GROUPS: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
              COMPRESSED: tl.constexpr, PRECOMPUTED: tl.constexpr, FP8: tl.constexpr,
              COLS: tl.constexpr, CHANNELS: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
              BR: tl.constexpr, BC: tl.constexpr, SCALE_COLS: tl.constexpr, MAXIMUM: tl.constexpr,
              PARTS: tl.constexpr, OBSERVE: tl.constexpr, SNAPSHOT: tl.constexpr, CHECK: tl.constexpr, BLOCK: tl.constexpr,
              center_shift=None, center_errors=None, M: tl.constexpr = 1, RECENTER: tl.constexpr = False):
    group, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    row = group // GROUPS
    offset = part * BLOCK + tl.arange(0, BLOCK)
    position = group % GROUPS * GROUP + offset
    valid = (position < K) & (offset < GROUP)
    _, value = _load(x, x_inverse, row, position, valid, K, SHAPE, False, META)
    rstd = tl.load(inverse + row)
    h = (value - tl.load(mean + row)) * rstd
    if COMPRESSED:
        if PRECOMPUTED:
            low, step = tl.load(minima + group), tl.load(steps + group)
            words = _affine_quantize_pack_with_metadata(h, valid, step, low, BITS, BLOCK)
        else:
            words, step, low = _affine_quantize_pack(h, valid, BITS, BLOCK)
            tl.store(steps + group, step)
            tl.store(minima + group, low)
        word_offset = part * (BLOCK // (32 // BITS)) + tl.arange(0, BLOCK // (32 // BITS))
        tl.store(data + group * WORDS + word_offset, words, word_offset < WORDS)
        if RECENTER:
            _record_error(words, step, low, h, valid, center_errors, center_shift,
                          row, group % GROUPS * PARTS + part, M, GROUPS * PARTS, BITS, BLOCK)
    y = h
    if HAS_WEIGHT:
        y *= _parameter(weight, position, valid, NORM, W_STRIDES)
    if HAS_BIAS:
        y += _parameter(bias, position, valid, NORM, B_STRIDES)
    logical = row * K + position
    if FP8:
        multiplier, scale_index, origin = _provided_scale(scales, logical, valid, COLS, CHANNELS, C, S, BR, BC, SCALE_COLS)
        tl.store(out_inverse + scale_index, tl.div_rn(1.0, multiplier), valid & origin)
        result = tl.minimum(tl.maximum(y * multiplier, -MAXIMUM), MAXIMUM)
    else:
        result = y
    destination = _destination(logical, CHANNELS, C, S)
    if FP8:
        tl.store(output + destination, result.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
    else:
        tl.store(output + destination, result.to(output.dtype.element_ty), valid)
    if OBSERVE:
        tl.store(observations + group * PARTS + part, tl.max(tl.where(valid, tl.abs(y), 0.0), 0))
    if SNAPSHOT:
        if group == 0 and part == 0:
            scale = tl.load(scales)
            tl.store(snapshot, scale)
            tl.store(snapshot + 1, tl.div_rn(1.0, scale))
    if CHECK:
        bad = valid & ~(tl.abs(h) < float("inf")) if COMPRESSED else valid & False
        if RECENTER:
            bad = bad | (valid & ((rstd <= 0) | ~(tl.abs(rstd) < float("inf"))))
        if FP8:
            bad = bad | (valid & (~(tl.abs(y) < float("inf")) | (multiplier <= 0) | ~(tl.abs(multiplier) < float("inf"))))
        tl.store(errors + group * PARTS + part, tl.max(bad.to(tl.int32), 0))


@triton.jit
def _output_amax(x, x_inverse, weight, bias, mean, inverse, partial,
                 K: tl.constexpr, SHAPE: tl.constexpr, NORM: tl.constexpr, META: tl.constexpr,
                 W_STRIDES: tl.constexpr, B_STRIDES: tl.constexpr, HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr,
                 ROWS: tl.constexpr, COLS: tl.constexpr, CHANNELS: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
                 BR: tl.constexpr, BC: tl.constexpr, SCALE_COLS: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    group, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    offset = part * BLOCK + tl.arange(0, BLOCK)
    row, col = group // SCALE_COLS * BR + offset // BC, group % SCALE_COLS * BC + offset % BC
    valid = (offset < BR * BC) & (row < ROWS) & (col < COLS)
    logical = _logical(row, col, COLS, CHANNELS, C, S)
    norm_row, position = logical // K, logical % K
    _, value = _load(x, x_inverse, row, col, valid, COLS, SHAPE, CHANNELS, META)
    y = (value - tl.load(mean + norm_row, valid, other=0)) * tl.load(inverse + norm_row, valid, other=0)
    if HAS_WEIGHT:
        y *= _parameter(weight, position, valid, NORM, W_STRIDES)
    if HAS_BIAS:
        y += _parameter(bias, position, valid, NORM, B_STRIDES)
    tl.store(partial + group * PARTS + part, tl.max(tl.where(valid, tl.abs(y), 0.0), 0))


@triton.jit
def _xhat_ranges(x, x_inverse, mean, inverse, lows, highs,
                 K: tl.constexpr, SHAPE: tl.constexpr, META: tl.constexpr, GROUP: tl.constexpr,
                 GROUPS: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    group, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    row = group // GROUPS
    offset = part * BLOCK + tl.arange(0, BLOCK)
    position = group % GROUPS * GROUP + offset
    valid = (offset < GROUP) & (position < K)
    _, value = _load(x, x_inverse, row, position, valid, K, SHAPE, False, META)
    h = (value - tl.load(mean + row)) * tl.load(inverse + row)
    tl.store(lows + group * PARTS + part, tl.min(tl.where(valid, h, float("inf")), 0))
    tl.store(highs + group * PARTS + part, tl.max(tl.where(valid, h, float("-inf")), 0))
