"""Inline xhat decode, weighted row reductions and bounded affine gradients."""

import triton
import triton.language as tl

from .common import _xhat, _xhat_rstd, _parameter, _logical, _destination
from ...activation.common import _load
from ...quantization.fp8 import _scale


@triton.jit
def _row_sums(dy, dy_inverse, data, step, minimum, x, x_inverse, mean, inverse, weight, sums,
               M: tl.constexpr, K: tl.constexpr, SHAPE: tl.constexpr, NORM: tl.constexpr,
               DY_META: tl.constexpr, X_META: tl.constexpr, W_STRIDES: tl.constexpr, HAS_WEIGHT: tl.constexpr,
               COMPRESSED: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
               PARTS: tl.constexpr, BLOCK: tl.constexpr,
               center_shift=None, RECENTER: tl.constexpr = False):
    row, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    position = part * BLOCK + tl.arange(0, BLOCK)
    valid = position < K
    _, gradient = _load(dy, dy_inverse, row, position, valid, K, SHAPE, False, DY_META)
    h = _xhat(data, step, minimum, x, x_inverse, mean, inverse, row, position, valid,
              K, SHAPE, X_META, COMPRESSED, GROUP, WORDS, BITS, center_shift, RECENTER)
    if HAS_WEIGHT:
        gradient = gradient * _parameter(weight, position, valid, NORM, W_STRIDES)
    tl.store(sums + row * PARTS + part, tl.sum(tl.where(valid, gradient, 0.0), 0))
    tl.store(sums + M * PARTS + row * PARTS + part, tl.sum(tl.where(valid, gradient * h, 0.0), 0))


@triton.jit
def _merge_sums(partial, output, DOMAINS: tl.constexpr, PARTS: tl.constexpr, NEXT: tl.constexpr, BLOCK: tl.constexpr):
    domain, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    offset = part * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(partial + domain * PARTS + offset, offset < PARTS, other=0)
    b = tl.load(partial + DOMAINS * PARTS + domain * PARTS + offset, offset < PARTS, other=0)
    tl.store(output + domain * NEXT + part, tl.sum(a, 0))
    tl.store(output + DOMAINS * NEXT + domain * NEXT + part, tl.sum(b, 0))


@triton.jit
def _dx_values(dy, dy_inverse, data, step, minimum, x, x_inverse, mean, inverse, weight, sums,
                row, position, valid, M: tl.constexpr, K: tl.constexpr, SHAPE: tl.constexpr, NORM: tl.constexpr,
                DY_META: tl.constexpr, X_META: tl.constexpr, W_STRIDES: tl.constexpr, HAS_WEIGHT: tl.constexpr,
                COMPRESSED: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                center_shift=None, RECENTER: tl.constexpr = False):
    _, gradient = _load(dy, dy_inverse, row, position, valid, K, SHAPE, False, DY_META)
    h, rstd = _xhat_rstd(data, step, minimum, x, x_inverse, mean, inverse, row, position, valid,
              K, SHAPE, X_META, COMPRESSED, GROUP, WORDS, BITS, center_shift, RECENTER)
    if HAS_WEIGHT:
        gradient = gradient * _parameter(weight, position, valid, NORM, W_STRIDES)
    return (gradient - tl.load(sums + row, valid, other=0) / K
            - h * tl.load(sums + M + row, valid, other=0) / K) * rstd


@triton.jit
def _input_gradient(dy, dy_inverse, data, step, minimum, x, x_inverse, mean, inverse, weight, sums,
                     output, scales, out_inverse, observations,
                     M: tl.constexpr, K: tl.constexpr, SHAPE: tl.constexpr, NORM: tl.constexpr,
                     DY_META: tl.constexpr, X_META: tl.constexpr, W_STRIDES: tl.constexpr, HAS_WEIGHT: tl.constexpr,
                     COMPRESSED: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                     ROWS: tl.constexpr, COLS: tl.constexpr, CHANNELS: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
                     FP8: tl.constexpr, CURRENT: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                     OBSERVE: tl.constexpr, MAXIMUM: tl.constexpr, BLOCK: tl.constexpr,
                     center_shift=None, RECENTER: tl.constexpr = False):
    pid = tl.program_id(0).to(tl.int64)
    offset = tl.arange(0, BLOCK)
    if FP8 and CURRENT:
        row, col = pid // triton.cdiv(COLS, BC) * BR + offset // BC, pid % triton.cdiv(COLS, BC) * BC + offset % BC
        valid = (offset < BR * BC) & (row < ROWS) & (col < COLS)
    else:
        index = pid * BLOCK + offset
        row, col = index // COLS, index % COLS
        valid = index < ROWS * COLS
    logical = _logical(row, col, COLS, CHANNELS, C, S)
    dx = _dx_values(dy, dy_inverse, data, step, minimum, x, x_inverse, mean, inverse, weight, sums,
                    logical // K, logical % K, valid, M, K, SHAPE, NORM, DY_META, X_META, W_STRIDES,
                    HAS_WEIGHT, COMPRESSED, GROUP, WORDS, BITS, center_shift, RECENTER)
    if FP8:
        if CURRENT:
            multiplier = _scale(tl.max(tl.where(valid, tl.abs(dx), 0.0), 0), MAXIMUM)
            tl.store(out_inverse + pid, tl.div_rn(1.0, multiplier))
        else:
            multiplier = tl.load(scales)
            if pid == 0:
                tl.store(out_inverse, tl.div_rn(1.0, multiplier))
        if OBSERVE:
            tl.store(observations + pid, tl.max(tl.where(valid, tl.abs(dx), 0.0), 0))
        converted = tl.minimum(tl.maximum(dx * multiplier, -MAXIMUM), MAXIMUM)
        tl.store(output + row * COLS + col, converted.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
    else:
        tl.store(output + row * COLS + col, dx.to(output.dtype.element_ty), valid)


@triton.jit
def _row_backward(dy, dy_inverse, data, step, minimum, x, x_inverse, mean, inverse, weight, output,
                   scales, out_inverse, observations,
                   M: tl.constexpr, K: tl.constexpr, SHAPE: tl.constexpr, NORM: tl.constexpr,
                   DY_META: tl.constexpr, X_META: tl.constexpr, W_STRIDES: tl.constexpr, HAS_WEIGHT: tl.constexpr,
                   COMPRESSED: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                   FP8: tl.constexpr, CURRENT: tl.constexpr, TENSORWISE: tl.constexpr,
                   COLS: tl.constexpr, CHANNELS: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
                   BC: tl.constexpr, SCALE_COLS: tl.constexpr, OBSERVE: tl.constexpr,
                   MAXIMUM: tl.constexpr, BLOCK: tl.constexpr,
                   center_shift=None, RECENTER: tl.constexpr = False):
    row = tl.program_id(0).to(tl.int64)
    position = tl.arange(0, BLOCK)
    valid = position < K
    _, gradient = _load(dy, dy_inverse, row, position, valid, K, SHAPE, False, DY_META)
    h, rstd = _xhat_rstd(data, step, minimum, x, x_inverse, mean, inverse, row, position, valid,
              K, SHAPE, X_META, COMPRESSED, GROUP, WORDS, BITS, center_shift, RECENTER)
    if HAS_WEIGHT:
        gradient = gradient * _parameter(weight, position, valid, NORM, W_STRIDES)
    a = tl.sum(tl.where(valid, gradient, 0.0), 0) / K
    b = tl.sum(tl.where(valid, gradient * h, 0.0), 0) / K
    dx = (gradient - a - h * b) * rstd
    logical = row * K + position
    if FP8:
        if CURRENT:
            if TENSORWISE:
                multiplier = _scale(tl.max(tl.where(valid, tl.abs(dx), 0.0), 0), MAXIMUM)
                tl.store(out_inverse, tl.div_rn(1.0, multiplier))
            else:
                values = tl.reshape(tl.where(valid, tl.abs(dx), 0.0), (BLOCK // BC, BC))
                multipliers = _scale(tl.max(values, 1), MAXIMUM)
                multiplier = tl.reshape(tl.broadcast_to(multipliers[:, None], (BLOCK // BC, BC)), (BLOCK,))
                tile = tl.arange(0, BLOCK // BC)
                tl.store(out_inverse + row * SCALE_COLS + tile, tl.div_rn(1.0, multipliers), tile < SCALE_COLS)
        else:
            multiplier = tl.load(scales)
            if row == 0:
                tl.store(out_inverse, tl.div_rn(1.0, multiplier))
        if OBSERVE:
            tl.store(observations + row, tl.max(tl.where(valid, tl.abs(dx), 0.0), 0))
        dx = tl.minimum(tl.maximum(dx * multiplier, -MAXIMUM), MAXIMUM)
        destination = _destination(logical, CHANNELS, C, S)
        tl.store(output + destination, dx.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
    else:
        tl.store(output + logical, dx.to(output.dtype.element_ty), valid)


@triton.jit
def _affine_partial(dy, dy_inverse, data, step, minimum, x, x_inverse, mean, inverse, partial,
                     M: tl.constexpr, K: tl.constexpr, SHAPE: tl.constexpr, DY_META: tl.constexpr, X_META: tl.constexpr,
                     COMPRESSED: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                     NEED_W: tl.constexpr, NEED_B: tl.constexpr, PARTS: tl.constexpr, RM: tl.constexpr, CK: tl.constexpr,
                     center_shift=None, RECENTER: tl.constexpr = False):
    feature, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    row, position = part * RM + tl.arange(0, RM), feature * CK + tl.arange(0, CK)
    valid = (row[:, None] < M) & (position[None, :] < K)
    _, gradient = _load(dy, dy_inverse, row[:, None], position[None, :], valid, K, SHAPE, False, DY_META)
    if NEED_W:
        h = _xhat(data, step, minimum, x, x_inverse, mean, inverse, row[:, None], position[None, :], valid,
                  K, SHAPE, X_META, COMPRESSED, GROUP, WORDS, BITS, center_shift, RECENTER)
        tl.store(partial + part * K + position, tl.sum(tl.where(valid, gradient * h, 0.0), 0), position < K)
    if NEED_B:
        tl.store(partial + PARTS * K + part * K + position, tl.sum(tl.where(valid, gradient, 0.0), 0), position < K)


@triton.jit
def _affine_merge(partial, output, K: tl.constexpr, PARTS: tl.constexpr, NEXT: tl.constexpr,
                   NEED_W: tl.constexpr, NEED_B: tl.constexpr, RP: tl.constexpr, CK: tl.constexpr):
    feature, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    rows, position = part * RP + tl.arange(0, RP), feature * CK + tl.arange(0, CK)
    valid = (rows[:, None] < PARTS) & (position[None, :] < K)
    if NEED_W:
        values = tl.load(partial + rows[:, None] * K + position[None, :], valid, other=0)
        tl.store(output + part * K + position, tl.sum(values, 0), position < K)
    if NEED_B:
        values = tl.load(partial + PARTS * K + rows[:, None] * K + position[None, :], valid, other=0)
        tl.store(output + NEXT * K + part * K + position, tl.sum(values, 0), position < K)
