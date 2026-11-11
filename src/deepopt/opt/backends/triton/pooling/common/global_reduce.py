"""Bounded multi-program global pooling; only the final stage packs/casts."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .addressing import _load_value
from .output import _store_value
from ...quantization.activation.indices import _pack_codes


@triton.jit
def _winner(values, codes, valid, SIGNED: tl.constexpr):
    positions = codes // 3 if SIGNED else codes
    nan = valid & (values != values)
    last_nan = tl.max(tl.where(nan, positions, -1), 1)
    maximum = tl.max(tl.where(valid & ~nan, values, -float("inf")), 1)
    first = tl.min(tl.where(valid & (values == maximum[:, None]), positions, 9223372036854775807), 1)
    chosen = valid & tl.where(last_nan[:, None] >= 0, nan & (positions == last_nan[:, None]), positions == first[:, None])
    lane = tl.min(tl.where(chosen, tl.arange(0, values.shape[1])[None, :].to(tl.int64), 9223372036854775807), 1)
    lane = tl.minimum(lane, values.shape[1] - 1).to(tl.int32)
    return tl.reshape(tl.gather(values, lane[:, None], 1), (values.shape[0],)), tl.reshape(tl.gather(codes, lane[:, None], 1), (values.shape[0],))


@triton.jit
def _global_partial(x, x_inverse, sums, arguments, SHAPE: tl.constexpr, META: tl.constexpr,
                     TOTAL: tl.constexpr, ELEMENTS: tl.constexpr, PARTS: tl.constexpr,
                     MAX: tl.constexpr, POWER: tl.constexpr, LP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    group, part = pid % triton.cdiv(TOTAL, 4), pid // triton.cdiv(TOTAL, 4)
    output = group * 4 + tl.arange(0, 4)
    position = (part * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    valid = (output[:, None] < TOTAL) & (position[None, :] < ELEMENTS)
    raw = _load_value(x, x_inverse, output[:, None] * ELEMENTS + position[None, :], valid, SHAPE, META)
    if MAX:
        value = (tl.abs(raw) if POWER > 0 else -tl.abs(raw)) if LP else raw
        code = position[None, :] + tl.full((4, BLOCK), 0, tl.int64)
        if LP:
            code = code * 3 + (raw > 0).to(tl.int64) + (raw < 0).to(tl.int64) * 2
        result, selected = _winner(value, code, valid, LP)
        tl.store(arguments + part * TOTAL + output, selected, output < TOTAL)
    else:
        value = libdevice.pow(raw, POWER) if LP else raw
        result = tl.sum(tl.where(valid, value, 0.0), 1)
    tl.store(sums + part * TOTAL + output, result, output < TOTAL)


@triton.jit
def _global_merge(sums, arguments, output, output_args, TOTAL: tl.constexpr, PARTS: tl.constexpr,
                   NEXT: tl.constexpr, MAX: tl.constexpr, LP: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    group, part = pid % triton.cdiv(TOTAL, 4), pid // triton.cdiv(TOTAL, 4)
    row = group * 4 + tl.arange(0, 4)
    column = part * BLOCK + tl.arange(0, BLOCK)
    valid = (row[:, None] < TOTAL) & (column[None, :] < PARTS)
    value = tl.load(sums + column[None, :] * TOTAL + row[:, None], valid, other=0)
    if MAX:
        code = tl.load(arguments + column[None, :] * TOTAL + row[:, None], valid, other=0)
        result, selected = _winner(value, code, valid, LP)
        tl.store(output_args + part * TOTAL + row, selected, row < TOTAL)
    else:
        result = tl.sum(tl.where(valid, value, 0.0), 1)
    tl.store(output + part * TOTAL + row, result, row < TOTAL)


@triton.jit
def _global_finish(sums, arguments, output, public, routing, factor, scales, inverse, observations, snapshot,
                    errors, TOTAL: tl.constexpr, PARTS: tl.constexpr, ELEMENTS: tl.constexpr,
                    DIVISOR: tl.constexpr, MAX: tl.constexpr, LP: tl.constexpr, POWER: tl.constexpr,
                    SAVE: tl.constexpr, PUBLIC: tl.constexpr, PACK: tl.constexpr, BITS: tl.constexpr,
                    WORD_BLOCK: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                    FP8: tl.constexpr, CURRENT: tl.constexpr, OBSERVE: tl.constexpr, SNAPSHOT: tl.constexpr,
                    MAXIMUM: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32)
    column = tl.arange(0, BLOCK)
    valid = (index[:, None] < TOTAL) & (column[None, :] < PARTS)
    values = tl.load(sums + column[None, :] * TOTAL + index[:, None], valid, other=0)
    if MAX:
        codes = tl.load(arguments + column[None, :] * TOTAL + index[:, None], valid, other=0)
        value, code = _winner(values, codes, valid, LP)
        if LP and POWER < 0:
            value = -value
        if PUBLIC:
            tl.store(public + index, code, index < TOTAL)
        if SAVE:
            code = tl.where(index < TOTAL, code, 0)
            if PACK:
                _pack_codes(code, routing, tl.program_id(0).to(tl.int64), BITS, WORD_BLOCK, TOTAL)
            else:
                tl.store(routing + index, code, index < TOTAL)
    else:
        z = tl.sum(tl.where(valid, values, 0.0), 1) / DIVISOR
        value = z
        if LP:
            base = z * ELEMENTS
            value = libdevice.pow(base, 1.0 / POWER)
            if SAVE:
                sign = (z > 0).to(tl.float32) - (z < 0).to(tl.float32)
                upstream = libdevice.pow(base, 1.0 / POWER - 1.0) * (ELEMENTS / POWER) * sign
                derivative = tl.where(tl.abs(z) <= 0, 0.0, upstream) * sign / DIVISOR
                tl.store(factor + index, derivative, index < TOTAL)
    _store_value(value, output, scales, inverse, observations, snapshot, errors, index, index < TOTAL,
                 ROWS, COLS, BR, BC, FP8, CURRENT, OBSERVE, SNAPSHOT, False, MAXIMUM, 32)
