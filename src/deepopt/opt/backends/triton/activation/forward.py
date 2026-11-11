"""Fused ReLU, precision conversion and mask production; bounded scale reductions."""

import triton
import triton.language as tl

from .common import _address, _load, _pack_bits
from ..quantization.fp8 import _scale


@triton.jit
def _forward(data, inverse, output, saved, scales, out_inverse, errors, observations, snapshot,
             ROWS: tl.constexpr, COLS: tl.constexpr, SHAPE: tl.constexpr, CHANNELS: tl.constexpr,
             META: tl.constexpr, FP8: tl.constexpr, REUSE: tl.constexpr, CURRENT: tl.constexpr,
             BR: tl.constexpr, BC: tl.constexpr, SCALE_COLS: tl.constexpr, MAXIMUM: tl.constexpr,
             SAVE: tl.constexpr, PACKED: tl.constexpr, INPLACE: tl.constexpr,
             CHECK: tl.constexpr, OBSERVE: tl.constexpr, SNAPSHOT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    index = pid * BLOCK + tl.arange(0, BLOCK)
    row, col = index // COLS, index % COLS
    valid = index < ROWS * COLS
    raw, value = _load(data, inverse, row, col, valid, COLS, SHAPE, CHANNELS, META)
    activated = tl.where(value < 0, 0.0, value)
    if REUSE:
        result = tl.where(value < 0, 0.0, raw)
    elif FP8:
        if CURRENT:
            scale = _scale(tl.max(tl.abs(activated), 0), MAXIMUM)
            tl.store(out_inverse, tl.div_rn(1.0, scale))
        else:
            scale = tl.load(scales + row // BR * SCALE_COLS + col // BC, valid, other=1.0)
            tl.store(out_inverse + row // BR * SCALE_COLS + col // BC, tl.div_rn(1.0, scale),
                     valid & (row % BR == 0) & (col % BC == 0))
        result = tl.minimum(tl.maximum(activated * scale, -MAXIMUM), MAXIMUM)
    else:
        result = activated
    destination = index
    if INPLACE:
        destination, _, _ = _address(row, col, COLS, SHAPE, CHANNELS, META)
    if FP8:
        tl.store(output + destination, result.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
    else:
        tl.store(output + destination, result.to(output.dtype.element_ty), valid)
    if SAVE:
        predicate = (~(value <= 0)) & valid
        if PACKED:
            words = _pack_bits(predicate, BLOCK)
            offsets = pid * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
            tl.store(saved + offsets, words, offsets < triton.cdiv(ROWS * COLS, 32))
        else:
            tl.store(saved + index, predicate, valid)
    if CHECK:
        bad = tl.max((valid & ~(tl.abs(activated) < float("inf"))).to(tl.int32), 0)
        if FP8 and not REUSE and not CURRENT:
            bad = tl.maximum(bad, tl.max((valid & ((scale <= 0) | ~(tl.abs(scale) < float("inf")))).to(tl.int32), 0))
        tl.store(errors + pid, bad)
    if OBSERVE:
        tl.store(observations + pid, tl.max(tl.abs(activated), 0))
    if SNAPSHOT:
        if pid == 0:
            s = tl.load(scales)
            tl.store(snapshot, s)
            tl.store(snapshot + 1, tl.div_rn(1.0, s))


@triton.jit
def _forward_blocks(data, inverse, output, saved, out_inverse, errors,
                    ROWS: tl.constexpr, COLS: tl.constexpr, SHAPE: tl.constexpr, CHANNELS: tl.constexpr,
                    META: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr, WIDTH: tl.constexpr,
                    GROUPS: tl.constexpr, SCALE_COLS: tl.constexpr, MAXIMUM: tl.constexpr,
                    SAVE: tl.constexpr, PACKED: tl.constexpr, CHECK: tl.constexpr, MASK_BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    tiles: tl.constexpr = triton.cdiv(COLS, WIDTH)
    row = pid // tiles * BR + tl.arange(0, BR)
    col = pid % tiles * WIDTH + tl.arange(0, WIDTH)
    valid = (row[:, None] < ROWS) & (col[None, :] < COLS)
    _, value = _load(data, inverse, row[:, None], col[None, :], valid, COLS, SHAPE, CHANNELS, META)
    activated = tl.where(value < 0, 0.0, value)
    grouped = tl.reshape(tl.abs(activated), (BR, GROUPS, BC))
    scale = _scale(tl.max(tl.max(grouped, 0), 1), MAXIMUM)
    inverse_offsets = pid // tiles * SCALE_COLS + pid % tiles * GROUPS + tl.arange(0, GROUPS)
    scale_valid = pid % tiles * GROUPS + tl.arange(0, GROUPS) < SCALE_COLS
    tl.store(out_inverse + inverse_offsets, tl.div_rn(1.0, scale), scale_valid)
    expanded = tl.reshape(tl.broadcast_to(scale[None, :, None], (BR, GROUPS, BC)), (BR, WIDTH))
    result = tl.minimum(tl.maximum(activated * expanded, -MAXIMUM), MAXIMUM)
    tl.store(output + row[:, None] * COLS + col[None, :],
             result.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
    if SAVE:
        predicate = (~(value <= 0)) & valid
        if PACKED:
            if COLS % 32 == 0 and WIDTH % 32 == 0:
                codes = tl.reshape(predicate.to(tl.int32), (BR, WIDTH // 32, 32))
                words = tl.sum(codes << tl.arange(0, 32)[None, None, :], 2)
                wc = pid % tiles * (WIDTH // 32) + tl.arange(0, WIDTH // 32)
                tl.store(saved + row[:, None] * (COLS // 32) + wc[None, :], words,
                         (row[:, None] < ROWS) & (wc[None, :] < COLS // 32))
            else:
                # Adjacent scale tiles can share a mask word at row tails.
                # Assign disjoint flat mask words to programs; reload only for
                # this unaligned case instead of racing partial-word stores.
                index = pid * MASK_BLOCK + tl.arange(0, MASK_BLOCK)
                mask_valid = index < ROWS * COLS
                _, mask_value = _load(data, inverse, index // COLS, index % COLS, mask_valid,
                                     COLS, SHAPE, CHANNELS, META)
                words = _pack_bits((~(mask_value <= 0)) & mask_valid, MASK_BLOCK)
                offsets = pid * (MASK_BLOCK // 32) + tl.arange(0, MASK_BLOCK // 32)
                tl.store(saved + offsets, words, offsets < triton.cdiv(ROWS * COLS, 32))
        else:
            tl.store(saved + row[:, None] * COLS + col[None, :], predicate, valid)
    if CHECK:
        tl.store(errors + pid, tl.max(tl.max((valid & ~(tl.abs(activated) < float("inf"))).to(tl.int32), 0), 0))


@triton.jit
def _partial_relu_amax(data, inverse, partials, ROWS: tl.constexpr, COLS: tl.constexpr,
                       SHAPE: tl.constexpr, CHANNELS: tl.constexpr, META: tl.constexpr,
                       BR: tl.constexpr, BC: tl.constexpr, SCALE_COLS: tl.constexpr,
                       PARTS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    group, part = pid // PARTS, pid % PARTS
    offsets = part * BLOCK + tl.arange(0, BLOCK)
    row = group // SCALE_COLS * BR + offsets // BC
    col = group % SCALE_COLS * BC + offsets % BC
    valid = (offsets < BR * BC) & (row < ROWS) & (col < COLS)
    _, value = _load(data, inverse, row, col, valid, COLS, SHAPE, CHANNELS, META)
    tl.store(partials + pid, tl.max(tl.abs(tl.where(value < 0, 0.0, value)), 0))
