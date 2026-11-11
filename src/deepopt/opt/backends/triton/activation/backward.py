"""Fused upstream-gradient decoding, mask extraction and gradient output."""

import triton
import triton.language as tl

from .common import _load


@triton.jit
def _backward(data, inverse, saved, output, ROWS: tl.constexpr, COLS: tl.constexpr,
              SHAPE: tl.constexpr, CHANNELS: tl.constexpr, META: tl.constexpr,
              PACKED: tl.constexpr, REUSE: tl.constexpr, FP8: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = index < ROWS * COLS
    raw, value = _load(data, inverse, index // COLS, index % COLS, valid, COLS, SHAPE, CHANNELS, META)
    if PACKED:
        word = tl.load(saved + index // 32, valid, other=0)
        predicate = ((word >> (index % 32)) & 1) != 0
    else:
        predicate = tl.load(saved + index, valid, other=0) != 0
    result = tl.where(predicate, raw if REUSE else value, 0.0)
    if FP8:
        tl.store(output + index, result.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
    else:
        tl.store(output + index, result.to(output.dtype.element_ty), valid)
