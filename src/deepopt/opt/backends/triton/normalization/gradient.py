"""Final normalization dX retiled for fused FP8 gradient storage."""

import triton
import triton.language as tl

from .common import _load_input, _load_xhat_rstd


@triton.jit
def _input_gradient_fp8(dy, dy_scale, data, step, minimum, x, x_scale, mean, inverse, weight, sums,
                        output, scale, out_inverse, observations,
                        C: tl.constexpr, S: tl.constexpr, G: tl.constexpr, BATCH: tl.constexpr,
                        DOMAINS: tl.constexpr, ELEMENTS: tl.constexpr, BATCH_STATS: tl.constexpr,
                        COMPRESSED: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                        HAS_WEIGHT: tl.constexpr, DY_META: tl.constexpr, X_META: tl.constexpr,
                        ROWS: tl.constexpr, COLS: tl.constexpr, LAST: tl.constexpr, CHANNELS: tl.constexpr,
                        BR: tl.constexpr, BC: tl.constexpr, CURRENT: tl.constexpr, OBSERVE: tl.constexpr,
                        MAXIMUM: tl.constexpr, BLOCK: tl.constexpr,
                        center_shift=None, RECENTER: tl.constexpr = False):
    pid = tl.program_id(0).to(tl.int64)
    offset = tl.arange(0, BLOCK)
    if CURRENT:
        row = pid // triton.cdiv(COLS, BC) * BR + offset // BC
        col = pid % triton.cdiv(COLS, BC) * BC + offset % BC
        valid = (offset < BR * BC) & (row < ROWS) & (col < COLS)
    else:
        index = pid * BLOCK + offset
        row, col = index // COLS, index % COLS
        valid = index < ROWS * COLS
    if CHANNELS:
        n, c, s = row // S, col, row % S
    else:
        logical = row * COLS + col
        n, c, s = logical // (C * S), logical // S % C, logical % S
    if BATCH:
        domain, position = c, n * S + s
    else:
        domain, position = n * G + c // (C // G), c % (C // G) * S + s
    gradient = _load_input(dy, dy_scale, n, c, s, valid, C, S, DY_META)
    if HAS_WEIGHT:
        gradient *= tl.load(weight + c, valid, other=1).to(tl.float32)
    if BATCH_STATS:
        h, rstd = _load_xhat_rstd(data, step, minimum, x, x_scale, mean, inverse, domain, position, n, c, s, valid,
                       C, S, ELEMENTS, COMPRESSED, GROUP, WORDS, BITS, X_META, center_shift, RECENTER)
        gradient -= tl.load(sums + domain, valid, other=0) / ELEMENTS
        gradient -= h * tl.load(sums + DOMAINS + domain, valid, other=0) / ELEMENTS
    else:
        rstd = tl.load(inverse + domain, valid, other=0)
    value = gradient * rstd
    if CURRENT:
        amax = tl.max(tl.where(valid, tl.abs(value), 0), 0)
        multiplier = tl.where(amax == 0, 1.0, tl.div_rn(MAXIMUM, tl.maximum(amax, 1e-12)))
        tl.store(out_inverse + pid, tl.div_rn(1.0, multiplier))
    else:
        multiplier = tl.load(scale)
        if pid == 0:
            tl.store(out_inverse, tl.div_rn(1.0, multiplier))
        if OBSERVE:
            tl.store(observations + pid, tl.max(tl.where(valid, tl.abs(value), 0), 0))
    converted = tl.minimum(tl.maximum(value * multiplier, -MAXIMUM), MAXIMUM)
    tl.store(output + row * COLS + col, converted.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
