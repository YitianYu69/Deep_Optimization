"""Inline FP8 gradient epilogues and scale-aware bias-gradient reduction."""

import triton
import triton.language as tl


@triton.jit
def _gradient_tile(values, rows, cols, valid, scale, inverse, observations,
                   M: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BR: tl.constexpr, BC: tl.constexpr, MAXIMUM: tl.constexpr,
                   CURRENT: tl.constexpr, OBSERVE: tl.constexpr):
    values = tl.where(valid, values, 0.0)
    if CURRENT:
        grouped = tl.reshape(tl.abs(values), (BM // BR, BR, BN // BC, BC))
        amax = tl.max(tl.max(grouped, 3), 1)
        multiplier = tl.where(amax == 0, 1.0, tl.div_rn(MAXIMUM, tl.maximum(amax, 1e-12)))
        rr = tl.min(rows, 0).to(tl.int64) + tl.arange(0, BM // BR) * BR
        cc = tl.min(cols, 0).to(tl.int64) + tl.arange(0, BN // BC) * BC
        tl.store(inverse + rr[:, None] // BR * triton.cdiv(N, BC) + cc[None, :] // BC,
                 tl.div_rn(1.0, multiplier), (rr[:, None] < M) & (cc[None, :] < N))
        multiplier = tl.reshape(tl.broadcast_to(multiplier[:, None, :, None], (BM // BR, BR, BN // BC, BC)), (BM, BN))
    else:
        multiplier = tl.load(scale)
        if tl.program_id(0) == 0:
            tl.store(inverse, tl.div_rn(1.0, multiplier))
        if OBSERVE:
            tl.store(observations + tl.program_id(0), tl.max(tl.max(tl.abs(values), 0), 0))
    return tl.minimum(tl.maximum(values * multiplier, -MAXIMUM), MAXIMUM)
