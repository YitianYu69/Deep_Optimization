"""Stable centered partial LayerNorm statistics; merges reuse norm Welford state."""

import triton
import triton.language as tl

from ...activation.common import _load
from .common import _row_statistics


@triton.jit
def _complete_statistics(x, x_inverse, mean, inverse, K: tl.constexpr, SHAPE: tl.constexpr,
                         META: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    mu, rstd = _row_statistics(x, x_inverse, row, K, SHAPE, META, EPS, BLOCK)
    tl.store(mean + row, mu)
    tl.store(inverse + row, rstd)


@triton.jit
def _partial_statistics(x, x_inverse, partial, M: tl.constexpr, K: tl.constexpr,
                        SHAPE: tl.constexpr, META: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    position = part * BLOCK + tl.arange(0, BLOCK)
    valid = position < K
    _, values = _load(x, x_inverse, row, position, valid, K, SHAPE, False, META)
    count = tl.sum(valid.to(tl.float32), 0)
    mu = tl.sum(tl.where(valid, values / count, 0.0), 0)
    delta = tl.where(valid, values - mu, 0.0)
    m2 = tl.sum(delta * delta, 0)
    index = row * PARTS + part
    tl.store(partial + index, count)
    tl.store(partial + M * PARTS + index, mu)
    tl.store(partial + 2 * M * PARTS + index, m2)
