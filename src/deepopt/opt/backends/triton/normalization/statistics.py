"""Stable hierarchical FP32 statistics and BatchNorm running-buffer updates."""

import triton
import triton.language as tl

from .common import _coordinates, _load_input


@triton.jit
def _partial_stats(x, scales, partial, C: tl.constexpr, S: tl.constexpr, G: tl.constexpr,
                   BATCH: tl.constexpr, DOMAINS: tl.constexpr, ELEMENTS: tl.constexpr,
                   PARTS: tl.constexpr, META: tl.constexpr, BLOCK: tl.constexpr):
    domain, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    position = part * BLOCK + tl.arange(0, BLOCK)
    valid = position < ELEMENTS
    n, c, s = _coordinates(domain, position, C, S, G, BATCH)
    value = _load_input(x, scales, n, c, s, valid, C, S, META)
    count = tl.sum(valid.to(tl.float32), 0)
    mean = tl.sum(value / count, 0)
    delta = tl.where(valid, value - mean, 0)
    m2 = tl.sum(delta * delta, 0)
    index = domain * PARTS + part
    tl.store(partial + index, count)
    tl.store(partial + DOMAINS * PARTS + index, mean)
    tl.store(partial + 2 * DOMAINS * PARTS + index, m2)


@triton.jit
def _merge_stats(partial, output, DOMAINS: tl.constexpr, PARTS: tl.constexpr,
                 NEXT: tl.constexpr, BLOCK: tl.constexpr):
    domain, part = tl.program_id(0).to(tl.int64), tl.program_id(1)
    offset = part * BLOCK + tl.arange(0, BLOCK)
    index = domain * PARTS + offset
    counts = tl.load(partial + index, offset < PARTS, other=0)
    means = tl.load(partial + DOMAINS * PARTS + index, offset < PARTS, other=0)
    m2s = tl.load(partial + 2 * DOMAINS * PARTS + index, offset < PARTS, other=0)
    count = tl.sum(counts, 0)
    mean = tl.sum((counts / count) * means, 0)
    delta = tl.where(offset < PARTS, means - mean, 0)
    m2 = tl.sum(m2s + counts * delta * delta, 0)
    target = domain * NEXT + part
    tl.store(output + target, count)
    tl.store(output + DOMAINS * NEXT + target, mean)
    tl.store(output + 2 * DOMAINS * NEXT + target, m2)


@triton.jit
def _finish_stats(partial, mean, inverse, running_mean, running_var, momentum,
                  DOMAINS: tl.constexpr, EPS: tl.constexpr, BATCH_STATS: tl.constexpr,
                  UPDATE: tl.constexpr, MOMENTUM_TENSOR: tl.constexpr, FACTOR: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = index < DOMAINS
    if BATCH_STATS:
        count = tl.load(partial + index, valid, other=1)
        mu = tl.load(partial + DOMAINS + index, valid, other=0)
        m2 = tl.load(partial + 2 * DOMAINS + index, valid, other=0)
        variance = m2 / count
        if UPDATE:
            factor = tl.load(momentum) if MOMENTUM_TENSOR else FACTOR
            old_mean = tl.load(running_mean + index, valid, other=0).to(tl.float32)
            old_var = tl.load(running_var + index, valid, other=0).to(tl.float32)
            tl.store(running_mean + index, old_mean * (1 - factor) + mu * factor, valid)
            tl.store(running_var + index, old_var * (1 - factor) + (m2 / (count - 1)) * factor, valid)
    else:
        mu = tl.load(running_mean + index, valid, other=0).to(tl.float32)
        variance = tl.load(running_var + index, valid, other=0).to(tl.float32)
    tl.store(mean + index, mu, valid)
    tl.store(inverse + index, tl.rsqrt(variance + EPS), valid)
