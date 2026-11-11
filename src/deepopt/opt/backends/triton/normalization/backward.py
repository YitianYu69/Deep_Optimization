"""Tile-local xhat decode, normalization reductions and affine gradients."""

import triton
import triton.language as tl

from .common import _coordinates, _load_input, _load_xhat, _load_xhat_rstd


@triton.jit
def _partial_backward(dy, dy_scale, data, step, minimum, x, x_scale, mean, inverse, weight, partial,
                      C: tl.constexpr, S: tl.constexpr, G: tl.constexpr, BATCH: tl.constexpr,
                      DOMAINS: tl.constexpr, ELEMENTS: tl.constexpr, PARTS: tl.constexpr,
                      COMPRESSED: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                      HAS_WEIGHT: tl.constexpr, DY_META: tl.constexpr, X_META: tl.constexpr, BLOCK: tl.constexpr,
                      center_shift=None, RECENTER: tl.constexpr = False):
    domain, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    position = part * BLOCK + tl.arange(0, BLOCK)
    valid = position < ELEMENTS
    n, c, s = _coordinates(domain, position, C, S, G, BATCH)
    gradient = _load_input(dy, dy_scale, n, c, s, valid, C, S, DY_META)
    h = _load_xhat(data, step, minimum, x, x_scale, mean, inverse, domain, position, n, c, s, valid,
                   C, S, ELEMENTS, COMPRESSED, GROUP, WORDS, BITS, X_META, center_shift, RECENTER)
    if HAS_WEIGHT:
        gradient = gradient * tl.load(weight + c, valid, other=1).to(tl.float32)
    index = domain * PARTS + part
    tl.store(partial + index, tl.sum(tl.where(valid, gradient, 0), 0))
    tl.store(partial + DOMAINS * PARTS + index, tl.sum(tl.where(valid, gradient * h, 0), 0))


@triton.jit
def _merge_sums(partial, output, DOMAINS: tl.constexpr, PARTS: tl.constexpr,
                NEXT: tl.constexpr, BLOCK: tl.constexpr):
    domain, part = tl.program_id(0).to(tl.int64), tl.program_id(1)
    offset = part * BLOCK + tl.arange(0, BLOCK)
    index = domain * PARTS + offset
    a = tl.load(partial + index, offset < PARTS, other=0)
    b = tl.load(partial + DOMAINS * PARTS + index, offset < PARTS, other=0)
    tl.store(output + domain * NEXT + part, tl.sum(a, 0))
    tl.store(output + DOMAINS * NEXT + domain * NEXT + part, tl.sum(b, 0))


@triton.jit
def _input_gradient(dy, dy_scale, data, step, minimum, x, x_scale, mean, inverse, weight, sums, dx,
                     C: tl.constexpr, S: tl.constexpr, G: tl.constexpr, BATCH: tl.constexpr,
                     DOMAINS: tl.constexpr, ELEMENTS: tl.constexpr, BATCH_STATS: tl.constexpr,
                     COMPRESSED: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                     HAS_WEIGHT: tl.constexpr, DY_META: tl.constexpr, X_META: tl.constexpr, BLOCK: tl.constexpr,
                     center_shift=None, RECENTER: tl.constexpr = False):
    domain, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    position = part * BLOCK + tl.arange(0, BLOCK)
    valid = position < ELEMENTS
    n, c, s = _coordinates(domain, position, C, S, G, BATCH)
    gradient = _load_input(dy, dy_scale, n, c, s, valid, C, S, DY_META)
    if HAS_WEIGHT:
        gradient = gradient * tl.load(weight + c, valid, other=1).to(tl.float32)
    if BATCH_STATS:
        h, rstd = _load_xhat_rstd(data, step, minimum, x, x_scale, mean, inverse, domain, position, n, c, s, valid,
                       C, S, ELEMENTS, COMPRESSED, GROUP, WORDS, BITS, X_META, center_shift, RECENTER)
        gradient = gradient - tl.load(sums + domain) / ELEMENTS - h * tl.load(sums + DOMAINS + domain) / ELEMENTS
    else:
        rstd = tl.load(inverse + domain)
    value = gradient * rstd
    tl.store(dx + (n * C + c) * S + s, value, valid)


@triton.jit
def _partial_affine(dy, dy_scale, data, step, minimum, x, x_scale, mean, inverse, partial,
                     N: tl.constexpr, C: tl.constexpr, S: tl.constexpr, G: tl.constexpr, BATCH: tl.constexpr,
                     ELEMENTS: tl.constexpr, PARTS: tl.constexpr, COMPRESSED: tl.constexpr,
                     GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                     NEED_DW: tl.constexpr, NEED_DB: tl.constexpr,
                     DY_META: tl.constexpr, X_META: tl.constexpr, BLOCK: tl.constexpr,
                     center_shift=None, RECENTER: tl.constexpr = False):
    c, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    offset = part * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N * S
    n, s = offset // S, offset % S
    gradient = _load_input(dy, dy_scale, n, c, s, valid, C, S, DY_META)
    if NEED_DW:
        if BATCH:
            domain, position = c, offset
        else:
            domain, position = n * G + c // (C // G), c % (C // G) * S + s
        h = _load_xhat(data, step, minimum, x, x_scale, mean, inverse, domain, position, n, c, s, valid,
                       C, S, ELEMENTS, COMPRESSED, GROUP, WORDS, BITS, X_META, center_shift, RECENTER)
        tl.store(partial + c * PARTS + part, tl.sum(tl.where(valid, gradient * h, 0), 0))
    else:
        tl.store(partial + c * PARTS + part, 0)
    tl.store(partial + C * PARTS + c * PARTS + part,
             tl.sum(tl.where(valid, gradient, 0), 0) if NEED_DB else 0)


@triton.jit
def _finish_affine(sums, dw, db, C: tl.constexpr, NEED_DW: tl.constexpr, NEED_DB: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if NEED_DW:
        tl.store(dw + c, tl.load(sums + c, c < C, other=0), c < C)
    if NEED_DB:
        tl.store(db + c, tl.load(sums + C + c, c < C, other=0), c < C)
