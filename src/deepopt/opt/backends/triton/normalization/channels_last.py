"""BatchNorm tiles with contiguous channel lanes and unchanged per-channel codes."""

import triton
import triton.language as tl

from .common import _load_input, _load_xhat
from ..quantization.activation.affine import _affine_step, _affine_quantize_pack_with_metadata


@triton.jit
def _channels_last_stats(x, scales, partial, C: tl.constexpr, S: tl.constexpr,
                        ELEMENTS: tl.constexpr, PARTS: tl.constexpr, META: tl.constexpr,
                        BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    channel = tl.program_id(0).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C)
    part = tl.program_id(1).to(tl.int64)
    position = part * BLOCK_S + tl.arange(0, BLOCK_S)
    valid = (position[:, None] < ELEMENTS) & (channel[None, :] < C)
    value = _load_input(x, scales, position[:, None] // S, channel[None, :], position[:, None] % S,
                        valid, C, S, META)
    count = tl.sum(valid.to(tl.float32), 0)
    mean = tl.sum(value / tl.maximum(count[None, :], 1.0), 0)
    delta = tl.where(valid, value - mean[None, :], 0.0)
    m2 = tl.sum(delta * delta, 0)
    index = channel * PARTS + part
    tl.store(partial + index, count, channel < C)
    tl.store(partial + C * PARTS + index, mean, channel < C)
    tl.store(partial + 2 * C * PARTS + index, m2, channel < C)


@triton.jit
def _channels_last_apply(x, scales, weight, bias, mean, inverse, output,
                        packed, steps, minima, center, errors,
                        C: tl.constexpr, S: tl.constexpr, ELEMENTS: tl.constexpr,
                        GROUP: tl.constexpr, GROUPS: tl.constexpr, WORDS: tl.constexpr,
                        BITS: tl.constexpr, META: tl.constexpr, OUT_STRIDES: tl.constexpr,
                        HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr, PACK: tl.constexpr,
                        RECENTER: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    channel = tl.program_id(0).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C)
    group = tl.program_id(1).to(tl.int64)
    offset = tl.arange(0, BLOCK_S)
    position = group * GROUP + offset
    valid = (offset[:, None] < GROUP) & (position[:, None] < ELEMENTS) & (channel[None, :] < C)
    n, spatial = position // S, position % S
    value = _load_input(x, scales, n[:, None], channel[None, :], spatial[:, None], valid, C, S, META)
    mu = tl.load(mean + channel, channel < C, other=0.0)
    inv = tl.load(inverse + channel, channel < C, other=0.0)
    h = (value - mu[None, :]) * inv[None, :]
    y = h
    if HAS_WEIGHT:
        y = y * tl.load(weight + channel, channel < C, other=1.0).to(tl.float32)[None, :]
    if HAS_BIAS:
        y = y + tl.load(bias + channel, channel < C, other=0.0).to(tl.float32)[None, :]
    # BatchNorm2d's output can be NCHW or channels-last. No logical codec
    # reordering follows from the coalesced input/affine traversal.
    address = (n[:, None] * OUT_STRIDES[0] + channel[None, :] * OUT_STRIDES[1]
               + (spatial[:, None] // META[2][-1]) * OUT_STRIDES[2]
               + (spatial[:, None] % META[2][-1]) * OUT_STRIDES[3])
    tl.store(output + address, y, valid)
    if PACK:
        low = tl.min(tl.where(valid, h, float("inf")), 0)
        high = tl.max(tl.where(valid, h, -float("inf")), 0)
        step = _affine_step(low, high, BITS)
        transposed, mask = tl.trans(h), tl.trans(valid)
        words = _affine_quantize_pack_with_metadata(transposed, mask, step[:, None], low[:, None],
                                                   BITS, BLOCK_S, BATCH=BLOCK_C)
        index = channel * GROUPS + group
        word = tl.arange(0, BLOCK_S // (32 // BITS))
        tl.store(packed + index[:, None] * WORDS + word[None, :], words,
                 (channel[:, None] < C) & (word[None, :] < WORDS))
        tl.store(steps + index, step, channel < C)
        tl.store(minima + index, low, channel < C)
        if RECENTER:
            lane = tl.arange(0, 32 // BITS)
            codes = (words[:, :, None] >> (lane[None, None, :] * BITS)) & ((1 << BITS) - 1)
            decoded = tl.reshape(codes, (BLOCK_C, BLOCK_S)).to(tl.float32) * step[:, None] + low[:, None]
            if BITS == 1:
                decoded = tl.where(step[:, None] < 0,
                    (low[:, None] * 0.5 + tl.reshape(codes, (BLOCK_C, BLOCK_S)).to(tl.float32) * -step[:, None]) * 2,
                    decoded)
            error = tl.sum(tl.where(mask, decoded - transposed, 0.0), 1)
            count = tl.sum(mask.to(tl.float32), 1)
            if GROUPS == 1:
                tl.store(center + channel, error / tl.maximum(count, 1.0), channel < C)
            else:
                tl.store(errors + index, error, channel < C)
                tl.store(errors + C * GROUPS + index, count, channel < C)


@triton.jit
def _channels_last_sums(dy, dy_scale, data, step, minimum, x, x_scale, mean, inverse, partial,
                       C: tl.constexpr, S: tl.constexpr, ELEMENTS: tl.constexpr, PARTS: tl.constexpr,
                       COMPRESSED: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                       DY_META: tl.constexpr, X_META: tl.constexpr, NEED_H: tl.constexpr,
                       BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr,
                       center_shift=None, RECENTER: tl.constexpr = False):
    channel = tl.program_id(0).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C)
    part = tl.program_id(1).to(tl.int64)
    position = part * BLOCK_S + tl.arange(0, BLOCK_S)
    valid = (position[:, None] < ELEMENTS) & (channel[None, :] < C)
    n, spatial = position[:, None] // S, position[:, None] % S
    gradient = _load_input(dy, dy_scale, n, channel[None, :], spatial, valid, C, S, DY_META)
    weighted = tl.zeros((BLOCK_C,), tl.float32)
    if NEED_H:
        h = _load_xhat(data, step, minimum, x, x_scale, mean, inverse,
            channel[None, :], position[:, None], n, channel[None, :], spatial, valid,
            C, S, ELEMENTS, COMPRESSED, GROUP, WORDS, BITS, X_META, center_shift, RECENTER)
        weighted = tl.sum(tl.where(valid, gradient * h, 0.0), 0)
    index = channel * PARTS + part
    tl.store(partial + index, weighted, channel < C)
    tl.store(partial + C * PARTS + index, tl.sum(tl.where(valid, gradient, 0.0), 0), channel < C)


@triton.jit
def _channels_last_dx(dy, dy_scale, data, step, minimum, x, x_scale, mean, inverse, weight, sums, dx,
                     C: tl.constexpr, S: tl.constexpr, ELEMENTS: tl.constexpr, BATCH_STATS: tl.constexpr,
                     COMPRESSED: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                     HAS_WEIGHT: tl.constexpr, DY_META: tl.constexpr, X_META: tl.constexpr,
                     DX_STRIDES: tl.constexpr, WIDTH: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr,
                     center_shift=None, RECENTER: tl.constexpr = False):
    channel = tl.program_id(0).to(tl.int64) * BLOCK_C + tl.arange(0, BLOCK_C)
    position = tl.program_id(1).to(tl.int64) * BLOCK_S + tl.arange(0, BLOCK_S)
    valid = (position[:, None] < ELEMENTS) & (channel[None, :] < C)
    n, spatial = position[:, None] // S, position[:, None] % S
    value = _load_input(dy, dy_scale, n, channel[None, :], spatial, valid, C, S, DY_META)
    if BATCH_STATS:
        h = _load_xhat(data, step, minimum, x, x_scale, mean, inverse,
            channel[None, :], position[:, None], n, channel[None, :], spatial, valid,
            C, S, ELEMENTS, COMPRESSED, GROUP, WORDS, BITS, X_META, center_shift, RECENTER)
        s1 = tl.load(sums + channel, channel < C, other=0.)
        s0 = tl.load(sums + C + channel, channel < C, other=0.)
        value = value - s0[None, :] / ELEMENTS - h * s1[None, :] / ELEMENTS
    if HAS_WEIGHT:
        value *= tl.load(weight + channel, channel < C, other=1.).to(tl.float32)[None, :]
    value *= tl.load(inverse + channel, channel < C, other=0.)[None, :]
    address = (n * DX_STRIDES[0] + channel[None, :] * DX_STRIDES[1]
               + spatial // WIDTH * DX_STRIDES[2] + spatial % WIDTH * DX_STRIDES[3])
    tl.store(dx + address, value, valid)
