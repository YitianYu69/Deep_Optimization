"""Normalize, apply affine parameters and pack xhat in the same tile."""

import triton
import triton.language as tl

from .common import _coordinates, _load_input
from .recenter import _record_error
from ..quantization.activation.affine import _affine_quantize_pack, _affine_quantize_pack_with_metadata


@triton.jit
def _apply(x, scales, weight, bias, mean, inverse, output, packed, step, minimum, low, high,
           C: tl.constexpr, S: tl.constexpr, G: tl.constexpr, BATCH: tl.constexpr,
           ELEMENTS: tl.constexpr, GROUPS: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr,
           BITS: tl.constexpr, PACK: tl.constexpr, PARTIAL: tl.constexpr, ENCODE: tl.constexpr,
           PARTS: tl.constexpr, HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr,
           META: tl.constexpr, OUT_STRIDES: tl.constexpr, BLOCK: tl.constexpr,
           center_shift=None, errors=None, DOMAINS: tl.constexpr = 1, RECENTER: tl.constexpr = False):
    group, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    domain, local_group = group // GROUPS, group % GROUPS
    offsets = part * BLOCK + tl.arange(0, BLOCK)
    position = local_group * GROUP + offsets
    valid = (offsets < GROUP) & (position < ELEMENTS)
    n, c, s = _coordinates(domain, position, C, S, G, BATCH)
    value = _load_input(x, scales, n, c, s, valid, C, S, META)
    h = (value - tl.load(mean + domain)) * tl.load(inverse + domain)
    if not ENCODE:
        y = h
        if HAS_WEIGHT:
            y = y * tl.load(weight + c, valid, other=1).to(tl.float32)
        if HAS_BIAS:
            y = y + tl.load(bias + c, valid, other=0).to(tl.float32)
        address = n * OUT_STRIDES[0] + c * OUT_STRIDES[1]
        for axis in tl.static_range(len(META[2])):
            address += s // META[3][axis] % META[2][axis] * OUT_STRIDES[axis + 2]
        tl.store(output + address, y, valid)
    if PACK:
        if PARTIAL:
            tl.store(low + group * PARTS + part, tl.min(tl.where(valid, h, float("inf")), 0))
            tl.store(high + group * PARTS + part, tl.max(tl.where(valid, h, float("-inf")), 0))
        else:
            if ENCODE:
                scale, start = tl.load(step + group), tl.load(minimum + group)
                words = _affine_quantize_pack_with_metadata(h, valid, scale, start, BITS, BLOCK)
            else:
                words, scale, start = _affine_quantize_pack(h, valid, BITS, BLOCK)
                tl.store(step + group, scale)
                tl.store(minimum + group, start)
            word_offset = part * (BLOCK // (32 // BITS)) + tl.arange(0, BLOCK // (32 // BITS))
            tl.store(packed + group * WORDS + word_offset, words, word_offset < WORDS)
            if RECENTER:
                _record_error(words, scale, start, h, valid, errors, center_shift,
                              domain, local_group * PARTS + part, DOMAINS, GROUPS * PARTS, BITS, BLOCK)
