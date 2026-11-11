"""Logical addressing and fused input/xhat decoding for normalization."""

from math import prod

import triton
import triton.language as tl

from .recenter import _correct_xhat

from deepopt.opt.quantization.activation_tensor import FP8Activation


def input_arguments(x):
    """Return actual storage tensors and pointer-free logical mapping."""
    spatial = tuple(x.shape[2:])
    divisors = tuple(prod(spatial[axis + 1:]) for axis in range(len(spatial)))
    if isinstance(x, FP8Activation):
        q = x.quantized()
        br, bc = q.block_shape or tuple(q.data.shape)
        meta = (1 if x._activation_layout == "channels" else 2, tuple(x.stride()), spatial, divisors,
                *q.data.stride(), br, bc, *q.scale_inv.stride(), x.shape[-1])
        return q.data, q.scale_inv, meta
    return x, x, (0, tuple(x.stride()), spatial, divisors, 0, 0, 1, 1, 0, 0, x.shape[-1])


@triton.jit
def _coordinates(domain, position, C: tl.constexpr, S: tl.constexpr, G: tl.constexpr, BATCH: tl.constexpr):
    if BATCH:
        n, c = position // S, domain
    else:
        n, c = domain // G, domain % G * (C // G) + position // S
    return n, c, position % S


@triton.jit
def _load_input(data, scales, n, c, s, valid, C: tl.constexpr, S: tl.constexpr, META: tl.constexpr):
    if META[0] == 0:
        address = n * META[1][0] + c * META[1][1]
        for axis in tl.static_range(len(META[2])):
            address += (s // META[3][axis] % META[2][axis]) * META[1][axis + 2]
        return tl.load(data + address, valid, other=0).to(tl.float32)
    else:
        if META[0] == 1:
            row, col = n * S + s, c
        else:
            logical = (n * C + c) * S + s
            row, col = logical // META[10], logical % META[10]
        q = tl.load(data + row * META[4] + col * META[5], valid, other=0).to(tl.float32)
        inverse = tl.load(scales + row // META[6] * META[8] + col // META[7] * META[9], valid, other=0)
        return q * inverse


@triton.jit
def _load_xhat(data, step, minimum, source, scales, mean, inverse, domain, position, n, c, s, valid,
               C: tl.constexpr, S: tl.constexpr, ELEMENTS: tl.constexpr, COMPRESSED: tl.constexpr,
               GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr, X_META: tl.constexpr,
               center_shift=None, RECENTER: tl.constexpr = False):
    if COMPRESSED:
        group = domain * tl.cdiv(ELEMENTS, GROUP) + position // GROUP
        offset = position % GROUP
        word = tl.load(data + group * WORDS + offset // (32 // BITS), valid, other=0)
        code = ((word >> (offset % (32 // BITS) * BITS)) & ((1 << BITS) - 1)).to(tl.float32)
        scale = tl.load(step + group, valid, other=0)
        low = tl.load(minimum + group, valid, other=0)
        value = code * scale + low
        if BITS == 1:
            value = tl.where(scale < 0, (low * 0.5 + code * -scale) * 2, value)
        if RECENTER:
            value, _ = _correct_xhat(value, inverse, center_shift, domain, valid, True)
        return value
    else:
        value = _load_input(source, scales, n, c, s, valid, C, S, X_META)
        index = domain + valid.to(tl.int64) * 0
        return (value - tl.load(mean + index, valid, other=0)) * tl.load(inverse + index, valid, other=0)


@triton.jit
def _load_xhat_rstd(data, step, minimum, source, scales, mean, inverse, domain, position, n, c, s, valid,
                    C: tl.constexpr, S: tl.constexpr, ELEMENTS: tl.constexpr, COMPRESSED: tl.constexpr,
                    GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr, X_META: tl.constexpr,
                    center_shift=None, RECENTER: tl.constexpr = False):
    h = _load_xhat(data, step, minimum, source, scales, mean, inverse, domain, position, n, c, s, valid,
                    C, S, ELEMENTS, COMPRESSED, GROUP, WORDS, BITS, X_META)
    return _correct_xhat(h, inverse, center_shift, domain, valid, RECENTER)
