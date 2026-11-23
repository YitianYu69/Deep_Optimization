"""Opt-in single-domain statistics, native output and normalized-state packing."""

import triton
import triton.language as tl

from .common import _coordinates, _load_input
from .activation import _activate
from .layer.common import _quantize_pack_xhat_rows, _statistics_from_values


@triton.jit
def _small_forward(x, scales, weight, bias, running_mean, running_var, momentum,
                   output, mean, inverse, data, step, minimum, center_shift,
                   C: tl.constexpr, S: tl.constexpr, G: tl.constexpr, BATCH: tl.constexpr,
                   ELEMENTS: tl.constexpr, META: tl.constexpr, OUT_STRIDES: tl.constexpr,
                   HAS_WEIGHT: tl.constexpr, HAS_BIAS: tl.constexpr, UPDATE: tl.constexpr,
                   TENSOR_MOMENTUM: tl.constexpr, MOMENTUM: tl.constexpr, EPS: tl.constexpr,
                   PACK: tl.constexpr, GROUP: tl.constexpr, GROUPS: tl.constexpr,
                   PAD_GROUPS: tl.constexpr, PAD: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                   RECENTER: tl.constexpr, BLOCK: tl.constexpr,
                   preactivation=None, ACTIVATION: tl.constexpr = "", APPROXIMATE: tl.constexpr = "none",
                   SAVE_PREACTIVATION: tl.constexpr = False):
    domain = tl.program_id(0).to(tl.int64)
    position = tl.arange(0, BLOCK)
    valid = position < ELEMENTS
    n, c, s = _coordinates(domain, position, C, S, G, BATCH)
    values = _load_input(x, scales, n, c, s, valid, C, S, META)
    mu, rstd = _statistics_from_values(values, valid, ELEMENTS, EPS)
    tl.store(mean + domain, mu)
    tl.store(inverse + domain, rstd)
    if UPDATE:
        factor = tl.load(momentum) if TENSOR_MOMENTUM else MOMENTUM
        # Derive variance with the same centered reduction, not by inverting
        # rstd (which would amplify rounding for nearly constant domains).
        delta = tl.where(valid, values - mu, 0.0)
        m2 = tl.sum(delta * delta, 0)
        unbiased = m2 / (ELEMENTS - 1)
        tl.store(running_mean + domain, tl.load(running_mean + domain) * (1.0 - factor) + mu * factor)
        tl.store(running_var + domain, tl.load(running_var + domain) * (1.0 - factor) + unbiased * factor)
    h = (values - mu) * rstd
    y = h
    if HAS_WEIGHT:
        y = y * tl.load(weight + c, valid, other=1.0).to(tl.float32)
    if HAS_BIAS:
        y = y + tl.load(bias + c, valid, other=0.0).to(tl.float32)
    address = n * OUT_STRIDES[0] + c * OUT_STRIDES[1]
    for axis in tl.static_range(len(META[2])):
        address += s // META[3][axis] % META[2][axis] * OUT_STRIDES[axis + 2]
    y = _activate(y, output, preactivation, address, valid, ACTIVATION, APPROXIMATE, SAVE_PREACTIVATION)
    tl.store(output + address, y, valid)
    if PACK:
        group = tl.arange(0, PAD_GROUPS)
        offset = tl.arange(0, PAD)
        logical = group[:, None] * GROUP + offset[None, :]
        mask = (logical < ELEMENTS) & (offset[None, :] < GROUP)
        if GROUP == PAD and PAD_GROUPS * PAD == BLOCK:
            normalized = tl.reshape(h, (PAD_GROUPS, PAD))
        else:
            nn, cc, ss = _coordinates(domain, logical, C, S, G, BATCH)
            values2 = _load_input(x, scales, nn, cc, ss, mask, C, S, META)
            normalized = (values2 - mu) * rstd
        _quantize_pack_xhat_rows(normalized, mask, data, step, minimum, domain,
                                GROUPS, PAD_GROUPS, PAD, WORDS, BITS, center_shift, RECENTER)
