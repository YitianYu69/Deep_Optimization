"""Register-local codec error and bounded normalization-domain reductions."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequantize_unpack_words(words, step, minimum, BITS: tl.constexpr, BLOCK: tl.constexpr):
    lanes = tl.arange(0, 32 // BITS)
    code = ((words[:, None] >> (lanes[None, :] * BITS)) & ((1 << BITS) - 1)).to(tl.float32)
    code = tl.reshape(code, (BLOCK,))
    value = code * step + minimum
    if BITS == 1:
        value = tl.where(step < 0, (minimum * 0.5 + code * -step) * 2, value)
    return value


@triton.jit
def _record_error(words, step, minimum, original, valid, partial, center_shift,
                  domain, part, DOMAINS: tl.constexpr, PARTS: tl.constexpr,
                  BITS: tl.constexpr, BLOCK: tl.constexpr):
    decoded = _dequantize_unpack_words(words, step, minimum, BITS, BLOCK)
    error = tl.sum(tl.where(valid, decoded - original, 0.0), 0)
    count = tl.sum(valid.to(tl.int32), 0)
    if PARTS == 1:
        tl.store(center_shift + domain, error / count)
    else:
        index = domain * PARTS + part
        tl.store(partial + index, error)
        tl.store(partial + DOMAINS * PARTS + index, count.to(tl.float32))


@triton.jit
def _correct_xhat(value, inverse, center_shift, domain, valid, RECENTER: tl.constexpr):
    # Affine tiles broadcast a row index across features; make pointer/mask
    # block shapes explicit rather than relying on scalar-load broadcasting.
    index = domain + valid.to(tl.int64) * 0
    rstd = tl.load(inverse + index, valid, other=0)
    if RECENTER:
        shift = tl.load(center_shift + index, valid, other=0)
        value = tl.where((tl.abs(rstd) < float("inf")) & (rstd > 0), value - shift, value)
    return value, rstd


@triton.jit
def _merge_center(partial, output, center_shift, DOMAINS: tl.constexpr, PARTS: tl.constexpr,
                    NEXT: tl.constexpr, FINAL: tl.constexpr, BLOCK: tl.constexpr):
    domain, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    offsets = part * BLOCK + tl.arange(0, BLOCK)
    index = domain * PARTS + offsets
    error = tl.sum(tl.load(partial + index, offsets < PARTS, other=0), 0)
    count = tl.sum(tl.load(partial + DOMAINS * PARTS + index, offsets < PARTS, other=0), 0)
    if FINAL:
        tl.store(center_shift + domain, error / count)
    else:
        tl.store(output + domain * NEXT + part, error)
        tl.store(output + DOMAINS * NEXT + domain * NEXT + part, count)


def reduce_center(partial, center_shift, domains, parts):
    """Return extra launches and total allocated partial bytes; no atomics."""
    launches = scratch = 0
    while parts > 1:
        block = min(256, triton.next_power_of_2(parts))
        nxt = triton.cdiv(parts, block)
        output = (torch.empty((2, domains, nxt), device=partial.device, dtype=torch.float32)
                  if nxt > 1 else center_shift)
        if nxt > 1:
            scratch += output.numel() * output.element_size()
        _merge_center[(domains, nxt)](partial, output, center_shift, domains, parts, nxt, nxt == 1, block,
                                      enable_fp_fusion=False)
        partial, parts = output, nxt
        launches += 1
    return launches, scratch
