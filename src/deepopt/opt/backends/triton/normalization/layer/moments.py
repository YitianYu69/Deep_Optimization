"""Packed-word LayerNorm moments with bounded, deterministic wide-row merges."""

import torch
import triton
import triton.language as tl

from ..recenter import _dequantize_unpack_words


@triton.jit
def _store_layer_moments(center_shift, moment_gain, row, mean, m2, target, count):
    # A constant decoded row has lost its direction. Zero is finite and centered;
    # inventing a direction or an infinite scale cannot recover that information.
    gain = tl.where(m2 > 0, tl.sqrt(target / tl.maximum(m2, count * 1e-12)), 0.0)
    tl.store(center_shift + row, mean)
    tl.store(moment_gain + row, gain)


@triton.jit
def _record_layer_moments(words, step, minimum, original, valid, partial, center_shift, moment_gain,
                          row, part, ROWS: tl.constexpr, PARTS: tl.constexpr,
                          BITS: tl.constexpr, BLOCK: tl.constexpr):
    decoded = _dequantize_unpack_words(words, step, minimum, BITS, BLOCK)
    count = tl.sum(valid.to(tl.int32), 0).to(tl.float32)
    mean = tl.sum(tl.where(valid, decoded, 0.0), 0) / tl.maximum(count, 1.0)
    delta = tl.where(valid, decoded - mean, 0.0)
    m2 = tl.sum(delta * delta, 0)
    target = tl.sum(tl.where(valid, original * original, 0.0), 0)
    if PARTS == 1:
        _store_layer_moments(center_shift, moment_gain, row, mean, m2, target, count)
    else:
        index = row * PARTS + part
        tl.store(partial + index, mean)
        tl.store(partial + ROWS * PARTS + index, m2)
        tl.store(partial + 2 * ROWS * PARTS + index, target)
        tl.store(partial + 3 * ROWS * PARTS + index, count)


@triton.jit
def _merge_layer_moments(partial, output, center_shift, moment_gain,
                         ROWS: tl.constexpr, PARTS: tl.constexpr, NEXT: tl.constexpr,
                         FINAL: tl.constexpr, BLOCK: tl.constexpr):
    row, part = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    offsets = part * BLOCK + tl.arange(0, BLOCK)
    index = row * PARTS + offsets
    valid = offsets < PARTS
    means = tl.load(partial + index, valid, other=0)
    m2s = tl.load(partial + ROWS * PARTS + index, valid, other=0)
    targets = tl.load(partial + 2 * ROWS * PARTS + index, valid, other=0)
    counts = tl.load(partial + 3 * ROWS * PARTS + index, valid, other=0)
    count = tl.sum(counts, 0)
    mean = tl.sum(means * counts, 0) / tl.maximum(count, 1.0)
    delta = means - mean
    # Merge centered M2 rather than subtracting two large raw moments.
    m2 = tl.sum(m2s + counts * delta * delta, 0)
    target = tl.sum(targets, 0)
    if FINAL:
        _store_layer_moments(center_shift, moment_gain, row, mean, m2, target, count)
    else:
        destination = row * NEXT + part
        tl.store(output + destination, mean)
        tl.store(output + ROWS * NEXT + destination, m2)
        tl.store(output + 2 * ROWS * NEXT + destination, target)
        tl.store(output + 3 * ROWS * NEXT + destination, count)


def reduce_layer_moments(partial, center_shift, moment_gain, rows, parts):
    """Merge bounded codec tiles without atomics or a dense decoded activation."""
    launches = scratch = 0
    while parts > 1:
        block = min(256, triton.next_power_of_2(parts))
        nxt = triton.cdiv(parts, block)
        output = (torch.empty((4, rows, nxt), device=partial.device, dtype=torch.float32)
                  if nxt > 1 else center_shift)
        if nxt > 1:
            scratch += output.numel() * output.element_size()
        _merge_layer_moments[(rows, nxt)](partial, output, center_shift, moment_gain,
            rows, parts, nxt, nxt == 1, block, enable_fp_fusion=False)
        partial, parts = output, nxt
        launches += 1
    return launches, scratch
