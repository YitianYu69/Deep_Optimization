"""Lossless boolean bit packing; no quantization or scaling metadata."""

from math import prod

import torch
import triton
import triton.language as tl


@triton.jit
def _mask_offsets(positions, SHAPE: tl.constexpr, STRIDES: tl.constexpr, CONTIGUOUS: tl.constexpr):
    if CONTIGUOUS:
        return positions
    else:
        remaining = positions
        address = tl.zeros(positions.shape, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            address += (remaining % SHAPE[axis]) * STRIDES[axis]
            remaining = remaining // SHAPE[axis]
        return address


@triton.jit
def pack_boolean_mask_kernel(mask, packed, COUNT: tl.constexpr, WORDS: tl.constexpr,
                             SHAPE: tl.constexpr, STRIDES: tl.constexpr, CONTIGUOUS: tl.constexpr,
                             BLOCK_WORDS: tl.constexpr):
    words = tl.program_id(0).to(tl.int64) * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
    shifts = tl.arange(0, 32)
    positions = words[:, None] * 32 + shifts[None, :]
    address = _mask_offsets(positions, SHAPE, STRIDES, CONTIGUOUS)
    values = tl.load(mask + address, positions < COUNT, other=0).to(tl.int32)
    bits = tl.sum(values << shifts[None, :], axis=1)
    tl.store(packed + words, bits, words < WORDS)


@triton.jit
def unpack_boolean_mask_kernel(packed, output, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    positions = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    words = tl.load(packed + positions // 32, positions < COUNT, other=0)
    values = ((words >> (positions % 32)) & 1) != 0
    tl.store(output + positions, values, positions < COUNT)


def pack_boolean_mask(mask):
    """Internal CUDA entry point; the public codec validates dtype and layout."""
    count = mask.numel()
    words = triton.cdiv(count, 32)
    with torch.cuda.device(mask.device):
        packed = torch.empty(words, dtype=torch.int32, device=mask.device)
        if count:
            pack_boolean_mask_kernel[(triton.cdiv(words, 128),)](
                mask, packed, count, words, tuple(mask.shape), tuple(mask.stride()), mask.is_contiguous(),
                BLOCK_WORDS=128, num_warps=4,
            )
    return packed


def unpack_boolean_mask(packed, shape):
    """Internal CUDA entry point producing contiguous, shape-preserving bools."""
    count = prod(shape)
    with torch.cuda.device(packed.device):
        output = torch.empty(shape, dtype=torch.bool, device=packed.device)
        if count:
            unpack_boolean_mask_kernel[(triton.cdiv(count, 1024),)](
                packed, output, count, BLOCK=1024, num_warps=4,
            )
    return output
