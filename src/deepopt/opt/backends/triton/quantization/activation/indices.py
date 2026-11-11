"""Exact routing packing and tile-local extraction; no quantization."""

import triton
import triton.language as tl


@triton.jit
def _pack_codes(codes, output, chunk, BITS: tl.constexpr, WORD_BLOCK: tl.constexpr, TOTAL: tl.constexpr):
    if BITS > 32:
        index = chunk * 32 + tl.arange(0, 32)
        tl.store(output + index, codes.to(tl.int64), index < TOTAL)
    elif BITS > 0:
        word = tl.arange(0, WORD_BLOCK)
        lane = tl.arange(0, 32)
        bit = word[:, None] * 32 + lane[None, :]
        source = tl.minimum(bit // BITS, 31)
        gathered = tl.gather(codes, tl.reshape(source, (WORD_BLOCK * 32,)), 0)
        gathered = tl.reshape(gathered, (WORD_BLOCK, 32))
        digit = ((gathered >> (bit % BITS)) & 1).to(tl.int32)
        value = tl.sum(tl.where(bit < 32 * BITS, digit << lane[None, :], 0), 1)
        tl.store(output + chunk * BITS + word, value, (word < BITS) & (chunk * BITS + word < triton.cdiv(TOTAL * BITS, 32)))


@triton.jit
def _unpack_codes(data, index, valid, BITS: tl.constexpr):
    if BITS == 0:
        return tl.full(index.shape, 0, tl.int64)
    elif BITS > 32:
        return tl.load(data + index, valid, other=0).to(tl.int64)
    else:
        bit = index % 32 * BITS
        word = index // 32 * BITS + bit // 32
        low = tl.load(data + word, valid, other=0).to(tl.uint32).to(tl.uint64)
        high = tl.load(data + word + 1, valid & (bit % 32 + BITS > 32), other=0).to(tl.uint32).to(tl.uint64)
        return (((low >> (bit % 32)) | (high << (32 - bit % 32))) & ((1 << BITS) - 1)).to(tl.int64)


@triton.jit
def _pack(input, output, N: tl.constexpr, SHAPE: tl.constexpr, STRIDES: tl.constexpr,
           BITS: tl.constexpr, WORD_BLOCK: tl.constexpr):
    chunk = tl.program_id(0).to(tl.int64)
    index = chunk * 32 + tl.arange(0, 32)
    remaining, address = index, tl.full((32,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        address += remaining % SHAPE[axis] * STRIDES[axis]
        remaining = remaining // SHAPE[axis]
    codes = tl.load(input + address, index < N, other=0).to(tl.int64)
    _pack_codes(codes, output, chunk, BITS, WORD_BLOCK, N)


@triton.jit
def _unpack(data, output, N: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = _unpack_codes(data, index, index < N, BITS)
    tl.store(output + index, value, index < N)


def pack(value, result):
    if result.bits and value.numel():
        _pack[(triton.cdiv(value.numel(), 32),)](value, result.data, value.numel(), tuple(value.shape), value.stride(),
                                                result.bits, triton.next_power_of_2(min(result.bits, 32)))


def unpack(value, result):
    if result.numel():
        _unpack[(triton.cdiv(result.numel(), 256),)](value.data, result, result.numel(), value.bits, 256)
