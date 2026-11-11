"""Lossless integer routing, in independently owned 32-value bitstreams."""

from dataclasses import dataclass
from math import prod

import torch
from torch._subclasses.fake_tensor import FakeTensor


@dataclass(frozen=True)
class PackedIndices:
    data: torch.Tensor
    shape: tuple
    value_count: int

    @property
    def bits(self):
        return (self.value_count - 1).bit_length()

    def __post_init__(self):
        if type(self.value_count) is not int or not 1 <= self.value_count <= 2**63:
            raise ValueError("value_count must be a positive integer no greater than 2**63")
        if not isinstance(self.shape, tuple) or any(type(v) is not int or v < 0 for v in self.shape):
            raise ValueError("index shape must contain nonnegative integers")
        bits = self.bits
        expected = ((prod(self.shape) * bits + 31) // 32 if bits <= 32 else prod(self.shape),)
        dtype = torch.int32 if bits <= 32 else torch.int64
        if (not isinstance(self.data, torch.Tensor) or self.data.dtype != dtype or self.data.layout != torch.strided
                or not self.data.is_contiguous() or tuple(self.data.shape) != expected):
            raise ValueError(f"packed indices require contiguous {dtype} storage of shape {expected}")


def empty_indices(shape, value_count, device):
    if type(value_count) is not int or not 1 <= value_count <= 2**63:
        raise ValueError("value_count must be a positive integer no greater than 2**63")
    bits = (value_count - 1).bit_length()
    data = torch.empty(((prod(shape) * bits + 31) // 32 if bits <= 32 else prod(shape),),
                       device=device, dtype=torch.int32 if bits <= 32 else torch.int64)
    return PackedIndices(data, tuple(shape), value_count)


@torch.no_grad()
def pack_indices(indices, value_count):
    if not isinstance(indices, torch.Tensor) or indices.dtype not in (torch.int32, torch.int64) or indices.layout != torch.strided:
        raise TypeError("indices must be a dense int32/int64 tensor")
    result = empty_indices(tuple(indices.shape), value_count, indices.device)
    fake = isinstance(indices, FakeTensor) or indices.device.type == "meta"
    if not fake and indices.numel():
        wide = indices.to(torch.int64)
        valid = ((wide >= 0) if value_count == 2**63 else (wide >= 0) & (wide < value_count)).all()
        if indices.device.type == "cuda":
            torch._assert_async(valid, "indices outside the specified value range")
        elif not valid:
            raise ValueError("indices outside the specified value range")
    if fake or not indices.numel() or not result.bits:
        return result
    if indices.device.type == "cuda":
        from ..backends.triton.quantization.activation.indices import pack

        with torch.cuda.device(indices.device):
            pack(indices, result)
        return result
    chunks = (indices.numel() + 31) // 32
    values = torch.zeros(chunks * 32, device=indices.device, dtype=torch.int64)
    values[:indices.numel()] = indices.reshape(-1)
    values = values.reshape(-1, 32)
    if result.bits > 32:
        result.data.copy_(values.reshape(-1)[:indices.numel()])
    else:
        bit = torch.arange(result.bits * 32, device=indices.device)
        digits = (values[:, bit // result.bits] >> (bit % result.bits)) & 1
        words = (digits.reshape(-1, result.bits, 32) << torch.arange(32, device=indices.device)).sum(-1).to(torch.int32)
        result.data.copy_(words.reshape(-1)[:result.data.numel()])
    return result


@torch.no_grad()
def unpack_indices(packed):
    if not isinstance(packed, PackedIndices):
        raise TypeError("packed must be PackedIndices")
    count, bits = prod(packed.shape), packed.bits
    result = torch.empty(packed.shape, dtype=torch.int64, device=packed.data.device)
    if isinstance(packed.data, FakeTensor) or result.device.type == "meta" or not count:
        return result
    if not bits:
        return result.zero_()
    if result.device.type == "cuda":
        from ..backends.triton.quantization.activation.indices import unpack

        with torch.cuda.device(result.device):
            unpack(packed, result)
        return result
    if bits > 32:
        return packed.data.reshape(-1)[:count].clone().reshape(packed.shape)
    words = torch.zeros(((count + 31) // 32) * bits, device=result.device, dtype=torch.int64)
    words[:packed.data.numel()] = packed.data.long()
    words = words.reshape(-1, bits)
    bit = torch.arange(32, device=result.device) * bits
    low = (words[:, bit // 32] & 0xffffffff) >> (bit % 32)
    high = words[:, (bit // 32 + 1).clamp_max(bits - 1)] & 0xffffffff
    value = (low | (high << (32 - bit % 32))) & ((1 << bits) - 1)
    return value.reshape(-1)[:count].reshape(packed.shape)
