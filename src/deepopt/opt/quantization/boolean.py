"""Lossless one-bit storage for boolean masks, independent of compute precision."""

from dataclasses import dataclass
from math import prod

import torch
from torch._subclasses.fake_tensor import FakeTensor


@dataclass(frozen=True)
class PackedBooleanMask:
    """Logical row-major mask bits in contiguous int32 words, low bit first.

    Only the original shape is needed to reconstruct a contiguous bool tensor.
    Unused high bits of the final word are zero when produced by this codec.
    """

    data: torch.Tensor
    shape: tuple[int, ...]

    def __post_init__(self):
        if (not isinstance(self.shape, tuple)
                or any(type(size) is not int or size < 0 for size in self.shape)):
            raise ValueError("shape must be a tuple of nonnegative integers")
        if not isinstance(self.data, torch.Tensor) or self.data.dtype != torch.int32:
            raise TypeError("packed mask data must be a torch.int32 tensor")
        if self.data.layout != torch.strided:
            raise ValueError("packed mask data must have strided layout")
        words = (prod(self.shape) + 31) // 32
        if tuple(self.data.shape) != (words,):
            raise ValueError(f"packed mask data must have shape {(words,)}")
        if not self.data.is_contiguous():
            raise ValueError("packed mask data must be contiguous")


@torch.no_grad()
def pack_boolean_mask(mask: torch.Tensor) -> PackedBooleanMask:
    """Pack bool values losslessly; CUDA reads input strides without copying.

    Scalars, empty tensors, transposes, slices and broadcast views are accepted.
    The packed result owns its storage and retains no reference to the input.
    """
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError("mask must be a torch.bool tensor")
    if mask.layout != torch.strided or mask.is_nested:
        raise ValueError("mask must have strided, non-nested layout")
    shape = tuple(mask.shape)
    words = (mask.numel() + 31) // 32
    if isinstance(mask, FakeTensor) or mask.device.type == "meta" or words == 0:
        data = torch.empty((words,), dtype=torch.int32, device=mask.device)
    elif mask.device.type == "cuda":
        from deepopt.opt.backends.triton.quantization.activation.boolean import pack_boolean_mask as backend_pack

        data = backend_pack(mask)
    else:
        # CPU reference: wide arithmetic preserves the sign bit of int32 words.
        values = torch.zeros(words * 32, dtype=torch.int64, device=mask.device)
        values[:mask.numel()] = mask.reshape(-1)
        shifts = torch.arange(32, dtype=torch.int64, device=mask.device)
        data = (values.reshape(words, 32) << shifts).sum(dim=1).to(torch.int32)
    return PackedBooleanMask(data, shape)


@torch.no_grad()
def unpack_boolean_mask(packed: PackedBooleanMask) -> torch.Tensor:
    """Unpack a mask into an independent, contiguous torch.bool tensor."""
    if not isinstance(packed, PackedBooleanMask):
        raise TypeError("packed must be a PackedBooleanMask")
    data, shape = packed.data, packed.shape
    if isinstance(data, FakeTensor) or data.device.type == "meta" or data.numel() == 0:
        return torch.empty(shape, dtype=torch.bool, device=data.device)
    if data.device.type == "cuda":
        from deepopt.opt.backends.triton.quantization.activation.boolean import unpack_boolean_mask as backend_unpack

        return backend_unpack(data, shape)
    shifts = torch.arange(32, dtype=torch.int64, device=data.device)
    bits = (data.to(torch.int64)[:, None] >> shifts) & 1
    return bits.reshape(-1)[:prod(shape)].to(torch.bool).reshape(shape)
