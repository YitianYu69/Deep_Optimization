"""Packed affine saved activations, independent of their compute dtype.

The codec changes saved values used by backward. It does not change forward
values. FP8 callers store values in the existing FP8-scaled domain and retain
the FP8 inverse scale separately.
"""

from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensor


def _validate_layout(shape, group_size, block_shape):
    if (not isinstance(shape, tuple) or len(shape) != 2
            or any(type(size) is not int or size <= 0 for size in shape)):
        raise ValueError("shape must contain two positive integers")
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if block_shape is not None and (
        not isinstance(block_shape, tuple) or len(block_shape) != 2
        or any(type(size) is not int or size <= 0 for size in block_shape)
    ):
        raise ValueError("block_shape must be None or two positive integers")


def _validate_bits(bits):
    if type(bits) is not int or bits not in (1, 2, 4, 8):
        raise ValueError("bits must be one of 1, 2, 4, or 8")


def _dimensions(shape, group_size, block_shape, bits=2):
    rows, cols = shape
    if block_shape is None:
        elements = group_size
        groups = (rows * cols + elements - 1) // elements
    else:
        block_rows, block_cols = block_shape
        elements = block_rows * block_cols
        groups = ((rows + block_rows - 1) // block_rows) * ((cols + block_cols - 1) // block_cols)
    values_per_word = 32 // bits
    return groups, elements, (elements + values_per_word - 1) // values_per_word


@dataclass(frozen=True)
class PackedActivation:
    """Packed integer payload and FP32 affine metadata for a 2D activation.

    For binary groups whose full step overflows FP32, a negative step marks
    the negated half-step. Encoding and decoding then operate on half-values.
    Normal groups use the ordinary nonnegative affine step.
    """

    data: torch.Tensor
    step: torch.Tensor
    minimum: torch.Tensor
    shape: tuple[int, int]
    group_size: int
    block_shape: tuple[int, int] | None = None
    bits: int = 2

    def __post_init__(self):
        _validate_layout(self.shape, self.group_size, self.block_shape)
        _validate_bits(self.bits)
        groups, _, words = _dimensions(self.shape, self.group_size, self.block_shape, self.bits)
        if self.data.dtype != torch.int32:
            raise TypeError("packed activation data must have dtype torch.int32")
        if tuple(self.data.shape) != (groups, words):
            raise ValueError(f"packed activation data must have shape {(groups, words)}")
        for name, metadata in (("step", self.step), ("minimum", self.minimum)):
            if metadata.dtype != torch.float32:
                raise TypeError(f"{name} must have dtype torch.float32")
            if tuple(metadata.shape) != (groups,):
                raise ValueError(f"{name} must have shape {(groups,)}")
            if metadata.device != self.data.device:
                raise ValueError("packed data and affine metadata must share a device")
        if not all(value.is_contiguous() for value in (self.data, self.step, self.minimum)):
            raise ValueError("packed data and affine metadata must be contiguous")


def _reference_groups(x, group_size, block_shape):
    """Return padded FP32 groups and their valid-element masks."""
    rows, cols = x.shape
    groups, elements, _ = _dimensions(tuple(x.shape), group_size, block_shape)
    values = x.float()
    if block_shape is None:
        padded = torch.zeros(groups * elements, dtype=torch.float32, device=x.device)
        padded[:x.numel()] = values.reshape(-1)
        valid = torch.arange(groups * elements, device=x.device) < x.numel()
        return padded.reshape(groups, elements), valid.reshape(groups, elements)
    block_rows, block_cols = block_shape
    row_groups, col_groups = (rows + block_rows - 1) // block_rows, (cols + block_cols - 1) // block_cols
    padded = torch.zeros((row_groups * block_rows, col_groups * block_cols), dtype=torch.float32, device=x.device)
    padded[:rows, :cols] = values
    valid = torch.zeros_like(padded, dtype=torch.bool)
    valid[:rows, :cols] = True
    def grouped(value):
        return value.reshape(row_groups, block_rows, col_groups, block_cols).permute(0, 2, 1, 3).reshape(groups, elements)
    return grouped(padded), grouped(valid)


@torch.no_grad()
def quantize_pack_activation(x: torch.Tensor, group_size=512, block_shape=None, *, bits=2) -> PackedActivation:
    """Affine-quantize and bit-pack finite 2D activations.

    Masked tails do not influence min/max.
    Tensorwise groups are chunks of the logical row-major matrix. With
    ``block_shape``, every rectangular tile has independent affine metadata.
    Neither grouping requires contiguous input or exact divisibility.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError("activation must be a 2D tensor")
    _validate_layout(tuple(x.shape), group_size, block_shape)
    _validate_bits(bits)
    if not x.is_floating_point():
        raise TypeError("activation must have a floating point dtype")
    groups, elements, words = _dimensions(tuple(x.shape), group_size, block_shape, bits)
    values_per_word, qmax = 32 // bits, (1 << bits) - 1
    if isinstance(x, FakeTensor) or x.device.type == "meta":
        data = torch.empty((groups, words), dtype=torch.int32, device=x.device)
        step = torch.empty(groups, dtype=torch.float32, device=x.device)
        minimum = torch.empty_like(step)
    elif x.device.type == "cuda":
        from deepopt.opt.backends.triton.quantization.activation.affine import quantize_pack_affine

        data, step, minimum = quantize_pack_affine(x, group_size, block_shape, bits=bits)
    else:
        values, valid = _reference_groups(x, group_size, block_shape)
        minimum = values.masked_fill(~valid, float("inf")).amin(dim=1)
        maximum = values.masked_fill(~valid, float("-inf")).amax(dim=1)
        span = maximum - minimum
        step = torch.where(torch.isinf(span), maximum / qmax - minimum / qmax, span / qmax)
        if bits == 1:
            step = torch.where(torch.isinf(span), -(maximum * 0.5 - minimum * 0.5), step)
        difference = values - minimum[:, None]
        normalized = torch.where(torch.isinf(difference),
                                 values / step[:, None] - minimum[:, None] / step[:, None],
                                 difference / step[:, None])
        if bits == 1:
            normalized = torch.where(step[:, None] < 0,
                                     (values * 0.5 - minimum[:, None] * 0.5) / -step[:, None],
                                     normalized)
        nonconstant = step[:, None] != 0 if bits == 1 else step[:, None] > 0
        scaled = torch.where(nonconstant, normalized, 0.0)
        codes = scaled.clamp(0, qmax).round().to(torch.int64).masked_fill(~valid, 0)
        padded_codes = torch.zeros((groups, words * values_per_word), dtype=torch.int64, device=x.device)
        padded_codes[:, :elements] = codes
        shifts = torch.arange(values_per_word, dtype=torch.int64, device=x.device) * bits
        data = (padded_codes.reshape(groups, words, values_per_word) << shifts).sum(dim=2).to(torch.int32)
    return PackedActivation(data, step, minimum, tuple(x.shape), group_size, block_shape, bits)


@torch.no_grad()
def dequantize_unpack_activation(packed: PackedActivation, dtype=torch.bfloat16, transpose=False):
    """Unpack codes and dequantize directly to the requested dtype/orientation.

    CUDA output rows have a multiple-of-16 stride suitable for FP8 GEMMs.
    FP8 reconstruction is clamped to its finite range before casting.
    """
    if not isinstance(packed, PackedActivation):
        raise TypeError("packed must be a PackedActivation")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                     torch.float8_e4m3fn, torch.float8_e5m2):
        raise TypeError("dtype must be a supported floating point dtype")
    if type(transpose) is not bool:
        raise TypeError("transpose must be a boolean")
    rows, cols = packed.shape
    shape = (cols, rows) if transpose else (rows, cols)
    if isinstance(packed.data, FakeTensor) or packed.data.device.type == "meta":
        stride = (((shape[1] + 15) // 16) * 16, 1) if packed.data.device.type == "cuda" else (shape[1], 1)
        return torch.empty_strided(shape, stride, dtype=dtype, device=packed.data.device)
    if packed.data.device.type == "cuda":
        from deepopt.opt.backends.triton.quantization.activation.affine import dequantize_unpack_affine

        return dequantize_unpack_affine(
            packed.data, packed.step, packed.minimum, packed.shape,
            packed.group_size, packed.block_shape, dtype=dtype, transpose=transpose, bits=packed.bits,
        )
    groups, elements, _ = _dimensions(packed.shape, packed.group_size, packed.block_shape, packed.bits)
    shifts = torch.arange(32 // packed.bits, dtype=torch.int64, device=packed.data.device) * packed.bits
    codes = ((packed.data.to(torch.int64)[:, :, None] >> shifts) & ((1 << packed.bits) - 1))
    codes = codes.reshape(groups, -1)[:, :elements]
    codes = codes.float()
    product = codes * packed.step[:, None]
    values = torch.where(torch.isinf(product),
                         (codes * (packed.step[:, None] * 0.5) + packed.minimum[:, None] * 0.5) * 2,
                         product + packed.minimum[:, None])
    if packed.bits == 1:
        values = torch.where(packed.step[:, None] < 0,
                             (packed.minimum[:, None] * 0.5 + codes * -packed.step[:, None]) * 2,
                             values)
    if packed.block_shape is None:
        values = values.reshape(-1)[:rows * cols].reshape(rows, cols)
    else:
        block_rows, block_cols = packed.block_shape
        row_groups, col_groups = (rows + block_rows - 1) // block_rows, (cols + block_cols - 1) // block_cols
        values = values.reshape(row_groups, col_groups, block_rows, block_cols).permute(0, 2, 1, 3)
        values = values.reshape(row_groups * block_rows, col_groups * block_cols)[:rows, :cols]
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        maximum = torch.finfo(dtype).max
        values = values.clamp(-maximum, maximum)
    values = values.to(dtype)
    return values.T.contiguous() if transpose else values
