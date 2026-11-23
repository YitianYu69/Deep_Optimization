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
def _quantize_pack_strided_activation(x, group_size=512, *, bits=2, source_dtype=None):
    """Logical row-major groups read directly from N-D storage and rounded in-register."""
    from math import prod

    if not isinstance(x, torch.Tensor) or x.ndim < 2 or not x.is_floating_point():
        raise ValueError("Strided activation packing requires a floating tensor of rank >= 2")
    shape = (prod(x.shape[:-1]), x.shape[-1])
    _validate_layout(shape, group_size, None)
    _validate_bits(bits)
    if x.device.type != "cuda" or isinstance(x, FakeTensor):
        return quantize_pack_activation(x.to(source_dtype or x.dtype).reshape(shape), group_size, bits=bits)
    if (x.ndim == 2 or x.is_contiguous()) and source_dtype in (None, x.dtype):
        return quantize_pack_activation(x.reshape(shape), group_size, bits=bits)
    from deepopt.opt.backends.dispatch import backend_module
    quantize_pack_strided = backend_module('activation_codec', 'quantization.activation.affine').quantize_pack_strided

    data, step, minimum = quantize_pack_strided(x, group_size, bits=bits, source_dtype=source_dtype)
    return PackedActivation(data, step, minimum, shape, group_size, None, bits)


def _joint_attention_pack_eligible(inputs, group_size):
    """Bound the opt-in kernel to matching ordinary, nonempty CUDA views."""
    return (len(inputs) == 3 and 0 < group_size <= 2048
            and all(type(x) is torch.Tensor for x in inputs)
            and inputs[0].device.type == "cuda" and inputs[0].ndim == 4 and inputs[0].numel() > 0
            and inputs[0].dtype in (torch.bfloat16, torch.float16)
            and all(x.shape == inputs[0].shape and x.stride() == inputs[0].stride()
                    and x.device == inputs[0].device and x.dtype == inputs[0].dtype for x in inputs)
            and all(s >= 0 for s in inputs[0].stride()))


@torch.no_grad()
def _quantize_pack_qkv_activation(inputs, group_size=512, *, bits=4):
    """Experimental single-launch Q/K/V packing; each logical group is unchanged.

    The generic path is also the CPU oracle and handles unsupported layouts.
    Selected by an explicit joint scope or the measured attention auto policy.
    """
    from math import prod
    _validate_bits(bits)
    if len(inputs) != 3:
        raise ValueError("Q/K/V packing requires exactly three operands")
    if not _joint_attention_pack_eligible(inputs, group_size):
        return tuple(_quantize_pack_strided_activation(x, group_size, bits=bits) for x in inputs)
    from deepopt.opt.backends.dispatch import backend_module
    quantize_pack_qkv = backend_module('activation_codec', 'quantization.activation.affine').quantize_pack_qkv
    data, step, minimum = quantize_pack_qkv(inputs, group_size, bits=bits)
    shape = (prod(inputs[0].shape[:-1]), inputs[0].shape[-1])
    return tuple(PackedActivation(data[i], step[i], minimum[i], shape, group_size, None, bits) for i in range(3))


@torch.no_grad()
def _dequantize_unpack_spatial_activation(packed, shape, *, dtype, memory_format=torch.contiguous_format):
    """Restore a spatial source without an intermediate dense channel-row copy."""
    from math import prod
    if (len(shape) not in (3, 4, 5) or packed.block_shape is not None
            or packed.shape != (shape[0] * prod(shape[2:]), shape[1])):
        raise ValueError("spatial unpack requires flat channel-row groups matching a batched spatial shape")
    if packed.data.device.type == "cuda" and not isinstance(packed.data, FakeTensor):
        from deepopt.opt.backends.dispatch import backend_module
        dequantize_unpack_spatial = backend_module('activation_codec', 'quantization.activation.affine').dequantize_unpack_spatial
        return dequantize_unpack_spatial(packed.data, packed.step, packed.minimum, tuple(shape),
                                         packed.group_size, dtype=dtype, bits=packed.bits,
                                         memory_format=memory_format)
    if isinstance(packed.data, FakeTensor) or packed.data.device.type == "meta":
        return torch.empty(shape, dtype=dtype, device=packed.data.device, memory_format=memory_format)
    matrix = dequantize_unpack_activation(packed, dtype=dtype, row_alignment=1)
    from ..nn.convolution.geometry import restore_matrix
    return restore_matrix(matrix, shape, memory_format)


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
        from deepopt.opt.backends.dispatch import backend_module
        quantize_pack_affine = backend_module('activation_codec', 'quantization.activation.affine').quantize_pack_affine

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
def dequantize_unpack_activation(packed: PackedActivation, dtype=torch.bfloat16, transpose=False, *, row_alignment=16):
    """Unpack codes and dequantize directly to the requested dtype/orientation.

    CUDA output rows default to a multiple-of-16 stride suitable for FP8 GEMMs.
    Spatial consumers may request compact rows with row_alignment=1.
    FP8 reconstruction is clamped to its finite range before casting.
    """
    if not isinstance(packed, PackedActivation):
        raise TypeError("packed must be a PackedActivation")
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                     torch.float8_e4m3fn, torch.float8_e5m2):
        raise TypeError("dtype must be a supported floating point dtype")
    if type(transpose) is not bool:
        raise TypeError("transpose must be a boolean")
    if type(row_alignment) is not int or row_alignment not in (1, 16):
        raise ValueError("row_alignment must be 1 or 16")
    rows, cols = packed.shape
    shape = (cols, rows) if transpose else (rows, cols)
    if isinstance(packed.data, FakeTensor) or packed.data.device.type == "meta":
        stride = (((shape[1] + row_alignment - 1) // row_alignment) * row_alignment, 1) if packed.data.device.type == "cuda" else (shape[1], 1)
        return torch.empty_strided(shape, stride, dtype=dtype, device=packed.data.device)
    if packed.data.device.type == "cuda":
        from deepopt.opt.backends.dispatch import backend_module
        dequantize_unpack_affine = backend_module('activation_codec', 'quantization.activation.affine').dequantize_unpack_affine

        return dequantize_unpack_affine(
            packed.data, packed.step, packed.minimum, packed.shape,
            packed.group_size, packed.block_shape, dtype=dtype, transpose=transpose, bits=packed.bits,
            row_alignment=row_alignment,
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
