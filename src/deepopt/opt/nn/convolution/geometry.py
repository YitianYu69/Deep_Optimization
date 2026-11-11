"""Dimension-independent geometry for spatial convolutions.

The source matrix has one row per input position and one column per channel.
It describes each unique activation once, rather than materializing the
overlapping patches used by a convolution GEMM.
"""

from dataclasses import dataclass
from math import prod

import torch


def _spatial_tuple(value, rank, name, *, minimum):
    values = (value,) * rank if type(value) is int else value
    if not isinstance(values, (tuple, list)) or len(values) != rank:
        raise ValueError(f"{name} must be an integer or a sequence of {rank} integers")
    if any(type(item) is not int or item < minimum for item in values):
        raise ValueError(f"{name} values must be integers greater than or equal to {minimum}")
    return tuple(values)


@dataclass(frozen=True)
class ConvGeometry:
    """Validated batched Conv1d/2d/3d shapes and symmetric zero padding."""

    input_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int
    synthetic_padding_right: tuple[int, ...] = ()

    @classmethod
    def from_tensors(cls, x, weight, stride=1, padding=0, dilation=1, groups=1):
        if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
            raise TypeError("convolution input and weight must be tensors")
        rank = weight.ndim - 2
        if rank not in (1, 2, 3) or x.ndim != rank + 2:
            raise ValueError("convolution requires batched input and 3D, 4D, or 5D weight")
        if type(groups) is not int or groups <= 0:
            raise ValueError("groups must be a positive integer")
        if x.shape[0] < 0 or any(size <= 0 for size in x.shape[1:]):
            raise ValueError("convolution channels and spatial dimensions must be positive")
        if any(size <= 0 for size in weight.shape):
            raise ValueError("convolution weight dimensions must be positive")
        if x.shape[1] != weight.shape[1] * groups:
            raise ValueError("input channels must equal weight input channels times groups")
        if weight.shape[0] % groups:
            raise ValueError("output channels must be divisible by groups")
        geometry = cls(tuple(x.shape), tuple(weight.shape),
                       _spatial_tuple(stride, rank, "stride", minimum=1),
                       _spatial_tuple(padding, rank, "padding", minimum=0),
                       _spatial_tuple(dilation, rank, "dilation", minimum=1), groups)
        if any(size <= 0 for size in geometry.output_spatial):
            raise ValueError("convolution kernel exceeds the padded input size")
        return geometry

    @property
    def rank(self):
        return len(self.weight_shape) - 2

    @property
    def input_spatial(self):
        return self.input_shape[2:]

    @property
    def valid_input_spatial(self):
        extra = self.synthetic_padding_right or (0,) * self.rank
        return tuple(size - padding for size, padding in zip(self.input_spatial, extra))

    @property
    def valid_input_spatial_3d(self):
        return (1,) * (3 - self.rank) + self.valid_input_spatial

    @property
    def output_spatial(self):
        return tuple((size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
                     for size, kernel, stride, padding, dilation in
                     zip(self.input_spatial, self.kernel_size, self.stride, self.padding, self.dilation))

    @property
    def kernel_size(self):
        return self.weight_shape[2:]

    @property
    def batch_size(self):
        return self.input_shape[0]

    @property
    def in_channels(self):
        return self.input_shape[1]

    @property
    def out_channels(self):
        return self.weight_shape[0]

    @property
    def in_channels_per_group(self):
        return self.weight_shape[1]

    @property
    def out_channels_per_group(self):
        return self.out_channels // self.groups

    @property
    def input_volume(self):
        return prod(self.input_spatial)

    @property
    def output_volume(self):
        return prod(self.output_spatial)

    @property
    def kernel_volume(self):
        return prod(self.kernel_size)

    @property
    def input_matrix_shape(self):
        return self.batch_size * self.input_volume, self.in_channels

    @property
    def grad_matrix_shape(self):
        return self.batch_size * self.output_volume, self.out_channels

    @property
    def weight_matrix_shape(self):
        return self.out_channels, self.in_channels_per_group * self.kernel_volume

    @property
    def output_shape(self):
        return (self.batch_size, self.out_channels, *self.output_spatial)

    @property
    def pointwise(self):
        return (self.groups == 1 and all(size == 1 for size in self.kernel_size)
                and all(stride == 1 for stride in self.stride)
                and all(padding == 0 for padding in self.padding))

    @property
    def input_spatial_3d(self):
        return (1,) * (3 - self.rank) + self.input_spatial

    @property
    def output_spatial_3d(self):
        return (1,) * (3 - self.rank) + self.output_spatial

    @property
    def kernel_size_3d(self):
        return (1,) * (3 - self.rank) + self.kernel_size

    @property
    def stride_3d(self):
        return (1,) * (3 - self.rank) + self.stride

    @property
    def padding_3d(self):
        return (0,) * (3 - self.rank) + self.padding

    @property
    def dilation_3d(self):
        return (1,) * (3 - self.rank) + self.dilation


def source_matrix(x):
    """Convert N,C,spatial source values to (N*spatial),C channel rows."""
    if not isinstance(x, torch.Tensor) or x.ndim not in (3, 4, 5):
        raise ValueError("source must be a batched Conv1d/2d/3d tensor")
    return x.movedim(1, -1).reshape(-1, x.shape[1])


def zero_synthetic_padding(x, geometry):
    """Restore literal SAME-padding zeros in a fresh decoded source tensor."""
    for axis, extra in enumerate(geometry.synthetic_padding_right, 2):
        if extra:
            x.narrow(axis, x.shape[axis] - extra, extra).zero_()
    return x


def restore_matrix(matrix, shape, memory_format=None):
    """Restore channel rows to the requested N,C,spatial shape."""
    shape = tuple(shape)
    if len(shape) not in (3, 4, 5) or matrix.ndim != 2:
        raise ValueError("matrix restoration requires a 2D matrix and a 3D, 4D, or 5D shape")
    expected = (shape[0] * prod(shape[2:]), shape[1])
    if tuple(matrix.shape) != expected:
        raise ValueError(f"matrix must have shape {expected}")
    result = matrix.reshape(shape[0], *shape[2:], shape[1]).movedim(-1, 1)
    return result if memory_format is None else _contiguous(result, memory_format)


def _contiguous(tensor, memory_format):
    # Tensor.contiguous's device guard runs before fake dispatch on CPU-only
    # builds. Clone dispatches correctly and performs the same copy when needed.
    return (tensor if tensor.is_contiguous(memory_format=memory_format)
            else torch.clone(tensor, memory_format=memory_format))
