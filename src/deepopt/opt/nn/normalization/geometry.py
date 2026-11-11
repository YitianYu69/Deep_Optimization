"""Logical normalization domains, independent of physical storage layout."""

from dataclasses import dataclass
from math import prod

import torch


def output_strides(x, layout=None):
    shape = tuple(x.shape)
    channels_last = layout == "channels" or layout is None and (
        x.ndim == 4 and x.is_contiguous(memory_format=torch.channels_last)
        or x.ndim == 5 and x.is_contiguous(memory_format=torch.channels_last_3d))
    if channels_last:
        return (prod(shape[1:]), 1, *(prod(shape[axis + 1:]) * shape[1] for axis in range(2, len(shape))))
    return tuple(prod(shape[axis + 1:]) for axis in range(len(shape)))


@dataclass(frozen=True)
class NormGeometry:
    kind: str
    shape: tuple
    groups: int = 1
    eps: float = 1e-5
    batch_stats: bool = True

    def __post_init__(self):
        if self.kind not in ("batch", "group"):
            raise ValueError("normalization kind must be batch or group")
        if len(self.shape) < 2 or self.kind == "batch" and len(self.shape) != 4:
            raise ValueError("BatchNorm2d requires N,C,H,W; GroupNorm requires N,C,*spatial")
        if any(size <= 0 for size in self.shape[1:]) or self.shape[0] < 0:
            raise ValueError("normalization channels and spatial dimensions must be positive")
        if type(self.groups) is not int or self.groups <= 0 or self.shape[1] % self.groups:
            raise ValueError("num_groups must be positive and divide the channels")
        if not self.eps >= 0 or self.kind == "batch" and self.batch_stats and self.eps == 0:
            raise ValueError("normalization eps must be positive for training BatchNorm and otherwise nonnegative")

    @property
    def channels(self):
        return self.shape[1]

    @property
    def spatial(self):
        return prod(self.shape[2:])

    @property
    def domains(self):
        return self.channels if self.kind == "batch" else self.shape[0] * self.groups

    @property
    def elements(self):
        return self.shape[0] * self.spatial if self.kind == "batch" else self.channels // self.groups * self.spatial

    @property
    def matrix_shape(self):
        return self.domains, self.elements

    def matrix(self, x):
        if self.kind == "batch":
            return x.movedim(1, 0).reshape(self.matrix_shape)
        return x.reshape(self.matrix_shape)

    def restore(self, values):
        if self.kind == "batch":
            return values.reshape(self.channels, self.shape[0], *self.shape[2:]).movedim(0, 1)
        return values.reshape(self.shape)
