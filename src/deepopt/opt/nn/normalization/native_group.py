"""Captured native-statistics GN contract for experimental staged kernels.

This describes the native recipe, separately from compressed xhat and BF16
storage-policy GN. Availability of these helpers is not CUDA qualification.
"""
from dataclasses import dataclass
from math import isfinite, prod

import torch


@dataclass(frozen=True)
class NativeGroupContract:
    shape: tuple
    groups: int
    eps: float
    compute_dtype: torch.dtype
    statistics_dtype: torch.dtype
    input_dtype: torch.dtype
    affine_dtype: torch.dtype
    cast_affine: bool = False

    def __post_init__(self):
        if len(self.shape) < 2 or not all(type(size) is int and size > 0 for size in self.shape):
            raise ValueError('Staged native GroupNorm requires nonempty geometry')
        if type(self.groups) is not int or self.groups < 1 or self.shape[1] % self.groups or not isfinite(self.eps) or self.eps < 0:
            raise ValueError('Invalid native GroupNorm groups/epsilon')
        if any(dtype not in (torch.bfloat16, torch.float32) for dtype in (
                self.compute_dtype, self.statistics_dtype, self.input_dtype, self.affine_dtype)):
            raise ValueError('Staged native GroupNorm supports BF16/FP32 only')

    @property
    def n(self):
        return self.shape[0]

    @property
    def channels(self):
        return self.shape[1]

    @property
    def spatial(self):
        return prod(self.shape[2:])

    @property
    def domains(self):
        return self.n * self.groups

    @property
    def elements(self):
        return self.channels // self.groups * self.spatial

    def receipt(self):
        return {'shape': self.shape, 'groups': self.groups, 'eps': self.eps,
            'compute_dtype': str(self.compute_dtype), 'saved_statistics_dtype': str(self.statistics_dtype),
            'input_gradient_dtype': str(self.input_dtype), 'affine_gradient_dtype': str(self.affine_dtype),
            'statistics_arithmetic': 'FP32', 'affine_reductions': 'FP32_ds_db_then_native_affine',
            'forward_statistics': 'FP32_internal_separate_from_rounded_saved_stats',
            'cuda_numerically_qualified': False}
