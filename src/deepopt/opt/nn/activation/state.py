"""Preprocessed activation coefficients, independent of FP8 scale metadata."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SavedActivation:
    data: torch.Tensor
    step: torch.Tensor | None = None
    minimum: torch.Tensor | None = None
    group_size: int = 512
    block_shape: tuple | None = None
    bits: int = 2
    mask: bool = False
    packed: bool = False

    @property
    def tensors(self):
        return (self.data,) if self.step is None else (self.data, self.step, self.minimum)

    @property
    def metadata(self):
        return self.group_size, self.block_shape, self.bits, self.mask, self.packed

    @classmethod
    def restore(cls, tensors, metadata):
        group, block, bits, mask, packed = metadata
        return cls(tensors[0], tensors[1] if len(tensors) > 1 else None,
                   tensors[2] if len(tensors) > 1 else None, group, block, bits, mask, packed)
