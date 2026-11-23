from __future__ import annotations

from ..ir import TensorValue

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .builder import GraphBuilder


@dataclass
class SymbolicTensor:
    value: TensorValue
    builder: "GraphBuilder"

    @property
    def shape(self):
        return self.value.meta.shape

    @property
    def dtype(self):
        return self.value.meta.dtype

    @property
    def device(self):
        return self.value.meta.device

    @property
    def stride(self):
        return self.value.meta.stride

    @property
    def requires_grad(self):
        return self.value.meta.requires_grad

    def __add__(self, other: Any):
        return self.builder.add(self, other)

    def __radd__(self, other: Any):
        return self.builder.add(other, self)

    def __sub__(self, other: Any):
        return self.builder.sub(self, other)

    def __rsub__(self, other: Any):
        return self.builder.sub(other, self)

    def __mul__(self, other: Any):
        return self.builder.mul(self, other)

    def __rmul__(self, other: Any):
        return self.builder.mul(other, self)

    def __truediv__(self, other: Any):
        return self.builder.div(self, other)

    def __rtruediv__(self, other: Any):
        return self.builder.div(other, self)

    def __neg__(self):
        return self.builder.neg(self)

    def __repr__(self):
        return f"SymbolicTensor({self.value})"
