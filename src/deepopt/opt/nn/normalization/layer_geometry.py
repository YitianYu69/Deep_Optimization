"""Suffix-aware LayerNorm domains, independent of FP8 transport orientation."""

from dataclasses import dataclass
from math import prod

from ..activation.geometry import ReLUGeometry


def normalized_shape_tuple(value):
    if type(value) is int:
        value = (value,)
    if not isinstance(value, (tuple, list)) or not value or any(type(v) is not int or v < 0 for v in value):
        raise ValueError("normalized_shape must contain at least one nonnegative integer")
    return tuple(value)


@dataclass(frozen=True)
class LayerNormGeometry:
    shape: tuple
    normalized_shape: tuple
    eps: float = 1e-5
    layout: str = "last"

    def __post_init__(self):
        if len(self.normalized_shape) > len(self.shape) or tuple(self.shape[-len(self.normalized_shape):]) != self.normalized_shape:
            raise ValueError("normalized_shape must match the exact trailing dimensions of the input")
        if self.layout not in ("last", "channels") or self.layout == "channels" and len(self.shape) < 2:
            raise ValueError("LayerNorm transport requires last layout, or channels for rank >= 2")

    @property
    def rows(self):
        return prod(self.shape[:-len(self.normalized_shape)])

    @property
    def elements(self):
        return prod(self.normalized_shape)

    @property
    def matrix_shape(self):
        return self.rows, self.elements

    @property
    def transport(self):
        return ReLUGeometry(self.shape, self.layout)

    def matrix(self, value):
        return value.reshape(self.matrix_shape)

    def restore(self, value):
        return value.reshape(self.shape)
