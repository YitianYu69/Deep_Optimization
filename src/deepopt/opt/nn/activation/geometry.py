"""Pointer-free logical/matrix addressing shared by ReLU forward and backward."""

from dataclasses import dataclass
from math import prod

from ...quantization.activation_tensor import FP8Activation


@dataclass(frozen=True)
class ReLUGeometry:
    shape: tuple
    layout: str = "last"

    @property
    def matrix_shape(self):
        if self.layout == "channels":
            return self.shape[0] * prod(self.shape[2:]), self.shape[1]
        cols = self.shape[-1] if self.shape else 1
        return prod(self.shape[:-1]) if self.shape else 1, cols

    def matrix(self, x):
        return (x.movedim(1, -1).reshape(self.matrix_shape) if self.layout == "channels"
                else x.reshape(self.matrix_shape))

    def restore(self, matrix):
        if self.layout == "channels":
            return matrix.reshape(self.shape[0], *self.shape[2:], self.shape[1]).movedim(-1, 1)
        return matrix.reshape(self.shape)


def tensor_arguments(x):
    """Return storage, inverse scales and immutable source address metadata."""
    shape = tuple(x.shape)
    channels, spatial = (shape[1], prod(shape[2:])) if len(shape) >= 2 else (1, 1)
    last = shape[-1] if shape else 1
    if isinstance(x, FP8Activation):
        data, scales = x._data, x._scale_inv
        br, bc = x._block_shape or tuple(data.shape)
        meta = (1 if x._activation_layout == "channels" else 2, (), False,
                channels, spatial, last, *data.stride(), br, bc, *scales.stride())
        return data, scales, meta
    return x, x, (0, tuple(x.stride()), x.is_contiguous(), channels, spatial, last, 0, 0, 1, 1, 0, 0)
