"""ReLU executor preserving the historical raw-mask return interface."""

from . import elementwise
import torch

from .common import cast, multiply, auxiliary_module
from deepopt.opt.nn.activation.state import SavedActivation


def forward(x, geometry, *, dtype, block_shape, packed, save_mask, inplace, reuse,
            scale=None, observe=False, check_finite=False, capture_scale=False):
    output, inverse, saved, snapshot, observations, info = elementwise.forward(x, geometry,
        kind="relu", approximate="none", dtype=dtype, block_shape=block_shape, packed=packed,
        bits=1, group_size=512, save_mask=save_mask, inplace=inplace, reuse=reuse,
        scale=scale, observe=observe, check_finite=check_finite, capture_scale=capture_scale)
    info["mask"] = "packed" if packed and save_mask else "bool" if save_mask else "none"
    return output, inverse, saved.data if saved is not None else None, snapshot, observations, info


def backward(gradient, mask, geometry, *, packed, dtype, reuse=False, gradient_output=None):
    if reuse:
        from deepopt.opt.quantization.activation_tensor import restore_activation
        raw = restore_activation(gradient._data, tuple(gradient.shape), gradient._activation_layout)
        unpack_boolean_mask = auxiliary_module("activation_codec", "quantization.activation.boolean").unpack_boolean_mask
        coefficient = unpack_boolean_mask(mask, geometry.matrix_shape) if packed else mask.reshape(geometry.matrix_shape)
        result = multiply(geometry.matrix(cast(raw, torch.float32)), coefficient, mask=True)
        return cast(result, dtype)
    return elementwise.backward(gradient, SavedActivation(mask, mask=True, packed=packed), geometry,
        dtype=dtype, gradient_output=gradient_output)
