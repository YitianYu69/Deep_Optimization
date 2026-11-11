"""Reference semantics for normalization-only saved-xhat mean correction."""

import torch

from ...quantization.activation import dequantize_unpack_activation


def record_center_shift(packed, original, center_shift):
    """Measure codec error, preserving the original domain's first moment."""
    if center_shift is not None:
        decoded = dequantize_unpack_activation(packed, dtype=torch.float32)
        center_shift.copy_((decoded - original).mean(1))


def corrected_xhat(value, inverse, center_shift):
    if center_shift is None:
        return value
    valid = torch.isfinite(inverse) & (inverse > 0)
    return torch.where(valid[:, None], value - center_shift[:, None], value)
