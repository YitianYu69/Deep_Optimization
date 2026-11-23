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


def record_layer_moments(packed, original, center_shift, moment_gain):
    """Restore zero mean and the actual forward xhat second moment.

    The target is measured before affine output, so it includes the layer's
    real epsilon without subtracting nearly equal values from one. A decoded
    constant row has no recoverable direction and safely reconstructs zeros.
    """
    decoded = dequantize_unpack_activation(packed, dtype=torch.float32)
    shift = decoded.mean(1)
    variance = (decoded - shift[:, None]).square().mean(1)
    target = original.square().mean(1)
    gain = torch.where(variance > 0, torch.sqrt(target / variance.clamp_min(1e-12)), 0.0)
    center_shift.copy_(shift)
    moment_gain.copy_(gain)


def corrected_layer_xhat(value, inverse, center_shift, moment_gain):
    value = corrected_xhat(value, inverse, center_shift)
    if moment_gain is not None:
        valid = torch.isfinite(inverse) & (inverse > 0)
        value = torch.where(valid[:, None], value * moment_gain[:, None], value)
    return value
