"""Compatibility scaling helpers backed by TileLang reductions."""

from ..quantization.fp8 import amax_scale


def block_amax_scalar(x, block_row, block_col, fp8_max):
    return amax_scale(x, fp8_max, (block_row, block_col), False)


def global_amax_scalar(x, fp8_max):
    scale, inverse = amax_scale(x, fp8_max, None, False)
    return scale.reshape(1), inverse.reshape(1)


def block_amax(x, block_row, block_col):
    from .reduction import block_amax as implementation
    return implementation(x, block_row, block_col)
