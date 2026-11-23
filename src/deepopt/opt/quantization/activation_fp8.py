"""Fused FP8 conversion and affine saved activations, with a CPU oracle."""

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .activation import _validate_bits, quantize_pack_activation
from .quantizer import quantize_fp8
from .tensor import QuantizedTensor
from .._execution import finite_checks_enabled
from deepopt.opt.scaling.amax import AmaxScaling


@torch.no_grad()
def quantize_pack_fp8_activation(x, *, fp8_dtype=torch.float8_e4m3fn,
                                block_shape=None, group_size=512, check_finite=None, bits=2, row_alignment=16):
    """Cast to FP8 and affine-quantize/pack the FP8-rounded saved operand.

    Tensorwise FP8 uses flat compression groups. Blockwise FP8 uses precisely
    its scaling rectangles, including their masked edges, instead of group_size.
    """
    checked = finite_checks_enabled() if check_finite is None else check_finite
    AmaxScaling(block_shape)._validate_input(x, fp8_dtype, checked)
    if type(group_size) is not int or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    _validate_bits(bits)
    if type(row_alignment) is not int or row_alignment not in (1, 16):
        raise ValueError("row_alignment must be 1 or 16")
    if x.device.type == "cuda" and not isinstance(x, FakeTensor):
        from deepopt.opt.backends.dispatch import backend_module
        backend_quantize_pack = backend_module('activation_codec', 'quantization.activation.fp8').quantize_pack_fp8_activation

        data, inverse, packed = backend_quantize_pack(x, fp8_dtype, block_shape, group_size, checked,
                                                     bits=bits, row_alignment=row_alignment)
        return QuantizedTensor(data, inverse, block_shape), packed
    quantized = quantize_fp8(x, fp8_dtype=fp8_dtype, block_shape=block_shape, check_finite=checked)
    if isinstance(x, FakeTensor) and x.device.type == "cuda":
        rows, cols = x.shape
        data = torch.empty_strided(x.shape, (((cols + row_alignment - 1) // row_alignment) * row_alignment, 1),
                                   dtype=fp8_dtype, device=x.device)
        quantized = QuantizedTensor(data, quantized.scale_inv, block_shape)
    packed = quantize_pack_activation(quantized.data.float(), group_size=group_size, block_shape=block_shape, bits=bits)
    return quantized, packed
