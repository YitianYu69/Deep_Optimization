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
                                block_shape=None, group_size=512, check_finite=None, bits=2):
    """Cast to FP8 and affine-quantize/pack the FP8-rounded saved operand.

    Tensorwise FP8 uses flat compression groups. Blockwise FP8 uses precisely
    its scaling rectangles, including their masked edges, instead of group_size.
    """
    checked = finite_checks_enabled() if check_finite is None else check_finite
    AmaxScaling(block_shape)._validate_input(x, fp8_dtype, checked)
    if type(group_size) is not int or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    _validate_bits(bits)
    if x.device.type == "cuda" and not isinstance(x, FakeTensor):
        from deepopt.opt.backends.triton.quantization.activation.fp8 import (
            quantize_pack_fp8_activation as backend_quantize_pack,
        )

        data, inverse, packed = backend_quantize_pack(x, fp8_dtype, block_shape, group_size, checked, bits=bits)
        return QuantizedTensor(data, inverse, block_shape), packed
    quantized = quantize_fp8(x, fp8_dtype=fp8_dtype, block_shape=block_shape, check_finite=checked)
    packed = quantize_pack_activation(quantized.data.float(), group_size=group_size, block_shape=block_shape, bits=bits)
    return quantized, packed
