"""Decode, independently scale, transpose and cast an FP8 gradient."""

from .._runtime import auxiliary_module


def transpose_quantize(q, dtype, blocks):
    from deepopt.opt.quantization.tensor import QuantizedTensor

    codec = auxiliary_module("quantization", "quantization.fp8")
    values = codec.dequantize(q.data, q.scale_inv, q.block_shape)
    data, inverse = codec.quantize(values.T, dtype, blocks, False)
    return QuantizedTensor(data, inverse, blocks)
