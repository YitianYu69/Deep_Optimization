"""Legacy unsigned amax activation codec using TileLang kernels."""

from functools import lru_cache
from math import prod

import torch

from ..._runtime import ceildiv, compile_kernel, device_key, dtype_name, require_cuda, runtime, storage_view


def _bits_consts(bits, group_size):
    if bits not in (1, 2, 3, 4, 8):
        raise ValueError(f"Unsupported bits: {bits}")
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    vpw = 32 // bits
    return vpw, ceildiv(group_size, vpw), (1 << bits) - 1


@lru_cache(maxsize=128)
def _encode(count, group, span, sm, sn, source_dtype, bits, device):
    _, T = runtime()
    vpw, words, qmax = _bits_consts(bits, group)
    groups = count // group

    @T.prim_func
    def kernel(X: T.Tensor((span,), source_dtype), P: T.Tensor((groups * words,), "int32"),
               S: T.Tensor((groups,), "float16")):
        with T.Kernel(ceildiv(groups, 32), threads=32) as bid:
            for lane in T.Parallel(32):
                g = T.cast(bid, "int64") * 32 + lane
                maximum = T.alloc_local((1,), "float32")
                word = T.alloc_local((1,), "uint32")
                maximum[0] = 0
                if g < groups:
                    for offset in T.serial(group):
                        maximum[0] = T.max(maximum[0], T.abs(T.cast(X[g * sm + offset * sn], "float32")))
                    multiplier = T.if_then_else(maximum[0] > 0, T.ieee_fdiv(T.float32(qmax), maximum[0]), 1.0)
                    S[g] = T.cast(T.ieee_fdiv(T.float32(1), multiplier), "float16")
                    for w in T.serial(words):
                        word[0] = 0
                        for digit in T.serial(vpw):
                            offset = w * vpw + digit
                            if offset < group:
                                value = T.cast(X[g * sm + offset * sn], "float32") * multiplier
                                lower = T.floor(value)
                                integer = T.cast(lower, "int32")
                                fraction = value - lower
                                rounded = integer + T.cast((fraction > 0.5) | ((fraction == 0.5) & ((integer & 1) != 0)), "int32")
                                code = T.min(T.max(rounded, 0), qmax)
                                word[0] = word[0] | (T.cast(code, "uint32") << (digit * bits))
                        P[g * words + w] = T.cast(word[0], "int32")

    return compile_kernel(kernel, device)


@lru_cache(maxsize=128)
def _decode(count, group, bits, output_dtype, device):
    _, T = runtime()
    vpw, words, qmax = _bits_consts(bits, group)
    groups = count // group

    @T.prim_func
    def kernel(P: T.Tensor((groups * words,), "int32"), S: T.Tensor((groups,), "float16"),
               Y: T.Tensor((count,), output_dtype)):
        with T.Kernel(ceildiv(count, 128), threads=128) as bid:
            for lane in T.Parallel(128):
                index = T.cast(bid, "int64") * 128 + lane
                if index < count:
                    g, offset = index // group, index % group
                    word = T.cast(P[g * words + offset // vpw], "uint32")
                    code = T.cast((word >> (offset % vpw * bits)) & T.uint32(qmax), "float32")
                    Y[index] = T.cast(code * T.cast(S[g], "float32"), output_dtype)

    return compile_kernel(kernel, device)


def amax_quantize_pack(x, group_size, bits):
    device = require_cuda(x)
    _, words, _ = _bits_consts(bits, group_size)
    grouped = x.view(-1, group_size)
    output = torch.empty((grouped.shape[0], words), device=device, dtype=torch.int32)
    scalar = torch.empty(grouped.shape[0], device=device, dtype=torch.float16)
    if grouped.numel():
        source = storage_view(grouped)
        with torch.cuda.device(device):
            _encode(x.numel(), group_size, source.numel(), *grouped.stride(), dtype_name(x.dtype), bits,
                    device_key(device))(source, output.view(-1), scalar)
    return output, scalar, x.shape


def amax_dequantize_unpack(packed, scalar, ori_shape, group_size, bits, output_dtype):
    device = require_cuda(packed, scalar)
    _bits_consts(bits, group_size)
    count = prod(ori_shape)
    output = torch.empty(ori_shape, device=device, dtype=output_dtype)
    if count:
        with torch.cuda.device(device):
            _decode(count, group_size, bits, dtype_name(output_dtype), device_key(device))(
                packed.view(-1), scalar, output.view(-1))
    return output
