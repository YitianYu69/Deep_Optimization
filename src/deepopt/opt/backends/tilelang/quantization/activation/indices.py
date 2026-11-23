"""Lossless arbitrary-width routing codes, including cross-word fields."""

from functools import lru_cache

import torch

from ..._runtime import ceildiv, compile_kernel, device_key, dtype_name, require_cuda, runtime, storage_view


@lru_cache(maxsize=128)
def _pack_kernel(shape, strides, span, source_dtype, count, bits, output_count, output_dtype, device):
    _, T = runtime()
    layout = tuple(zip(shape[::-1], strides[::-1]))

    @T.prim_func
    def kernel(X: T.Tensor((span,), source_dtype), Y: T.Tensor((output_count,), output_dtype)):
        with T.Kernel(ceildiv(output_count, 128), threads=128) as bid:
            for lane in T.Parallel(128):
                word_index = T.cast(bid, "int64") * 128 + lane
                address = T.alloc_local((1,), "int64")
                remaining = T.alloc_local((1,), "int64")
                word = T.alloc_local((1,), "uint32")
                word[0] = 0
                if word_index < output_count:
                    if bits > 32:
                        remaining[0] = word_index
                        address[0] = 0
                        for dimension, stride in layout:
                            address[0] = address[0] + remaining[0] % dimension * stride
                            remaining[0] = remaining[0] // dimension
                        Y[word_index] = T.cast(X[address[0]], output_dtype)
                    else:
                        for digit in T.serial(32):
                            bit = word_index * 32 + digit
                            index = bit // bits
                            if index < count:
                                remaining[0] = index
                                address[0] = 0
                                for dimension, stride in layout:
                                    address[0] = address[0] + remaining[0] % dimension * stride
                                    remaining[0] = remaining[0] // dimension
                                value = T.cast(X[address[0]], "uint64")
                                word[0] = word[0] | (T.cast((value >> (bit % bits)) & T.uint64(1), "uint32") << digit)
                        Y[word_index] = T.cast(word[0], output_dtype)

    return compile_kernel(kernel, device)


@lru_cache(maxsize=128)
def _unpack_kernel(count, bits, source_count, source_dtype, output_dtype, device):
    _, T = runtime()
    mask = (1 << bits) - 1 if bits <= 32 else 0

    @T.prim_func
    def kernel(X: T.Tensor((source_count,), source_dtype), Y: T.Tensor((count,), output_dtype)):
        with T.Kernel(ceildiv(count, 128), threads=128) as bid:
            for lane in T.Parallel(128):
                index = T.cast(bid, "int64") * 128 + lane
                value = T.alloc_local((1,), "uint64")
                value[0] = 0
                if index < count:
                    if bits > 32:
                        Y[index] = T.cast(X[index], output_dtype)
                    elif bits == 0:
                        Y[index] = T.cast(0, output_dtype)
                    else:
                        bit = index * bits
                        low = T.cast(T.cast(X[bit // 32], "uint32"), "uint64")
                        value[0] = low >> (bit % 32)
                        if bit % 32 + bits > 32:
                            high = T.cast(T.cast(X[bit // 32 + 1], "uint32"), "uint64")
                            value[0] = value[0] | (high << (32 - bit % 32))
                        Y[index] = T.cast(value[0] & T.uint64(mask), output_dtype)

    return compile_kernel(kernel, device)


def pack(value, result):
    device = require_cuda(value, result.data)
    if result.bits and value.numel():
        source = storage_view(value)
        with torch.cuda.device(device):
            _pack_kernel(tuple(value.shape), tuple(value.stride()), source.numel(), dtype_name(value.dtype),
                         value.numel(), result.bits, result.data.numel(), dtype_name(result.data.dtype), device_key(device))(
                             source, result.data.view(-1))


def unpack(value, result):
    device = require_cuda(value.data, result)
    if result.numel():
        with torch.cuda.device(device):
            _unpack_kernel(result.numel(), value.bits, value.data.numel(), dtype_name(value.data.dtype),
                           dtype_name(result.dtype), device_key(device))(value.data.view(-1), result.view(-1))
