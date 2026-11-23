"""Low-bit affine codecs with low-bit-first int32 payloads and FP32 metadata."""

from functools import lru_cache
from math import prod

import torch

from ..._runtime import ceildiv, compile_kernel, device_key, dtype_name, require_cuda, runtime, storage_view


def _bits_consts(bits, group_size):
    if bits not in (1, 2, 4, 8):
        raise ValueError("bits must be one of 1, 2, 4, or 8")
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    return 32 // bits, ceildiv(group_size, 32 // bits), (1 << bits) - 1


@lru_cache(maxsize=256)
def _pack(shape, strides, span, rows, cols, group, br, bc, bits, source_dtype, rounded_dtype, device):
    _, T = runtime()
    vpw, words, qmax = _bits_consts(bits, group)
    groups = ceildiv(rows, br) * ceildiv(cols, bc) if br else ceildiv(rows * cols, group)
    layout = tuple(zip(shape[::-1], strides[::-1]))

    @T.prim_func
    def kernel(X: T.Tensor((span,), source_dtype), P: T.Tensor((groups * words,), "int32"),
               S: T.Tensor((groups,), "float32"), L: T.Tensor((groups,), "float32")):
        with T.Kernel(ceildiv(groups, 32), threads=32) as bid:
            for lane in T.Parallel(32):
                g = T.cast(bid, "int64") * 32 + lane
                low = T.alloc_local((1,), "float32")
                high = T.alloc_local((1,), "float32")
                address = T.alloc_local((1,), "int64")
                remaining = T.alloc_local((1,), "int64")
                word = T.alloc_local((1,), "uint32")
                low[0] = T.infinity("float32")
                high[0] = -T.infinity("float32")
                if g < groups:
                    for offset in T.serial(group):
                        if br:
                            row = g // ceildiv(cols, bc) * br + offset // bc
                            col = g % ceildiv(cols, bc) * bc + offset % bc
                        else:
                            row = (g * group + offset) // cols
                            col = (g * group + offset) % cols
                        if row < rows and col < cols:
                            remaining[0] = row * cols + col
                            address[0] = 0
                            for dimension, stride in layout:
                                address[0] = address[0] + remaining[0] % dimension * stride
                                remaining[0] = remaining[0] // dimension
                            value = T.cast(T.cast(X[address[0]], rounded_dtype), "float32")
                            low[0] = T.min(low[0], value)
                            high[0] = T.max(high[0], value)
                    span_value = high[0] - low[0]
                    if bits == 1:
                        step = T.if_then_else(span_value == T.infinity("float32"),
                                              -(high[0] * 0.5 - low[0] * 0.5), span_value)
                    else:
                        step = T.if_then_else(span_value == T.infinity("float32"),
                                              T.ieee_fdiv(high[0], T.float32(qmax)) - T.ieee_fdiv(low[0], T.float32(qmax)),
                                              T.ieee_fdiv(span_value, T.float32(qmax)))
                    S[g] = step
                    L[g] = low[0]
                    for w in T.serial(words):
                        word[0] = T.uint32(0)
                        for digit in T.serial(vpw):
                            offset = w * vpw + digit
                            if br:
                                row = g // ceildiv(cols, bc) * br + offset // bc
                                col = g % ceildiv(cols, bc) * bc + offset % bc
                            else:
                                row = (g * group + offset) // cols
                                col = (g * group + offset) % cols
                            if offset < group and row < rows and col < cols:
                                remaining[0] = row * cols + col
                                address[0] = 0
                                for dimension, stride in layout:
                                    address[0] = address[0] + remaining[0] % dimension * stride
                                    remaining[0] = remaining[0] // dimension
                                value = T.cast(T.cast(X[address[0]], rounded_dtype), "float32")
                                difference = value - low[0]
                                normalized = T.if_then_else(difference == T.infinity("float32"),
                                    T.ieee_fdiv(value, step) - T.ieee_fdiv(low[0], step), T.ieee_fdiv(difference, step))
                                if bits == 1:
                                    normalized = T.if_then_else(step < 0,
                                        T.ieee_fdiv(value * 0.5 - low[0] * 0.5, -step), normalized)
                                    normalized = T.if_then_else(step != 0, normalized, 0.0)
                                else:
                                    normalized = T.if_then_else(step > 0, normalized, 0.0)
                                clipped = T.min(T.max(normalized, 0.0), T.float32(qmax))
                                lower = T.floor(clipped)
                                integer = T.cast(lower, "int32")
                                fraction = clipped - lower
                                code = integer + T.cast((fraction > 0.5) | ((fraction == 0.5) & ((integer & 1) != 0)), "int32")
                                word[0] = word[0] | (T.cast(code, "uint32") << (digit * bits))
                        P[g * words + w] = T.cast(word[0], "int32")

    return compile_kernel(kernel, device)


def quantize_pack_affine(x, group_size=512, block_shape=None, *, bits=2, implementation=None,
                         groups_per_program=None, logical_shape=None, num_warps=None, source_dtype=None):
    device = require_cuda(x)
    group = prod(block_shape) if block_shape is not None else group_size
    _, words, _ = _bits_consts(bits, group)
    rows, cols = tuple(logical_shape) if logical_shape is not None else x.shape
    if logical_shape is not None and (block_shape is not None or rows * cols != x.numel()):
        raise ValueError("Strided logical packing requires matching flat groups")
    if implementation not in (None, "auto", "grouped", "batched", "serial"):
        raise ValueError("implementation must be auto, grouped, batched, or serial")
    if groups_per_program is not None and (type(groups_per_program) is not int or groups_per_program not in (1, 2, 4, 8)):
        raise ValueError("groups_per_program must be 1, 2, 4, or 8")
    if num_warps is not None and num_warps not in ("auto", 1, 2, 4, 8):
        raise ValueError("num_warps must be auto, 1, 2, 4, or 8")
    if source_dtype is not None and source_dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        raise ValueError("source_dtype must be a native convolution floating dtype")
    br, bc = block_shape or (0, 0)
    groups = ceildiv(rows, br) * ceildiv(cols, bc) if br else ceildiv(rows * cols, group)
    packed = torch.empty((groups, words), device=device, dtype=torch.int32)
    step = torch.empty(groups, device=device, dtype=torch.float32)
    minimum = torch.empty_like(step)
    from deepopt.opt.backends._shared.activation_codec import observe_pack
    observe_pack(x, (rows, cols), group, bits, block_shape, "tilelang_serial", 32, 1,
                 {"SOURCE_DTYPE": source_dtype} if source_dtype is not None else {})
    if groups:
        source = storage_view(x)
        with torch.cuda.device(device):
            _pack(tuple(x.shape), tuple(x.stride()), source.numel(), rows, cols, group, br, bc, bits,
                  dtype_name(x.dtype), dtype_name(source_dtype or x.dtype), device_key(device))(
                      source, packed.view(-1), step, minimum)
    return packed, step, minimum


def quantize_pack_strided(x, group_size=512, *, bits=2, source_dtype=None):
    return quantize_pack_affine(x, group_size, bits=bits,
                                logical_shape=(prod(x.shape[:-1]), x.shape[-1]), source_dtype=source_dtype)


def quantize_pack_qkv(inputs, group_size=512, *, bits=4):
    q, k, v = inputs
    require_cuda(q, k, v)
    groups = ceildiv(q.numel(), group_size)
    _, words, _ = _bits_consts(bits, group_size)
    packed = torch.empty((3, groups, words), device=q.device, dtype=torch.int32)
    steps = torch.empty((3, groups), device=q.device, dtype=torch.float32)
    minima = torch.empty_like(steps)
    for index, value in enumerate(inputs):
        from deepopt.opt.backends._shared.activation_codec import observe_pack
        observe_pack(value, (prod(value.shape[:-1]), value.shape[-1]), group_size, bits, None,
                     "tilelang_serial_qkv", 32, 1, {})
        source = storage_view(value)
        rows, cols = prod(value.shape[:-1]), value.shape[-1]
        with torch.cuda.device(q.device):
            _pack(tuple(value.shape), tuple(value.stride()), source.numel(), rows, cols, group_size, 0, 0, bits,
                  dtype_name(value.dtype), dtype_name(value.dtype), device_key(value.device))(
                      source, packed[index].view(-1), steps[index], minima[index])
    return packed, steps, minima


@lru_cache(maxsize=256)
def _unpack(rows, cols, group, br, bc, bits, output_shape, output_strides, transpose, spatial, out_dtype, device):
    _, T = runtime()
    vpw, words, qmax = _bits_consts(bits, group)
    groups = ceildiv(rows, br) * ceildiv(cols, bc) if br else ceildiv(rows * cols, group)
    count = rows * cols
    output_span = 1 + sum((size - 1) * stride for size, stride in zip(output_shape, output_strides)) if count else 0
    maximum = float(torch.finfo(getattr(torch, out_dtype)).max) if out_dtype.startswith("float8_") else 0.0
    spatial_layout = tuple(zip(output_shape[2:][::-1], output_strides[2:][::-1]))

    @T.prim_func
    def kernel(P: T.Tensor((groups * words,), "int32"), S: T.Tensor((groups,), "float32"),
               L: T.Tensor((groups,), "float32"), Y: T.Tensor((output_span,), out_dtype)):
        with T.Kernel(ceildiv(count, 128), threads=128) as bid:
            for lane in T.Parallel(128):
                index = T.cast(bid, "int64") * 128 + lane
                if index < count:
                    row, col = index // cols, index % cols
                    if br:
                        g = row // br * ceildiv(cols, bc) + col // bc
                        offset = row % br * bc + col % bc
                    else:
                        g, offset = index // group, index % group
                    word = T.cast(P[g * words + offset // vpw], "uint32")
                    code = T.cast((word >> (offset % vpw * bits)) & T.uint32(qmax), "float32")
                    product = code * S[g]
                    value = T.if_then_else(product == T.infinity("float32"),
                        (code * (S[g] * 0.5) + L[g] * 0.5) * 2.0, product + L[g])
                    if bits == 1:
                        value = T.if_then_else(S[g] < 0, (L[g] * 0.5 + code * -S[g]) * 2.0, value)
                    if maximum > 0:
                        value = T.min(T.max(value, -maximum), maximum)
                    if spatial:
                        # row is batch/spatial, col is channel.
                        batch, pixel = row // spatial, row % spatial
                        address = T.alloc_local((1,), "int64")
                        remaining = T.alloc_local((1,), "int64")
                        address[0] = batch * output_strides[0] + col * output_strides[1]
                        remaining[0] = pixel
                        for dimension, stride in spatial_layout:
                            address[0] = address[0] + remaining[0] % dimension * stride
                            remaining[0] = remaining[0] // dimension
                        Y[address[0]] = T.cast(value, out_dtype)
                    elif transpose:
                        Y[col * output_strides[0] + row * output_strides[1]] = T.cast(value, out_dtype)
                    else:
                        Y[row * output_strides[0] + col * output_strides[1]] = T.cast(value, out_dtype)

    return compile_kernel(kernel, device)


def dequantize_unpack_affine(packed, step, minimum, shape, group_size=512, block_shape=None, *,
                            dtype=torch.bfloat16, transpose=False, bits=2, implementation="auto",
                            tile_shape=None, num_warps=4, row_alignment=16):
    device = require_cuda(packed, step, minimum)
    group = prod(block_shape) if block_shape is not None else group_size
    _bits_consts(bits, group)
    if implementation not in ("auto", "grouped", "tiled", "copy"):
        raise ValueError("implementation must be auto, grouped, tiled, or copy")
    if implementation in ("tiled", "copy") and not transpose:
        raise ValueError("tiled/copy controls require transpose=True")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    if tile_shape is not None and (len(tile_shape) != 2 or any(type(v) is not int or v not in (16, 32, 64, 128) for v in tile_shape)
                                   or prod(tile_shape) > 8192):
        raise ValueError("tile_shape requires two power-of-two dimensions 16..128 and at most 8192 elements")
    if type(row_alignment) is not int or row_alignment not in (1, 16):
        raise ValueError("row_alignment must be 1 or 16")
    rows, cols = shape
    output_shape = (cols, rows) if transpose else (rows, cols)
    strides = (ceildiv(output_shape[1], row_alignment) * row_alignment, 1)
    output = torch.empty_strided(output_shape, strides, dtype=dtype, device=device)
    if output.numel():
        with torch.cuda.device(device):
            _unpack(rows, cols, group, *(block_shape or (0, 0)), bits, output_shape, strides,
                    transpose, 0, dtype_name(dtype), device_key(device))(
                        packed.view(-1), step, minimum, storage_view(output))
    return output


def dequantize_unpack_spatial(packed, step, minimum, shape, group_size, *, dtype, bits, memory_format):
    device = require_cuda(packed, step, minimum)
    output = torch.empty(shape, device=device, dtype=dtype, memory_format=memory_format)
    spatial = prod(shape[2:])
    if output.numel():
        with torch.cuda.device(device):
            _unpack(shape[0] * spatial, shape[1], group_size, 0, 0, bits, tuple(shape), tuple(output.stride()),
                    False, spatial, dtype_name(dtype), device_key(device))(
                        packed.view(-1), step, minimum, storage_view(output))
    return output


def affine_min_max_quantize_pack(x, bits, group_size):
    if not x.numel():
        raise ValueError("x must be non-empty")
    packed, step, minimum = quantize_pack_affine(x, group_size, bits=bits, logical_shape=(1, x.numel()))
    return packed, step, minimum, x.shape


def affine_min_max_dequantize_unpack(packed, scalar, min_x, bits, group_size, ori_shape, output_dtype=torch.float16):
    return dequantize_unpack_affine(packed, scalar, min_x, (1, prod(ori_shape)), group_size,
                                    dtype=output_dtype, bits=bits).reshape(ori_shape)
