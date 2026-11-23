"""Strided FP8 conversion with FP32 statistics and caller-owned buffers."""

from functools import lru_cache

import torch

from .._runtime import ceildiv, compile_kernel, device_key, dtype_name, require_cuda, runtime, storage_view


@lru_cache(maxsize=128)
def _statistics(m, n, sm, sn, span, source_dtype, br, bc, maximum, device):
    _, T = runtime()
    rows, cols = ceildiv(m, br), ceildiv(n, bc)
    groups = rows * cols

    @T.prim_func
    def kernel(X: T.Tensor((span,), source_dtype), S: T.Tensor((groups,), "float32"),
               I: T.Tensor((groups,), "float32"), E: T.Tensor((groups,), "int32")):
        with T.Kernel(ceildiv(groups, 32), threads=32) as bid:
            for lane in T.Parallel(32):
                group = T.cast(bid, "int64") * 32 + lane
                amax = T.alloc_local((1,), "float32")
                bad = T.alloc_local((1,), "int32")
                amax[0] = 0.0
                bad[0] = 0
                if group < groups:
                    for j in T.serial(br * bc):
                        row = group // cols * br + j // bc
                        col = group % cols * bc + j % bc
                        if row < m and col < n:
                            value = T.cast(X[row * sm + col * sn], "float32")
                            amax[0] = T.if_then_else(amax[0] != amax[0], amax[0],
                                                    T.if_then_else(value != value, value, T.max(amax[0], T.abs(value))))
                            if not (T.abs(value) < T.infinity("float32")):
                                bad[0] = 1
                    denominator = T.if_then_else(amax[0] != amax[0], amax[0], T.max(amax[0], 1e-12))
                    multiplier = T.if_then_else(amax[0] == 0.0, 1.0, T.ieee_fdiv(T.float32(maximum), denominator))
                    S[group] = multiplier
                    I[group] = T.ieee_fdiv(T.float32(1.0), multiplier)
                    E[group] = bad[0]

    return compile_kernel(kernel, device)


@lru_cache(maxsize=128)
def _scales_from_amax(groups, maximum, device):
    _, T = runtime()

    @T.prim_func
    def kernel(A: T.Tensor((groups,), "float32"), K: T.Tensor((groups,), "int32"),
               S: T.Tensor((groups,), "float32"), I: T.Tensor((groups,), "float32"),
               E: T.Tensor((groups,), "int32")):
        with T.Kernel(ceildiv(groups, 128), threads=128) as block:
            for lane in T.Parallel(128):
                group = T.cast(block, "int64") * 128 + lane
                if group < groups:
                    magnitude = T.if_then_else(K[group] == 2, T.float32(float("nan")), A[group])
                    denominator = T.if_then_else(K[group] == 2, magnitude, T.max(magnitude, 1e-12))
                    multiplier = T.if_then_else(magnitude == 0.0, 1.0,
                        T.ieee_fdiv(T.float32(maximum), denominator))
                    S[group] = multiplier
                    I[group] = T.ieee_fdiv(T.float32(1.0), multiplier)
                    E[group] = T.cast(K[group] != 0, "int32")

    return compile_kernel(kernel, device)


@lru_cache(maxsize=256)
def _cast(m, n, sm, sn, span, source_dtype, out_dtype, br, bc, srs, scs, scale_span,
          output_stride, transpose, maximum, device, multiply=True):
    _, T = runtime()
    out_span = n * output_stride if transpose else m * output_stride

    @T.prim_func
    def kernel(X: T.Tensor((span,), source_dtype), S: T.Tensor((scale_span,), "float32"),
               Y: T.Tensor((out_span,), out_dtype)):
        with T.Kernel(ceildiv(out_span, 128), threads=128) as bid:
            for lane in T.Parallel(128):
                pos = T.cast(bid, "int64") * 128 + lane
                if pos < out_span:
                    row = pos % output_stride if transpose else pos // output_stride
                    col = pos // output_stride if transpose else pos % output_stride
                    if row < m and col < n:
                        value = T.cast(X[row * sm + col * sn], "float32")
                        factor = S[(row // br) * srs + (col // bc) * scs]
                        scaled = value * factor
                        if multiply:
                            scaled = T.if_then_else(scaled != scaled, scaled, T.min(T.max(scaled, -maximum), maximum))
                        Y[pos] = T.cast(scaled, out_dtype)
                    else:
                        Y[pos] = T.cast(0, out_dtype)

    return compile_kernel(kernel, device)


def amax_scale(x, fp8_max, block_shape, check_finite):
    device = require_cuda(x)
    m, n = x.shape
    br, bc = block_shape or (max(m, 1), max(n, 1))
    # Clip work while preserving the number and location of scale blocks.
    br, bc = min(br, max(m, 1)), min(bc, max(n, 1))
    shape = (ceildiv(m, br), ceildiv(n, bc))
    scale = torch.empty(shape, device=device, dtype=torch.float32)
    inverse = torch.empty_like(scale)
    errors = torch.empty(shape, device=device, dtype=torch.int32)
    if scale.numel():
        from ..scaling.reduction import AMAX_SERIAL_LIMIT, amax_partials
        with torch.cuda.device(device):
            if br * bc <= AMAX_SERIAL_LIMIT:
                source = storage_view(x)
                _statistics(m, n, *x.stride(), source.numel(), dtype_name(x.dtype), br, bc, float(fp8_max), device_key(device))(
                    source, scale.view(-1), inverse.view(-1), errors.view(-1))
            else:
                maxima, kinds = amax_partials(x, br, bc)
                _scales_from_amax(scale.numel(), float(fp8_max), device_key(device))(
                    maxima, kinds, scale.view(-1), inverse.view(-1), errors.view(-1))
        if check_finite and errors.any().item():
            raise ValueError("AmaxScaling requires finite inputs")
    return scale, inverse


def cast_scaled(x, scale, dtype, block_shape=None, *, transpose=False, row_alignment=1, clamp=True):
    device = require_cuda(x, scale)
    m, n = x.shape
    if scale.numel() == 1:
        scale = scale.reshape(1, 1)
    br, bc = block_shape or (max(m, 1), max(n, 1))
    shape = (n, m) if transpose else (m, n)
    stride = ceildiv(shape[1], row_alignment) * row_alignment
    storage = torch.empty((shape[0], stride), device=device, dtype=dtype)
    output = storage[:, :shape[1]]
    if storage.numel():
        source, factors = storage_view(x), storage_view(scale)
        maximum = float(torch.finfo(dtype).max) if clamp else 0.0
        with torch.cuda.device(device):
            _cast(m, n, *x.stride(), source.numel(), dtype_name(x.dtype), dtype_name(dtype), br, bc,
                  *scale.stride(), factors.numel(), stride, transpose, maximum, device_key(device), clamp)(
                      source, factors, storage.view(-1))
    return output


def quantize(x, fp8_dtype, block_shape, check_finite):
    scale, inverse = amax_scale(x, torch.finfo(fp8_dtype).max, block_shape, check_finite)
    return cast_scaled(x, scale, fp8_dtype, block_shape), inverse


def dequantize(values, inverse, block_shape):
    return cast_scaled(values, inverse, torch.float32, block_shape, clamp=False)


def quantize_tensorwise_pair(x, fp8_dtype, check_finite, *, transpose_only=False, statistics=None, autotune=True):
    if type(autotune) is not bool:
        raise ValueError("autotune must be a boolean")
    scale, inverse = amax_scale(x, torch.finfo(fp8_dtype).max, None, check_finite) if statistics is None else statistics
    normal = None if transpose_only else cast_scaled(x, scale, fp8_dtype, row_alignment=16)
    transposed = cast_scaled(x, scale, fp8_dtype, transpose=True, row_alignment=16)
    return normal, inverse, transposed


def quantize_gradient_pair_and_bias(x, fp8_dtype, check_finite, *, transpose_only=False, autotune=True):
    if type(autotune) is not bool:
        raise ValueError("autotune must be a boolean")
    from ..gradient.reduction import sum_columns

    normal, inverse, transposed = quantize_tensorwise_pair(x, fp8_dtype, check_finite,
                                                         transpose_only=transpose_only, autotune=autotune)
    return normal, inverse, transposed, sum_columns(x)


def _run_scaling(x, fp8_max, block_shape, check_finite, fp8_dtype=None):
    if fp8_dtype is None:
        return amax_scale(x, fp8_max, block_shape, check_finite)
    return quantize(x, fp8_dtype, block_shape, check_finite)
