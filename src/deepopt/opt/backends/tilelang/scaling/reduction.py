"""Strided amax with bounded parallel tiles and explicit nonfinite precedence."""

from functools import lru_cache

import torch

from .._runtime import ceildiv, compile_kernel, device_key, dtype_name, require_cuda, runtime, storage_view


AMAX_SERIAL_LIMIT = 256
AMAX_CHUNK = 1024


@lru_cache(maxsize=128)
def _partial_amax(m, n, sm, sn, span, source_dtype, br, bc, parts, device):
    _, T = runtime()
    columns = ceildiv(n, bc)
    groups = ceildiv(m, br) * columns

    @T.prim_func
    def kernel(X: T.Tensor((span,), source_dtype), A: T.Tensor((groups, parts), "float32"),
               K: T.Tensor((groups, parts), "int32")):
        with T.Kernel(groups * parts, threads=256) as block:
            program = T.cast(block, "int64")
            group, part = program // parts, program % parts
            values = T.alloc_fragment((AMAX_CHUNK,), "float32")
            kinds = T.alloc_fragment((AMAX_CHUNK,), "int32")
            maximum = T.alloc_fragment((1,), "float32")
            kind = T.alloc_fragment((1,), "int32")
            for lane in T.Parallel(AMAX_CHUNK):
                offset = part * AMAX_CHUNK + lane
                row = group // columns * br + offset // bc
                col = group % columns * bc + offset % bc
                values[lane] = 0.0
                kinds[lane] = 0
                if (offset < br * bc) & (row < m) & (col < n):
                    value = T.cast(X[row * sm + col * sn], "float32")
                    # Reducing clean magnitudes and an integer classification
                    # makes NaN win independently of the GPU's max intrinsic.
                    values[lane] = T.if_then_else(value != value, 0.0, T.abs(value))
                    kinds[lane] = T.if_then_else(value != value, 2,
                        T.if_then_else(T.abs(value) == T.infinity("float32"), 1, 0))
            T.reduce_max(values, maximum, dim=0, clear=True)
            T.reduce_max(kinds, kind, dim=0, clear=True)
            for lane in T.Parallel(1):
                A[group, part] = maximum[0]
                K[group, part] = kind[0]

    return compile_kernel(kernel, device)


@lru_cache(maxsize=128)
def _merge_amax(groups, count, following, device):
    _, T = runtime()

    @T.prim_func
    def kernel(A: T.Tensor((groups, count), "float32"), K: T.Tensor((groups, count), "int32"),
               NextA: T.Tensor((groups, following), "float32"),
               NextK: T.Tensor((groups, following), "int32")):
        with T.Kernel(groups * following, threads=256) as block:
            program = T.cast(block, "int64")
            group, part = program // following, program % following
            values = T.alloc_fragment((AMAX_CHUNK,), "float32")
            kinds = T.alloc_fragment((AMAX_CHUNK,), "int32")
            maximum = T.alloc_fragment((1,), "float32")
            kind = T.alloc_fragment((1,), "int32")
            for lane in T.Parallel(AMAX_CHUNK):
                offset = part * AMAX_CHUNK + lane
                values[lane] = 0.0
                kinds[lane] = 0
                if offset < count:
                    values[lane] = A[group, offset]
                    kinds[lane] = K[group, offset]
            T.reduce_max(values, maximum, dim=0, clear=True)
            T.reduce_max(kinds, kind, dim=0, clear=True)
            for lane in T.Parallel(1):
                NextA[group, part] = maximum[0]
                NextK[group, part] = kind[0]

    return compile_kernel(kernel, device)


def amax_partials(x, block_row, block_col):
    """Return complete magnitudes and kinds; each CTA reads at most 1024 values.

    Scratch stores one FP32 magnitude and one int32 kind per chunk. Successive
    levels shrink by 1024; no stage scans an entire large source group serially.
    Kind 2 denotes NaN and takes priority over kind 1 (infinity).
    """
    device = require_cuda(x)
    m, n = x.shape
    groups = ceildiv(m, block_row) * ceildiv(n, block_col)
    parts = ceildiv(block_row * block_col, AMAX_CHUNK)
    maxima = torch.empty((groups, parts), device=device, dtype=torch.float32)
    kinds = torch.empty((groups, parts), device=device, dtype=torch.int32)
    if groups:
        source = storage_view(x)
        with torch.cuda.device(device):
            _partial_amax(m, n, *x.stride(), source.numel(), dtype_name(x.dtype), block_row, block_col,
                parts, device_key(device))(source, maxima, kinds)
            while parts > 1:
                following = ceildiv(parts, AMAX_CHUNK)
                reduced = torch.empty((groups, following), device=device, dtype=torch.float32)
                reduced_kinds = torch.empty((groups, following), device=device, dtype=torch.int32)
                _merge_amax(groups, parts, following, device_key(device))(maxima, kinds, reduced, reduced_kinds)
                maxima, kinds, parts = reduced, reduced_kinds, following
    return maxima.reshape(groups), kinds.reshape(groups)


@lru_cache(maxsize=128)
def _finish_amax(groups, floor, device):
    _, T = runtime()

    @T.prim_func
    def kernel(A: T.Tensor((groups,), "float32"), K: T.Tensor((groups,), "int32"),
               Y: T.Tensor((groups,), "float32")):
        with T.Kernel(ceildiv(groups, 128), threads=128) as block:
            for lane in T.Parallel(128):
                group = T.cast(block, "int64") * 128 + lane
                if group < groups:
                    Y[group] = T.if_then_else(K[group] == 2, T.float32(float("nan")), T.max(A[group], floor))

    return compile_kernel(kernel, device)


@lru_cache(maxsize=128)
def _amax(m, n, sm, sn, span, source_dtype, br, bc, floor, device):
    _, T = runtime()
    columns = ceildiv(n, bc)
    groups = ceildiv(m, br) * columns

    @T.prim_func
    def kernel(X: T.Tensor((span,), source_dtype), A: T.Tensor((groups,), "float32")):
        with T.Kernel(ceildiv(groups, 32), threads=32) as bid:
            for lane in T.Parallel(32):
                group = T.cast(bid, "int64") * 32 + lane
                maximum = T.alloc_local((1,), "float32")
                maximum[0] = 0.0
                if group < groups:
                    for offset in T.serial(br * bc):
                        row = group // columns * br + offset // bc
                        col = group % columns * bc + offset % bc
                        if row < m and col < n:
                            value = T.cast(X[row * sm + col * sn], "float32")
                            maximum[0] = T.if_then_else(maximum[0] != maximum[0], maximum[0],
                                T.if_then_else(value != value, value, T.max(maximum[0], T.abs(value))))
                    A[group] = T.if_then_else(maximum[0] != maximum[0], maximum[0], T.max(maximum[0], floor))

    return compile_kernel(kernel, device)


def block_amax(x, block_row, block_col, *, minimum=1e-12):
    device = require_cuda(x)
    m, n = x.shape
    br, bc = min(block_row, max(m, 1)), min(block_col, max(n, 1))
    output = torch.empty((ceildiv(m, br), ceildiv(n, bc)), device=device, dtype=torch.float32)
    if output.numel():
        with torch.cuda.device(device):
            if br * bc <= AMAX_SERIAL_LIMIT:
                source = storage_view(x)
                _amax(m, n, *x.stride(), source.numel(), dtype_name(x.dtype), br, bc, float(minimum), device_key(device))(
                    source, output.view(-1))
            else:
                maxima, kinds = amax_partials(x, br, bc)
                _finish_amax(output.numel(), float(minimum), device_key(device))(maxima, kinds, output.view(-1))
    return output
