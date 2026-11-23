"""Scale-aware FP32 bias-gradient reduction."""

from functools import lru_cache

import torch

from .._runtime import ceildiv, compile_kernel, device_key, dtype_name, require_cuda, runtime, storage_view


@lru_cache(maxsize=128)
def _columns(m, n, sm, sn, span, source_dtype, scale_span, srs, scs, br, bc, scaled, device):
    _, T = runtime()

    @T.prim_func
    def kernel(X: T.Tensor((span,), source_dtype), S: T.Tensor((scale_span,), "float32"),
               Y: T.Tensor((n,), "float32")):
        with T.Kernel(ceildiv(n, 128), threads=128) as bid:
            for lane in T.Parallel(128):
                col = T.cast(bid, "int64") * 128 + lane
                accumulator = T.alloc_local((1,), "float32")
                accumulator[0] = 0.0
                if col < n:
                    for row in T.serial(m):
                        value = T.cast(X[T.cast(row, "int64") * sm + col * sn], "float32")
                        if scaled:
                            value = value * S[(row // br) * srs + (col // bc) * scs]
                        accumulator[0] = accumulator[0] + value
                    Y[col] = accumulator[0]

    return compile_kernel(kernel, device)


def sum_columns(x, inverse=None, block_shape=None):
    from deepopt.kernel_policy import current_kernel_policy, kernel_backend
    if current_kernel_policy().backend == "mixed" and kernel_backend("gradient") != "tilelang":
        from types import SimpleNamespace
        from .._runtime import auxiliary_module
        factors = inverse if inverse is not None else torch.ones((1, 1), device=x.device, dtype=torch.float32)
        operand = SimpleNamespace(data=x, scale_inv=factors, block_shape=block_shape)
        return auxiliary_module("gradient", "gradient").bias(operand)
    device = require_cuda(x, inverse)
    m, n = x.shape
    source = storage_view(x)
    output = torch.empty(n, device=device, dtype=torch.float32)
    factors = inverse if inverse is not None else torch.empty((1, 1), device=device, dtype=torch.float32)
    if factors.numel() == 1:
        factors = factors.reshape(1, 1)
    scales = storage_view(factors)
    br, bc = block_shape or (max(m, 1), max(n, 1))
    if n:
        with torch.cuda.device(device):
            _columns(m, n, *x.stride(), source.numel(), dtype_name(x.dtype), scales.numel(),
                     *factors.stride(), br, bc, inverse is not None, device_key(device))(source, scales, output)
    return output


def bias(q):
    return sum_columns(q.data, q.scale_inv, q.block_shape)
