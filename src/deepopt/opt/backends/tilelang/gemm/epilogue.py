"""FP32 bias addition and native output casting."""

from functools import lru_cache

import torch

from .._runtime import ceildiv, compile_kernel, device_key, dtype_name, require_cuda, runtime, storage_view


@lru_cache(maxsize=128)
def _bias(rows, cols, sm, sn, span, source_dtype, bias_dtype, bs, bspan, output_dtype, device):
    _, T = runtime()

    @T.prim_func
    def kernel(X: T.Tensor((span,), source_dtype), B: T.Tensor((bspan,), bias_dtype),
               Y: T.Tensor((rows * cols,), output_dtype)):
        with T.Kernel(ceildiv(rows * cols, 128), threads=128) as bid:
            for lane in T.Parallel(128):
                index = T.cast(bid, "int64") * 128 + lane
                if index < rows * cols:
                    row, col = index // cols, index % cols
                    Y[index] = T.cast(T.cast(X[row * sm + col * sn], "float32") + T.cast(B[col * bs], "float32"), output_dtype)

    return compile_kernel(kernel, device)


def bias_output(product, bias, dtype):
    device = require_cuda(product, bias)
    output = torch.empty(product.shape, device=device, dtype=dtype)
    source, b = storage_view(product), storage_view(bias)
    if output.numel():
        with torch.cuda.device(device):
            _bias(*product.shape, *product.stride(), source.numel(), dtype_name(product.dtype),
                  dtype_name(bias.dtype), bias.stride(0), b.numel(), dtype_name(dtype), device_key(device))(
                      source, b, output.view(-1))
    return output
