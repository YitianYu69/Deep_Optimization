"""Correctness-first FP8 GEMM with independent formats and FP32 scales.

The generic-scale path deliberately uses FP32 SIMT arithmetic. Hardware MXFP8
GEMM consumes a different scale representation; treating arbitrary FP32 scales
as E8M0 or replacing Triton's TF32x3 dot with one TF32 dot loses precision.
"""

from functools import lru_cache

import torch

from .._runtime import ceildiv, compile_kernel, device_key, dtype_name, require_cuda, runtime, storage_view


@lru_cache(maxsize=128)
def _copy_rows(rows, cols, sm, sn, span, padded, source_dtype, device):
    _, T = runtime()

    @T.prim_func
    def kernel(X: T.Tensor((span,), source_dtype), Y: T.Tensor((rows * padded,), source_dtype)):
        with T.Kernel(ceildiv(rows * padded, 128), threads=128) as bid:
            for lane in T.Parallel(128):
                index = T.cast(bid, "int64") * 128 + lane
                if index < rows * padded:
                    row, col = index // padded, index % padded
                    if col < cols:
                        Y[index] = X[row * sm + col * sn]
                    else:
                        Y[index] = T.cast(0, source_dtype)

    return compile_kernel(kernel, device)


def _pack_fp8_rows(data, block_k, *, implementation="auto"):
    if implementation not in ("auto", "flat", "tiled"):
        raise ValueError("FP8 row packing implementation must be auto, flat, or tiled")
    device = require_cuda(data)
    rows, cols = data.shape
    if data.stride(1) == 1 and data.stride(0) >= cols and data.stride(0) % block_k == 0 and data.data_ptr() % 16 == 0:
        return data
    padded = ceildiv(cols, block_k) * block_k
    output = torch.empty((rows, padded), device=device, dtype=data.dtype)
    if output.numel():
        source = storage_view(data)
        with torch.cuda.device(device):
            _copy_rows(rows, cols, *data.stride(), source.numel(), padded, dtype_name(data.dtype), device_key(device))(
                source, output.view(-1))
    return output


@lru_cache(maxsize=256)
def _gemm(m, n, k, asm, asn, bsm, bsn, aspan, bspan, adtype, bdtype,
          assm, assn, bssm, bssn, ascale_span, bscale_span, abr, abc, bbr, bbc,
          tensorwise, bias_dtype, bias_stride, bias_span, out_dtype, output_stride, device):
    _, T = runtime()

    @T.prim_func
    def kernel(A: T.Tensor((aspan,), adtype), B: T.Tensor((bspan,), bdtype),
               AS: T.Tensor((ascale_span,), "float32"), BS: T.Tensor((bscale_span,), "float32"),
               Bias: T.Tensor((max(bias_span, 1),), bias_dtype), Y: T.Tensor((m * output_stride,), out_dtype)):
        with T.Kernel(ceildiv(m * n, 128), threads=128) as bid:
            for lane in T.Parallel(128):
                index = T.cast(bid, "int64") * 128 + lane
                accumulator = T.alloc_local((1,), "float32")
                accumulator[0] = 0.0
                if index < m * n:
                    row, col = index // n, index % n
                    for reduction in T.serial(k):
                        kk = T.cast(reduction, "int64")
                        left = T.cast(A[row * asm + kk * asn], "float32")
                        right = T.cast(B[col * bsm + kk * bsn], "float32")
                        if not tensorwise:
                            left = left * AS[(row // abr) * assm + (kk // abc) * assn]
                            right = right * BS[(col // bbr) * bssm + (kk // bbc) * bssn]
                        accumulator[0] = accumulator[0] + left * right
                    value = accumulator[0]
                    if tensorwise:
                        value = (value * AS[0]) * BS[0]
                    if bias_span:
                        value = value + T.cast(Bias[col * bias_stride], "float32")
                    Y[row * output_stride + col] = T.cast(value, out_dtype)

    return compile_kernel(kernel, device)


@lru_cache(maxsize=256)
def _gemm_fp8(m, n, k, asm, asn, bsm, bsn, aspan, bspan, adtype, bdtype,
              assm, assn, bssm, bssn, ascale_span, bscale_span, abr, abc, bbr, bbc,
              tensorwise, bias_dtype, bias_stride, bias_span, out_dtype, output_stride, device):
    """FP8 matrix instructions with FP32 scaling outside each K32 product.

    Every scale is constant along its 32-element reduction panel. FP32 scale
    values are applied in software; this does not reinterpret them as MX scales.
    """
    _, T = runtime()
    bm, bn, bk = 64, 64, 32

    @T.prim_func
    def kernel(A: T.Tensor((aspan,), adtype), B: T.Tensor((bspan,), bdtype),
               AS: T.Tensor((ascale_span,), "float32"), BS: T.Tensor((bscale_span,), "float32"),
               Bias: T.Tensor((max(bias_span, 1),), bias_dtype), Y: T.Tensor((m * output_stride,), out_dtype)):
        with T.Kernel(ceildiv(n, bn), ceildiv(m, bm), threads=128) as (bx, by):
            left_shared = T.alloc_shared((bm, bk), adtype)
            right_shared = T.alloc_shared((bn, bk), bdtype)
            partial = T.alloc_fragment((bm, bn), "float32")
            accumulator = T.alloc_fragment((bm, bn), "float32")
            T.clear(accumulator)
            for panel in T.serial(ceildiv(k, bk)):
                for i, kk in T.Parallel(bm, bk):
                    row = T.cast(by, "int64") * bm + i
                    reduction = T.cast(panel, "int64") * bk + kk
                    if row < m and reduction < k:
                        left_shared[i, kk] = A[row * asm + reduction * asn]
                    else:
                        left_shared[i, kk] = T.cast(0, adtype)
                for j, kk in T.Parallel(bn, bk):
                    col = T.cast(bx, "int64") * bn + j
                    reduction = T.cast(panel, "int64") * bk + kk
                    if col < n and reduction < k:
                        right_shared[j, kk] = B[col * bsm + reduction * bsn]
                    else:
                        right_shared[j, kk] = T.cast(0, bdtype)
                T.gemm(left_shared, right_shared, partial, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(bm, bn):
                    row = T.cast(by, "int64") * bm + i
                    col = T.cast(bx, "int64") * bn + j
                    if row < m and col < n:
                        value = partial[i, j]
                        if not tensorwise:
                            sa = AS[(row // abr) * assm + ((panel * bk) // abc) * assn]
                            sb = BS[(col // bbr) * bssm + ((panel * bk) // bbc) * bssn]
                            value = (value * sa) * sb
                        accumulator[i, j] = accumulator[i, j] + value
            for i, j in T.Parallel(bm, bn):
                row = T.cast(by, "int64") * bm + i
                col = T.cast(bx, "int64") * bn + j
                if row < m and col < n:
                    value = accumulator[i, j]
                    if tensorwise:
                        value = (value * AS[0]) * BS[0]
                    if bias_span:
                        value = value + T.cast(Bias[col * bias_stride], "float32")
                    Y[row * output_stride + col] = T.cast(value, out_dtype)

    return compile_kernel(kernel, device)


@torch.no_grad()
def gemm_nt(a, b, *, out_dtype=torch.float32, bias=None, num_stages="auto", autotune=True, gradient_output=None):
    if type(autotune) is not bool:
        raise ValueError("autotune must be a boolean")
    if num_stages != "auto" and (type(num_stages) is not int or num_stages not in (1, 2, 3, 4, 5)):
        raise ValueError("num_stages must be 'auto', 1, 2, 3, 4, or 5")
    if out_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Output dtype must be FP32, BF16, or FP16")
    device = require_cuda(a.data, b.data, a.scale_inv, b.scale_inv, bias)
    if torch.version.hip is not None:
        raise ValueError("This backend requires an Nvidia CUDA GPU")
    if torch.cuda.get_device_capability(device) < (9, 0):
        raise ValueError("This backend requires compute capability >= 9")
    m, k = a.data.shape
    n, bk = b.data.shape
    if bk != k:
        raise ValueError("Operands must have the same reduction dimension")
    if bias is not None:
        if tuple(bias.shape) != (n,):
            raise ValueError("Bias must have shape [N]")
        if bias.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError("Bias must be FP32, BF16, or FP16")
    result_dtype = torch.float32 if gradient_output is not None else out_dtype
    alignment = 16 // torch.empty((), dtype=result_dtype).element_size()
    stride = ceildiv(n, alignment) * alignment
    output_storage = torch.empty((m, stride), device=device, dtype=result_dtype)
    output = output_storage[:, :n]
    av, bv = storage_view(a.data), storage_view(b.data)
    asi, bsi = storage_view(a.scale_inv), storage_view(b.scale_inv)
    bias_arg = torch.empty(1, device=device, dtype=torch.float32) if bias is None else storage_view(bias)
    native_fp8 = all(block is None or block[1] % 32 == 0 for block in (a.block_shape, b.block_shape))
    factory = _gemm_fp8 if native_fp8 else _gemm
    with torch.cuda.device(device):
        factory(m, n, k, *a.data.stride(), *b.data.stride(), av.numel(), bv.numel(),
              dtype_name(a.data.dtype), dtype_name(b.data.dtype), *a.scale_inv.stride(), *b.scale_inv.stride(),
              asi.numel(), bsi.numel(), *(a.block_shape or (m, k)), *(b.block_shape or (n, k)),
              a.block_shape is None and b.block_shape is None, dtype_name(bias_arg.dtype),
              bias.stride(0) if bias is not None else 1, bias_arg.numel() if bias is not None else 0,
              dtype_name(result_dtype), stride, device_key(device))(
                  av, bv, asi, bsi, bias_arg, output_storage.view(-1))
    if gradient_output is not None:
        from ..gradient.epilogue import output as gradient_epilogue
        return gradient_epilogue(output, gradient_output)
    return output
