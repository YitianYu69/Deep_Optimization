"""
Tritin FP8 GEMM with TMA operand loads.
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor 

from math import gcd

from ..quantization.tensor import QuantizedTensor


@triton.jit
def _gemm_nt_kernel(
    a_desc, b_desc, c_desc,
    a_scale_inv_ptr, b_scale_inv_ptr,
    bias_ptr,
    M, N, K,
    A_BLOCK_ROWS: tl.constexpr, A_BLOCK_COLS: tl.constexpr,
    B_BLOCK_ROWS: tl.constexpr, B_BLOCK_COLS: tl.constexpr,
    A_SCALE_COLS: tl.constexpr, B_SCALE_COLS: tl.constexpr,
    TENSORWISE: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr
):

    start_m = tl.program_id(0) * BLOCK_M
    start_n = tl.program_id(1) * BLOCK_N

    rows = start_m + tl.arange(0, BLOCK_M)
    cols = start_n + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_tile in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=PIPELINE_STAGES):
        start_k = k_tile * BLOCK_K

        a_tile = a_desc.load([start_m, start_k])
        b_tile = b_desc.load([start_n, start_k])

        if TENSORWISE:
            acc = tl.dot(a_tile, b_tile.T, acc, out_dtype=tl.float32, max_num_imprecise_acc=32)
        elif NATIVE_FP8:
            partial = tl.dot(a_tile, b_tile.T, out_dtype=tl.float32, max_num_imprecise_acc=32)

            a_scale_offset = (rows // A_BLOCK_ROWS) * A_SCALE_COLS + start_k // A_BLOCK_COLS
            b_scale_offset = (cols // B_BLOCK_ROWS) * B_SCALE_COLS + start_k // B_BLOCK_COLS

            a_scale_inv = tl.load(a_scale_inv_ptr + a_scale_offset, mask=rows<M, other=0.0)
            b_scale_inv = tl.load(b_scale_inv_ptr + b_scale_offset, mask=cols<N, other=0.0)

            scaled_partial = (partial * a_scale_inv[:, None]) * b_scale_inv[None, :]
            acc = acc + scaled_partial
        else:
            # Transposing rowwise blocks can make scales vary at every K.
            # Apply those scales before reduction without requantizing FP8.
            ks = start_k + tl.arange(0, BLOCK_K)
            a_scale_offset = ((rows // A_BLOCK_ROWS)[:, None] * A_SCALE_COLS
                              + (ks // A_BLOCK_COLS)[None, :])
            b_scale_offset = ((cols // B_BLOCK_ROWS)[:, None] * B_SCALE_COLS
                              + (ks // B_BLOCK_COLS)[None, :])
            a_scales = tl.load(a_scale_inv_ptr + a_scale_offset,
                               mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
            b_scales = tl.load(b_scale_inv_ptr + b_scale_offset,
                               mask=(cols[:, None] < N) & (ks[None, :] < K), other=0.0)
            a_values = a_tile.to(tl.float32) * a_scales
            b_values = b_tile.to(tl.float32) * b_scales
            acc = tl.dot(a_values, b_values.T, acc, out_dtype=tl.float32,
                         input_precision="tf32x3")
    
    if TENSORWISE:
        a_scale_inv = tl.load(a_scale_inv_ptr)
        b_scale_inv = tl.load(b_scale_inv_ptr)
        acc = (acc * a_scale_inv) * b_scale_inv

    if HAS_BIAS:
        cols = start_n + tl.arange(0, BLOCK_N)
        bias = tl.load(bias_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]
    
    c_desc.store([start_m, start_n], acc.to(OUTPUT_DTYPE))




def _pack_fp8_rows(data: torch.Tensor, block_k: int) -> torch.Tensor:
    """
    Make FP8 rows contiguous and pad K with zeros
    """

    rows, cols = data.shape
    padded_cols = triton.cdiv(cols, block_k) * block_k

    if (cols == padded_cols and data.is_contiguous() and data.stride(1) == 1 and data.data_ptr() % 16 == 0):
        return data
    
    packed = torch.zeros(
        (rows, padded_cols),
        device=data.device, dtype=data.dtype
    )
    packed[:, :cols].copy_(data)
    return packed




@torch.no_grad()
def gemm_nt(
    a: QuantizedTensor,
    b: QuantizedTensor,
    *,
    out_dtype: torch.dtype = torch.float32,
    bias: torch.Tensor | None = None,
    num_stages: int = 3
) -> torch.Tensor:

    """
    Compute A @ B.t with independent FP8 formats and TMA loads/stores.

    Scale blocks aligned along K use native FP8 dot products. Other layouts
    dequantize tiles to FP32 before a TF32x3 dot, retaining the stored FP8
    values and their scales without additional FP8 rounding.
    """
    
    if a.data.device != b.data.device:
        raise ValueError("Operands must be on the same device")
    if a.data.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("This backend requires an Nvidia CUDA GPU")
    if torch.cuda.get_device_capability(a.data.device)[0] < 9:
        raise ValueError("This backend requires compute cabaility >= 9")
    if a.data.shape[1] != b.data.shape[1]:
        raise ValueError("Operands must have the same reduction dimension")
    if type(num_stages) is not int or num_stages not in (1, 2, 3, 4):
        raise ValueError("num_stages must be 1, 2, 3, 4")
    
    output_types = {
        torch.float32 : tl.float32,
        torch.bfloat16 : tl.bfloat16,
        torch.float16 : tl.float16
    }
    if out_dtype not in output_types:
        raise ValueError("Output dtype must be FP32, BF16, or FP16")

    m, k = a.data.shape
    n, _ = b.data.shape
    
    if bias is not None:
        if bias.device != a.data.device:
            raise ValueError("Bias must be on the operands' device")
        if tuple(bias.shape) != (n,):
            raise ValueError("Bias must have shape [N]")
        if bias.dtype not in output_types:
            raise TypeError("Bias must be FP32, BF16, or FP16")

    native_fp8 = all(operand.block_shape is None or operand.block_shape[1] % 32 == 0
                     for operand in (a, b))
    if native_fp8:
        block_m, block_n, block_k, num_warps = 128, 128, 64, 8
        for operand in (a, b):
            if operand.block_shape is not None:
                # No dot interval may cross either operand's scale boundary.
                block_k = gcd(block_k, operand.block_shape[1])
    else:
        # FP32 operands require more resources than native FP8 tiles.
        block_m, block_n, block_k, num_warps = 64, 64, 32, 4
    
    padded_k = triton.cdiv(k, block_k) * block_k

    # Represent tensorwise scaling as a single logical scale block when the other operands is blockwise
    a_block_rows, a_block_cols = a.block_shape or (m, padded_k)
    b_block_rows, b_block_cols = b.block_shape or (n, padded_k)

    tensorwise = a.block_shape is None and b.block_shape is None
    with torch.cuda.device(a.data.device):
        a_packed = _pack_fp8_rows(a.data, block_k)
        b_packed = _pack_fp8_rows(b.data, block_k)

        a_scale_inv = a.scale_inv.contiguous()
        b_scale_inv = b.scale_inv.contiguous()

        a_desc = TensorDescriptor.from_tensor(
            a_packed, [block_m, block_k]
        )
        b_desc = TensorDescriptor.from_tensor(
            b_packed, [block_n, block_k]
        )

        padded_m = triton.cdiv(m, block_m) * block_m
        padded_n = triton.cdiv(n, block_n) * block_n

        output_storage = torch.empty(
            (padded_m, padded_n), dtype=out_dtype, device=a.data.device
        )
        c_desc = TensorDescriptor.from_tensor(
            output_storage, [block_m, block_n]
        )

        bias_arg = output_storage if bias is None else bias.contiguous()

        grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
        _gemm_nt_kernel[grid](
            a_desc, b_desc, c_desc,
            a_scale_inv, b_scale_inv,
            bias_arg,
            m, n, k,
            A_BLOCK_ROWS=a_block_rows, A_BLOCK_COLS=a_block_cols,
            B_BLOCK_ROWS=b_block_rows, B_BLOCK_COLS=b_block_cols,
            A_SCALE_COLS=a_scale_inv.shape[1], B_SCALE_COLS=b_scale_inv.shape[1],
            TENSORWISE=tensorwise,
            NATIVE_FP8=native_fp8,
            HAS_BIAS=bias is not None,
            OUTPUT_DTYPE=output_types[out_dtype],
            BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_K=block_k,
            PIPELINE_STAGES=num_stages,
            num_warps=num_warps, num_stages=num_stages
        )
    
    return output_storage[:m, :n]
