"""Autotuned Triton FP8 GEMM with TMA operand loads."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor 

from ..quantization.tensor import QuantizedTensor
from ._triton_autotune import (
    AUTOTUNE_KEY, AUTOTUNE_TILES, MAX_BLOCK_M, MAX_BLOCK_N, MAX_BLOCK_K,
    prune_gemm_configs, set_descriptor_block_shapes,
)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                      num_warps=warps, num_stages=stages, pre_hook=set_descriptor_block_shapes)
        for bm, bn, bk, warps in AUTOTUNE_TILES
        for stages in (1, 2, 3, 4)
    ],
    key=list(AUTOTUNE_KEY),
    prune_configs_by={"early_config_prune": prune_gemm_configs},
)
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
    A_FP8_FORMAT: tl.constexpr, B_FP8_FORMAT: tl.constexpr,
    DEVICE_INDEX: tl.constexpr, DEVICE_CAPABILITY: tl.constexpr,
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
    capability = torch.cuda.get_device_capability(a.data.device)
    if capability[0] < 9:
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
    # Allocate once for every candidate. Descriptor hooks and the launch grid
    # follow the selected tiles, including on autotuner cache hits.
    padded_k = triton.cdiv(k, MAX_BLOCK_K) * MAX_BLOCK_K

    # Represent tensorwise scaling as a single logical scale block when the other operands is blockwise
    a_block_rows, a_block_cols = a.block_shape or (m, padded_k)
    b_block_rows, b_block_cols = b.block_shape or (n, padded_k)

    tensorwise = a.block_shape is None and b.block_shape is None
    with torch.cuda.device(a.data.device):
        a_packed = _pack_fp8_rows(a.data, MAX_BLOCK_K)
        b_packed = _pack_fp8_rows(b.data, MAX_BLOCK_K)

        a_scale_inv = a.scale_inv.contiguous()
        b_scale_inv = b.scale_inv.contiguous()

        a_desc = TensorDescriptor.from_tensor(
            a_packed, [64, 32]
        )
        b_desc = TensorDescriptor.from_tensor(
            b_packed, [64, 32]
        )

        padded_m = triton.cdiv(m, MAX_BLOCK_M) * MAX_BLOCK_M
        padded_n = triton.cdiv(n, MAX_BLOCK_N) * MAX_BLOCK_N

        output_storage = torch.empty(
            (padded_m, padded_n), dtype=out_dtype, device=a.data.device
        )
        c_desc = TensorDescriptor.from_tensor(
            output_storage, [64, 64]
        )

        bias_arg = output_storage if bias is None else bias.contiguous()

        def grid(meta):
            return (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))

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
            A_FP8_FORMAT=str(a.data.dtype), B_FP8_FORMAT=str(b.data.dtype),
            DEVICE_INDEX=torch.cuda.current_device(), DEVICE_CAPABILITY=capability[0] * 10 + capability[1],
            PIPELINE_STAGES=num_stages,
        )
    
    return output_storage[:m, :n]
