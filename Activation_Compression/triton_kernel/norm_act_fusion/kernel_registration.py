import torch
from torch import Tensor
from torch.library import triton_op, wrap_triton
import triton

from typing import Tuple, Optional

from .bn_act_fusion import (bn_relu_fwd_norm_fused_quant_pack_kernel, bn_relu_bwd_reduce_fused_dequant_unpack_kernel, bn_relu_bwd_dx_fused_dequant_unpack_kernel)


def _grid_2d(M, BLOCK_SIZE, C):
    return (triton.cdiv(M, BLOCK_SIZE), C)

def _bits_consts(bits: int, group_size: int):
    if bits not in (1, 2, 4, 8):
        raise ValueError(f"bits must be one of (1,2,4,8). got {bits}")
    vpw = 32 // bits
    nwords = group_size // vpw          
    qmax = (1 << bits) - 1
    return vpw, nwords, qmax

@triton_op("act_lib::bn_relu_fwd_norm_fused_quant_pack", mutates_args={})
def bn_relu_fwd_norm_fused_quant_pack_impl(
    X: Tensor,
    Sum: Tensor,
    Sum_Square: Tensor,
    Weight: Tensor,
    Bias: Tensor,
    mean: Optional[Tensor] = None,
    var: Optional[Tensor] = None,
    *,
    M: int, HW: int,
    stride_n: int, stride_c: int,
    relu: bool, relu6: bool,
    BLOCK_SIZE: int,
    BITS: int,
    sync: bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

    GROUP_SIZE = BLOCK_SIZE // 2
    VPW, NWORDS, QMAX = _bits_consts(BITS, GROUP_SIZE)
    VPW_RELU, NWORDS_RELU, QMAX_RELU = _bits_consts(1, BLOCK_SIZE)

    TOTAL_GROUPS = M // GROUP_SIZE; TOTAL_GROUPS_RELU = M // BLOCK_SIZE
    TOTAL_WORDS = TOTAL_GROUPS * NWORDS; TOTAL_WORDS_RELU = TOTAL_GROUPS_RELU * NWORDS_RELU

    C = X.shape[1]

    y = torch.empty_like(X, dtype=torch.bfloat16, device='cuda')
    x_hat_packed = torch.empty((C, TOTAL_WORDS), dtype=torch.int32, device='cuda')
    relu_mask_packed = torch.empty((C, TOTAL_WORDS_RELU), dtype=torch.int32, device='cuda')
    scale = torch.empty((C, TOTAL_GROUPS), dtype=torch.bfloat16, device='cuda')
    min = torch.empty((C, TOTAL_GROUPS), dtype=torch.bfloat16, device='cuda')

    if not sync:
        mean = torch.empty((C,), dtype=torch.float32, device='cuda')
        var = torch.empty((C,), dtype=torch.float32, device='cuda')

    grid = _grid_2d(M, BLOCK_SIZE, C)
    wrap_triton(bn_relu_fwd_norm_fused_quant_pack_kernel)[grid](
        X, Sum, Sum_Square, Weight, Bias,
        x_hat_packed, relu_mask_packed, y,
        mean, var,
        scale, min,
        M, HW,
        stride_n, stride_c, x_hat_packed.stride(0), scale.stride(0), relu_mask_packed.stride(0),
        relu=relu, relu6=relu6,
        BLOCK_SIZE=BLOCK_SIZE, GROUP_SIZE=GROUP_SIZE, 
        NWORDS=NWORDS, NWORDS_RELU=NWORDS_RELU, VPW=VPW, VPW_RELU=VPW_RELU,
        BITS=BITS, QMAX=QMAX, sync=sync,
        TOTAL_WORDS=TOTAL_WORDS, TOTAL_WORDS_RELU=TOTAL_WORDS_RELU
    )

    if not sync:
        return y, x_hat_packed, relu_mask_packed, scale, min, mean, var
    else:
        return y, x_hat_packed, relu_mask_packed, scale, min, None, None



@triton_op("act_lib::bn_relu_bwd_reduce_fused_dequant_unpack", mutates_args={})
def bn_relu_bwd_reduce_fused_dequant_unpack_impl(
    X_hat_packed: Tensor,
    ReLU_mask_packed: Tensor,
    DY: Tensor,
    Scale: Tensor,
    Min: Tensor,
    *,
    M: int, HW: int,
    stride_n: int, stride_c: int,
    BLOCK_SIZE: int,
    BITS: int
) -> Tuple[Tensor, Tensor]:
    
    VPW, NWORDS, QMAX = _bits_consts(BITS, BLOCK_SIZE // 2)
    VPW_RELU, NWORDS_RELU, QMAX_RELU = _bits_consts(1, BLOCK_SIZE)

    TOTAL_GROUPS = M // (BLOCK_SIZE // 2); TOTAL_GROUPS_RELU = M // BLOCK_SIZE
    TOTAL_WORDS = TOTAL_GROUPS * NWORDS; TOTAL_WORDS_RELU = TOTAL_GROUPS_RELU * NWORDS_RELU

    C = DY.shape[1]

    DW = torch.zeros((C,), dtype=torch.float32, device='cuda')
    DB = torch.zeros((C,), dtype=torch.float32, device='cuda')

    grid = _grid_2d(M, BLOCK_SIZE, C)
    wrap_triton(bn_relu_bwd_reduce_fused_dequant_unpack_kernel)[grid](
        X_hat_packed, ReLU_mask_packed, DY,
        Scale, Min,
        DB, DW,
        M, HW,
        stride_n, stride_c, X_hat_packed.stride(0), Scale.stride(0), ReLU_mask_packed.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        NWORDS=NWORDS, NWORDS_RELU=NWORDS_RELU,
        VPW=VPW, VPW_RELU=VPW_RELU,
        BITS=BITS,
        TOTAL_WORDS=TOTAL_WORDS, TOTAL_WORDS_RELU=TOTAL_WORDS_RELU
    )

    return DW, DB



@triton_op("act_lib::bn_relu_bwd_dx_fused_dequant_unpack",mutates_args={})
def bn_relu_bwd_dx_fused_dequant_unpack_impl(
    X_hat_packed: Tensor,
    ReLU_mask_packed: Tensor,
    DY: Tensor,
    DW: Tensor,
    DB: Tensor,
    Weight: Tensor,
    Var: Tensor,
    Scale: Tensor,
    Min: Tensor,
    *,
    M: int, HW: int,
    stride_n: int, stride_c: int,
    BLOCK_SIZE: int,
    BITS: int
) -> Tensor:
    
    VPW, NWORDS, QMAX = _bits_consts(BITS, BLOCK_SIZE // 2)
    VPW_RELU, NWORDS_RELU, QMAX_RELU = _bits_consts(1, BLOCK_SIZE)

    TOTAL_GROUPS = M // (BLOCK_SIZE // 2); TOTAL_GROUPS_RELU = M // BLOCK_SIZE
    TOTAL_WORDS = TOTAL_GROUPS * NWORDS; TOTAL_WORDS_RELU = TOTAL_GROUPS_RELU * NWORDS_RELU

    C = DY.shape[1]

    DX = torch.empty_like(DY, dtype=torch.bfloat16, device='cuda')

    grid = _grid_2d(M, BLOCK_SIZE, C)
    wrap_triton(bn_relu_bwd_dx_fused_dequant_unpack_kernel)[grid](
        X_hat_packed, ReLU_mask_packed, DY,
        Scale, Min,
        DW, DB, Weight, Var,
        DX,
        M, HW,
        stride_n, stride_c, X_hat_packed.stride(0), Scale.stride(0), ReLU_mask_packed.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        NWORDS=NWORDS, NWORDS_RELU=NWORDS_RELU,
        VPW=VPW, VPW_RELU=VPW_RELU,
        BITS=BITS,
        TOTAL_WORDS=TOTAL_WORDS, TOTAL_WORDS_RELU=TOTAL_WORDS_RELU
    )

    return DX
