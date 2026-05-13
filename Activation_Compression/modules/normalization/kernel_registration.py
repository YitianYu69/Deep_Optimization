import torch 
from torch.library import triton_op, wrap_triton

from .kernel_implementation import (_bn_fwd_reduce_kernel, _bn_fwd_norm_fused, _bn_bwd_reduce_kernel, _bn_bwd_dx_kernel, 
                                    bn_fwd_norm_quant_pack_fused_kernel, bn_bwd_reduce_dequant_unpack_fused_kernel, bn_bwd_dx_dequant_unpack_fused_kernel,
                                    layer_norm_fwd_kernel, layer_norm_bwd_dx_kernel, layer_norm_bwd_dwdb_kernel)
from ...act_triton_kernel import _bits_consts

import triton

from typing import Tuple, Optional



def _grid_2d(C: int, M: int, BLOCK_M: int):
    return (triton.cdiv(M, BLOCK_M), C)


# -------------------------------------------
# Batch Norm 2d Triton kernel registerations
# -------------------------------------------

@triton_op("act_lib::bn_fwd_reduce", mutates_args=())
def bn_fwd_reduce(
    X: torch.Tensor,
    *,
    M: int,
    HW: int,
    stride_n: int,
    stride_c: int,
    BLOCK_M: int,
    num_warps: int,
    num_stages: int,
) -> Tuple[torch.Tensor, torch.Tensor]:

    C = X.shape[1]
    sum_ = torch.zeros((C,), device=X.device, dtype=torch.float32)
    sumsq_ = torch.zeros((C,), device=X.device, dtype=torch.float32)

    grid = _grid_2d(C, M, BLOCK_M)
    wrap_triton(_bn_fwd_reduce_kernel)[grid](
        X, sum_, sumsq_,
        M, HW,
        stride_n, stride_c,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return sum_, sumsq_


@triton_op("act_lib::bn_fwd_norm", mutates_args=())
def bn_fwd_norm(
    X: torch.Tensor,
    SUM: torch.Tensor,
    SUM_SQUARE: torch.Tensor,
    GAMMA: torch.Tensor,
    BETA: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    var: Optional[torch.Tensor] = None,
    *,
    M: int,
    HW: int,
    stride_n: int,
    stride_c: int,
    BLOCK_M: int,
    sync: bool,
    num_warps: int,
    num_stages: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    C = X.shape[1]
    y = torch.empty_like(X, dtype=torch.bfloat16)
    x_hat = torch.empty_like(X, dtype=torch.bfloat16)

    if not sync:
        var = torch.empty((C,), dtype=torch.float32, device='cuda')
        mean = torch.empty((C,), dtype=torch.float32, device='cuda')

    grid = _grid_2d(C, M, BLOCK_M)
    wrap_triton(_bn_fwd_norm_fused)[grid](
        X, SUM, SUM_SQUARE, y, x_hat,
        GAMMA, BETA, var, mean,
        M, HW,
        stride_n, stride_c,
        BLOCK_SIZE=BLOCK_M,
        sync=sync,
        num_warps=num_warps,
        num_stages=num_stages
    )
    
    if not sync:
        return y.contiguous(), x_hat, mean, var
    else:
        return y.contiguous(), x_hat, None, None

@triton_op("act_lib::bn_fwd_norm_quant_pack_fused", mutates_args={})
def bn_fwd_norm_quant_pack_fused_impl(
    X: torch.Tensor,
    SUM: torch.Tensor,
    SUM_SQUARE: torch.Tensor,
    GAMMA: torch.Tensor,
    BETA: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    var: Optional[torch.Tensor] = None,
    *,
    M: int,
    HW: int,
    stride_n: int,
    stride_c: int,
    BLOCK_M: int,
    bits: int,
    sync: bool,
    avg_alam: bool,
    alam_bits: int,
    num_warps: int,
    num_stages: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    GROUP_SIZE = BLOCK_M // 2
    VPW, NWORDS, QMAX = _bits_consts(bits, GROUP_SIZE)
    TOTAL_GROUPS = M // GROUP_SIZE
    TOTAL_WORDS = TOTAL_GROUPS * NWORDS
    C = X.shape[1]

    y = torch.empty_like(X, dtype=torch.bfloat16, device='cuda')
    scale = torch.empty((C, TOTAL_GROUPS), dtype=torch.bfloat16, device='cuda')
    min = torch.empty((C, TOTAL_GROUPS), dtype=torch.bfloat16, device='cuda')

    if avg_alam:
        ALAM_NOWRDS = triton.cdiv(NWORDS, alam_bits)
        SUB_GROUP = int(ALAM_NOWRDS * VPW)
        TOTAL_WORDS = TOTAL_GROUPS * ALAM_NOWRDS
        x_hat_packed = torch.empty((C, TOTAL_WORDS), dtype=torch.int32, device='cuda')
    else:
        ALAM_NOWRDS = NWORDS
        SUB_GROUP = 0  # dummy value
        x_hat_packed = torch.empty((C, TOTAL_WORDS), dtype=torch.int32, device='cuda')


    if not sync:
        mean = torch.empty((C,), dtype=torch.float32, device='cuda')
        var = torch.empty((C,), dtype=torch.float32, device='cuda')

    grid = _grid_2d(C, M, BLOCK_M)
    wrap_triton(bn_fwd_norm_quant_pack_fused_kernel)[grid](
        X, SUM, SUM_SQUARE, GAMMA, BETA,
        y, x_hat_packed, scale, min,
        mean, var,
        M, HW,
        stride_n, stride_c, x_hat_packed.stride(0), scale.stride(0),
        BLOCK_SIZE=BLOCK_M, GROUP_SIZE=GROUP_SIZE,
        BITS=bits, VPW=VPW, NWORDS=NWORDS, TOTAL_WORDS=TOTAL_WORDS, QMAX=QMAX,
        sync=sync,
        AVG_ALAM=avg_alam, ALAM_BITS=alam_bits, ALAM_NWORDS=ALAM_NOWRDS, SUB_GROUP=SUB_GROUP,
        num_warps=num_warps,
        num_stages=num_stages
    )

    if not sync:
        return y.contiguous(), x_hat_packed, scale, min, mean, var
    else:
        return y.contiguous(), x_hat_packed, scale, min, None, None


@triton_op("act_lib::bn_bwd_reduce", mutates_args=())
def bn_bwd_reduce(
    X: torch.Tensor,
    DY: torch.Tensor,
    *,
    M: int,
    HW: int,
    stride_n: int,
    stride_c: int,
    BLOCK_M: int,
    num_warps: int,
    num_stages: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    C = X.shape[1]

    dbeta  = torch.zeros((C,), device=X.device, dtype=torch.float32)
    dgamma = torch.zeros((C,), device=X.device, dtype=torch.float32)

    grid = _grid_2d(C, M, BLOCK_M)
    wrap_triton(_bn_bwd_reduce_kernel)[grid](
        X, DY,
        dbeta, dgamma,
        M, HW,
        stride_n, stride_c,
        BLOCK_M=BLOCK_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dgamma.contiguous(), dbeta.contiguous()


@triton_op("act_lib::bn_bwd_dx", mutates_args=())
def bn_bwd_dx(
    X: torch.Tensor,
    DY: torch.Tensor,
    VAR: torch.Tensor,
    GAMMA: torch.Tensor,
    DBETA: torch.Tensor,
    DGAMMA: torch.Tensor,
    *,
    M: int,
    HW: int,
    stride_n: int,
    stride_c: int,
    BLOCK_M: int,
    sync: bool,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    C = X.shape[1]
    dx = torch.empty_like(X, dtype=torch.bfloat16)

    grid = _grid_2d(C, M, BLOCK_M)
    wrap_triton(_bn_bwd_dx_kernel)[grid](
        X, DY, VAR, GAMMA,
        DBETA, DGAMMA,
        dx,
        M, HW,
        stride_n, stride_c,
        BLOCK_M=BLOCK_M,
        sync=sync,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dx.contiguous()


@triton_op("act_lib::bn_bwd_reduce_dequant_unpack_fused", mutates_args=())
def bn_bwd_reduce_dequant_unpack_fused_impl(
    x_hat_packed: torch.Tensor, dy: torch.Tensor, scale: torch.Tensor, min: torch.Tensor,
    *,
    M: int, HW: int, stride_n: int, stride_c: int, BLOCK_M: int, bits: int,
    avg_alam: bool, alam_bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    VPW, NWORDS, QMAX = _bits_consts(bits, (BLOCK_M // 2))
    TOTAL_GROUPS = M // (BLOCK_M // 2)
    TOTAL_WORDS = TOTAL_GROUPS * NWORDS
    C = dy.shape[1]

    if avg_alam:
        ALAM_NWORDS = triton.cdiv(NWORDS, alam_bits)
        SUB_GROUP = int(ALAM_NWORDS * VPW)
        TOTAL_WORDS = TOTAL_GROUPS * ALAM_NWORDS
    else:
        ALAM_NWORDS = NWORDS
        SUB_GROUP = 0  # dummy value

    DW = torch.zeros((C,), dtype=torch.float32, device='cuda')
    DB = torch.zeros((C,), dtype=torch.float32, device='cuda')

    grid = _grid_2d(C, M, BLOCK_M)
    wrap_triton(bn_bwd_reduce_dequant_unpack_fused_kernel)[grid](
        x_hat_packed, dy, scale, min, 
        DW, DB,
        M, HW,
        stride_n, stride_c, x_hat_packed.stride(0), scale.stride(0),
        BLOCK_SIZE=BLOCK_M, BITS=bits, VPW=VPW, NWORDS=NWORDS, TOTAL_WORDS=TOTAL_WORDS,
        AVG_ALAM=avg_alam, ALAM_BITS=alam_bits, ALAM_NWORDS=ALAM_NWORDS, SUB_GROUP=SUB_GROUP
    )

    return DW.contiguous(), DB.contiguous()


@triton_op("act_lib::bn_bwd_dx_dequant_unpack_fused", mutates_args=())
def bn_bwd_dx_dequant_unpack_fused_impl(
    x_hat_packed: torch.Tensor, dy: torch.Tensor, scale: torch.Tensor, min: torch.Tensor,
    DW: torch.Tensor, DB: torch.Tensor, W: torch.Tensor, VAR: torch.Tensor,
    *,
    M: int, HW: int, stride_n: int, stride_c: int, BLOCK_M: int, bits: int, sync: bool,
    avg_alam: bool, alam_bits: int
) -> torch.Tensor:
    
    VPW, NWORDS, QMAX = _bits_consts(bits, (BLOCK_M // 2))
    TOTAL_GROUPS = M // (BLOCK_M // 2)
    TOTAL_WORDS = TOTAL_GROUPS * NWORDS

    if avg_alam:
        ALAM_NWORDS = triton.cdiv(NWORDS, alam_bits)
        SUB_GROUP = int(ALAM_NWORDS * VPW)
        TOTAL_WORDS = TOTAL_GROUPS * ALAM_NWORDS
    else:
        ALAM_NWORDS = NWORDS
        SUB_GROUP = 0  # dummy value

    C = dy.shape[1]
    dx = torch.empty_like(dy, dtype=torch.bfloat16)
    
    grid = _grid_2d(C, M, BLOCK_M)
    wrap_triton(bn_bwd_dx_dequant_unpack_fused_kernel)[grid](
        x_hat_packed, dy, scale, min, DW, DB, W, VAR,
        dx,
        M, HW,
        stride_n, stride_c, x_hat_packed.stride(0), scale.stride(0),
        BLOCK_SIZE=BLOCK_M, BITS=bits, VPW=VPW, NWORDS=NWORDS, TOTAL_WORDS=TOTAL_WORDS,
        sync=sync,
        AVG_ALAM=avg_alam, ALAM_BITS=alam_bits, ALAM_NWORDS=ALAM_NWORDS, SUB_GROUP=SUB_GROUP
    )

    return dx.contiguous()



# -------------------------------------------
# Layer Norm 2d Triton kernel registerations
# -------------------------------------------

@triton_op("act_lib::layer_norm_fwd", mutates_args=())
def layer_norm_fwd_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    
    y = torch.empty_like(x)

    x_args = x.view(-1, x.shape[-1])
    M, C = x_args.shape
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    x_hat = torch.empty_like(x_args)

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(C))
    if C > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    # kernel
    wrap_triton(layer_norm_fwd_kernel)[(M,)](
        x_args, y.view(M, C), weight, bias, x_hat, rstd,
        x_args.stride(0), C, 1e-8,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1
    )

    return y.view_as(x).contiguous(), x_hat, rstd, BLOCK_SIZE, num_warps



@triton_op("act_lib::layer_norm_bwd_dx", mutates_args=())
def layer_norm_bwd_dx_impl(
    dy: torch.Tensor, weight: torch.Tensor, x_hat: torch.Tensor, rstd: torch.Tensor,
    *,
    BLOCK_SIZE_C: int, num_warps: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    
    C = weight.shape[0]
    GROUP_SIZE_M = 64
    if C <= 8192: GROUP_SIZE_M = 96
    if C <= 4096: GROUP_SIZE_M = 128
    if C <= 1024: GROUP_SIZE_M = 256

    locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=x_hat.device)
    _dw = torch.empty((GROUP_SIZE_M, C), dtype=torch.float32, device=x_hat.device)
    _db = torch.empty((GROUP_SIZE_M, C), dtype=torch.float32, device=x_hat.device)
    dx = torch.empty_like(x_hat)

    M, C = x_hat.shape
    wrap_triton(layer_norm_bwd_dx_kernel)[(M,)](
        dx, dy.view_as(x_hat), _dw, _db,  weight, x_hat, rstd,
        locks, x_hat.stride(0), C, 
        BLOCK_SIZE_C=BLOCK_SIZE_C, GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps
    )

    return dx.view_as(dy).contiguous(), _dw, _db, M, C


@triton_op("act_lib::layer_norm_bwd_dwdb", mutates_args=())
def layer_norm_bwd_dwdb_impl(
    _dw: torch.Tensor, _db: torch.Tensor,
    M: int, C: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    dw = torch.empty((C,), dtype=torch.float32, device=_dw.device)
    db = torch.empty((C,), dtype=torch.float32, device=_dw.device)

    grid = lambda META: triton.cdiv(C, META['BLOCK_SIZE_C'])
    wrap_triton(layer_norm_bwd_dwdb_kernel)[grid](
        _dw, _db, dw, db,
        M, C,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=128,
        num_ctas=1
    )

    return dw.contiguous(), db.contiguous()