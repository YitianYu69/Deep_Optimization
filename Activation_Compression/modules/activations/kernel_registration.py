from modules.tensor_act_reshape_utils import spatical_aware_act_tensor_reshape_back
import torch 
from torch.library import triton_op, wrap_triton

from .kernel_implementation import (
    relu_variance_kernel, relu_variance_fwd_fused_pack_kernel, relu_bwd_fused_unpack_kernel,
    silu_triton, silu_fwd_fused_quan_pack_triton, silu_bwd_fused_dequan_unpack_triton,
    gelu_triton, gelu_fwd_fused_quan_pack_triton, gelu_bwd_fused_dequan_unpack_triton)

from modules.tensor_act_reshape_utils import spatical_aware_act_tensor_reshape, spatical_aware_act_tensor_reshape_back

from ACT6.act_triton_kernel import _bits_consts

import triton

from typing import Tuple


# ------
# ReLU
# ------
@triton_op("act_lib::relu_triton", mutates_args={})
def relu_triton(x: torch.Tensor, relu: bool, relu6: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.is_cuda and x.is_contiguous

    n_elts = x.numel()
    relu_mask = torch.empty(x.shape, dtype=torch.int8, device='cuda')
    y = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elts, meta['BLOCK_SIZE']),)
    wrap_triton(relu_variance_kernel)[grid](
        x, y, relu_mask,
        n_elts,
        relu=relu,
        relu6=relu6,
        BLOCK_SIZE=1024
    )

    return y, relu_mask


@triton_op("act_lib::relu_fwd_fused_pack", mutates_args={})
def relu_fwd_fused_pack(x: torch.Tensor, relu: bool, relu6: bool, bits: int, group_size: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    assert x.is_cuda and x.is_contiguous

    VPW = 32 // bits
    NWORDS = group_size // VPW

    input_shape = x.shape
    N = x.shape[0]
    x2 = x.view(N, -1, group_size)
    G = x2.shape[1]

    NG = N * G
    x3 = x2.view(NG, group_size)

    y = torch.empty((NG, group_size), dtype=x.dtype, device='cuda')
    packed = torch.empty((NG, NWORDS), dtype=torch.int32, device='cuda')

    grid = (NG,)
    wrap_triton(relu_variance_fwd_fused_pack_kernel)[grid](
        x3, packed, y,
        x3.stride(0), x3.stride(1),
        packed.stride(0), packed.stride(1),
        y.stride(0), y.stride(1),
        relu=relu,
        relu6=relu6,
        BITS=bits,
        VPW=VPW,
        NWORDS=NWORDS
    )
    return y.view(*input_shape), packed, N, G

@triton_op("act_lib::relu_bwd_fused_unpack", mutates_args={})
def relu_bwd_fused_unpack(packed: torch.Tensor, dy: torch.Tensor, bits: int, N: int, G: int, group_size: int) -> torch.Tensor:
    NG = N * G
    VPW = 32 // bits
    NWORDS = group_size // VPW
    output_shape = dy.shape

    dy2 = dy.view(-1, group_size)
    dx = torch.empty((N * G, group_size), dtype=torch.bfloat16, device='cuda')

    grid = (NG,)
    wrap_triton(relu_bwd_fused_unpack_kernel)[grid](
        packed, dy2, dx,
        packed.stride(0), packed.stride(1),
        dy2.stride(0), dy2.stride(1),
        dx.stride(0), dx.stride(1),
        BITS=bits,
        VPW=VPW,
        NWORDS=NWORDS
    )
    return dx.view(*output_shape)


@torch.library.register_fake("act_lib::relu_triton")
def relu_triton_fake(x: torch.Tensor):
    relu_mask = torch.empty_like(x, dtype=torch.int8)
    y = torch.empty_like(x)

    return y, relu_mask


@torch.library.register_fake("act_lib::relu_fwd_fused_pack")
def relu_fwd_fused_pack_fake(x: torch.Tensor, bits: int, group_size: int):
    N = x.shape[0]
    x2 = x.view(N, -1, group_size)
    G = x2.shape[1]

    NG = N * G
    VPW = 32 // bits
    NWORDS = group_size // VPW

    y = torch.empty(x.shape, dtype=x.dtype)
    packed = torch.empty((NG, NWORDS), dtype=torch.int32)
    return y, packed, N, G

@torch.library.register_fake("act_lib::relu_bwd_fused_unpack")
def relu_bwd_fused_unpack_fake(packed: torch.Tensor, dy: torch.Tensor, bits: int, N: int, G: int, group_size: int):
    dx = torch.empty_like(dy, dtype=torch.bfloat16)

    return dx




# ------
# SiLU
# ------

@triton_op("act_lib::silu_triton", mutates_args={})
def silu_triton_impl(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    n_elt = x.numel()
    y = torch.empty_like(x, dtype=torch.bfloat16, device='cuda')
    act = torch.empty_like(x, dtype=torch.bfloat16, device='cuda')

    grid = (triton.cdiv(n_elt, 1024),)
    wrap_triton(silu_triton)[grid](
        x, y, act,
        n_elt,
        BLOCK_SIZE=1024
    )

    return y, act


@triton_op("act_lib::silu_fwd_quan_pack", mutates_args={})
def silu_fwd_fused_quan_pack_impl(x: torch.Tensor,
                                  bits: int, group_size: int,
                                  avg_alam: bool, alam_bits: int, pack_only: bool
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int, bool, bool]:

    seed = 42
    # input_shape = x.shape
    VPW, NWORDS, QMAX = _bits_consts(bits, group_size)

    # B, C, H, W

    # N = x.shape[0]
    # x2 = x.view(N, -1, group_size) # [B, (C * H * W) // group_size, group_size] -> [B, (C * H * W) // group_size, sub-group_size, 4]
    # G = x2.shape[1]
    # NG = N * G

    x2, N, G, new_H, new_W, ori_shape_new, spatical_padding_eligibility, spatical_reshape_eligibility, channel_mean = spatical_aware_act_tensor_reshape(x, Group_Size=group_size, act_padding=False, pack_only=pack_only)
    NG = N * G
    x3 = x2.view(NG, group_size)

    y = torch.empty((NG, group_size), dtype=torch.bfloat16, device='cuda')

    if avg_alam:
        ALAM_NWORDS = triton.cdiv(NWORDS, alam_bits)
        SUB_GROUP = int(ALAM_NWORDS * VPW)
        packed = torch.empty((NG, ALAM_NWORDS), dtype=torch.int32, device='cuda')
    else:
        ALAM_NWORDS = NWORDS
        SUB_GROUP = int(NWORDS * VPW)
        packed = torch.empty((NG, NWORDS), dtype=torch.int32, device='cuda')

    scale = torch.zeros((NG,), dtype=torch.bfloat16, device='cuda')
    min = torch.zeros((NG,), dtype=torch.bfloat16, device='cuda')

    grid = (NG,)
    wrap_triton(silu_fwd_fused_quan_pack_triton)[grid](
        x3, y, packed,
        scale, min,
        seed,
        x3.stride(0), x3.stride(1),
        y.stride(0), y.stride(1),
        packed.stride(0), packed.stride(1),
        BITS=bits, VPW=VPW, NWORDS=NWORDS,
        AVG_ALAM=avg_alam, ALAM_BITS=alam_bits, ALAM_NWORDS=ALAM_NWORDS, SUB_GROUP=SUB_GROUP,
        QMAX=QMAX
    )

    y2 = spatical_aware_act_tensor_reshape_back(y, ori_shape_new, group_size, new_H, new_W, False, spatical_padding_eligibility, spatical_reshape_eligibility, channel_mean)
    return y2, packed, scale, min, N, G, new_H, new_W, spatical_padding_eligibility, spatical_reshape_eligibility
    # return y.view(*x.shape), packed, scale, min, N, G, 0, 0, False, False


@triton_op("act_lib::silu_bwd_fused_dequan_unpack", mutates_args={})
def silu_bwd_fused_dequan_unpack_impl(packed: torch.Tensor, dy: torch.Tensor, 
                                      scale: torch.Tensor, min: torch.Tensor, 
                                      bits: int, N: int, G: int, group_size: int, pack_only: bool,
                                      new_H: int, new_W: int, spatical_padding_eligibility: bool, spatical_reshape_eligibility: bool, 
                                      avg_alam: bool, alam_bits: int) -> torch.Tensor:
    seed = 42
    # output_shape = dy.shape
    VPW, NWORDS, _ = _bits_consts(bits, group_size)

    # dy2 = dy.view(N*G, group_size)
    dy2, N, G, _, _, output_shape, _, _, channel_mean = spatical_aware_act_tensor_reshape(dy, Group_Size=group_size, act_padding=False, pack_only=pack_only)
    dy3 = dy2.view(N*G, group_size)
    dx = torch.empty((N*G, group_size), dtype=torch.bfloat16, device='cuda')

    if avg_alam:
        ALAM_NWORDS = triton.cdiv(NWORDS, alam_bits)
        SUB_GROUP = int(ALAM_NWORDS * VPW)
    else:
        ALAM_NWORDS = NWORDS
        SUB_GROUP = int(NWORDS * VPW)

    grid = (N*G,)
    wrap_triton(silu_bwd_fused_dequan_unpack_triton)[grid](
        dy3, packed, dx,
        scale, min,
        seed,
        dy3.stride(0), dy3.stride(1),
        packed.stride(0), packed.stride(1),
        dx.stride(0), dx.stride(1),
        BITS=bits, VPW=VPW, NWORDS=NWORDS,
        AVG_ALAM=avg_alam, ALAM_BITS=alam_bits, ALAM_NWORDS=ALAM_NWORDS, SUB_GROUP=SUB_GROUP
    )

    dx2 = spatical_aware_act_tensor_reshape_back(dx, output_shape, group_size, new_H, new_W, False, spatical_padding_eligibility, spatical_reshape_eligibility, channel_mean)
    return dx2




# ------
# GELU
# ------
@triton_op("act_lib::gelu_triton", mutates_args={})
def gelu_triton_impl(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    n_elt = x.numel()
    y = torch.empty_like(x, dtype=torch.bfloat16, device='cuda')
    act = torch.empty_like(x, dtype=torch.bfloat16, device='cuda')

    grid = (triton.cdiv(n_elt, 1024),)
    wrap_triton(gelu_triton)[grid](
        x, y, act,
        n_elt,
        BLOCK_SIZE=1024
    )

    return y, act

@triton_op("act_lib::gelu_fwd_fused_quan_pack", mutates_args={})
def gelu_fwd_fused_quan_pack_impl(x: torch.Tensor, bits: int, group_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:

    VPW, NWORDS, QMAX = _bits_consts(bits, group_size)

    input_shape = x.shape
    N = x.shape[0]
    x2 = x.view(N, -1, group_size)
    G = x2.shape[1]
    x3 = x2.view(N*G, group_size)

    NG = N * G

    y = torch.empty((NG, group_size), dtype=torch.bfloat16, device='cuda')
    packed = torch.empty((NG, NWORDS), dtype=torch.int32, device='cuda')
    scale = torch.zeros((NG,), dtype=torch.bfloat16, device='cuda')
    min = torch.zeros((NG,), dtype=torch.bfloat16, device='cuda')

    grid = (NG,)
    wrap_triton(gelu_fwd_fused_quan_pack_triton)[grid](
        x3, y, packed,
        scale, min,
        x3.stride(0), x3.stride(1),
        y.stride(0), y.stride(1),
        packed.stride(0), packed.stride(1),
        BITS=bits,
        VPW=VPW,
        NWORDS=NWORDS,
        QMAX=QMAX
    )

    return y.view(*input_shape), packed, scale, min, N, G


@triton_op("act_lib::gelu_bwd_fused_dequan_unpack", mutates_args={})
def gelu_bwd_fused_dequan_unpack_impl(packed: torch.Tensor, dy: torch.Tensor, scale: torch.Tensor, min: torch.Tensor, bits: int, N: int, G: int, group_size: int) -> torch.Tensor:

    VPW, NWORDS, _ = _bits_consts(bits, group_size)

    output_shape = dy.shape

    NG = N * G
    dy2 = dy.reshape(NG, group_size)

    dx = torch.empty((NG, group_size), dtype=torch.bfloat16, device='cuda')

    grid = (NG,)
    wrap_triton(gelu_bwd_fused_dequan_unpack_triton)[grid](
        dy2, dx, packed,
        scale, min,
        dy2.stride(0), dy2.stride(1),
        dx.stride(0), dx.stride(1),
        packed.stride(0), packed.stride(1),
        BITS=bits,
        VPW=VPW,
        NWORDS=NWORDS
    )

    return dx.view(*output_shape)
