from curses import meta
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

import torch.nn.functional as F
from torch.library import triton_op, wrap_triton
from torch.amp import custom_fwd, custom_bwd

from typing import Tuple 



@triton.jit
def norm(x, NOWORDS, VPW):
    sum_square = tl.sum(x * x)
    rms = tl.sqrt(sum_square / (NOWORDS * VPW))
    return x / (rms + 1e-8)



@triton.jit
def quant_pack_kernel(
    X_ptr, P_ptr, S_ptr, Min_ptr,
    seed,
    stride_x0: tl.constexpr, stride_x1: tl.constexpr,
    stride_p0: tl.constexpr, stride_p1: tl.constexpr,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr,
    QMAX: tl.constexpr,
    AVG_ALAM: tl.constexpr,
    ALAM_BITS: tl.constexpr,
    ALAM_NWORDS: tl.constexpr,
    SUB_GROUP: tl.constexpr,
    CLAMP: tl.constexpr,
    CLAMP_ALPHA: tl.constexpr
):
    pid = tl.program_id(0)

    # ---- block pointer: [NWORDS, VPW] = 256 values ----
    x_block_ptr = tl.make_block_ptr(
        base=X_ptr + pid * stride_x0,
        shape=(NWORDS * VPW,),
        strides=(stride_x1,),
        offsets=(0,),
        block_shape=(NWORDS * VPW,),
        order=(0,),
    )

    # ---- single global load ----
    x = tl.load(x_block_ptr).to(tl.float32)   # [NWORDS * VPW]

    if CLAMP:
        x = tl.clamp(x, -CLAMP_ALPHA, CLAMP_ALPHA)

    x = libdevice.tanh(x / 0.5) * 0.5

    xmax = tl.max(x, axis=0)
    xmin = tl.min(x, axis=0)

    rng = xmax - xmin
    scale = (rng / tl.full([], QMAX, tl.float32))
    scale = tl.where(rng > 0.0, scale, tl.full([], 1.0, tl.float32))

    tl.store(S_ptr + pid, scale.to(tl.bfloat16))
    tl.store(Min_ptr + pid, xmin.to(tl.bfloat16))


    inv_scale = tl.full([], 1.0, tl.float32) / scale

    # ---- optional average over 4 values ----
    # if AVG_ALAM:
        # noise_shape = tl.arange(0, SUB_GROUP)
        # sgd_noise = tl.rand(seed + pid, noise_shape)

    # x = libdevice.tanh(x * 0.5) / 0.5
        # x = tl.reshape(x, (SUB_GROUP, ALAM_BITS))
        # x = tl.sum(x, axis=1) / ALAM_BITS

        # alpha1 = 0.25
        # alpha2 = 0.25
        # beta_max = 0.5

        # mu = tl.sum(x, axis=1) * (1.0 / ALAM_BITS)  # mean
        # mx = tl.max(x, axis=1)
        # mv = tl.min(x, axis=1)

        # ms2 = tl.sum(x * x, axis=1) * (1.0 / ALAM_BITS)
        # rms = tl.sqrt(ms2 + 1e-8)
        # srms = tl.where(mu >= 0.0, rms, -rms)

        # y1 = (1.0 - alpha1 - alpha2) * mu + alpha1 * mx + alpha2 * mv

        # # cancellation indicator: small |mu| vs rms
        # c = tl.abs(mu) / (rms + 1e-8)          # in [0, 1-ish]
        # beta = (1.0 - c) * beta_max           # more RMS when c is small
        # x = (1.0 - beta) * y1 + beta * srms
    # else:
        # noise_shape = tl.arange(0, NWORDS * VPW)
        # sgd_noise = tl.rand(seed + pid, noise_shape)

    # ---- scalar min / max over all 256 values ----
    # xmin = tl.min(x, axis=0) 
    # xmax = tl.max(x, axis=0)


    # ---- pack ----
    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    # quantize all at once
    qf = (x - xmin) * inv_scale + (0.5 - 1e-6)

    if AVG_ALAM:
        qf = tl.reshape(qf, (SUB_GROUP, ALAM_BITS))
        qf = tl.sum(qf, axis=1) / ALAM_BITS


    # if AVG_ALAM:
    #     qf = tl.reshape(qf, (SUB_GROUP, ALAM_BITS))

        # qf_mean = tl.sum(qf, axis=1) / ALAM_BITS
        # k = tl.arange(0, ALAM_BITS)
        # pattern = tl.where(k < (ALAM_BITS // 2), 1.0, -1.0).to(tl.float32)
        # qf = qf * pattern[None, :]
        # qf = tl.sum(qf, axis=1)
        # qf = tl.max(qf, axis=1)
        # qf = tl.sum(qf, axis=1) / ALAM_BITS
        # score = tl.sum(qf * qf, axis=0) / ALAM_BITS
        # score = tl.softmax(score * 2, dim=0)
        # qf = tl.sum(qf * score[None, :], axis=1)


        # alpha1 = 0.25
        # alpha2 = 0.25
        # beta_max = 0.5

        # mu = tl.sum(qf, axis=1) * (1.0 / ALAM_BITS)  # mean
        # mx = tl.max(qf, axis=1)
        # mv = tl.min(qf, axis=1)

        # ms2 = tl.sum(qf * qf, axis=1) * (1.0 / ALAM_BITS)
        # rms = tl.sqrt(ms2 + 1e-8)
        # srms = tl.where(mu >= 0.0, rms, -rms)

        # y1 = (1.0 - alpha1 - alpha2) * mu + alpha1 * mx + alpha2 * mv

        # # cancellation indicator: small |mu| vs rms
        # c = tl.abs(mu) / (rms + 1e-8)          # in [0, 1-ish]
        # beta = (1.0 - c) * beta_max           # more RMS when c is small
        # qf = (1.0 - beta) * y1 + beta * srms

    qi = qf.to(tl.int32)
    qi = tl.maximum(qi, 0)
    qi = tl.minimum(qi, QMAX)

    qi = tl.reshape(qi, (ALAM_NWORDS, VPW))
    words = tl.sum(qi << shifts[None, :], axis=1)

    p_block_ptr = tl.make_block_ptr(
        base = P_ptr + pid * stride_p0,
        shape=(ALAM_NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        block_shape=(ALAM_NWORDS,),
        order=(0,)
    )

    tl.store(p_block_ptr, words.to(tl.int32))


@triton.jit
def dequant_unpack_kernel(
    P_ptr, S_ptr, M_ptr, Y_ptr,
    seed,
    stride_p0: tl.constexpr, stride_p1: tl.constexpr,
    stride_y0: tl.constexpr, stride_y1: tl.constexpr,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr,
    AVG_ALAM: tl.constexpr,
    ALAM_BITS: tl.constexpr,
    ALAM_NWORDS: tl.constexpr,
    SUB_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)

    p_block_ptr = tl.make_block_ptr(
        base=P_ptr + pid * stride_p0,
        shape=(ALAM_NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        block_shape=(ALAM_NWORDS,),
        order=(0,)
    )

    word = tl.load(p_block_ptr)

    scale = tl.load(S_ptr + pid)
    scale_dtype = scale.dtype
    scale = scale.to(tl.float32)
    xmin  = tl.load(M_ptr + pid).to(tl.float32)

    mask = (1 << BITS) - 1
    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    q = ((word[:, None] >> shifts[None, :]) & mask) # [ALAM_NWORDS, VPW]

    if AVG_ALAM:
        q = tl.reshape(q, (ALAM_NWORDS * VPW,))
        q = tl.broadcast_to(q[:, None], (SUB_GROUP, ALAM_BITS))
        q = tl.reshape(q, (SUB_GROUP * ALAM_BITS,))

        noise_shape = tl.arange(0, SUB_GROUP * ALAM_BITS)
        sgd_noise = tl.randn(seed + pid, noise_shape)
        q += (sgd_noise - 0.5)
    else:
        q = q.to(tl.float32)

    q = q * scale + xmin

    # q = q * tl.sigmoid(q)
    # q = libdevice.tanh(q * 0.5) / 0.5
    # q = q * (1.0 / (tl.abs(q) + 1e-6))
    # q = q + 0.5 * tl.sin(q * 3.14159)

    q_flat = tl.reshape(q, (NWORDS * VPW,))

    y_block_ptr = tl.make_block_ptr(
        base=Y_ptr + pid * stride_y0,
        shape=(NWORDS * VPW,),
        strides=(stride_y1,),
        offsets=(0,),
        block_shape=(NWORDS * VPW,),
        order=(0,)
    )

    tl.store(y_block_ptr, q_flat.to(scale_dtype))




@triton.jit
def pack_kernel(
    X_ptr, P_ptr,
    stride_x0: tl.constexpr, stride_x1: tl.constexpr,
    stride_p0: tl.constexpr, stride_p1: tl.constexpr,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr
):
    pid = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        base = X_ptr + pid * stride_x0,
        shape=(NWORDS, VPW),
        strides=(stride_x1 * VPW, stride_x1),
        offsets=(0, 0),
        block_shape=(NWORDS, VPW),
        order=(0, 1),
    )

    x = tl.load(x_block_ptr).to(tl.int8)

    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    words = tl.sum(x << shifts[None, :], axis=1).to(tl.int32)

    p_block_ptr = tl.make_block_ptr(
        base = P_ptr + pid * stride_p0,
        shape=(NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        block_shape=(NWORDS,),
        order=(0,)
    )
    tl.store(p_block_ptr, words)
    

@triton.jit
def unpack_kernel(
    P_ptr, Y_ptr,
    stride_p0: tl.constexpr, stride_p1: tl.constexpr,
    stride_y0: tl.constexpr, stride_y1: tl.constexpr,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr
):
    pid = tl.program_id(0)

    p_block_ptr = tl.make_block_ptr(
        base= P_ptr + pid * stride_p0,
        shape=(NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        block_shape=(NWORDS,),
        order=(0,)
    )

    word = tl.load(p_block_ptr)

    mask = (1 << BITS) - 1
    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    q = (word[:, None] >> shifts[None, :]) & mask
    q_flatten = tl.reshape(q, (NWORDS * VPW,))

    y_block_ptr = tl.make_block_ptr(
        base=Y_ptr + pid * stride_y0,
        shape=(NWORDS * VPW,),
        strides=(stride_y1,),
        offsets=(0,),
        block_shape=(NWORDS * VPW,),
        order=(0,)
    )

    tl.store(y_block_ptr, q_flatten.to(tl.int8))





def _bits_consts(bits: int, group_size: int):
    if bits not in (1, 2, 4, 8):
        raise ValueError(f"bits must be one of (1,2,4,8). got {bits}")
    vpw = 32 // bits
    nwords = group_size // vpw          
    qmax = (1 << bits) - 1
    return vpw, nwords, qmax

@triton_op("act_lib::quant_pack_triton", mutates_args={})
def quant_pack_triton(x: torch.Tensor, bits: int, group_size: int, avg_alam: bool, alam_bits: int, clamp: bool, clamp_alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seed = 42
    N, G, _ = x.shape
    NG = N * G
    VPW, NWORDS, QMAX = _bits_consts(bits, group_size)
    x2 = x.view(NG, group_size).contiguous()

    if avg_alam:
        ALAM_NWORDS = triton.cdiv(NWORDS, alam_bits)
        SUB_GROUP = int(ALAM_NWORDS * VPW)
        packed = torch.empty((NG, ALAM_NWORDS), dtype=torch.int32, device=x.device)
    else:
        packed = torch.empty((NG, NWORDS), dtype=torch.int32, device=x.device)
        ALAM_NWORDS = NWORDS
        SUB_GROUP = 0  # dummy value

    scale = torch.empty((NG,), dtype=torch.bfloat16, device=x.device)
    min = torch.empty((NG,), dtype=torch.bfloat16, device=x.device)

    grid = (NG,)
    wrap_triton(quant_pack_kernel)[grid](
        x2, packed, scale, min,
        seed,
        x2.stride(0), x2.stride(1),
        packed.stride(0), packed.stride(1),
        BITS=bits, VPW=VPW, NWORDS=NWORDS, QMAX=QMAX,
        AVG_ALAM=avg_alam, ALAM_BITS=alam_bits, ALAM_NWORDS=ALAM_NWORDS, SUB_GROUP=SUB_GROUP,
        CLAMP=clamp, CLAMP_ALPHA=clamp_alpha,
        num_warps=4,
    )

    return packed, scale.view(N, G), min.view(N, G)



@triton_op("act_lib::dequant_unpack_triton", mutates_args={})
def dequant_unpack_triton(packed: torch.Tensor, scale: torch.Tensor, xmin: torch.Tensor, bits: int, group_size: int, avg_alam: bool, alam_bits: int) -> torch.Tensor:
    seed = 42
    N, G = scale.shape
    NG = N * G
    VPW, NWORDS, _ = _bits_consts(bits, group_size)

    ALAM_NWORDS = triton.cdiv(NWORDS, alam_bits) if avg_alam else NWORDS
    SUB_GROUP = int(ALAM_NWORDS * VPW) if avg_alam else 0  # dummy value

    y = torch.empty((NG, group_size), device=packed.device, dtype=torch.bfloat16)

    grid = (NG,)
    wrap_triton(dequant_unpack_kernel)[grid](
        packed, scale, xmin, y,
        seed,
        packed.stride(0), packed.stride(1),
        y.stride(0), y.stride(1),
        BITS=bits, VPW=VPW, NWORDS=NWORDS,
        AVG_ALAM=avg_alam, ALAM_BITS=alam_bits, ALAM_NWORDS=ALAM_NWORDS, SUB_GROUP=SUB_GROUP,
        num_warps=4,
    )

    return y.view(N, G, group_size)


@triton_op("act_lib::pack_triton", mutates_args={})
def pack_triton(x: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    assert x.is_cuda and x.is_contiguous
    assert x.shape[-1] == group_size

    N, G, _ = x.shape
    NG = N * G
    VPW, NWORDS, QMAX = _bits_consts(bits, group_size)

    x2 = x.view(NG, group_size)
    packed = torch.zeros((NG, NWORDS), dtype=torch.int32, device='cuda')

    grid = (NG,)
    wrap_triton(pack_kernel)[grid](
        x2, packed,
        x2.stride(0), x2.stride(1),
        packed.stride(0), packed.stride(1),
        bits, VPW, NWORDS
    )
    return packed

@triton_op("act_lib::unpack_triton", mutates_args={})
def unpack_triton(packed: torch.Tensor, bits: int, N: int, G: int, group_size: int) -> torch.Tensor:
    NG = packed.shape[0]
    VPW, NWORDS, QMAX = _bits_consts(bits, group_size)

    unpacked = torch.zeros((NG, group_size), dtype=torch.int8, device='cuda')

    grid = (NG,)
    wrap_triton(unpack_kernel)[grid](
        packed, unpacked,
        packed.stride(0), packed.stride(1),
        unpacked.stride(0), unpacked.stride(1),
        bits, VPW, NWORDS
    )
    return unpacked.view(N, G, group_size)



@torch.library.register_fake("act_lib::quant_pack_triton")
def quant_pack_triton_fake(x: torch.Tensor, beta: float, bits: int, group_size: int):
    N, G = x.shape[:2]
    NG = N * G
    VPW, NWORDS, QMAX = _bits_consts(bits, group_size)

    packed = x.new_empty((NG, NWORDS), dtype=torch.float32, device=x.device)
    scaler = x.new_empty((NG,), dtype=x.dtype, device=x.device)
    xmin = x.new_empty((NG,), dtype=x.dtype, device=x.device)
    return packed, scaler, xmin

@torch.library.register_fake("act_lib::dequant_unpack_triton")
def dequant_unpack_triton_fake(packed_flat: torch.Tensor, scale: torch.Tensor, xmin: torch.Tensor, bits: int, group_size: int):
    N, G = scale.shape
    NG = N * G

    y = torch.empty((NG, group_size), device=packed_flat.device, dtype=scale.dtype)
    return y 


@torch.library.register_fake("act_lib::pack_triton")
def pack_triton_fake(x: torch.Tensor, bits: int, group_size: int):
    N, G, _ = x.shape
    NG = N * G
    VPW, NWORDS, QMAX = _bits_consts(bits, group_size)

    packed = x.new_empty((NG, NWORDS), dtype=torch.int32, device=x.device)

    return packed

@torch.library.register_fake("act_lib::unpack_triton")
def unpack_triton_fake(packed: torch.Tensor, bits: int, N: int, G: int, group_size: int):
    NG = packed.shape[0]

    unpacked = packed.new_empty((NG, group_size), dtype=torch.int8, device=packed.device)

    return unpacked.view(N, G, group_size)



