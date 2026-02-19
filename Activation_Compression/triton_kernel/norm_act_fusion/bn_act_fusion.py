import torch

import triton
import triton.language as tl


@triton.jit
def bn_relu_fwd_norm_fused_quant_pack_kernel(
    x_ptr, sum_ptr, sum_square_ptr, w_ptr, b_ptr,
    x_hat_packed_ptr, relu_mask_packed_ptr, y_ptr,
    mean_ptr, var_ptr,
    scale_ptr, min_ptr,
    M, HW,
    stride_n, stride_c, stride_packed_c, stride_stats_c, stride_packed_relu_c,
    relu: tl.constexpr, relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NWORDS: tl.constexpr,
    NWORDS_RELU: tl.constexpr,
    VPW: tl.constexpr,
    VPW_RELU: tl.constexpr,
    BITS: tl.constexpr,
    QMAX: tl.constexpr,
    TOTAL_WORDS: tl.constexpr,
    TOTAL_WORDS_RELU: tl.constexpr,
    sync: tl.constexpr
):
    c = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    M = M.to(tl.int64)
    HW = HW.to(tl.int64)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < M

    packed_offs = pid * NWORDS * 2 + tl.arange(0, NWORDS * 2).to(tl.int64)
    packed_mask = packed_offs < TOTAL_WORDS

    packed_relu_off = pid * NWORDS_RELU + tl.arange(0, NWORDS_RELU)
    packed_relu_mask = packed_relu_off < (TOTAL_WORDS_RELU)

    n = offs // HW
    s = offs - n * HW

    x_ptr += n * stride_n + c * stride_c + s
    y_ptr += n * stride_n + c * stride_c + s
    x_hat_packed_ptr += c * stride_packed_c + packed_offs
    relu_mask_packed_ptr += c * stride_packed_relu_c + packed_relu_off
    scale_ptr += c * stride_stats_c + pid * 2 + tl.arange(0, 2).to(tl.int64)
    min_ptr += c * stride_stats_c + pid * 2 + tl.arange(0, 2).to(tl.int64)

    w = tl.load(w_ptr + c).to(tl.float32)
    b = tl.load(b_ptr + c).to(tl.float32)

    if not sync:
        sum = tl.load(sum_ptr + c).to(tl.float32)
        sum_q = tl.load(sum_square_ptr + c).to(tl.float32)

        mean = sum / M
        var = (sum_q / M) - mean * mean
        var = tl.maximum(var, 0.0)
        invrstd = 1.0 / tl.sqrt(var + 1e-5)

        if pid == 0:
            tl.store(mean_ptr + c, mean)
            tl.store(var_ptr + c, var)
    else:
        mean = tl.load(mean_ptr + c).to(tl.float32)
        invrstd = tl.load(var_ptr + c).to(tl.float32)

    x = tl.load(x_ptr, mask=mask, other=0.).to(tl.float32)
    x_hat = (x - mean) * invrstd
    y_bn = x_hat * w + b

    if relu:
        relu_mask = y_bn > 0
        y = tl.where(relu_mask, y_bn, 0.0)
    elif relu6:
        relu_mask = (y_bn > 0) & (y_bn < 6.0)
        y = tl.where(relu_mask, y_bn, 0.0)
    tl.store(y_ptr, y.to(tl.bfloat16), mask=mask)

    # x_hat = tl.clamp(x_hat, -4.0, 4.0)
    x_hat = tl.reshape(x_hat, (2, GROUP_SIZE))
    mask = tl.reshape(mask, (2, GROUP_SIZE))

    neg_inf = tl.full((), -float("inf"), tl.float32)
    pos_inf = tl.full((),  float("inf"), tl.float32)

    x_for_max = tl.where(mask, x_hat, neg_inf)
    x_for_min = tl.where(mask, x_hat, pos_inf)

    max = tl.max(x_for_max, axis=1)
    min = tl.min(x_for_min, axis=1)

    rng = max - min
    scale = tl.where(rng > 0.0, rng / QMAX, 1.0)

    eps  = tl.full((), 1e-6, tl.float32)
    half = tl.full((), 0.5,  tl.float32)

    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    qf = tl.where(mask, (x_hat - min[:, None]) / scale[:, None] + (half - eps), 0.0)
    # dequan_qf = qf * scale[:, None] + min[:, None]

    # x_hat2_mean = x_hat * x_hat
    # x_hat2_mean = tl.sum(x_hat2_mean, axis=1) / GROUP_SIZE

    # dequan_qf2_mean = dequan_qf * dequan_qf
    # dequan_qf2_mean = tl.sum(dequan_qf2_mean, axis=1) / GROUP_SIZE

    # ratio = (dequan_qf2_mean) / (x_hat2_mean + 1e-6) # (2, )
    # mask_ratio = ratio > 1.2
    # gain = 1.0 / tl.sqrt(ratio)
    # gain = tl.where(mask_ratio, gain, 1.0)
    # scale *= gain

    qf = qf.to(tl.int32)
    qf = tl.maximum(qf, 0)
    qf = tl.minimum(qf, QMAX)
    qf = tl.reshape(qf, (2, NWORDS, VPW))

    words = tl.sum(qf << shifts[None, None, :], axis=2)
    words = tl.reshape(words, (2 * NWORDS,))

    tl.store(x_hat_packed_ptr, words, mask=packed_mask)

    shifts = tl.arange(0, VPW_RELU)
    relu_mask = tl.reshape(relu_mask, (NWORDS_RELU, VPW_RELU))
    relu_mask_packed = tl.sum(relu_mask << shifts[None, :], axis=1).to(tl.int32)
    tl.store(relu_mask_packed_ptr, relu_mask_packed, mask=packed_relu_mask)
    tl.store(scale_ptr, scale.to(tl.bfloat16))
    tl.store(min_ptr, min.to(tl.bfloat16))







@triton.jit
def bn_relu_bwd_reduce_fused_dequant_unpack_kernel(
    x_hat_packed_ptr, relu_mask_packed_ptr, dy_ptr,
    scale_ptr, min_ptr,
    sum_dy_ptr, sum_dy_xhat_ptr,
    M, HW,
    stride_n, stride_c, stride_packed_c, stride_stats_c, stride_packed_relu_c,
    BLOCK_SIZE: tl.constexpr,
    NWORDS: tl.constexpr,
    NWORDS_RELU: tl.constexpr,
    VPW: tl.constexpr,
    VPW_RELU: tl.constexpr,
    BITS: tl.constexpr,
    TOTAL_WORDS: tl.constexpr,
    TOTAL_WORDS_RELU: tl.constexpr
):
    c = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    M = M.to(tl.int64)
    HW = HW.to(tl.int64)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    offs_mask = offs < M

    packed_offs = pid * NWORDS * 2 + tl.arange(0, NWORDS * 2).to(tl.int64)
    packed_mask = packed_offs < TOTAL_WORDS

    packed_relu_offs = pid * NWORDS_RELU + tl.arange(0, NWORDS_RELU).to(tl.int64)
    packed_relu_mask = packed_relu_offs < TOTAL_WORDS_RELU

    n = offs // HW
    s = offs - n * HW

    x_hat_packed_ptr += c * stride_packed_c + packed_offs
    relu_mask_packed_ptr += c * stride_packed_relu_c + packed_relu_offs
    dy_ptr += n * stride_n + c * stride_c + s

    x_hat_packed = tl.load(x_hat_packed_ptr, mask=packed_mask, other=0.).to(tl.int32)
    relu_mask_packed = tl.load(relu_mask_packed_ptr, mask=packed_relu_mask, other=0.).to(tl.int32)

    j = tl.arange(0, VPW); j_relu = tl.arange(0, VPW_RELU)
    shifts = (j * BITS).to(tl.int32)
    unpack_mask = (1 << BITS) - 1

    x_hat_packed = tl.reshape(x_hat_packed, (2, NWORDS))
    x_hat_packed = (x_hat_packed[:, :, None] >> shifts[None, None, :]) & unpack_mask
    
    g = pid * 2 + tl.arange(0, 2).to(tl.int32)
    scale = tl.load(scale_ptr + c * stride_stats_c + g).to(tl.float32)
    min = tl.load(min_ptr + c * stride_stats_c + g).to(tl.float32)
    x_hat = x_hat_packed * scale[:, None, None] + min[:, None, None]

    x_hat = tl.reshape(x_hat, (2 * NWORDS * VPW,))
    relu_mask = (relu_mask_packed[:, None] >> j_relu[None, :]) & 1
    relu_mask = tl.reshape(relu_mask, (NWORDS_RELU * VPW_RELU,))

    dy = tl.load(dy_ptr, mask=offs_mask, other=0.).to(tl.float32)

    sum_dy = tl.sum(dy * relu_mask, axis=0)
    sum_dy_xhat = tl.sum(dy * relu_mask * x_hat, axis=0)

    tl.atomic_add(sum_dy_ptr + c, sum_dy)
    tl.atomic_add(sum_dy_xhat_ptr + c, sum_dy_xhat)



@triton.jit
def bn_relu_bwd_dx_fused_dequant_unpack_kernel(
    x_hat_packed_ptr, relu_mask_packed_ptr, dy_ptr,
    scale_ptr, min_ptr,
    dw_ptr, db_ptr, w_ptr, var_ptr,
    dx_ptr,
    M, HW,
    stride_n, stride_c, stride_packed_c, stride_stats_c, stride_packed_relu_c,
    BLOCK_SIZE: tl.constexpr,
    NWORDS: tl.constexpr,
    NWORDS_RELU: tl.constexpr,
    VPW: tl.constexpr,
    VPW_RELU: tl.constexpr,
    BITS: tl.constexpr,
    TOTAL_WORDS: tl.constexpr,
    TOTAL_WORDS_RELU: tl.constexpr
):
    c = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    M = M.to(tl.int64)
    HW = HW.to(tl.int64)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    offs_mask = offs < M

    packed_offs = pid * NWORDS * 2 + tl.arange(0, NWORDS * 2).to(tl.int64)
    packed_offs_mask = packed_offs < TOTAL_WORDS

    packed_relu_offs = pid * NWORDS_RELU + tl.arange(0, NWORDS_RELU).to(tl.int32)
    packed_relu_offs_mask = packed_relu_offs < TOTAL_WORDS_RELU

    n = offs // HW
    s = offs - n * HW

    x_hat_packed_ptr += c * stride_packed_c + packed_offs
    relu_mask_packed_ptr += c * stride_packed_relu_c + packed_relu_offs
    dy_ptr += n * stride_n + c * stride_c + s
    dx_ptr += n * stride_n + c * stride_c + s

    x_hat_packed = tl.load(x_hat_packed_ptr, mask=packed_offs_mask, other=0.).to(tl.int32)
    relu_mask = tl.load(relu_mask_packed_ptr, mask=packed_relu_offs_mask, other=0).to(tl.int32)

    j = tl.arange(0, VPW); j_relu = tl.arange(0, VPW_RELU)
    shifts = (j * BITS).to(tl.int32)
    unpack_mask = (1 << BITS) - 1

    x_hat_packed = tl.reshape(x_hat_packed, (2, NWORDS))
    x_hat_packed = (x_hat_packed[:, :, None] >> shifts[None, None, :]) & unpack_mask

    g = pid * 2 + tl.arange(0, 2).to(tl.int64)
    scale = tl.load(scale_ptr + c * stride_stats_c + g).to(tl.float32)
    min = tl.load(min_ptr + c * stride_stats_c + g).to(tl.float32)

    x_hat = x_hat_packed * scale[:, None, None] + min[:, None, None]
    x_hat = tl.reshape(x_hat, (2 * NWORDS * VPW,))
    relu_mask = (relu_mask[:, None] >> j_relu[None, :]) & 1
    relu_mask = tl.reshape(relu_mask, (NWORDS_RELU * VPW_RELU))

    dy = tl.load(dy_ptr, mask=offs_mask, other=0.).to(tl.float32)
    dy = dy * relu_mask

    w = tl.load(w_ptr + c).to(tl.float32)
    var = tl.load(var_ptr + c).to(tl.float32)
    dw = tl.load(dw_ptr + c).to(tl.float32)
    db = tl.load(db_ptr + c).to(tl.float32)

    invrstd = 1.0 / tl.sqrt(var + 1e-5)

    dx = (dy - (db / M) - x_hat * (dw / M)) * w * invrstd
    tl.store(dx_ptr, dx.to(tl.bfloat16), mask=offs_mask)
