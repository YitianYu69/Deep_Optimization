import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def softshrink(x, lambd):
    return tl.where(x > lambd, x - lambd, tl.where(x < -lambd, x + lambd, 0.0))


@triton.jit
def _bn_fwd_reduce_kernel(
    X, SUM, SUMSQ,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    stride_c = tl.full((), stride_c, tl.int64)
    stride_n = tl.full((), stride_n, tl.int64)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptrs = X + n * stride_n + c * stride_c + s
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_x  = tl.sum(x, axis=0)
    sum_x2 = tl.sum(x * x, axis=0)

    tl.atomic_add(SUM   + c, sum_x)
    tl.atomic_add(SUMSQ + c, sum_x2)



@triton.jit
def _bn_fwd_norm_fused(
    x_ptr, sum_ptr, sum_square_ptr, y_ptr, x_hat_ptr,
    w_ptr, b_ptr, var_ptr, mean_ptr,
    M, HW,
    stride_n, stride_c,
    BLOCK_SIZE: tl.constexpr,
    sync: tl.constexpr
): 
    c = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    M = tl.full((), M, tl.int64)
    HW = tl.full((), HW, tl.int64)
    stride_c = tl.full((), stride_c, tl.int64)
    stride_n = tl.full((), stride_n, tl.int64)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_ptr += n * stride_n + c * stride_c + s
    y_ptr += n * stride_n + c * stride_c + s
    x_hat_ptr += n * stride_n + c * stride_c + s

    if not sync:
        sum = tl.load(sum_ptr + c).to(tl.float32)
        sum_square = tl.load(sum_square_ptr + c).to(tl.float32)

        mean = sum / M
        var = (sum_square / M) - mean * mean
        var = tl.maximum(var, 0.0)
        invrstd = 1.0 / tl.sqrt(var + 1e-7)

        if pid == 0:
            tl.store(var_ptr + c, var)
            tl.store(mean_ptr + c, mean)
    else:
        mean = tl.load(mean_ptr + c).to(tl.float32)
        invrstd = tl.load(var_ptr + c).to(tl.float32)

    x = tl.load(x_ptr, mask=mask, other=0.).to(tl.float32)
    w = tl.load(w_ptr + c).to(tl.float32)
    b = tl.load(b_ptr + c).to(tl.float32)

    x_hat = (x - mean) * invrstd
    y = x_hat * w + b

    tl.store(x_hat_ptr, x_hat.to(tl.bfloat16), mask=mask)
    tl.store(y_ptr, y.to(tl.bfloat16), mask=mask)


@triton.jit
def bn_fwd_norm_quant_pack_fused_kernel(
    x_ptr, sum_ptr, sum_square_ptr, w_ptr, b_ptr,
    y_ptr, x_hat_packed_ptr, scale_ptr, min_ptr,
    mean_ptr, var_ptr,
    M, HW,
    stride_n, stride_c, stride_packed_c, stride_stats_c,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr,
    TOTAL_WORDS: tl.constexpr,
    QMAX: tl.constexpr,
    sync: tl.constexpr,
    AVG_ALAM: tl.constexpr,
    ALAM_BITS: tl.constexpr,
    ALAM_NWORDS: tl.constexpr,
    SUB_GROUP: tl.constexpr,
):
    c = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    M = tl.full((), M, tl.int64)
    HW = tl.full((), HW, tl.int64)
    stride_c = tl.full((), stride_c, tl.int64)
    stride_n = tl.full((), stride_n, tl.int64)
    stride_packed_c = tl.full((), stride_packed_c, tl.int64)
    stride_stats_c = tl.full((), stride_stats_c, tl.int64)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < M

    packed_offs = pid  * ALAM_NWORDS * 2 + tl.arange(0, ALAM_NWORDS * 2).to(tl.int64)
    packed_mask = packed_offs < TOTAL_WORDS

    n = offs // HW
    s = offs - n * HW

    x_ptr += n * stride_n + c * stride_c + s
    y_ptr += n * stride_n + c * stride_c + s
    x_hat_packed_ptr = x_hat_packed_ptr + c * stride_packed_c + packed_offs

    x = tl.load(x_ptr, mask=mask, other=0.).to(tl.float32)
    w = tl.load(w_ptr + c).to(tl.float32)
    b = tl.load(b_ptr + c).to(tl.float32)

    if not sync:
        sum = tl.load(sum_ptr + c).to(tl.float32)
        sum_square = tl.load(sum_square_ptr + c).to(tl.float32)

        mean = sum / M
        var = (sum_square / M) - mean * mean
        var = tl.maximum(var, 0.0)
        invrstd = 1.0 / tl.sqrt(var + 1e-5)

        if pid == 0:
            tl.store(mean_ptr + c, mean)
            tl.store(var_ptr + c, var)
    else:
        mean = tl.load(mean_ptr + c).to(tl.float32)
        invrstd = tl.load(var_ptr + c).to(tl.float32)

    x_hat = (x - mean) * invrstd
    y = x_hat * w + b

    tl.store(y_ptr, y.to(tl.bfloat16), mask=mask)

    x_hat = tl.reshape(x_hat, (2, GROUP_SIZE))
    mask = tl.reshape(mask, (2, GROUP_SIZE))

    x_hat = tl.clamp(x_hat, -3, 3)

    x_hat = libdevice.tanh(x_hat * 0.5) / 0.5
    # x_hat = x_hat * tl.sigmoid(x_hat)

    max = tl.max(x_hat, axis=1)
    min = tl.min(x_hat, axis=1)

    # x_hat = x_hat * tl.sigmoid(x_hat)
    # x_hat = softshrink(x_hat, 0.7)
    
    # if AVG_ALAM:
        # x_hat = tl.reshape(x_hat, (2, SUB_GROUP, ALAM_BITS))
        # x_hat = tl.sum(x_hat, axis=2) / ALAM_BITS
    # x_hat = libdevice.tanh(x_hat * 0.5) / 0.5

    # max = tl.max(x_hat, axis=1)
    # min = tl.min(x_hat, axis=1)

    rng = max - min
    scale = rng / QMAX
    scale = tl.where(rng > 0.0, scale, 1.0)

    inv_scale = 1.0 / scale

    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    eps  = tl.full((), 1e-6, tl.float32)
    half = tl.full((), 0.5,  tl.float32)

    if AVG_ALAM:
        qf = (x_hat - min[:, None]) * inv_scale[:, None] + (half - eps)
        qf = tl.reshape(qf, (2, SUB_GROUP, ALAM_BITS))

        # k = tl.arange(0, ALAM_BITS)
        # qf_mean = tl.sum(qf, axis=2) / ALAM_BITS
        # pattern = tl.where(k < (ALAM_BITS // 2), 1.0, -1.0).to(tl.float32)
        # qf = qf * pattern[None, None, :]
        # qf = tl.sum(qf, axis=2)
        # qf = tl.max(qf, axis=2) 
        qf = tl.sum(qf, axis=2) / ALAM_BITS
        # _qf = tl.reshape(qf, (2 * SUB_GROUP, ALAM_BITS))
        # score = tl.sum(_qf * _qf, axis=0) / ALAM_BITS
        # score = tl.softmax(score * 2, dim=0)
        # qf = tl.sum(qf * score[None, None, :], axis=2)

        # alpha1 = 0.25
        # alpha2 = 0.25
        # beta_max = 0.5

        # mu = tl.sum(qf, axis=2) * (1.0 / ALAM_BITS)
        # mx = tl.max(qf, axis=2)
        # mv = tl.min(qf, axis=2)

        # ms2 = tl.sum(qf * qf, axis=2) * (1.0 / ALAM_BITS)
        # rms = tl.sqrt(ms2 + 1e-8)
        # srms = tl.where(mu >= 0.0, rms, -rms)

        # y1 = (1.0 - alpha1 - alpha2) * mu + alpha1 * mx + alpha2 * mv

        # # cancellation indicator: small |mu| vs rms
        # _c = tl.abs(mu) / (rms + 1e-8)          # in [0, 1-ish]
        # beta = (1.0 - _c) * beta_max           # more RMS when c is small
        # qf = (1.0 - beta) * y1 + beta * srms
    else:
        qf = (x_hat - min[:, None]) * inv_scale[:, None] + (half - eps)

    qf = qf.to(tl.int32)
    qf = tl.maximum(qf, 0)
    qf = tl.minimum(qf, QMAX)
    qf = tl.reshape(qf, (2, ALAM_NWORDS, VPW))

    words = tl.sum(qf << shifts[None, None, :], axis=2)
    words = tl.reshape(words, (2 * ALAM_NWORDS,))

    g = pid * 2 + tl.arange(0, 2).to(tl.int64)
    tl.store(scale_ptr + c * stride_stats_c + g, scale.to(tl.bfloat16))
    tl.store(min_ptr + c * stride_stats_c + g, min.to(tl.bfloat16))
    tl.store(x_hat_packed_ptr, words.to(tl.int32), mask=packed_mask)


@triton.jit
def _bn_bwd_reduce_kernel(
    X_hat, DY,
    DBETA, DGAMMA,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
):
    c = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    M = tl.full((), M, tl.int64)
    HW = tl.full((), HW, tl.int64)
    stride_c = tl.full((), stride_c, tl.int64)
    stride_n = tl.full((), stride_n, tl.int64)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    sum_dy      = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA  + c, sum_dy)
    tl.atomic_add(DGAMMA + c, sum_dy_xhat)


@triton.jit
def _bn_bwd_dx_kernel(
    X_hat, DY, VAR, GAMMA,
    DBETA, DGAMMA,
    DX,
    M, HW,
    stride_n, stride_c,
    BLOCK_M: tl.constexpr,
    sync: tl.constexpr
):
    c = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    M = tl.full((), M, tl.int64)
    HW = tl.full((), HW, tl.int64)
    stride_c = tl.full((), stride_c, tl.int64)
    stride_n = tl.full((), stride_n, tl.int64)

    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    mask = offs < M

    n = offs // HW
    s = offs - n * HW

    x_hat_ptrs  = X_hat  + n * stride_n + c * stride_c + s
    dy_ptrs = DY + n * stride_n + c * stride_c + s
    dx_ptrs = DX + n * stride_n + c * stride_c + s

    x_hat  = tl.load(x_hat_ptrs,  mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptrs, mask=mask, other=0.0).to(tl.float32)

    if not sync:
        var = tl.load(VAR + c).to(tl.float32)
        invrstd = 1.0 / tl.sqrt(var + 1e-7)
    else:
        invrstd = tl.load(VAR + c).to(tl.float32)

    gamma  = tl.load(GAMMA  + c).to(tl.float32)

    dbeta  = tl.load(DBETA  + c).to(tl.float32)
    dgamma = tl.load(DGAMMA + c).to(tl.float32)

    m = tl.full((), M, tl.float32)

    dx = (dy - dbeta / m - x_hat * (dgamma / m)) * gamma * invrstd
    tl.store(dx_ptrs, dx.to(tl.bfloat16), mask=mask)




@triton.jit
def bn_bwd_reduce_dequant_unpack_fused_kernel(
    x_hat_packed_ptr, dy_ptr, scale_ptr, min_ptr,
    DGAMMA_ptr, DBETA_ptr,
    M, HW,
    stride_n, stride_c, stride_packed_c, stride_stats_c,
    BLOCK_SIZE: tl.constexpr,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr,
    TOTAL_WORDS: tl.constexpr,
    AVG_ALAM: tl.constexpr,
    ALAM_BITS: tl.constexpr,
    ALAM_NWORDS: tl.constexpr,
    SUB_GROUP: tl.constexpr,
):
    c = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    M = tl.full((), M, tl.int64)
    HW = tl.full((), HW, tl.int64)
    stride_c = tl.full((), stride_c, tl.int64)
    stride_n = tl.full((), stride_n, tl.int64)
    stride_packed_c = tl.full((), stride_packed_c, tl.int64)
    stride_stats_c = tl.full((), stride_stats_c, tl.int64)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < M

    packed_offs = pid * ALAM_NWORDS * 2 + tl.arange(0, ALAM_NWORDS * 2).to(tl.int64)
    packed_mask = packed_offs < TOTAL_WORDS

    n = offs // HW
    s = offs - n * HW

    x_hat_packed_ptr = x_hat_packed_ptr + c * stride_packed_c + packed_offs
    dy_ptr += n * stride_n + c * stride_c + s

    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)
    bit_mask = (1 << BITS) - 1

    x_hat_packed = tl.load(x_hat_packed_ptr, mask=packed_mask).to(tl.int32)
    x_hat_packed = tl.reshape(x_hat_packed, (2, ALAM_NWORDS))

    x_hat = ((x_hat_packed[:, :, None] >> shifts[None, None, :]) & bit_mask).to(tl.float32)

    if AVG_ALAM:
        x_hat = tl.reshape(x_hat, (2, ALAM_NWORDS * VPW))
        x_hat = tl.broadcast_to(x_hat[:, :, None], (2, SUB_GROUP, ALAM_BITS))
        x_hat = tl.reshape(x_hat, (2, SUB_GROUP * ALAM_BITS))

        noise_shape = tl.arange(0, SUB_GROUP * ALAM_BITS)
        sgd_noise = tl.rand(42 + pid, noise_shape)
        x_hat = x_hat + (sgd_noise - 0.5)
    else:
        x_hat = tl.reshape(x_hat, (2, NWORDS * VPW))

    g = pid * 2 + tl.arange(0, 2).to(tl.int64)
    scale = tl.load(scale_ptr + c * stride_stats_c + g).to(tl.float32)
    min = tl.load(min_ptr + c * stride_stats_c + g).to(tl.float32)

    x_hat = x_hat * scale[:, None] + min[:, None]

    # x_hat = x_hat * tl.sigmoid(x_hat)
    # x_hat = libdevice.tanh(x_hat * 0.5) / 0.5
    # x_hat = x_hat * (1.0 / (tl.abs(x_hat) + 1e-6))
    # x_hat = x_hat + tl.sin(x_hat * 3.14159)

    x_hat = tl.reshape(x_hat, (2 * NWORDS * VPW,))
    dy = tl.load(dy_ptr, mask=mask, other=0.).to(tl.float32)

    sum_dy = tl.sum(dy, axis=0)
    sum_dy_xhat = tl.sum(dy * x_hat, axis=0)

    tl.atomic_add(DBETA_ptr + c, sum_dy)
    tl.atomic_add(DGAMMA_ptr + c, sum_dy_xhat)



@triton.jit
def bn_bwd_dx_dequant_unpack_fused_kernel(
    x_hat_packed_ptr, dy_ptr, scale_ptr, min_ptr, DGAMMA_ptr, DBETA_ptr, GAMMA_ptr, var_ptr,
    dx_ptr,
    M, HW,
    stride_n, stride_c, stride_packed_c, stride_stats_c,
    BLOCK_SIZE: tl.constexpr,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr,
    TOTAL_WORDS: tl.constexpr,
    sync: tl.constexpr,
    AVG_ALAM: tl.constexpr,
    ALAM_BITS: tl.constexpr,
    ALAM_NWORDS: tl.constexpr,
    SUB_GROUP: tl.constexpr,
):
    c = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    M = tl.full((), M, tl.int64)
    HW = tl.full((), HW, tl.int64)
    stride_c = tl.full((), stride_c, tl.int64)
    stride_n = tl.full((), stride_n, tl.int64)
    stride_packed_c = tl.full((), stride_packed_c, tl.int64)
    stride_stats_c = tl.full((), stride_stats_c, tl.int64)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < M

    packed_offs = pid * ALAM_NWORDS * 2 + tl.arange(0, ALAM_NWORDS * 2).to(tl.int64)
    packed_mask = packed_offs < TOTAL_WORDS

    n = offs // HW
    s = offs - n * HW

    x_hat_packed_ptr += c * stride_packed_c + packed_offs
    dy_ptr += n * stride_n + c * stride_c + s
    dx_ptr += n * stride_n + c * stride_c + s

    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)
    bit_mask = (1 << BITS) - 1

    x_hat_packed = tl.load(x_hat_packed_ptr, mask=packed_mask).to(tl.int32)
    x_hat_packed = tl.reshape(x_hat_packed, (2, ALAM_NWORDS))

    x_hat = ((x_hat_packed[:, :, None] >> shifts[None, None, :]) & bit_mask).to(tl.float32)

    if AVG_ALAM:
        x_hat = tl.reshape(x_hat, (2, ALAM_NWORDS * VPW))
        x_hat = tl.broadcast_to(x_hat[:, :, None], (2, SUB_GROUP, ALAM_BITS))
        x_hat = tl.reshape(x_hat, (2, SUB_GROUP * ALAM_BITS))
    else:
        x_hat = tl.reshape(x_hat, (2, NWORDS * VPW))

    g = pid * 2 + tl.arange(0, 2).to(tl.int64)
    scale = tl.load(scale_ptr + c * stride_stats_c + g).to(tl.float32)
    min = tl.load(min_ptr + c * stride_stats_c + g).to(tl.float32)

    x_hat = x_hat * scale[:, None] + min[:, None]
    x_hat = tl.reshape(x_hat, (2 * NWORDS * VPW,))

    dy = tl.load(dy_ptr, mask=mask, other=0.).to(tl.float32)
    dw = tl.load(DGAMMA_ptr + c).to(tl.float32)
    db = tl.load(DBETA_ptr + c).to(tl.float32)
    w = tl.load(GAMMA_ptr + c).to(tl.float32)

    if not sync:
        var = tl.load(var_ptr + c).to(tl.float32)
        invrstd = 1.0 / tl.sqrt(var + 1e-7)
    else:
        invrstd = tl.load(var_ptr + c).to(tl.float32)

    dx = (dy - (db / M) - x_hat * (dw / M)) * w * invrstd
    tl.store(dx_ptr, dx.to(tl.bfloat16), mask=mask)
