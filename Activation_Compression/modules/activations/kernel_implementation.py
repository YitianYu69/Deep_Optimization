import triton
import triton.language as tl
from triton.language.extra import libdevice

# ------
# ReLU
# ------

@triton.jit
def relu_variance_kernel(
    X_ptr, Y_ptr, Mask_prt,
    n_elts,
    relu: tl.constexpr,
    relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
): 
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elts

    x = tl.load(X_ptr + offs, mask=mask, other=0.)

    if relu:
        mask_ = x > 0
        y = tl.where(mask_, x, 0.)
    elif relu6:
        mask_ = (x > 0) & (x < 6)
        y = tl.where(mask_, x, 0.)

    tl.store(Y_ptr + offs, y, mask=mask)
    tl.store(Mask_prt + offs, mask_.to(tl.int8), mask=mask)


@triton.jit
def relu_variance_fwd_fused_pack_kernel(
    X_ptr, P_ptr, Y_ptr,
    stride_x0, stride_x1,
    stride_p0, stride_p1,
    stride_y0, stride_y1,
    relu: tl.constexpr,
    relu6: tl.constexpr,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)

    x_block_ptr = tl.make_block_ptr(
        base=X_ptr + pid * stride_x0,
        shape=(NWORDS * VPW,),
        strides=(stride_x1, ),
        offsets=(0,),
        block_shape=(NWORDS * VPW,),
        order=(0,)
    )

    x = tl.load(x_block_ptr)

    if relu:
        mask_ = x > 0
        x = tl.where(mask_, x, 0)
    elif relu6:
        mask_ = (x > 0) & (x < 6)
        x = tl.where(mask_, x, 0)

    mask_ = tl.reshape(mask_, (NWORDS, VPW))
    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)
    p_relu_mask = tl.sum(mask_ << shifts[None, :], axis=1).to(tl.int32)

    p_block_ptr = tl.make_block_ptr(
        base=P_ptr + pid * stride_p0,
        shape=(NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        block_shape=(NWORDS,),
        order=(0,)
    )

    y_block_ptr = tl.make_block_ptr(
        base=Y_ptr + pid * stride_y0,
        shape=(NWORDS * VPW,),
        strides=(stride_y1,),
        offsets=(0,),
        block_shape=(NWORDS * VPW,),
        order=(0,)
    )

    tl.store(p_block_ptr, p_relu_mask)
    tl.store(y_block_ptr, x.to(tl.bfloat16))


@triton.jit
def relu_bwd_fused_unpack_kernel(
    P_ptr, DY_ptr, DX_ptr,
    stride_p0, stride_p1,
    stride_dy0, stride_dy1,
    stride_dx0, stride_dx1,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)

    p_block_ptr = tl.make_block_ptr(
        base=P_ptr + pid * stride_p0,
        shape=(NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        block_shape=(NWORDS,),
        order=(0,)
    )

    dy_block_ptr = tl.make_block_ptr(
        base=DY_ptr + pid * stride_dy0,
        shape=(NWORDS, VPW),
        strides=(VPW, stride_dy1),
        offsets=(0,0),
        block_shape=(NWORDS, VPW),
        order=(0,1)
    )

    packed = tl.load(p_block_ptr)

    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    relu_mask = ((packed[:, None] >> shifts[None, :]) & 1).to(tl.float32) # [NWORDS, VPW]
    dy = tl.load(dy_block_ptr).to(tl.float32)

    dx = dy * relu_mask
    dx = tl.reshape(dx, (NWORDS * VPW,))

    dx_block_ptr = tl.make_block_ptr(
        base=DX_ptr + pid * stride_dx0,
        shape=(NWORDS * VPW,),
        strides=(stride_dx1,),
        offsets=(0,),
        block_shape=(NWORDS * VPW,),
        order=(0,)
    )

    tl.store(dx_block_ptr, dx.to(tl.bfloat16))





# ------
# SiLU
# ------

@triton.jit
def silu_triton(
    X_ptr, Y_ptr, ACT_ptr,
    n_elt,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elt

    x = tl.load(X_ptr + offs, mask=mask, other=0.).to(tl.float32)

    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    y = sigmoid * x
    act = sigmoid + x * sigmoid * (1 - sigmoid)

    tl.store(Y_ptr + offs, y.to(tl.bfloat16), mask=mask)
    tl.store(ACT_ptr + offs, act.to(tl.bfloat16), mask=mask)




@triton.jit
def norm(x, NOWORDS, VPW):
    sum_square = tl.sum(x * x)
    rms = tl.sqrt(sum_square / (NOWORDS * VPW))
    return x / (rms + 1e-8)



@triton.jit
def silu_fwd_fused_quan_pack_triton(
    X_ptr, Y_ptr, Packed_ptr,
    Scale_ptr, Min_ptr,
    seed,
    stride_x0, stride_x1,
    stride_y0, stride_y1,
    stride_p0, stride_p1,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr,
    QMAX: tl.constexpr,
    AVG_ALAM: tl.constexpr,
    ALAM_BITS: tl.constexpr,
    ALAM_NWORDS: tl.constexpr,
    SUB_GROUP: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    x_block_ptr = tl.make_block_ptr(
        base=X_ptr + pid * stride_x0,
        shape=(NWORDS * VPW,),
        strides=(stride_x1,),
        offsets=(0,),
        block_shape=(NWORDS * VPW,),
        order=(0,)
    )

    x = tl.load(x_block_ptr).to(tl.float32)

    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid

    y_block_ptr = tl.make_block_ptr(
        base=Y_ptr + pid * stride_y0,
        shape=(NWORDS * VPW,),
        strides=(stride_y1,),
        offsets=(0,),
        block_shape=(NWORDS * VPW,),
        order=(0,)
    )
    tl.store(y_block_ptr, y.to(tl.bfloat16))

    act = sigmoid + x * sigmoid * (1.0 - sigmoid)

    act = tl.clamp(act, -3, 3)
    act = libdevice.tanh(act / 0.5) * 0.5

    max = tl.max(act, axis=0)
    min = tl.min(act, axis=0)

    rng = max - min
    rng = tl.maximum(rng, 1e-4)
    scale = rng / QMAX
    scale = tl.where(scale > 0, scale, 1.0)

    tl.store(Scale_ptr + pid, scale.to(tl.bfloat16))
    tl.store(Min_ptr + pid, min.to(tl.bfloat16))

    inv_scale = 1.0 / scale 


        # noise_shape = tl.arange(0, SUB_GROUP)
        # sgd_noise = tl.rand(seed + pid, noise_shape)
    # act = libdevice.tanh(act * 0.5) / 0.5

        # act = norm(act, NWORDS, VPW)

        # act = tl.reshape(act, (SUB_GROUP, ALAM_BITS))
        # act = tl.sum(act, axis=1) / ALAM_BITS

        # alpha1 = 0.25
        # alpha2 = 0.25
        # beta_max = 0.5

        # mu = tl.sum(act, axis=1) * (1.0 / ALAM_BITS)
        # mx = tl.max(act, axis=1)
        # mv = tl.min(act, axis=1)

        # ms2 = tl.sum(act * act, axis=1) * (1.0 / ALAM_BITS)
        # rms = tl.sqrt(ms2 + 1e-8)
        # srms = tl.where(mu >= 0.0, rms, -rms)

        # y1 = (1.0 - alpha1 - alpha2) * mu + alpha1 * mx + alpha2 * mv

        # # cancellation indicator: small |mu| vs rms
        # c = tl.abs(mu) / (rms + 1e-8)          # in [0, 1-ish]
        # beta = (1.0 - c) * beta_max           # more RMS when c is small
        # act = (1.0 - beta) * y1 + beta * srms
    # else:
        # noise_shape = tl.arange(0, NWORDS * VPW)
        # sgd_noise = tl.rand(seed + pid, noise_shape)

    # max = tl.max(act, axis=0)
    # min = tl.min(act, axis=0)


    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    qf = (act - min) * inv_scale + (0.5 - 1e-6)

    if AVG_ALAM:
        qb = tl.reshape(qf, (SUB_GROUP, ALAM_BITS))
        qb = tl.sum(qb, axis=1) / ALAM_BITS



    # if AVG_ALAM:
    #     qf = tl.reshape(qf, (SUB_GROUP, ALAM_BITS))

        # k = tl.arange(0, ALAM_BITS)
        # qf_mean = tl.sum(qf, axis=1) / ALAM_BITS
        # pattern = tl.where((k < (ALAM_BITS // 2)), 1.0, -1.0).to(tl.float32)
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

        # mu = tl.sum(qf, axis=1) * (1.0 / ALAM_BITS)
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

    packed_block_ptr = tl.make_block_ptr(
        base=Packed_ptr + pid * stride_p0,
        shape=(ALAM_NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        block_shape=(ALAM_NWORDS,),
        order=(0,)
    )
    tl.store(packed_block_ptr, words.to(tl.int32))


@triton.jit
def silu_bwd_fused_dequan_unpack_triton(
    DY_ptr, Packed_ptr, DX_ptr,
    Scale_ptr, Min_ptr,
    seed,
    stride_dy0, stride_dy1,
    stride_p0, stride_p1,
    stride_dx0, stride_dx1,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr,
    AVG_ALAM: tl.constexpr,
    ALAM_BITS: tl.constexpr,
    ALAM_NWORDS: tl.constexpr,
    SUB_GROUP: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    packed_block_ptr = tl.make_block_ptr(
        base=Packed_ptr + pid * stride_p0,
        shape=(ALAM_NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        block_shape=(ALAM_NWORDS,),
        order=(0,)
    )

    packed = tl.load(packed_block_ptr)
    scale = tl.load(Scale_ptr + pid).to(tl.float32)
    min = tl.load(Min_ptr + pid).to(tl.float32)

    mask = (1 << BITS) - 1
    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    act = ((packed[:, None] >> shifts[None, :]) & mask) # [ALAM_NWORDS, VPW]

    if AVG_ALAM:
        act = tl.reshape(act, (ALAM_NWORDS * VPW,))
        act = tl.broadcast_to(act[:, None], (SUB_GROUP, ALAM_BITS))
        act = tl.reshape(act, (SUB_GROUP * ALAM_BITS,))

        noise_shape = tl.arange(0, SUB_GROUP * ALAM_BITS)
        sgd_noise = tl.rand(seed + pid, noise_shape)
        act = act.to(tl.float32) + (sgd_noise - 0.5)
    else:
        act = act.to(tl.float32)

    act = act * scale + min


    # act = act * tl.sigmoid(act)
    # act = act + 0.5 * tl.sin(act * 3.14159)
    # act = libdevice.tanh(act * 0.5) / 0.5
    # act = act * (1.0 / tl.abs(act + 1e-6))

    act = tl.reshape(act, (NWORDS * VPW,))

    dy_block_ptr = tl.make_block_ptr(
        base=DY_ptr + pid * stride_dy0,
        shape=(NWORDS * VPW,),
        strides=(stride_dy1,),
        offsets=(0,),
        block_shape=(NWORDS * VPW,),
        order=(0,)
    )

    dy = tl.load(dy_block_ptr).to(tl.float32)
    dx = dy * act

    dx_block_ptr = tl.make_block_ptr(
        base=DX_ptr + pid * stride_dx0,
        shape=(NWORDS * VPW,),
        strides=(stride_dx1,),
        offsets=(0,),
        block_shape=(NWORDS * VPW,),
        order=(0,)
    )

    tl.store(dx_block_ptr, dx.to(tl.bfloat16))





# -----------
# GELU
# -----------
@triton.jit
def gelu_triton(
    X_ptr, Y_ptr, ACT_ptr,
    n_elt,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_elt

    x = tl.load(X_ptr + offs, mask=mask, other=0.).to(tl.float32)

    # tanh approximation constants
    a = 0.7978845608028654  # sqrt(2/pi)
    b = 0.044715

    x2 = x * x
    x3 = x * x * x
    u = a * (x + b * x3)
    t = libdevice.tanh(u)

    y = 0.5 * x * (1.0 + t)
    tl.store(Y_ptr + offs, y.to(tl.bfloat16), mask=mask)

    # sech^2(u) = 1 - tanh(u)^2
    dt = 1.0 - t * t
    du_dx = a * (1.0 + 3.0 * b * x2)

    # dy/dx:
    # 0.5*(1+t) + 0.5*x*dt*du_dx
    dydx = 0.5 * (1.0 + t) + 0.5 * x * dt * du_dx
    tl.store(ACT_ptr + offs, dydx.to(tl.bfloat16), mask=mask)


@triton.jit
def gelu_fwd_fused_quan_pack_triton(
    X_ptr, Y_ptr, Packed_ptr,
    Scale_ptr, Min_ptr,
    stride_x0, stride_x1,
    stride_y0, stride_y1,
    stride_p0, stride_p1,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr,
    QMAX: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)

    x_block_ptr = tl.make_block_ptr(
        base=X_ptr + pid * stride_x0,
        shape=(NWORDS, VPW),
        strides=(VPW, stride_x1),
        offsets=(0, 0),
        block_shape=(NWORDS, VPW),
        order=(0, 1)
    )

    x = tl.load(x_block_ptr).to(tl.float32)

    # tanh approximation constants
    a = 0.7978845608028654  # sqrt(2/pi)
    b = 0.044715

    x2 = x * x
    x3 = x * x2

    u = a * (x + b * x3)
    t = libdevice.tanh(u)

    y = 0.5 * x * (1.0 + t)

    y_block_ptr = tl.make_block_ptr(
        base=Y_ptr + pid * stride_y0,
        block_shape=(NWORDS, VPW),
        strides=(VPW, stride_y1),
        offsets=(0, 0),
        shape=(NWORDS, VPW),
        order=(0, 1)
    )
    tl.store(y_block_ptr, y.to(tl.bfloat16))

    dt = 1.0 - t * t
    du_dx = a * (1.0 + 3.0 * b * x2)
    dydx = 0.5 * (1.0 + t) + 0.5 * x * dt * du_dx

    dydx = tl.clamp(dydx, -3.0, 3.0)
    dydx = libdevice.tanh(dydx / 0.5) * 0.5

    max = tl.max(dydx, axis=1)
    min = tl.min(dydx, axis=1)

    max = tl.max(max, axis=0)
    min = tl.min(min, axis=0)

    rng = max - min
    rng = tl.maximum(rng, 1e-4)
    scale = rng / QMAX
    scale = tl.where(scale > 0.0, scale, 1.0)

    tl.store(Scale_ptr + pid, scale.to(tl.bfloat16))
    tl.store(Min_ptr + pid, min.to(tl.bfloat16))

    inv_scale = 1.0 / scale

    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    eps  = 1e-6
    half = 0.5

    qf = (dydx - min) * inv_scale + (half - eps)
    qf = qf.to(tl.int32)
    qf = tl.maximum(qf, 0)
    qf = tl.minimum(qf, QMAX)

    words = tl.sum(qf << shifts[None, :], axis=1)

    packed_block_ptr = tl.make_block_ptr(
        base=Packed_ptr + pid * stride_p0,
        block_shape=(NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        shape=(NWORDS,),
        order=(0,)
    )
    tl.store(packed_block_ptr, words)


@triton.jit
def gelu_bwd_fused_dequan_unpack_triton(
    DY_ptr, DX_ptr, Packed_ptr,
    Scale_ptr, Min_ptr,
    stride_dy0, stride_dy1,
    stride_dx0, stride_dx1,
    stride_p0, stride_p1,
    BITS: tl.constexpr,
    VPW: tl.constexpr,
    NWORDS: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)

    packed_block_ptr = tl.make_block_ptr(
        base=Packed_ptr + pid * stride_p0,
        block_shape=(NWORDS,),
        strides=(stride_p1,),
        offsets=(0,),
        shape=(NWORDS,),
        order=(0,)
    )

    packed = tl.load(packed_block_ptr).to(tl.int32)

    mask = (1 << BITS) - 1
    j = tl.arange(0, VPW)
    shifts = (j * BITS).to(tl.int32)

    act = ((packed[:, None] >> shifts[None, :]) & mask).to(tl.float32)
    scale = tl.load(Scale_ptr + pid).to(tl.float32)
    min = tl.load(Min_ptr + pid).to(tl.float32)

    act = act * scale + min

    dy_block_ptr = tl.make_block_ptr(
        base=DY_ptr + pid * stride_dy0,
        block_shape=(NWORDS, VPW),
        strides=(VPW, stride_dy1),
        offsets=(0, 0),
        shape=(NWORDS, VPW),
        order=(0, 1)
    )

    dy = tl.load(dy_block_ptr).to(tl.float32)

    dx = dy * act

    dx_block_ptr = tl.make_block_ptr(
        base=DX_ptr + pid * stride_dx0,
        block_shape=(NWORDS, VPW),
        strides=(VPW, stride_dx1),
        offsets=(0, 0),
        shape=(NWORDS, VPW),
        order=(0, 1)
    )

    tl.store(dx_block_ptr, dx.to(tl.bfloat16))

