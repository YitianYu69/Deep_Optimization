"""Numerically explicit TileLang normalization reductions and affine maps."""

from functools import lru_cache
import torch

from .._runtime import runtime, compile_kernel
from ..activation.common import cast, pack, unpack


@lru_cache(maxsize=256)
def _stats_kernel(rows, cols, eps, batch_stats, update, device):
    _, T = runtime()

    @T.prim_func
    def kernel(X: T.Tensor((rows, cols), "float32"), M: T.Tensor((rows,), "float32"),
               R: T.Tensor((rows,), "float32"), RM: T.Tensor((rows,), "float32"),
               RV: T.Tensor((rows,), "float32"), Momentum: T.Tensor((1,), "float32")):
        with T.Kernel(rows, threads=32) as domain:
            row = T.cast(domain, "int64")
            mu = T.alloc_var("float32")
            var = T.alloc_var("float32")
            if T.get_thread_binding() == 0:
                if batch_stats:
                    mu = 0.0
                    for j in T.serial(cols):
                        mu += X[row, j]
                    mu = mu / cols
                    var = 0.0
                    for j in T.serial(cols):
                        diff = X[row, j] - mu
                        var += diff * diff
                    var = var / cols
                    if update:
                        RM[row] = RM[row] + Momentum[0] * (mu - RM[row])
                        RV[row] = RV[row] + Momentum[0] * (var * (cols / (cols - 1)) - RV[row])
                else:
                    mu = RM[row]
                    var = RV[row]
                M[row] = mu
                R[row] = T.rsqrt(var + eps)

    return compile_kernel(kernel, device)


def statistics(values, eps, *, batch_stats=True, running_mean=None, running_var=None, momentum=None):
    values = values.contiguous()
    rows, cols = values.shape
    mean = torch.empty(rows, device=values.device, dtype=torch.float32)
    inverse = torch.empty_like(mean)
    rm = cast(running_mean, torch.float32) if running_mean is not None else torch.empty_like(mean)
    rv = cast(running_var, torch.float32) if running_var is not None else torch.empty_like(mean)
    update = running_mean is not None and batch_stats and momentum is not None
    m = cast(momentum.reshape(1), torch.float32) if isinstance(momentum, torch.Tensor) else torch.tensor(
        [0.0 if momentum is None else momentum], device=values.device, dtype=torch.float32)
    if rows and cols:
        _stats_kernel(rows, cols, eps, batch_stats, update, str(values.device))(values, mean, inverse, rm, rv, m)
    if update:
        running_mean.copy_(cast(rm, running_mean.dtype))
        running_var.copy_(cast(rv, running_var.dtype))
    return mean, inverse


@lru_cache(maxsize=256)
def _xhat_kernel(rows, cols, reconstruct, correct, gain, device):
    _, T = runtime()

    @T.prim_func
    def kernel(X: T.Tensor((rows, cols), "float32"), M: T.Tensor((rows,), "float32"),
               R: T.Tensor((rows,), "float32"), Shift: T.Tensor((rows,), "float32"),
               Gain: T.Tensor((rows,), "float32"), Y: T.Tensor((rows, cols), "float32")):
        with T.Kernel(T.ceildiv(rows * cols, 256), threads=256) as bx:
            for t in T.Parallel(256):
                p = T.cast(bx, "int64") * 256 + t
                if p < rows * cols:
                    row, col = p // cols, p % cols
                    h = T.alloc_var("float32")
                    h = (X[row, col] - M[row]) * R[row] if reconstruct else X[row, col]
                    if correct:
                        if (R[row] > 0) & (R[row] < T.infinity("float32")):
                            h = h - Shift[row]
                            if gain:
                                h = h * Gain[row]
                    Y[row, col] = h

    return compile_kernel(kernel, device)


def xhat(values, mean, inverse, *, reconstruct=True, center_shift=None, moment_gain=None):
    values = values.contiguous()
    result = torch.empty_like(values)
    rows, cols = values.shape
    if rows * cols:
        if inverse is None:
            inverse = torch.empty(rows, dtype=torch.float32, device=values.device)
        if mean is None:
            mean = inverse
        _xhat_kernel(rows, cols, reconstruct, center_shift is not None, moment_gain is not None,
            str(values.device))(values, mean, inverse, center_shift if center_shift is not None else mean,
            moment_gain if moment_gain is not None else mean, result)
    return result


@lru_cache(maxsize=256)
def _moments_kernel(rows, cols, moments, device):
    _, T = runtime()

    @T.prim_func
    def kernel(X: T.Tensor((rows, cols), "float32"), D: T.Tensor((rows, cols), "float32"),
               S: T.Tensor((rows,), "float32"), G: T.Tensor((rows,), "float32")):
        with T.Kernel(rows, threads=32) as domain:
            row = T.cast(domain, "int64")
            original_mean = T.alloc_var("float32")
            decoded_mean = T.alloc_var("float32")
            variance = T.alloc_var("float32")
            target = T.alloc_var("float32")
            if T.get_thread_binding() == 0:
                original_mean = 0.0
                decoded_mean = 0.0
                for j in T.serial(cols):
                    original_mean += X[row, j]
                    decoded_mean += D[row, j]
                original_mean = original_mean / cols
                decoded_mean = decoded_mean / cols
                S[row] = decoded_mean if moments else decoded_mean - original_mean
                if moments:
                    variance = 0.0
                    target = 0.0
                    for j in T.serial(cols):
                        diff = D[row, j] - decoded_mean
                        variance += diff * diff
                        target += X[row, j] * X[row, j]
                    G[row] = T.if_then_else(variance > 0, T.sqrt(target / T.max(variance, 1e-12 * cols)), 0.0)

    return compile_kernel(kernel, device)


def compress(h, compression, center_shift=None, moment_gain=None, group_size=None):
    if not compression["enabled"]:
        return None
    group = group_size or compression["group_size"]
    packed = pack(h, group, (1, group), compression["bits"])
    if center_shift is not None and h.numel():
        decoded = unpack(packed).contiguous()
        _moments_kernel(*h.shape, moment_gain is not None, str(h.device))(h.contiguous(), decoded,
            center_shift, moment_gain if moment_gain is not None else center_shift)
    return packed


@lru_cache(maxsize=256)
def _affine_kernel(count, channels, spatial, multiply_weight, add_bias, device):
    _, T = runtime()

    @T.prim_func
    def kernel(X: T.Tensor((count,), "float32"), W: T.Tensor((channels,), "float32"),
               B: T.Tensor((channels,), "float32"), Y: T.Tensor((count,), "float32")):
        with T.Kernel(T.ceildiv(count, 256), threads=256) as bx:
            for t in T.Parallel(256):
                p = T.cast(bx, "int64") * 256 + t
                if p < count:
                    c = (p // spatial) % channels
                    value = T.alloc_var("float32")
                    value = X[p]
                    if multiply_weight:
                        value = value * W[c]
                    if add_bias:
                        value = value + B[c]
                    Y[p] = value

    return compile_kernel(kernel, device)


def affine(values, weight, bias, channels, spatial):
    source = values.contiguous()
    result = torch.empty_like(source)
    dummy = torch.empty(channels, device=values.device)
    w = cast(weight, torch.float32).reshape(-1) if weight is not None else dummy
    b = cast(bias, torch.float32).reshape(-1) if bias is not None else dummy
    if source.numel():
        _affine_kernel(source.numel(), channels, spatial, weight is not None, bias is not None,
            str(values.device))(source.view(-1), w, b, result.view(-1))
    return result


@lru_cache(maxsize=256)
def _dx_kernel(rows, cols, batch_stats, device):
    _, T = runtime()

    @T.prim_func
    def kernel(U: T.Tensor((rows, cols), "float32"), H: T.Tensor((rows, cols), "float32"),
               R: T.Tensor((rows,), "float32"), Y: T.Tensor((rows, cols), "float32")):
        with T.Kernel(rows, threads=32) as domain:
            row = T.cast(domain, "int64")
            total = T.alloc_var("float32")
            product = T.alloc_var("float32")
            if T.get_thread_binding() == 0:
                total = 0.0
                product = 0.0
                if batch_stats:
                    for j in T.serial(cols):
                        total += U[row, j]
                        product += U[row, j] * H[row, j]
                for j in T.serial(cols):
                    if batch_stats:
                        Y[row, j] = (U[row, j] - total / cols - H[row, j] * product / cols) * R[row]
                    else:
                        Y[row, j] = U[row, j] * R[row]

    return compile_kernel(kernel, device)


def input_gradient(u, h, inverse, batch_stats=True):
    u, h = u.contiguous(), h.contiguous()
    result = torch.empty_like(u)
    if u.numel():
        _dx_kernel(*u.shape, batch_stats, str(u.device))(u, h, inverse, result)
    return result


@lru_cache(maxsize=256)
def _affine_gradient_kernel(batch, channels, spatial, need_dw, need_db, device):
    _, T = runtime()
    count = batch * channels * spatial

    @T.prim_func
    def kernel(D: T.Tensor((count,), "float32"), H: T.Tensor((count,), "float32"),
               DW: T.Tensor((channels,), "float32"), DB: T.Tensor((channels,), "float32")):
        with T.Kernel(channels, threads=32) as channel:
            c = T.cast(channel, "int64")
            dw = T.alloc_var("float32")
            db = T.alloc_var("float32")
            if T.get_thread_binding() == 0:
                dw = 0.0
                db = 0.0
                for i in T.serial(batch * spatial):
                    p = (i // spatial * channels + c) * spatial + i % spatial
                    if need_dw:
                        dw += D[p] * H[p]
                    if need_db:
                        db += D[p]
                DW[c] = dw
                DB[c] = db

    return compile_kernel(kernel, device)


def affine_gradient(dy, h, channels, spatial, need_dw, need_db):
    dw = torch.empty(channels, device=dy.device)
    db = torch.empty_like(dw)
    if channels and (need_dw or need_db):
        _affine_gradient_kernel(dy.numel() // (channels * spatial), channels, spatial, need_dw,
            need_db, str(dy.device))(dy.contiguous().view(-1), h.contiguous().view(-1), dw, db)
    return dw if need_dw else None, db if need_db else None
