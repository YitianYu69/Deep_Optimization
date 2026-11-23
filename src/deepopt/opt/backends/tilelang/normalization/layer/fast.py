"""Cooperative suffix LayerNorm kernels; other layouts keep the generic path.

Row statistics use a centered two-pass FP32 reduction. Affine gradients use
bounded, deterministic partials, never atomics or an invocation-global cache.
The kernels receive explicit outputs and run on the caller's current stream.
"""

from functools import lru_cache

import torch

from ..._runtime import ceildiv, compile_kernel, device_key, dtype_name, require_cuda, runtime
from ...activation.common import dense, unpack


MAX_COLUMNS = 8192
AFFINE_WORKSPACE_LIMIT = 8 * 1024 * 1024
AFFINE_ROWS = 32
AFFINE_COLUMNS = 32
MAX_PARTS = 128
NATIVE = (torch.float16, torch.bfloat16, torch.float32)


def supported(value, geometry):
    """Do not reinterpret channel-oriented FP8 transport as suffix rows."""
    if geometry.layout != "last" or not 0 < geometry.elements <= MAX_COLUMNS or not geometry.rows:
        return False
    if hasattr(value, "_data"):
        return value._activation_layout == "last" and value._logical_dtype in NATIVE
    return value.dtype in NATIVE and value.is_contiguous()


def matrix(value, geometry):
    # Native tensors retain their dtype: widening every element on the host
    # would restore the full-size temporary eliminated by the fused kernels.
    return geometry.matrix(dense(value) if hasattr(value, "_data") else value).contiguous()


def _vector(value, columns, dummy):
    return value.reshape(-1).contiguous() if value is not None else dummy


@lru_cache(maxsize=128)
def _forward(rows, cols, source_dtype, weight_dtype, bias_dtype, output_dtype,
             has_weight, has_bias, save_h, eps, device):
    _, T = runtime()
    width = 1 << (cols - 1).bit_length()
    hrows, hcols = (rows, cols) if save_h else (1, 1)

    @T.prim_func
    def kernel(X: T.Tensor((rows, cols), source_dtype), W: T.Tensor((cols,), weight_dtype),
               B: T.Tensor((cols,), bias_dtype), Y: T.Tensor((rows, cols), output_dtype),
               M: T.Tensor((rows,), "float32"), R: T.Tensor((rows,), "float32"),
               H: T.Tensor((hrows, hcols), "float32")):
        with T.Kernel(rows, threads=256) as row_block:
            row = T.cast(row_block, "int64")
            delta = T.alloc_fragment((width,), "float32")
            squared = T.alloc_fragment((width,), "float32")
            total = T.alloc_fragment((1,), "float32")
            variance = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(width):
                delta[j] = 0.0
                if j < cols:
                    delta[j] = T.cast(X[row, j], "float32") - T.cast(X[row, 0], "float32")
            T.reduce_sum(delta, total, dim=0)
            for j in T.Parallel(width):
                # Saved mean is FP32. Variance and reconstructed xhat must
                # share that rounded center, including large-offset rows.
                squared[j] = 0.0
                if j < cols:
                    center = T.cast(X[row, 0], "float32") + total[0] / cols
                    difference = T.cast(X[row, j], "float32") - center
                    squared[j] = difference * difference
            T.reduce_sum(squared, variance, dim=0)
            for j in T.Parallel(width):
                if j < cols:
                    mu = T.cast(X[row, 0], "float32") + total[0] / cols
                    inverse = T.rsqrt(variance[0] / cols + eps)
                    h = (T.cast(X[row, j], "float32") - mu) * inverse
                    value = T.alloc_var("float32")
                    value = h
                    if save_h:
                        H[row, j] = h
                    if has_weight:
                        value = value * T.cast(W[j], "float32")
                    if has_bias:
                        value = value + T.cast(B[j], "float32")
                    Y[row, j] = T.cast(value, output_dtype)
                    if j == 0:
                        M[row] = mu
                        R[row] = inverse

    return compile_kernel(kernel, device)


def forward(value, weight, bias, geometry, output_dtype, *, save_h=False):
    source = matrix(value, geometry)
    device = require_cuda(source, weight, bias)
    rows, cols = geometry.matrix_shape
    dummy = torch.empty(cols, device=device, dtype=torch.float32)
    w, b = _vector(weight, cols, dummy), _vector(bias, cols, dummy)
    result = torch.empty((rows, cols), device=device, dtype=output_dtype)
    mean = torch.empty(rows, device=device, dtype=torch.float32)
    inverse = torch.empty_like(mean)
    h = torch.empty((rows, cols) if save_h else (1, 1), device=device, dtype=torch.float32)
    _forward(rows, cols, dtype_name(source.dtype), dtype_name(w.dtype), dtype_name(b.dtype),
             dtype_name(output_dtype), weight is not None, bias is not None, save_h, float(geometry.eps),
             device_key(device))(source, w, b, result, mean, inverse, h)
    return result, mean, inverse, h if save_h else None


@lru_cache(maxsize=128)
def _dx(rows, cols, source_dtype, dy_dtype, weight_dtype, output_dtype, reconstruct,
        correct, gain, has_weight, device):
    _, T = runtime()
    width = 1 << (cols - 1).bit_length()

    @T.prim_func
    def kernel(X: T.Tensor((rows, cols), source_dtype), D: T.Tensor((rows, cols), dy_dtype),
               W: T.Tensor((cols,), weight_dtype), M: T.Tensor((rows,), "float32"),
               R: T.Tensor((rows,), "float32"), S: T.Tensor((rows,), "float32"),
               G: T.Tensor((rows,), "float32"), Y: T.Tensor((rows, cols), output_dtype)):
        with T.Kernel(rows, threads=256) as row_block:
            row = T.cast(row_block, "int64")
            h = T.alloc_fragment((width,), "float32")
            u = T.alloc_fragment((width,), "float32")
            product = T.alloc_fragment((width,), "float32")
            total = T.alloc_fragment((1,), "float32")
            dot = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(width):
                h[j] = 0.0
                u[j] = 0.0
                if j < cols:
                    h[j] = T.cast(X[row, j], "float32")
                    if reconstruct:
                        h[j] = (h[j] - M[row]) * R[row]
                    if correct:
                        if (R[row] > 0) & (R[row] < T.infinity("float32")):
                            h[j] = h[j] - S[row]
                            if gain:
                                h[j] = h[j] * G[row]
                    u[j] = T.cast(D[row, j], "float32")
                    if has_weight:
                        u[j] = u[j] * T.cast(W[j], "float32")
                product[j] = u[j] * h[j]
            T.reduce_sum(u, total, dim=0)
            T.reduce_sum(product, dot, dim=0)
            for j in T.Parallel(width):
                if j < cols:
                    Y[row, j] = T.cast((u[j] - total[0] / cols - h[j] * dot[0] / cols) * R[row], output_dtype)

    return compile_kernel(kernel, device)


def affine_parts(rows, cols, need_dw, need_db):
    count = int(need_dw) + int(need_db)
    return min(ceildiv(rows, AFFINE_ROWS), MAX_PARTS,
               max(1, AFFINE_WORKSPACE_LIMIT // (cols * 4 * max(count, 1))))


@lru_cache(maxsize=128)
def _affine_partial(rows, cols, parts, source_dtype, dy_dtype, reconstruct, correct, gain,
                    need_dw, need_db, device):
    _, T = runtime()
    # One joint allocation, containing only the requested gradient partials.
    buffers = int(need_dw) + int(need_db)
    xrows, xcols = (rows, cols) if need_dw else (1, 1)

    @T.prim_func
    def kernel(X: T.Tensor((xrows, xcols), source_dtype), D: T.Tensor((rows, cols), dy_dtype),
               M: T.Tensor((rows,), "float32"), R: T.Tensor((rows,), "float32"),
               S: T.Tensor((rows,), "float32"), G: T.Tensor((rows,), "float32"),
               P: T.Tensor((buffers, parts, cols), "float32")):
        with T.Kernel(parts, T.ceildiv(cols, AFFINE_COLUMNS), threads=256) as (part, column_block):
            values = T.alloc_fragment((AFFINE_ROWS, AFFINE_COLUMNS), "float32")
            products = T.alloc_fragment((AFFINE_ROWS, AFFINE_COLUMNS), "float32")
            db = T.alloc_fragment((AFFINE_COLUMNS,), "float32")
            dw = T.alloc_fragment((AFFINE_COLUMNS,), "float32")
            T.clear(db)
            T.clear(dw)
            for chunk in T.serial(T.ceildiv(rows, parts * AFFINE_ROWS)):
                for i, j in T.Parallel(AFFINE_ROWS, AFFINE_COLUMNS):
                    row = T.cast(part, "int64") * AFFINE_ROWS + T.cast(chunk, "int64") * parts * AFFINE_ROWS + i
                    col = T.cast(column_block, "int64") * AFFINE_COLUMNS + j
                    values[i, j] = 0.0
                    products[i, j] = 0.0
                    if row < rows and col < cols:
                        values[i, j] = T.cast(D[row, col], "float32")
                        if need_dw:
                            h = T.alloc_var("float32")
                            h = T.cast(X[row, col], "float32")
                            if reconstruct:
                                h = (h - M[row]) * R[row]
                            if correct:
                                if (R[row] > 0) & (R[row] < T.infinity("float32")):
                                    h = h - S[row]
                                    if gain:
                                        h = h * G[row]
                            products[i, j] = values[i, j] * h
                if need_dw:
                    T.reduce_sum(products, dw, dim=0, clear=False)
                if need_db:
                    T.reduce_sum(values, db, dim=0, clear=False)
            for j in T.Parallel(AFFINE_COLUMNS):
                col = T.cast(column_block, "int64") * AFFINE_COLUMNS + j
                if col < cols:
                    if need_dw:
                        P[0, part, col] = dw[j]
                    if need_db:
                        P[buffers - 1, part, col] = db[j]

    return compile_kernel(kernel, device)


@lru_cache(maxsize=128)
def _affine_finish(cols, parts, buffers, device):
    _, T = runtime()
    width = 1 << (parts - 1).bit_length()

    @T.prim_func
    def kernel(P: T.Tensor((buffers, parts, cols), "float32"), Y: T.Tensor((buffers, cols), "float32")):
        with T.Kernel(buffers, T.ceildiv(cols, AFFINE_COLUMNS), threads=256) as (kind, column_block):
            values = T.alloc_fragment((width, AFFINE_COLUMNS), "float32")
            total = T.alloc_fragment((AFFINE_COLUMNS,), "float32")
            for i, j in T.Parallel(width, AFFINE_COLUMNS):
                col = T.cast(column_block, "int64") * AFFINE_COLUMNS + j
                values[i, j] = 0.0
                if i < parts and col < cols:
                    values[i, j] = P[kind, i, col]
            T.reduce_sum(values, total, dim=0)
            for j in T.Parallel(AFFINE_COLUMNS):
                col = T.cast(column_block, "int64") * AFFINE_COLUMNS + j
                if col < cols:
                    Y[kind, col] = total[j]

    return compile_kernel(kernel, device)


def backward(gradient, weight, geometry, *, packed, x, mean, inverse, need_dx, need_dw, need_db,
             input_dtype, gradient_output=None, center_shift=None, moment_gain=None):
    dy = matrix(gradient, geometry)
    device = require_cuda(dy, weight, inverse)
    rows, cols = geometry.matrix_shape
    h = (unpack(packed).contiguous() if packed is not None else matrix(x, geometry)) if need_dx or need_dw else torch.empty((1, 1), device=device)
    dummy = torch.empty(rows, device=device, dtype=torch.float32)
    mu, inv = mean if mean is not None else dummy, inverse if inverse is not None else dummy
    shift, gain = center_shift if center_shift is not None else dummy, moment_gain if moment_gain is not None else dummy
    dx = None
    if need_dx:
        w = _vector(weight, cols, torch.empty(cols, device=device, dtype=torch.float32))
        output_dtype = input_dtype if gradient_output is None else torch.float32
        dx = torch.empty((rows, cols), device=device, dtype=output_dtype)
        _dx(rows, cols, dtype_name(h.dtype), dtype_name(dy.dtype), dtype_name(w.dtype), dtype_name(output_dtype),
            packed is None, center_shift is not None, moment_gain is not None, weight is not None,
            device_key(device))(h, dy, w, mu, inv, shift, gain, dx)
        dx = geometry.restore(dx)
        if gradient_output is not None:
            dx = gradient_output.convert(dx)
    dw = db = None
    workspace = 0
    if need_dw or need_db:
        parts = affine_parts(rows, cols, need_dw, need_db)
        count = int(need_dw) + int(need_db)
        partials = torch.empty((count, parts, cols), device=device, dtype=torch.float32)
        result = torch.empty((count, cols), device=device, dtype=torch.float32)
        _affine_partial(rows, cols, parts, dtype_name(h.dtype), dtype_name(dy.dtype), packed is None,
            center_shift is not None, moment_gain is not None, need_dw, need_db, device_key(device))(
                h if need_dw else h[:1, :1], dy, mu, inv, shift, gain, partials)
        _affine_finish(cols, parts, count, device_key(device))(partials, result)
        dw = result[0].reshape(geometry.normalized_shape) if need_dw else None
        db = result[-1].reshape(geometry.normalized_shape) if need_db else None
        workspace = partials.numel() * partials.element_size()
    return dx, dw, db, workspace
