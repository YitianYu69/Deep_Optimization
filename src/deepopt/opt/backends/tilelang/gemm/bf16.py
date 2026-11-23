"""BF16 TileLang tensor-core products and exact-format saved-input packing.

All factories are lazy. These initial schedules require target-GPU performance
qualification; selecting TileLang never imports a Triton implementation.
"""

from functools import lru_cache

import torch

from .._runtime import runtime, compile_kernel, storage_view, dtype_name, require_cuda, ceildiv


@lru_cache(maxsize=256)
def _product(m, n, k, asm, ask, bsn, bsk, aspan, bspan, adtype, bdtype,
             bias_dtype, has_bias, bias_stride, bias_span, kind, approx, save, derivative, dapprox,
             dsm, dsn, dspan, ddtype, dpacked, dgroup, dwords, dbits,
             residual, rsm, rsn, rspan, packed, group, words, bits, tile, device):
    _, T = runtime()
    groups = ceildiv(k * n, group) if packed else 1
    dgroups = ceildiv(m * n, dgroup) if dpacked else 1
    bm, bn, bk = tile, tile, 32

    @T.prim_func
    def kernel(A: T.Tensor((aspan,), adtype), B: T.Tensor((bspan,), bdtype),
               Bias: T.Tensor((bias_span,), bias_dtype),
               D: T.Tensor((dspan,), ddtype), DS: T.Tensor((dgroups,), "float32"),
               DL: T.Tensor((dgroups,), "float32"), S: T.Tensor((groups,), "float32"),
               L: T.Tensor((groups,), "float32"), Y: T.Tensor((m * n,), "bfloat16"),
               Saved: T.Tensor((m * n,), "bfloat16"), R: T.Tensor((rspan,), "bfloat16")):
        with T.Kernel(ceildiv(m, bm), ceildiv(n, bn), threads=128) as (by, bx):
            left = T.alloc_shared((bm, bk), "bfloat16")
            right = T.alloc_shared((bn, bk), "bfloat16")
            acc = T.alloc_fragment((bm, bn), "float32")
            T.clear(acc)
            for panel in T.serial(ceildiv(k, bk)):
                for i, r in T.Parallel(bm, bk):
                    row = T.cast(by, "int64") * bm + i
                    reduction = T.cast(panel, "int64") * bk + r
                    if row < m and reduction < k:
                        left[i, r] = T.cast(A[row * asm + reduction * ask], "bfloat16")
                    else:
                        left[i, r] = T.cast(0, "bfloat16")
                for j, r in T.Parallel(bn, bk):
                    col = T.cast(bx, "int64") * bn + j
                    reduction = T.cast(panel, "int64") * bk + r
                    if col < n and reduction < k:
                        if packed:
                            logical = reduction * n + col
                            g, within = logical // group, logical % group
                            word = T.cast(B[g * words + within // (32 // bits)], "uint32")
                            code = T.cast((word >> ((within % (32 // bits)) * bits)) & ((1 << bits) - 1), "float32")
                            product = code * S[g]
                            value = T.if_then_else(product == T.infinity("float32"),
                                (code * (S[g] * 0.5) + L[g] * 0.5) * 2.0, product + L[g])
                            if bits == 1:
                                value = T.if_then_else(S[g] < 0, (L[g] * 0.5 + code * -S[g]) * 2.0, value)
                            right[j, r] = T.cast(value, "bfloat16")
                        else:
                            right[j, r] = T.cast(B[col * bsn + reduction * bsk], "bfloat16")
                    else:
                        right[j, r] = T.cast(0, "bfloat16")
                T.gemm(left, right, acc, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                row, col = T.cast(by, "int64") * bm + i, T.cast(bx, "int64") * bn + j
                if row < m and col < n:
                    value = acc[i, j]
                    if has_bias:
                        value = value + T.cast(T.cast(Bias[col * bias_stride], "bfloat16"), "float32")
                    rounded = T.cast(value, "bfloat16")
                    if residual:
                        rounded = T.cast(T.cast(rounded, "float32") + T.cast(R[row * rsm + col * rsn], "float32"), "bfloat16")
                    if save:
                        Saved[row * n + col] = rounded
                    x = T.cast(rounded, "float32")
                    if kind == "silu":
                        x = x / (1.0 + T.exp(-x))
                    elif kind == "gelu":
                        if approx == "tanh":
                            x = 0.5 * x * (1.0 + T.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
                        else:
                            x = x * (0.5 * (1.0 + T.erf(x * 0.7071067811865476)))
                    result = T.cast(x, "bfloat16")
                    if derivative != "none" or dpacked:
                        if dpacked:
                            logical = row * n + col
                            dg, within = logical // dgroup, logical % dgroup
                            word = T.cast(D[dg * dwords + within // (32 // dbits)], "uint32")
                            code = T.cast((word >> ((within % (32 // dbits)) * dbits)) & ((1 << dbits) - 1), "float32")
                            p = code * DS[dg]
                            coefficient = T.if_then_else(p == T.infinity("float32"),
                                (code * (DS[dg] * 0.5) + DL[dg] * 0.5) * 2.0, p + DL[dg])
                            if dbits == 1:
                                coefficient = T.if_then_else(DS[dg] < 0, (DL[dg] * 0.5 + code * -DS[dg]) * 2.0, coefficient)
                        else:
                            source = T.cast(D[row * dsm + col * dsn], "float32")
                        if not dpacked and derivative == "silu":
                            sigmoid = 1.0 / (1.0 + T.exp(-source))
                            coefficient = sigmoid * (1.0 + source * (1.0 - sigmoid))
                        elif not dpacked and dapprox == "tanh":
                            t = T.tanh(0.7978845608028654 * (source + 0.044715 * source * source * source))
                            coefficient = 0.5 * (1.0 + t) + 0.5 * source * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 0.134145 * source * source)
                        elif not dpacked:
                            coefficient = 0.5 * (1.0 + T.erf(source * 0.7071067811865476)) + source * 0.3989422804014327 * T.exp(-0.5 * source * source)
                        result = T.cast(T.cast(result, "float32") * coefficient, "bfloat16")
                    Y[row * n + col] = result

    return compile_kernel(kernel, device)


def product(a, b, *, bias=None, activation=None, derivative_input=None,
            derivative_activation=None, packed_b=None, tile=64, num_stages=3,
            return_preactivation=False, residual=None, derivative_packed=None):
    if num_stages != 3:
        raise ValueError("The initial TileLang BF16 schedule supports bf16_num_stages=3")
    device = require_cuda(a, b, bias, derivative_input)
    m, k = a.shape
    n = b.shape[0] if packed_b is None else packed_b.shape[1]
    av = storage_view(a)
    rhs = storage_view(b) if packed_b is None else packed_b.data.view(-1)
    output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    saved = torch.empty_like(output) if return_preactivation else output
    d = (derivative_packed.data.view(-1) if derivative_packed is not None else
         storage_view(derivative_input) if derivative_input is not None else output.view(-1))
    r = storage_view(residual) if residual is not None else output.view(-1)
    bias_arg = storage_view(bias) if bias is not None else output.view(-1)
    # Scale metadata is unused for ordinary BF16 products.
    dummy = torch.empty(1, device=device, dtype=torch.float32)
    with torch.cuda.device(device):
        _product(m, n, k, *a.stride(), *(b.stride() if b is not None else (0, 0)),
                 av.numel(), rhs.numel(), dtype_name(a.dtype), dtype_name(rhs.dtype),
                 dtype_name(bias_arg.dtype), bias is not None, bias.stride(0) if bias is not None else 1,
                 bias_arg.numel(), *(activation or ("none", "none")),
                 return_preactivation, *(derivative_activation or ("none", "none")),
                 *(derivative_input.stride() if derivative_input is not None else (0, 0)), d.numel(),
                 dtype_name(d.dtype), derivative_packed is not None,
                 derivative_packed.group_size if derivative_packed is not None else 1,
                 derivative_packed.data.shape[1] if derivative_packed is not None else 1,
                 derivative_packed.bits if derivative_packed is not None else 2,
                 residual is not None, *(residual.stride() if residual is not None else (0, 0)), r.numel(),
                 packed_b is not None, packed_b.group_size if packed_b is not None else 1,
                 packed_b.data.shape[1] if packed_b is not None else 1,
                 packed_b.bits if packed_b is not None else 2, tile, str(device))(
                     av, rhs, bias_arg, d, derivative_packed.step if derivative_packed is not None else dummy,
                     derivative_packed.minimum if derivative_packed is not None else dummy,
                     packed_b.step if packed_b is not None else dummy,
                     packed_b.minimum if packed_b is not None else dummy, output.view(-1), saved.view(-1), r)
    return (output, saved) if return_preactivation else output


@lru_cache(maxsize=128)
def _prepare(count, group, bits, kind, approx, dtype, save_derivative, checked, device):
    _, T = runtime()
    words, qmax, vpw = ceildiv(group, 32 // bits), (1 << bits) - 1, 32 // bits
    groups = ceildiv(count, group)

    @T.prim_func
    def kernel(X: T.Tensor((count,), dtype), Y: T.Tensor((count,), "bfloat16"),
               P: T.Tensor((groups * words,), "int32"), S: T.Tensor((groups,), "float32"),
               L: T.Tensor((groups,), "float32"), DP: T.Tensor((groups * words,), "int32"),
               DS: T.Tensor((groups,), "float32"), DL: T.Tensor((groups,), "float32"),
               Errors: T.Tensor((groups,), "int32")):
        with T.Kernel(ceildiv(groups, 128), threads=128) as bid:
            low = T.alloc_local((1,), "float32")
            high = T.alloc_local((1,), "float32")
            word = T.alloc_local((1,), "uint32")
            dlow = T.alloc_local((1,), "float32")
            dhigh = T.alloc_local((1,), "float32")
            bad = T.alloc_local((1,), "int32")
            for lane in T.Parallel(128):
                g = T.cast(bid, "int64") * 128 + lane
                if g < groups:
                    low[0] = T.infinity("float32")
                    high[0] = -T.infinity("float32")
                    dlow[0] = T.infinity("float32")
                    dhigh[0] = -T.infinity("float32")
                    bad[0] = 0
                    for index in T.serial(group):
                        offset = g * group + index
                        if offset < count:
                            x = T.cast(X[offset], "float32")
                            source = x
                            if save_derivative:
                                if kind == "silu":
                                    sigmoid = 1.0 / (1.0 + T.exp(-source))
                                    coefficient = sigmoid * (1.0 + source * (1.0 - sigmoid))
                                elif approx == "tanh":
                                    t = T.tanh(0.7978845608028654 * (source + 0.044715 * source * source * source))
                                    coefficient = 0.5 * (1.0 + t) + 0.5 * source * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 0.134145 * source * source)
                                else:
                                    coefficient = 0.5 * (1.0 + T.erf(source * 0.7071067811865476)) + source * 0.3989422804014327 * T.exp(-0.5 * source * source)
                                dlow[0] = T.min(dlow[0], coefficient)
                                dhigh[0] = T.max(dhigh[0], coefficient)
                                if checked:
                                    bad[0] = bad[0] | T.cast(~(T.abs(coefficient) < T.infinity("float32")), "int32")
                            if kind == "silu":
                                x = x / (1.0 + T.exp(-x))
                            elif kind == "gelu":
                                if approx == "tanh":
                                    x = 0.5 * x * (1.0 + T.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
                                else:
                                    x = x * (0.5 * (1.0 + T.erf(x * 0.7071067811865476)))
                            rounded = T.cast(x, "bfloat16")
                            Y[offset] = rounded
                            low[0] = T.min(low[0], T.cast(rounded, "float32"))
                            high[0] = T.max(high[0], T.cast(rounded, "float32"))
                    span = high[0] - low[0]
                    step = T.if_then_else(span == T.infinity("float32"),
                        T.ieee_fdiv(high[0], qmax) - T.ieee_fdiv(low[0], qmax), T.ieee_fdiv(span, qmax))
                    if bits == 1:
                        step = T.if_then_else(span == T.infinity("float32"), -(high[0] * 0.5 - low[0] * 0.5), span)
                    S[g] = step
                    L[g] = low[0]
                    for w in T.serial(words):
                        word[0] = T.uint32(0)
                        for digit in T.serial(vpw):
                            index = w * vpw + digit
                            offset = g * group + index
                            if index < group and offset < count:
                                value = T.cast(Y[offset], "float32")
                                difference = value - low[0]
                                normalized = T.if_then_else(difference == T.infinity("float32"),
                                    T.ieee_fdiv(value, step) - T.ieee_fdiv(low[0], step), T.ieee_fdiv(difference, step))
                                if bits == 1:
                                    normalized = T.if_then_else(step < 0,
                                        T.ieee_fdiv(value * 0.5 - low[0] * 0.5, -step), normalized)
                                    normalized = T.if_then_else(step != 0, normalized, 0.0)
                                else:
                                    normalized = T.if_then_else(step > 0, normalized, 0.0)
                                clipped = T.min(T.max(normalized, 0.0), T.float32(qmax))
                                floor = T.floor(clipped)
                                integer = T.cast(floor, "int32")
                                fraction = clipped - floor
                                code = integer + T.cast((fraction > 0.5) | ((fraction == 0.5) & ((integer & 1) != 0)), "int32")
                                word[0] = word[0] | (T.cast(code, "uint32") << (digit * bits))
                        P[g * words + w] = T.cast(word[0], "int32")
                    if save_derivative:
                        derivative_step = T.ieee_fdiv(dhigh[0] - dlow[0], qmax)
                        DS[g] = derivative_step
                        DL[g] = dlow[0]
                        if checked:
                            Errors[g] = bad[0]
                        for w in T.serial(words):
                            word[0] = T.uint32(0)
                            for digit in T.serial(vpw):
                                index = w * vpw + digit
                                offset = g * group + index
                                if index < group and offset < count:
                                    source = T.cast(X[offset], "float32")
                                    if kind == "silu":
                                        sigmoid = 1.0 / (1.0 + T.exp(-source))
                                        coefficient = sigmoid * (1.0 + source * (1.0 - sigmoid))
                                    elif approx == "tanh":
                                        t = T.tanh(0.7978845608028654 * (source + 0.044715 * source * source * source))
                                        coefficient = 0.5 * (1.0 + t) + 0.5 * source * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 0.134145 * source * source)
                                    else:
                                        coefficient = 0.5 * (1.0 + T.erf(source * 0.7071067811865476)) + source * 0.3989422804014327 * T.exp(-0.5 * source * source)
                                    normalized = T.if_then_else(derivative_step > 0,
                                        T.ieee_fdiv(coefficient - dlow[0], derivative_step), 0.0)
                                    clipped = T.min(T.max(normalized, 0.0), T.float32(qmax))
                                    floor = T.floor(clipped)
                                    integer = T.cast(floor, "int32")
                                    fraction = clipped - floor
                                    code = integer + T.cast((fraction > 0.5) | ((fraction == 0.5) & ((integer & 1) != 0)), "int32")
                                    word[0] = word[0] | (T.cast(code, "uint32") << (digit * bits))
                            DP[g * words + w] = T.cast(word[0], "int32")

    return compile_kernel(kernel, device)


def prepare(value, *, activation, compression, derivative_compression=None):
    from deepopt.opt.quantization.activation import PackedActivation
    from deepopt.opt._execution import finite_checks_enabled

    device = require_cuda(value)
    group, bits = compression["group_size"], compression["bits"]
    groups, words = ceildiv(value.numel(), group), ceildiv(group, 32 // bits)
    output = torch.empty_like(value, dtype=torch.bfloat16)
    payload = torch.empty((groups, words), device=device, dtype=torch.int32)
    step = torch.empty(groups, device=device, dtype=torch.float32)
    minimum = torch.empty_like(step)
    saved = derivative_compression is not None
    dpayload = torch.empty_like(payload) if saved else payload
    dstep, dminimum = (torch.empty_like(step), torch.empty_like(step)) if saved else (step, minimum)
    checked = saved and finite_checks_enabled()
    errors = torch.empty(groups, device=device, dtype=torch.int32) if checked else payload.view(-1)
    with torch.cuda.device(device):
        _prepare(value.numel(), group, bits, *(activation or ("none", "none")), dtype_name(value.dtype), saved, checked, str(device))(
            value.view(-1), output.view(-1), payload.view(-1), step, minimum, dpayload.view(-1), dstep, dminimum, errors)
    if checked and errors.any().item():
        raise ValueError("Activation derivative compression requires finite coefficients")
    result = output, PackedActivation(payload, step, minimum, tuple(value.shape), group, bits=bits)
    return (*result, PackedActivation(dpayload, dstep, dminimum, tuple(value.shape), group, bits=bits)) if saved else result
