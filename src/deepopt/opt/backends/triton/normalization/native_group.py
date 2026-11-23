"""Experimental staged GN with distinct internal and native saved statistics.

No activation-sized FP32 staging output or derivative map is allocated. All
statistic/ds/db reductions use FP32 scratch. Target CUDA numerics and actual
replacement launches are mandatory before enabling this route by default.
"""
import torch
import triton
import triton.language as tl

from deepopt.opt.nn.normalization.geometry import output_strides
from .common import input_arguments, _load_input, _coordinates
from .statistics import _partial_stats, _merge_stats
from .activation import _activate, _adjust_spatial_gradient


@triton.jit
def _native_stats_finish(partial, mean, inverse, saved_mean, saved_inverse,
                         DOMAINS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    domain = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = domain < DOMAINS
    count = tl.load(partial + domain, valid, other=1)
    mu = tl.load(partial + DOMAINS + domain, valid, other=0)
    variance = tl.load(partial + 2 * DOMAINS + domain, valid, other=0) / count
    inv = tl.rsqrt(variance + EPS)
    tl.store(mean + domain, mu, valid)
    tl.store(inverse + domain, inv, valid)
    # Native BF16 statistics are a saved boundary, not the forward arithmetic.
    tl.store(saved_mean + domain, mu, valid)
    tl.store(saved_inverse + domain, inv, valid)


@triton.jit
def _native_group_apply(x, scales, weight, bias, mean, inverse, output, preactivation,
                        C: tl.constexpr, S: tl.constexpr, G: tl.constexpr, ELEMENTS: tl.constexpr,
                        META: tl.constexpr, OUT_STRIDES: tl.constexpr, HAS_WEIGHT: tl.constexpr,
                        HAS_BIAS: tl.constexpr, ACTIVATION: tl.constexpr, APPROXIMATE: tl.constexpr,
                        SAVE_PREACTIVATION: tl.constexpr, BLOCK: tl.constexpr):
    domain = tl.program_id(0).to(tl.int64)
    position = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = position < ELEMENTS
    n, c, s = _coordinates(domain, position, C, S, G, False)
    value = _load_input(x, scales, n, c, s, valid, C, S, META)
    mu = tl.load(mean + domain).to(tl.float32)
    inv = tl.load(inverse + domain).to(tl.float32)
    gamma = tl.load(weight + c, valid, other=1).to(tl.float32) if HAS_WEIGHT else 1.0
    beta = tl.load(bias + c, valid, other=0).to(tl.float32) if HAS_BIAS else 0.0
    multiplier = inv * gamma
    offset = -mu * multiplier + beta
    value = value * multiplier + offset
    address = n * OUT_STRIDES[0] + c * OUT_STRIDES[1]
    for axis in tl.static_range(len(META[2])):
        address += s // META[3][axis] % META[2][axis] * OUT_STRIDES[axis + 2]
    value = _activate(value, output, preactivation, address, valid, ACTIVATION, APPROXIMATE, SAVE_PREACTIVATION)
    tl.store(output + address, value, valid)


@triton.jit
def _native_ds_db(dy, dy_scales, x, x_scales, partial, derivative_input,
                  N: tl.constexpr, C: tl.constexpr, S: tl.constexpr, PARTS: tl.constexpr,
                  DY_META: tl.constexpr, X_META: tl.constexpr, DERIVATIVE_META: tl.constexpr,
                  ACTIVATION: tl.constexpr, APPROXIMATE: tl.constexpr, NEED_SOURCE: tl.constexpr, BLOCK: tl.constexpr):
    nc = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1).to(tl.int64)
    n, c = nc // C, nc % C
    s = part * BLOCK + tl.arange(0, BLOCK)
    valid = s < S
    gradient = _load_input(dy, dy_scales, n, c, s, valid, C, S, DY_META)
    gradient = _adjust_spatial_gradient(gradient, derivative_input, n, c, s, valid, C, S,
                                        DERIVATIVE_META, ACTIVATION, APPROXIMATE)
    value = _load_input(x, x_scales, n, c, s, valid, C, S, X_META) if NEED_SOURCE else 0.0
    tl.store(partial + nc * PARTS + part, tl.sum(tl.where(valid, gradient * value, 0), 0))
    tl.store(partial + N * C * PARTS + nc * PARTS + part, tl.sum(tl.where(valid, gradient, 0), 0))


@triton.jit
def _native_merge_ds_db(partial, output, NC: tl.constexpr, PARTS: tl.constexpr,
                        NEXT: tl.constexpr, BLOCK: tl.constexpr):
    nc = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1).to(tl.int64)
    offset = part * BLOCK + tl.arange(0, BLOCK)
    ds = tl.load(partial + nc * PARTS + offset, offset < PARTS, other=0)
    db = tl.load(partial + NC * PARTS + nc * PARTS + offset, offset < PARTS, other=0)
    tl.store(output + nc * NEXT + part, tl.sum(ds, 0))
    tl.store(output + NC * NEXT + nc * NEXT + part, tl.sum(db, 0))


@triton.jit
def _native_group_coefficients(ds_db, weight, mean, inverse, coefficients,
                               N: tl.constexpr, C: tl.constexpr, S: tl.constexpr, G: tl.constexpr,
                               HAS_WEIGHT: tl.constexpr, BLOCK: tl.constexpr):
    domain = tl.program_id(0).to(tl.int64)
    n, group = domain // G, domain % G
    offset = tl.arange(0, BLOCK)
    c = group * (C // G) + offset
    valid = offset < C // G
    gamma = tl.load(weight + c, valid, other=1).to(tl.float32) if HAS_WEIGHT else 1.0
    ds = tl.load(ds_db + n * C + c, valid, other=0)
    db = tl.load(ds_db + N * C + n * C + c, valid, other=0)
    ds = tl.sum(tl.where(valid, ds * gamma, 0), 0)
    db = tl.sum(tl.where(valid, db * gamma, 0), 0)
    mu = tl.load(mean + domain).to(tl.float32)
    inv = tl.load(inverse + domain).to(tl.float32)
    c2 = (db * mu - ds) * inv * inv * inv / ((C // G) * S)
    c3 = -c2 * mu - db * inv / ((C // G) * S)
    tl.store(coefficients + domain, c2)
    tl.store(coefficients + N * G + domain, c3)


@triton.jit
def _native_group_dx(dy, dy_scales, x, x_scales, weight, inverse, coefficients, dx, derivative_input,
                     N: tl.constexpr, C: tl.constexpr, S: tl.constexpr, G: tl.constexpr,
                     HAS_WEIGHT: tl.constexpr, DY_META: tl.constexpr, X_META: tl.constexpr,
                     DERIVATIVE_META: tl.constexpr, ACTIVATION: tl.constexpr, APPROXIMATE: tl.constexpr,
                     BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N * C * S
    n, c, s = offset // (C * S), offset // S % C, offset % S
    domain = n * G + c // (C // G)
    gradient = _load_input(dy, dy_scales, n, c, s, valid, C, S, DY_META)
    gradient = _adjust_spatial_gradient(gradient, derivative_input, n, c, s, valid, C, S,
                                        DERIVATIVE_META, ACTIVATION, APPROXIMATE)
    value = _load_input(x, x_scales, n, c, s, valid, C, S, X_META)
    gamma = tl.load(weight + c, valid, other=1).to(tl.float32) if HAS_WEIGHT else 1.0
    inv = tl.load(inverse + domain, valid, other=0).to(tl.float32)
    c2 = tl.load(coefficients + domain, valid, other=0)
    c3 = tl.load(coefficients + N * G + domain, valid, other=0)
    tl.store(dx + offset, gradient * (inv * gamma) + value * c2 + c3, valid)


@triton.jit
def _native_affine_finish(ds_db, mean, inverse, dw, dbias, N: tl.constexpr, C: tl.constexpr,
                          G: tl.constexpr, NEED_DW: tl.constexpr, NEED_DB: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.program_id(0).to(tl.int64)
    n = tl.arange(0, BLOCK).to(tl.int64)
    valid = n < N
    ds = tl.load(ds_db + n * C + c, valid, other=0)
    db = tl.load(ds_db + N * C + n * C + c, valid, other=0)
    if NEED_DW:
        domain = n * G + c // (C // G)
        mu = tl.load(mean + domain, valid, other=0).to(tl.float32)
        inv = tl.load(inverse + domain, valid, other=0).to(tl.float32)
        tl.store(dw + c, tl.sum(tl.where(valid, (ds - db * mu) * inv, 0), 0))
    if NEED_DB:
        tl.store(dbias + c, tl.sum(tl.where(valid, db, 0), 0))


def _reduce(partial, domains, parts, planes, kernel, scratch):
    while parts > 1:
        block = min(256, triton.next_power_of_2(parts))
        next_parts = triton.cdiv(parts, block)
        merged = torch.empty((planes, domains, next_parts), device=partial.device, dtype=torch.float32)
        scratch.append(merged.numel() * 4)
        kernel[(domains, next_parts)](partial, merged, domains, parts, next_parts, block, enable_fp_fusion=False)
        partial, parts = merged, next_parts
    return partial


def _validate(x, weight, bias, contract):
    if tuple(x.shape) != contract.shape or x.device.type != 'cuda':
        raise ValueError('Native staged GN requires matching CUDA geometry')
    if any(t is not None and t.device != x.device for t in (weight, bias)):
        raise ValueError('Native staged GN operands must share a device')
    affine_dtypes = (contract.affine_dtype, contract.compute_dtype) if contract.cast_affine else (contract.affine_dtype,)
    if any(t is not None and (tuple(t.shape) != (contract.channels,) or t.dtype not in affine_dtypes)
           for t in (weight, bias)):
        raise ValueError('Native staged GN affine schema differs from the captured contract')


def _validate_activation(tensor, activation, preactivation, contract):
    if activation is not None and activation not in (('silu', 'none'), ('gelu', 'none'), ('gelu', 'tanh')):
        raise ValueError('Unsupported native staged GN activation')
    if preactivation is not None and (preactivation.device != tensor.device or
            tuple(preactivation.shape) != contract.shape or preactivation.dtype != contract.compute_dtype):
        raise ValueError('Native staged GN preactivation ABI differs from its captured contract')


@torch.no_grad()
def forward(x, weight, bias, contract, *, activation=None, preactivation=None, execution_info=None):
    _validate(x, weight, bias, contract)
    _validate_activation(x, activation, preactivation, contract)
    with torch.cuda.device(x.device):
        source = x.to(contract.compute_dtype)
        gamma = weight.to(contract.compute_dtype) if contract.cast_affine and weight is not None else weight
        beta = bias.to(contract.compute_dtype) if contract.cast_affine and bias is not None else bias
        actual, scale, metadata = input_arguments(source)
        parts = triton.cdiv(contract.elements, 1024)
        partial = torch.empty((3, contract.domains, parts), device=x.device, dtype=torch.float32)
        scratch = [partial.numel() * 4]
        _partial_stats[(contract.domains, parts)](actual, scale, partial, contract.channels, contract.spatial,
            contract.groups, False, contract.domains, contract.elements, parts, metadata, 1024, enable_fp_fusion=False)
        partial = _reduce(partial, contract.domains, parts, 3, _merge_stats, scratch)
        mean = torch.empty(contract.domains, device=x.device, dtype=torch.float32)
        inverse = torch.empty_like(mean)
        saved_mean = torch.empty_like(mean, dtype=contract.statistics_dtype)
        saved_inverse = torch.empty_like(saved_mean)
        _native_stats_finish[(triton.cdiv(contract.domains, 128),)](partial, mean, inverse, saved_mean, saved_inverse,
            contract.domains, contract.eps, 128, enable_fp_fusion=False)
        output = torch.empty_strided(contract.shape, output_strides(source), device=x.device, dtype=contract.compute_dtype)
        _native_group_apply[(contract.domains, triton.cdiv(contract.elements, 512))](actual, scale,
            gamma if gamma is not None else mean, beta if beta is not None else mean, mean, inverse, output,
            preactivation if preactivation is not None else output, contract.channels, contract.spatial,
            contract.groups, contract.elements, metadata, output.stride(), gamma is not None, beta is not None,
            activation[0] if activation else '', activation[1] if activation else 'none', preactivation is not None, 512,
            enable_fp_fusion=False)
        if execution_info is not None:
            execution_info.update(contract.receipt(), native_group_staged=True,
                forward_scratch_bytes=sum(scratch) + mean.numel() * 8, forward_global_fp32_staging_output=False,
                output_storage_dtype=str(output.dtype),
                source_conversion_bytes=source.numel() * source.element_size() if source is not x else 0,
                saved_statistics_bytes=saved_mean.numel() * saved_mean.element_size() * 2,
                retained_preactivation_bytes=preactivation.numel() * preactivation.element_size() if preactivation is not None else 0)
        return output, saved_mean, saved_inverse, source, gamma


@torch.no_grad()
def backward(gradient, source, mean, inverse, weight, contract, *, needs=(True, True, True),
             derivative_input=None, derivative_activation=None, execution_info=None):
    needs_source = needs[0] or needs[1]
    if needs_source and (source is None or mean is None or inverse is None):
        raise ValueError('Native staged GN missing saved source/statistics')
    source = source if needs_source else gradient
    _validate(source, weight, None, contract)
    _validate_activation(gradient, derivative_activation, derivative_input, contract)
    if derivative_activation is not None and derivative_input is None:
        raise ValueError('Native staged GN derivative requires exact preactivation')
    if any(t is not None and t.device != source.device for t in (gradient, mean, inverse, derivative_input)):
        raise ValueError('Native staged GN backward operands must share a device')
    if (source.dtype != contract.compute_dtype or gradient.dtype != contract.compute_dtype or
            tuple(gradient.shape) != contract.shape or (needs_source and any(t.dtype != contract.statistics_dtype or
                t.numel() != contract.domains for t in (mean, inverse)))):
        raise ValueError('Native staged GN saved-statistic or gradient ABI differs from its captured contract')
    with torch.cuda.device(source.device):
        dy, dy_scale, dy_meta = input_arguments(gradient)
        x, x_scale, x_meta = input_arguments(source)
        derivative_meta = input_arguments(derivative_input)[2] if derivative_input is not None else ()
        mean = mean if mean is not None else gradient
        inverse = inverse if inverse is not None else gradient
        activation = derivative_activation[0] if derivative_activation else ''
        approximate = derivative_activation[1] if derivative_activation else 'none'
        parts = triton.cdiv(contract.spatial, 1024)
        nc = contract.n * contract.channels
        partial = torch.empty((2, nc, parts), device=source.device, dtype=torch.float32)
        scratch = [partial.numel() * 4]
        _native_ds_db[(nc, parts)](dy, dy_scale, x, x_scale, partial,
            derivative_input if derivative_input is not None else gradient, contract.n, contract.channels,
            contract.spatial, parts, dy_meta, x_meta, derivative_meta, activation, approximate, needs_source, 1024,
            enable_fp_fusion=False)
        sums = _reduce(partial, nc, parts, 2, _native_merge_ds_db, scratch)
        dx = dw = db = None
        if needs[0]:
            coefficients = torch.empty((2, contract.domains), device=source.device, dtype=torch.float32)
            scratch.append(coefficients.numel() * 4)
            _native_group_coefficients[(contract.domains,)](sums, weight if weight is not None else mean,
                mean, inverse, coefficients, contract.n, contract.channels, contract.spatial, contract.groups,
                weight is not None, triton.next_power_of_2(contract.channels // contract.groups), enable_fp_fusion=False)
            dx = torch.empty(contract.shape, device=source.device, dtype=contract.input_dtype)
            _native_group_dx[(triton.cdiv(dx.numel(), 512),)](dy, dy_scale, x, x_scale,
                weight if weight is not None else mean, inverse, coefficients, dx,
                derivative_input if derivative_input is not None else gradient,
                contract.n, contract.channels, contract.spatial, contract.groups, weight is not None,
                dy_meta, x_meta, derivative_meta, activation, approximate, 512, enable_fp_fusion=False)
        if needs[1] or needs[2]:
            dw = torch.empty(contract.channels, device=source.device, dtype=contract.affine_dtype) if needs[1] else None
            db = torch.empty_like(dw) if needs[2] and dw is not None else torch.empty(
                contract.channels, device=source.device, dtype=contract.affine_dtype) if needs[2] else None
            _native_affine_finish[(contract.channels,)](sums, mean, inverse,
                dw if dw is not None else mean, db if db is not None else mean, contract.n,
                contract.channels, contract.groups, needs[1], needs[2], triton.next_power_of_2(contract.n),
                enable_fp_fusion=False)
        if execution_info is not None:
            execution_info.update(contract.receipt(), native_group_staged=True,
                backward_scratch_bytes=sum(scratch), materialized_activation_gradient_bytes=0,
                input_gradient_storage_dtype=str(dx.dtype) if dx is not None else None,
                activation_derivative_fused=derivative_activation is not None)
        return dx, dw, db
