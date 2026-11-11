"""Pooling value/routing/factor launches with bounded temporary storage."""

from math import isinf, prod
import warnings

import torch
import triton

from .window.forward import _forward as _pool_forward
from .power.factors import _coefficient
from .unpool.forward import _zero, _scatter, _check_indices
from .common.launch import kernel_geometry, output_storage, finish_output
from .common.global_reduce import _global_partial, _global_merge, _global_finish
from ..quantization.activation.affine import _finish_min_max
from ..activation.elementwise_forward import _reduce_ranges
from ...reference.pooling import coefficient_geometry
from ....nn.activation.geometry import ReLUGeometry, tensor_arguments
from ....nn.pooling.state import PoolState
from ....quantization.activation import PackedActivation
from ....quantization.indices import empty_indices


def coefficient_state(x, compression, input_block_shape, power, need_state, checked, info, *, allocate_only=False):
    transport, block = coefficient_geometry(x, dict(compression=compression, input_block_shape=input_block_shape))
    if not need_state:
        return None, transport.layout
    rows, cols = transport.matrix_shape
    data, inverse, meta = tensor_arguments(x)
    packed = compression["enabled"]
    bits, group = compression["bits"], compression["group_size"]
    br, bc = block or (0, 0)
    elements = br * bc if br else group
    groups = triton.cdiv(rows, br) * triton.cdiv(cols, bc) if br else triton.cdiv(rows * cols, group)
    words = triton.cdiv(elements, 32 // bits)
    if packed:
        result = PackedActivation(torch.empty((groups, words), device=x.device, dtype=torch.int32),
                                  torch.empty(groups, device=x.device, dtype=torch.float32), torch.empty(groups, device=x.device, dtype=torch.float32),
                                  (rows, cols), group, block, bits)
        output, steps, minima = result.data, result.step, result.minimum
    else:
        result = torch.empty((rows, cols), device=x.device, dtype=torch.float32)
        output, steps, minima = result, result, result
    if allocate_only:
        return result, transport.layout
    size = max(32 // bits, triton.next_power_of_2(elements))
    size = min(4096, size)
    parts = triton.cdiv(elements, size)
    low = high = data
    prec = packed and parts > 1
    args = (data, inverse, output, steps, minima)
    constants = (tuple(x.shape), meta, transport.layout == "channels", rows, cols, br, bc,
                 elements, power, packed, bits, words, parts)
    if prec:
        low, high = torch.empty((groups, parts), device=x.device, dtype=torch.float32), torch.empty((groups, parts), device=x.device, dtype=torch.float32)
        _coefficient[(groups, parts)](*args, low, high, *constants, True, False, size, enable_fp_fusion=False)
        info["packing_launches"] += 1
        info["scratch_bytes"] += low.numel() * 8
        n = parts
        while n > 1024:
            nxt = triton.cdiv(n, 1024)
            lo, hi = torch.empty((groups, nxt), device=x.device, dtype=torch.float32), torch.empty((groups, nxt), device=x.device, dtype=torch.float32)
            _reduce_ranges[(groups, nxt)](low, high, lo, hi, n, nxt, 1024)
            low, high, n = lo, hi, nxt
            info["packing_launches"] += 1
            info["scratch_bytes"] += lo.numel() * 8
        _finish_min_max[(groups,)](low, high, steps, minima, n, bits, triton.next_power_of_2(n), enable_fp_fusion=False)
        info["packing_launches"] += 1
    _coefficient[(groups, parts)](*args, low, high, *constants, False, prec, size, enable_fp_fusion=False)
    info["packing_launches"] += 1
    if packed and checked and (not torch.isfinite(steps).all() or not torch.isfinite(minima).all()):
        raise ValueError("LP derivative compression requires finite preprocessed coefficients")
    return result, transport.layout


def power_program_block(g):
    # Forward reduction tiling must not depend on compression bits/groups.
    return 32 if g.layout == "channels" else min(128, max(1, 512 // triton.next_power_of_2(prod(g.kernel))))


def fused_coefficient_block(x, g, compression, input_block_shape, need_state):
    if not need_state or not compression["enabled"] or g.family != "power" or isinf(g.power):
        return None
    transport, block = coefficient_geometry(x, dict(compression=compression, input_block_shape=input_block_shape))
    group, volume = compression["group_size"], prod(g.kernel)
    if g.stride != g.kernel or any(g.padding) or any(d != 1 for d in g.dilation) or any(i % k for i, k in zip(g.input_spatial, g.kernel)):
        return None
    if block == (1, 32) and transport.layout == g.layout == "channels" and g.batched and g.channels % 32 == 0 and volume <= 128:
        return 32
    if block is not None or transport.layout != "last" or g.layout != "last" or group > 4096 or group % volume:
        return None
    outputs = power_program_block(g)
    if group != outputs * volume or any(i % k for i, k in zip(g.input_spatial, g.kernel)):
        return None
    if any(k > 1 and outputs % prod(g.output_spatial[axis + 1:]) for axis, k in enumerate(g.kernel)):
        return None
    return outputs


def global_reduction(g):
    return g.input_elements > 16384 and g.output_elements == 1 and (
        g.family.startswith("adaptive") or g.family in ("max", "average", "power") and
        g.input_spatial == g.kernel and not any(g.padding) and all(d == 1 for d in g.dilation))


def launch_global(source, source_inverse, source_meta, g, output, public, routing, factor,
                  scales, inverse, observations, snapshot, need_state, compression, bits,
                  dtype, fp8, br, bc, observe, capture_scale, info):
    total = prod(g.output_shape)
    parts = triton.cdiv(g.input_elements, 1024)
    maximum, lp = g.routes, g.family == "power"
    sums = torch.empty((parts, total), device=source.device, dtype=torch.float32)
    arguments = torch.empty((parts, total), device=source.device, dtype=torch.int64) if maximum else sums
    info["scratch_bytes"] += sums.numel() * (12 if maximum else 4)
    _global_partial[(triton.cdiv(total, 4) * parts,)](source, source_inverse, sums, arguments,
        g.shape, source_meta, total, g.input_elements, parts, maximum, g.power, lp, 1024, enable_fp_fusion=False)
    info["forward_launches"] += 1
    while parts > 128:
        nxt = triton.cdiv(parts, 128)
        next_sums = torch.empty((nxt, total), device=source.device, dtype=torch.float32)
        next_args = torch.empty((nxt, total), device=source.device, dtype=torch.int64) if maximum else next_sums
        _global_merge[(triton.cdiv(total, 4) * nxt,)](sums, arguments, next_sums, next_args, total, parts, nxt,
                                                     maximum, lp, 128, enable_fp_fusion=False)
        sums, arguments, parts = next_sums, next_args, nxt
        info["scratch_bytes"] += sums.numel() * (12 if maximum else 4)
        info["forward_launches"] += 1
    rows, cols = g.transport.matrix_shape
    _global_finish[(triton.cdiv(total, 32),)](sums, arguments, output, public if public is not None else output,
        routing, factor if factor is not None else output, scales if scales is not None else output,
        inverse if fp8 else output, observations if observations is not None else output,
        snapshot if snapshot is not None else output, output, total, parts, g.input_elements,
        g.divisor if g.divisor is not None else g.input_elements, maximum, lp, g.power, need_state, public is not None,
        compression["enabled"], bits, triton.next_power_of_2(max(1, min(bits, 32))), rows, cols, br, bc,
        fp8, scales is None, observe and fp8, capture_scale and fp8, torch.finfo(dtype).max if fp8 else 0.0,
        triton.next_power_of_2(parts), enable_fp_fusion=False)
    info["forward_launches"] += 1
    info["global_reduction"] = True


def forward(x, g, **options):
    with torch.cuda.device(x.device):
        return _forward(x, g, **options)


def _forward(x, g, *, options, dtype, block_shape, compression, check_finite, need_state,
             input_block_shape=None, samples=None, scale=None, observe=False, capture_scale=False):
    info = dict(backend="triton", forward_launches=0, packing_launches=0, output_cast_launches=0, scale_launches=0,
                scratch_bytes=0, workspace_dtype=None, deterministic_strategy="input_owned_gather",
                scale_mode="provided" if scale is not None else "current" if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else "none")
    if not prod(g.output_shape):
        output = torch.empty(g.transport.matrix_shape, device=x.device, dtype=dtype)
        inv = None
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            r, c = g.transport.matrix_shape
            br, bc = block_shape or (max(r, 1), max(c, 1))
            grid = (1, 1) if block_shape is None else (triton.cdiv(r, br), triton.cdiv(c, bc))
            inv = torch.ones(grid, device=x.device, dtype=torch.float32)
        public = torch.empty(g.output_shape, device=x.device, dtype=torch.int64) if options.get("return_indices", False) else None
        return output, inv, PoolState(public_indices=public), None, None, info
    if g.family == "unpool" and torch.are_deterministic_algorithms_enabled():
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn("CUDA max_unpool may be nondeterministic for repeated indices", UserWarning, stacklevel=2)
        else:
            raise RuntimeError("CUDA max_unpool does not have a deterministic implementation for repeated indices")
    source, source_inverse, source_meta = tensor_arguments(x)
    sample_data = samples.contiguous() if samples is not None else source
    route_transport = ReLUGeometry(g.shape, g.layout) if g.family == "unpool" else g.transport
    routing = None
    if need_state and g.routes:
        routing = (empty_indices(route_transport.matrix_shape, g.route_values, x.device) if compression["enabled"] else
                   torch.empty(route_transport.matrix_shape, device=x.device, dtype=torch.int64))
    route_data = routing.data if compression["enabled"] and routing is not None else routing if routing is not None else source
    bits = routing.bits if compression["enabled"] and routing is not None else 0
    finite_power = g.family == "power" and not isinf(g.power)
    fused_block = fused_coefficient_block(x, g, compression, input_block_shape, need_state)
    large_global = global_reduction(g)
    program_block = power_program_block(g) if finite_power and not large_global else 32
    coefficient, coefficient_layout = None, "last"
    if fused_block is not None:
        coefficient, coefficient_layout = coefficient_state(x, compression, input_block_shape, g.power, need_state,
                                                            check_finite, info, allocate_only=True)
    info["coefficient_fused"] = fused_block is not None
    output, inverse, observations, snapshot, inline, br, bc = output_storage(
        g.transport.matrix_shape, dtype, block_shape, scale, observe, capture_scale, check_finite, x.device,
        force_workspace=g.family == "unpool" and (scale is None or observe or capture_scale), program_block=program_block)
    fp8 = inverse is not None
    public_matrix = torch.empty(g.transport.matrix_shape, device=x.device, dtype=torch.int64) if options.get("return_indices", False) else None
    factor = torch.empty(g.transport.matrix_shape, device=x.device, dtype=torch.float32) if finite_power and need_state else None
    rows, cols = g.transport.matrix_shape
    if g.family == "unpool":
        programs = triton.cdiv(x.numel(), 32)
        errors = torch.empty(programs, device=x.device, dtype=torch.int32)
        nscales = inverse.numel() if fp8 else 0
        _zero[(triton.cdiv(max(output.numel(), nscales), 256),)](output, inverse if fp8 else output,
            scale if scale is not None else output, output.numel(), nscales, fp8, 256)
        _, _, index_meta = tensor_arguments(options["indices"])
        _scatter[(programs,)](source, source_inverse, options["indices"], output, route_data,
            scale if scale is not None else output, output, output, errors,
            g.shape, g.output_shape, source_meta, index_meta, g.layout == "channels", g.input_elements,
            g.output_elements, x.numel(), need_state, compression["enabled"], bits, triton.next_power_of_2(max(1, min(bits, 32))),
            cols, br, bc, fp8, False, False, torch.finfo(dtype).max if fp8 else 0.0, enable_fp_fusion=False)
        _check_indices[(triton.cdiv(programs, 256),)](errors, programs, 256, debug=True)
        info["forward_launches"] = 2
        info["validation_launches"] = 1
        info["scratch_bytes"] += errors.numel() * errors.element_size()
        # Inline unpool is only valid with precomputed scales. Current scales
        # and delayed observations must measure the completed scatter result.
        observations = snapshot = None
    elif large_global:
        launch_global(source, source_inverse, source_meta, g, output, public_matrix, route_data, factor,
                      scale, inverse, observations, snapshot, need_state, compression, bits, dtype, fp8, br, bc,
                      observe, capture_scale, info)
    else:
        programs = triton.cdiv(output.numel(), program_block)
        _pool_forward[(programs,)](source, source_inverse, sample_data, output,
            public_matrix if public_matrix is not None else output, route_data, factor if factor is not None else output,
            scale if scale is not None else output, inverse if fp8 else output,
            observations if observations is not None else output, snapshot if snapshot is not None else output, output,
            g.shape, g.output_shape, source_meta, kernel_geometry(g), g.layout == "channels", output.numel(),
            g.window_volume, need_state, public_matrix is not None, compression["enabled"], bits,
            triton.next_power_of_2(max(1, min(bits, 32))), rows, cols, br, bc, fp8, scale is None,
            observe and fp8, capture_scale and fp8, False, torch.finfo(dtype).max if fp8 else 0.0,
            min(128, triton.next_power_of_2(g.window_volume)),
            coefficient=coefficient.data if coefficient is not None else None,
            coefficient_step=coefficient.step if coefficient is not None else None,
            coefficient_minimum=coefficient.minimum if coefficient is not None else None,
            INLINE_COEFF=fused_block is not None, CGROUP=compression["group_size"], CBITS=compression["bits"],
            CWORDS=coefficient.data.shape[1] if coefficient is not None else 1,
            CBLOCK=max(32 // compression["bits"], triton.next_power_of_2(compression["group_size"])),
            BLOCK=program_block, BLOCK_COEFF=coefficient is not None and coefficient.block_shape == (1, 32),
            num_warps=4, enable_fp_fusion=False)
        info["forward_launches"] = 1
    if finite_power and coefficient is None:
        coefficient, coefficient_layout = coefficient_state(x, compression, input_block_shape, g.power,
                                                              need_state, check_finite, info)
    elif coefficient is not None and check_finite:
        if not torch.isfinite(coefficient.step).all() or not torch.isfinite(coefficient.minimum).all():
            raise ValueError("LP derivative compression requires finite preprocessed coefficients")
    if not inline:
        info["scratch_bytes"] += output.numel() * 4
        output, inverse, snapshot, observations = finish_output(output, dtype, block_shape, scale, observe,
                                                               capture_scale, check_finite, info)
    public = g.transport.restore(public_matrix) if public_matrix is not None else None
    state = PoolState(routing, coefficient, factor, public, samples if need_state else None, coefficient_layout)
    return output, inverse, state, snapshot, observations, info
