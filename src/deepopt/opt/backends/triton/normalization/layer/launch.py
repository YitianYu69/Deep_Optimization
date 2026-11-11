"""Bounded LayerNorm launch policy and native/FP8 forward storage."""

from math import prod

import torch
import triton

from .forward import _row_forward, _epilogue, _output_amax, _xhat_ranges
from .statistics import _partial_statistics, _complete_statistics
from ..recenter import reduce_center
from ..statistics import _merge_stats, _finish_stats
from ...activation.elementwise_forward import _reduce_ranges
from ...quantization.activation.affine import _finish_min_max
from ...quantization.fp8 import _reduce_amax, _block_scales
from deepopt.opt.nn.activation.geometry import tensor_arguments
from deepopt.opt.nn.normalization.layer_state import LayerNormState
from deepopt.opt.quantization.activation import PackedActivation


def _reduce_scales(partial, groups, parts, inverse, maximum, info):
    while parts > 1024:
        nxt = triton.cdiv(parts, 1024)
        output = torch.empty((groups, nxt), dtype=torch.float32, device=partial.device)
        _reduce_amax[(groups * nxt,)](partial, output, parts, nxt, BLOCK=1024)
        partial, parts = output, nxt
        info["scale_launches"] += 1
    scales = torch.empty_like(inverse)
    _block_scales[(groups,)](partial, scales, inverse, parts,
                             BLOCK=triton.next_power_of_2(parts), FP8_MAX=maximum)
    info["scale_launches"] += 1
    return scales


def forward(x, geometry, **options):
    with torch.cuda.device(x.device):
        return _forward(x, geometry, **options)


def _forward(x, geometry, *, weight, bias, dtype, block_shape, compression, check_finite,
             scale=None, observe=False, capture_scale=False, center_shift=None):
    m, k = geometry.matrix_shape
    rows, cols = geometry.transport.matrix_shape
    data, x_inverse, meta = tensor_arguments(x)
    fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    maximum = torch.finfo(dtype).max if fp8 else 0.0
    br, bc = block_shape or (max(1, rows), max(1, cols))
    sr, sc = (1, 1) if block_shape is None else (triton.cdiv(rows, br), triton.cdiv(cols, bc))
    output = torch.empty((rows, cols), device=x.device, dtype=dtype)
    out_inverse = torch.empty((sr, sc), device=x.device, dtype=torch.float32) if fp8 else None
    mean, inverse = torch.empty(m, device=x.device, dtype=torch.float32), torch.empty(m, device=x.device, dtype=torch.float32)
    info = {"backend": "triton", "forward_launches": int(m * k > 0), "statistics_launches": 0,
            "scale_launches": 0, "compression_launches": 0, "workspace_dtype": None,
            "recenter_launches": 0, "recenter_scratch_bytes": 0,
            "recenter_partial_launches": 0, "recenter_partial_tiles": 0, "recenter_finish_launches": 0,
            "recenter_state_bytes": m * 4 if center_shift is not None else 0,
            "scale_mode": "provided" if scale is not None else "current" if fp8 else "none"}
    if not m * k:
        return output, out_inverse, LayerNormState(mean, inverse), None, None, info
    compressed, bits = compression["enabled"], compression["bits"]
    size = compression["group_size"] if compressed else (max(32, bc) if fp8 and block_shape is not None and br == 1 else 512)
    group = min(size, k)
    groups, pad = triton.cdiv(k, group), max(32 // bits, triton.next_power_of_2(group))
    pgroups = triton.next_power_of_2(groups)
    words = triton.cdiv(group, 32 // bits)
    packed = None
    if compressed:
        packed = PackedActivation(torch.empty((m * groups, words), device=x.device, dtype=torch.int32),
                                  torch.empty(m * groups, device=x.device, dtype=torch.float32),
                                  torch.empty(m * groups, device=x.device, dtype=torch.float32),
                                  (m, k), group, (1, group), bits)
    saved, steps, minima = (packed.data, packed.step, packed.minimum) if packed is not None else (output, output, output)
    channels = geometry.layout == "channels"
    c, s = (geometry.shape[1], prod(geometry.shape[2:])) if len(geometry.shape) >= 2 else (1, 1)
    w, b = weight if weight is not None else output, bias if bias is not None else output
    ws, bs = weight.stride() if weight is not None else (), bias.stride() if bias is not None else ()
    small = pgroups * pad <= 8192
    local = (fp8 and scale is None and small and (block_shape is None and m == 1 or
             block_shape is not None and not channels and k == cols and br == 1 and
             (group == pad or groups == 1) and bc <= pad and pad % bc == 0))
    direct = small and (not fp8 or scale is not None or local)
    prepass = small and not direct and block_shape is None
    snapshot = torch.empty(2, device=x.device, dtype=torch.float32) if capture_scale else None
    observations = None
    error_buffers = []
    if direct or prepass:
        observations = torch.empty(m, device=x.device, dtype=torch.float32) if observe or prepass else None
        errors = torch.empty(m, device=x.device, dtype=torch.int32) if check_finite else output
        _row_forward[(m,)](
            data, x_inverse, w, b, output, mean, inverse, saved, steps, minima,
            scale if scale is not None else output, out_inverse if fp8 else output,
            observations if observations is not None else output, snapshot if snapshot is not None else output, errors,
            m, k, geometry.shape, geometry.normalized_shape, meta, ws, bs, weight is not None, bias is not None,
            geometry.eps, group, groups, pgroups, pad, words, bits, compressed, fp8, prepass, local,
            block_shape is None, cols, channels, c, s, br, bc, sc, maximum, observe, capture_scale,
            check_finite, triton.next_power_of_2(k),
            center_shift=center_shift, RECENTER=center_shift is not None,
            num_warps=8 if pgroups * pad >= 4096 else 4, enable_fp_fusion=False)
        if check_finite:
            error_buffers.append(errors)
        if prepass:
            info["statistics_launches"] += 1
            scale = _reduce_scales(observations, 1, m, out_inverse, maximum, info)
    elif k <= 8192:
        _complete_statistics[(m,)](data, x_inverse, mean, inverse, k, geometry.shape, meta,
                                   geometry.eps, triton.next_power_of_2(k),
                                   num_warps=8 if k >= 4096 else 4, enable_fp_fusion=False)
        info["statistics_launches"] += 1
    else:
        parts = triton.cdiv(k, 4096)
        partial = torch.empty((3, m, parts), device=x.device, dtype=torch.float32)
        _partial_statistics[(m, parts)](data, x_inverse, partial, m, k, geometry.shape, meta, parts, 4096)
        info["statistics_launches"] += 1
        while parts > 1:
            nxt = triton.cdiv(parts, 128)
            reduced = torch.empty((3, m, nxt), device=x.device, dtype=torch.float32)
            _merge_stats[(m, nxt)](partial, reduced, m, parts, nxt, 128)
            partial, parts = reduced, nxt
            info["statistics_launches"] += 1
        _finish_stats[(triton.cdiv(m, 256),)](partial, mean, inverse, output, output, output,
                                            m, geometry.eps, True, False, False, 0.0, 256)
        info["statistics_launches"] += 1

    if fp8 and not direct and scale is None:
        parts = triton.cdiv(br * bc, 4096)
        partial = torch.empty((sr * sc, parts), device=x.device, dtype=torch.float32)
        _output_amax[(sr * sc, parts)](data, x_inverse, w, b, mean, inverse, partial,
                                      k, geometry.shape, geometry.normalized_shape, meta, ws, bs,
                                      weight is not None, bias is not None, rows, cols, channels, c, s,
                                      br, bc, sc, parts, 4096, enable_fp_fusion=False)
        info["scale_launches"] += 1
        scale = _reduce_scales(partial, sr * sc, parts, out_inverse, maximum, info)
    if not direct:
        # A tensorwise row prepass already packed xhat; do not encode it twice.
        save_here = compressed and not prepass
        block = min(4096 if group > 16384 else 16384, max(32 // bits, triton.next_power_of_2(group)))
        parts = triton.cdiv(group, block)
        prec = save_here and parts > 1
        if prec:
            lo = torch.empty((m * groups, parts), device=x.device, dtype=torch.float32)
            hi = torch.empty_like(lo)
            _xhat_ranges[(m * groups, parts)](data, x_inverse, mean, inverse, lo, hi, k, geometry.shape,
                                             meta, group, groups, parts, block, enable_fp_fusion=False)
            info["compression_launches"] += 1
            n = parts
            while n > 1024:
                nxt = triton.cdiv(n, 1024)
                nlo, nhi = torch.empty((m * groups, nxt), device=x.device, dtype=torch.float32), torch.empty((m * groups, nxt), device=x.device, dtype=torch.float32)
                _reduce_ranges[(m * groups, nxt)](lo, hi, nlo, nhi, n, nxt, 1024)
                lo, hi, n = nlo, nhi, nxt
                info["compression_launches"] += 1
            _finish_min_max[(m * groups,)](lo, hi, steps, minima, n, bits, triton.next_power_of_2(n), enable_fp_fusion=False)
            info["compression_launches"] += 1
        observations = torch.empty(m * groups * parts, device=x.device, dtype=torch.float32) if observe else None
        errors = torch.empty(m * groups * parts, device=x.device, dtype=torch.int32) if check_finite else output
        recenter_here = save_here and center_shift is not None
        center_parts = groups * parts
        center_errors = (torch.empty((2, m, center_parts), device=x.device, dtype=torch.float32)
                         if recenter_here and center_parts > 1 else output)
        _epilogue[(m * groups, parts)](
            data, x_inverse, w, b, mean, inverse, output, saved, steps, minima,
            scale if scale is not None else output, out_inverse if fp8 else output,
            observations if observe else output, snapshot if snapshot is not None else output, errors,
            k, geometry.shape, geometry.normalized_shape, meta, ws, bs, weight is not None, bias is not None,
            group, groups, words, bits, save_here, prec, fp8, cols, channels, c, s, br, bc, sc,
            maximum, parts, observe, capture_scale, check_finite, block,
            center_shift=center_shift, center_errors=center_errors, M=m, RECENTER=recenter_here, enable_fp_fusion=False)
        if recenter_here and center_parts > 1:
            launches, scratch = reduce_center(center_errors, center_shift, m, center_parts)
            info["recenter_launches"] += launches
            info["recenter_partial_launches"] = 1
            info["recenter_partial_tiles"] = m * center_parts
            info["recenter_finish_launches"] += launches
            info["recenter_scratch_bytes"] += scratch + center_errors.numel() * center_errors.element_size()
        if check_finite:
            error_buffers.append(errors)
    if check_finite and any(e.any().item() for e in error_buffers):
        raise ValueError("LayerNorm casting/compression requires finite values and positive finite scales")
    return output, out_inverse, LayerNormState(mean, inverse, packed), snapshot, observations, info
