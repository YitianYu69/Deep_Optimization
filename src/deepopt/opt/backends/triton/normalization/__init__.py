"""Fused normalization executors with bounded hierarchical scratch storage."""

import torch
import triton

from deepopt.opt.quantization.activation import PackedActivation
from deepopt.opt.nn.normalization.geometry import output_strides
from .common import input_arguments
from .recenter import reduce_center
from .statistics import _partial_stats, _merge_stats, _finish_stats
from .forward import _apply
from .backward import _partial_backward, _merge_sums, _input_gradient, _partial_affine, _finish_affine
from ..quantization.activation.affine import _finish_min_max


def _reduce(partial, domains, parts, planes, kernel):
    while parts > 1:
        block = min(256, triton.next_power_of_2(parts))
        next_parts = triton.cdiv(parts, block)
        result = torch.empty((planes, domains, next_parts), dtype=torch.float32, device=partial.device)
        kernel[(domains, next_parts)](partial, result, domains, parts, next_parts, block, enable_fp_fusion=False)
        partial, parts = result, next_parts
    return partial


@torch.no_grad()
def _forward(x, weight, bias, geometry, compression, *, running_mean=None, running_var=None, momentum=None,
            out_dtype=None, output_layout=None, center_shift=None, execution_info=None):
    source, scales, meta = input_arguments(x)
    c, s, g, domains, elements = (geometry.channels, geometry.spatial, geometry.groups,
                                  geometry.domains, geometry.elements)
    batch = geometry.kind == "batch"
    mean = torch.empty(domains, device=x.device, dtype=torch.float32)
    inverse = torch.empty_like(mean)
    partial = mean
    if geometry.batch_stats:
        parts = triton.cdiv(elements, 1024)
        partial = torch.empty((3, domains, parts), device=x.device, dtype=torch.float32)
        _partial_stats[(domains, parts)](source, scales, partial, c, s, g, batch, domains, elements, parts,
                                         meta, 1024, enable_fp_fusion=False)
        partial = _reduce(partial, domains, parts, 3, _merge_stats)
    tensor_momentum = isinstance(momentum, torch.Tensor)
    _finish_stats[(triton.cdiv(domains, 128),)](
        partial, mean, inverse, running_mean if running_mean is not None else mean,
        running_var if running_var is not None else mean, momentum if tensor_momentum else mean,
        domains, geometry.eps, geometry.batch_stats, running_mean is not None and geometry.batch_stats and momentum is not None,
        tensor_momentum, 0.0 if tensor_momentum or momentum is None else momentum, 128, enable_fp_fusion=False)
    output = torch.empty_strided(geometry.shape, output_strides(x, output_layout), device=x.device,
                                 dtype=out_dtype or x.dtype)
    pack = compression["enabled"]
    group = compression["group_size"] if pack else min(512, elements)
    groups, bits = triton.cdiv(elements, group), compression["bits"]
    words = triton.cdiv(group, 32 // bits)
    packed = None
    data = step = minimum = low = high = mean
    parts = 1
    if pack:
        data = torch.empty((domains * groups, words), dtype=torch.int32, device=x.device)
        step = torch.empty(domains * groups, dtype=torch.float32, device=x.device)
        minimum = torch.empty_like(step)
        packed = PackedActivation(data, step, minimum, geometry.matrix_shape, group, (1, group), bits)
    large = group > 4096 and pack
    block = 1024 if large else max(32 // bits, triton.next_power_of_2(group))
    if large:
        parts = triton.cdiv(group, block)
        low = torch.empty((domains * groups, parts), dtype=torch.float32, device=x.device)
        high = torch.empty_like(low)
    recenter = center_shift is not None and pack
    center_parts = groups * parts
    center_errors = (torch.empty((2, domains, center_parts), device=x.device, dtype=torch.float32)
                     if recenter and center_parts > 1 else mean)
    correction = dict(center_shift=center_shift, errors=center_errors, DOMAINS=domains, RECENTER=recenter)
    args = (source, scales, mean if weight is None else weight, mean if bias is None else bias,
            mean, inverse, output, data, step, minimum, low, high,
            c, s, g, batch, elements, groups, group, words, bits)
    _apply[(domains * groups, parts)](*args, pack, large, False, parts, weight is not None, bias is not None,
                                      meta, output.stride(), block, **correction, enable_fp_fusion=False)
    if large:
        _finish_min_max[(domains * groups,)](low, high, step, minimum, parts, bits,
                                             triton.next_power_of_2(parts), enable_fp_fusion=False)
        _apply[(domains * groups, parts)](*args, True, False, True, parts, weight is not None, bias is not None,
                                          meta, output.stride(), block, **correction, enable_fp_fusion=False)
    launches = scratch = 0
    if recenter and center_parts > 1:
        launches, scratch = reduce_center(center_errors, center_shift, domains, center_parts)
        scratch += center_errors.numel() * center_errors.element_size()
    if execution_info is not None:
        execution_info.update(recenter_launches=launches, recenter_scratch_bytes=scratch,
                              recenter_partial_launches=int(recenter and center_parts > 1),
                              recenter_partial_tiles=domains * center_parts if recenter and center_parts > 1 else 0,
                              recenter_finish_launches=launches,
                              recenter_state_bytes=domains * 4 if recenter else 0)
    return output, mean, inverse, packed


@torch.no_grad()
def _backward(gradient, weight, geometry, *, packed=None, x=None, mean=None, inverse=None,
             need_dx=True, need_dw=True, need_db=True, input_dtype=None, gradient_output=None, center_shift=None):
    dy, dy_scales, dy_meta = input_arguments(gradient)
    source, scales, x_meta = input_arguments(x if x is not None else gradient)
    c, s, g, domains, elements = (geometry.channels, geometry.spatial, geometry.groups,
                                  geometry.domains, geometry.elements)
    batch = geometry.kind == "batch"
    dummy = gradient._data if hasattr(gradient, "_data") else gradient
    data, step, minimum = (packed.data, packed.step, packed.minimum) if packed is not None else (dummy, dummy, dummy)
    group, words, bits = ((packed.group_size, packed.data.shape[1], packed.bits) if packed is not None else (1, 1, 2))
    mu, inv = (dummy if mean is None else mean), (dummy if inverse is None else inverse)
    w = dummy if weight is None else weight
    common = (packed is not None, group, words, bits)
    correction = dict(center_shift=center_shift, RECENTER=center_shift is not None)
    sums = dummy
    if need_dx and geometry.batch_stats:
        parts = triton.cdiv(elements, 1024)
        partial = torch.empty((2, domains, parts), dtype=torch.float32, device=gradient.device)
        _partial_backward[(domains, parts)](dy, dy_scales, data, step, minimum, source, scales, mu, inv, w, partial,
            c, s, g, batch, domains, elements, parts, *common, weight is not None, dy_meta, x_meta, 1024,
            **correction, enable_fp_fusion=False)
        sums = _reduce(partial, domains, parts, 2, _merge_sums)
    dx = dw = db = None
    fused = (gradient_output is not None and not gradient_output.checked and (gradient_output.scale is not None or
             gradient_output.block_shape is not None and gradient_output.block_shape[0] * gradient_output.block_shape[1] <= 16384))
    if need_dx and fused:
        from .gradient import _input_gradient_fp8
        from deepopt.opt.nn.activation.geometry import ReLUGeometry

        out = gradient_output
        rows, cols = ReLUGeometry(geometry.shape, out.layout).matrix_shape
        br, bc = out.block_shape or (rows, cols)
        current = out.scale is None
        block = max(32, triton.next_power_of_2(br * bc)) if current else 512
        programs = triton.cdiv(rows, br) * triton.cdiv(cols, bc) if current else triton.cdiv(rows * cols, block)
        out.allocate(rows, cols, gradient.device, programs)
        values = torch.empty((rows, cols), dtype=out.dtype, device=gradient.device)
        _input_gradient_fp8[(programs,)](
            dy, dy_scales, data, step, minimum, source, scales, mu, inv, w, sums,
            values, out.scale if out.scale is not None else inv, out.inverse,
            out.observations if out.observations is not None else inv,
            c, s, g, batch, domains, elements, geometry.batch_stats, *common,
            weight is not None, dy_meta, x_meta, rows, cols, geometry.shape[-1], out.layout == "channels",
            br, bc, current, out.observations is not None, torch.finfo(out.dtype).max, block, **correction, enable_fp_fusion=False)
        dx = out.wrap(values)
    elif need_dx:
        dx = torch.empty(geometry.shape, dtype=input_dtype or gradient.dtype, device=gradient.device)
        _input_gradient[(domains, triton.cdiv(elements, 512))](
            dy, dy_scales, data, step, minimum, source, scales, mu, inv, w, sums, dx,
            c, s, g, batch, domains, elements, geometry.batch_stats, *common,
            weight is not None, dy_meta, x_meta, 512, **correction, enable_fp_fusion=False)
        if gradient_output is not None:
            dx = gradient_output.convert(dx)
    if need_dw or need_db:
        parts = triton.cdiv(geometry.shape[0] * s, 1024)
        partial = torch.empty((2, c, parts), dtype=torch.float32, device=gradient.device)
        _partial_affine[(c, parts)](dy, dy_scales, data, step, minimum, source, scales, mu, inv, partial,
            geometry.shape[0], c, s, g, batch, elements, parts, *common, need_dw, need_db, dy_meta, x_meta, 1024,
            **correction, enable_fp_fusion=False)
        reduced = _reduce(partial, c, parts, 2, _merge_sums)
        dw = torch.empty(c, device=gradient.device, dtype=torch.float32) if need_dw else None
        db = torch.empty(c, device=gradient.device, dtype=torch.float32) if need_db else None
        _finish_affine[(triton.cdiv(c, 128),)](reduced, dw if need_dw else dummy, db if need_db else dummy,
                                             c, need_dw, need_db, 128)
    return dx, dw, db


def forward(x, *args, **kwargs):
    with torch.cuda.device(x.device):
        return _forward(x, *args, **kwargs)


def backward(gradient, *args, **kwargs):
    with torch.cuda.device(gradient.device):
        return _backward(gradient, *args, **kwargs)
