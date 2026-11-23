"""Fused normalization executors with bounded hierarchical scratch storage."""

import torch
import triton

from deepopt.opt.quantization.activation import PackedActivation
from deepopt.opt.nn.normalization.geometry import output_strides
from deepopt.opt.backends._shared.normalization import normalization_variant, measured_norm_device, measured_group_forward, channels_last_batch
from .common import input_arguments
from .recenter import reduce_center
from .statistics import _partial_stats, _merge_stats, _finish_stats
from .forward import _apply
from .small import _small_forward
from .backward import _partial_backward, _merge_sums, _input_gradient, _partial_affine, _finish_affine
from .channels_last import _channels_last_stats, _channels_last_apply, _channels_last_sums, _channels_last_dx
from ..quantization.activation.affine import _finish_min_max


def _reduce(partial, domains, parts, planes, kernel, info=None):
    while parts > 1:
        block = min(256, triton.next_power_of_2(parts))
        next_parts = triton.cdiv(parts, block)
        result = torch.empty((planes, domains, next_parts), dtype=torch.float32, device=partial.device)
        if info is not None:
            info["backward_scratch_bytes"] += result.numel() * result.element_size()
            info["backward_reduction_launches"] += 1
        kernel[(domains, next_parts)](partial, result, domains, parts, next_parts, block, enable_fp_fusion=False)
        partial, parts = result, next_parts
    return partial


@torch.no_grad()
def _forward(x, weight, bias, geometry, compression, *, running_mean=None, running_var=None, momentum=None,
            out_dtype=None, output_layout=None, center_shift=None, execution_info=None,
            activation=None, preactivation=None, output=None, activation_saved=None):
    if activation_saved is not None:
        from deepopt.opt.backends._shared.normalization_activation import validate_group_silu_residuals

        validate_group_silu_residuals(x, activation_saved, geometry, compression, activation)
        if output is not None or output_layout is not None or preactivation is not None or out_dtype != torch.bfloat16:
            raise ValueError("GroupNorm/SiLU residual fusion requires a fresh BF16 output without preactivation storage")
    if output is not None:
        from deepopt.opt.backends._shared.normalization_output import validate_normalization_output

        if normalization_variant() not in ("auto", "baseline"):
            raise ValueError("Normalization output destination does not support this executor variant")
        validate_normalization_output(output, x, weight, bias, geometry, dtype=out_dtype or x.dtype,
            output_layout=output_layout, activation=activation, preactivation=preactivation)
    if activation and (geometry.kind != "group" or output_layout is not None):
        raise ValueError("Spatial activation epilogues require GroupNorm with native dense output")
    fused_activation = dict(preactivation=preactivation, ACTIVATION=activation[0] if activation else "",
                            APPROXIMATE=activation[1] if activation else "none",
                            SAVE_PREACTIVATION=preactivation is not None)
    activation_errors = None
    if activation_saved is not None:
        from math import prod
        from deepopt.opt._execution import finite_checks_enabled

        activation_errors = (torch.empty((prod(geometry.shape) // 1024,), dtype=torch.int32, device=x.device)
                             if finite_checks_enabled() else None)
        fused_activation.update(activation_saved=activation_saved[0], activation_step=activation_saved[1],
            activation_minimum=activation_saved[2], SAVE_ACTIVATION_COEFFICIENT=True,
            activation_errors=activation_errors, CHECK_ACTIVATION_COEFFICIENT=activation_errors is not None)
    if execution_info is not None:
        execution_info.update(activation_epilogue=activation[0] if activation else None,
            retained_preactivation_bytes=preactivation.numel() * preactivation.element_size() if preactivation is not None else 0)
    source, scales, meta = input_arguments(x)
    c, s, g, domains, elements = (geometry.channels, geometry.spatial, geometry.groups,
                                  geometry.domains, geometry.elements)
    batch = geometry.kind == "batch"
    pack = compression["enabled"]
    group = compression["group_size"] if pack else min(512, elements)
    coalesced = channels_last_batch(geometry, meta, group, x.device)
    mean = torch.empty(domains, device=x.device, dtype=torch.float32)
    inverse = torch.empty_like(mean)
    measured_small = (normalization_variant() == "auto" and meta[0] == 0 and x.is_contiguous()
                      and weight is not None and bias is not None
                      and weight.is_contiguous() and bias.is_contiguous()
                      and measured_group_forward(geometry, x.dtype, out_dtype or x.dtype, compression,
                                                 center_shift is not None, x.device))
    candidate = (not coalesced and (normalization_variant() in ("candidate", "small_forward") or measured_small) and geometry.batch_stats and 0 < elements <= 4096
                 and (out_dtype or x.dtype) not in (torch.float8_e4m3fn, torch.float8_e5m2))
    if not candidate:
        partial = mean
        if geometry.batch_stats:
            parts = triton.cdiv(elements, 1024)
            partial = torch.empty((3, domains, parts), device=x.device, dtype=torch.float32)
            if coalesced:
                _channels_last_stats[(triton.cdiv(c, 4), parts)](source, scales, partial, c, s, elements, parts,
                                                               meta, 1024, 4, num_warps=4, enable_fp_fusion=False)
            else:
                _partial_stats[(domains, parts)](source, scales, partial, c, s, g, batch, domains, elements, parts,
                                                 meta, 1024, enable_fp_fusion=False)
            partial = _reduce(partial, domains, parts, 3, _merge_stats)
        tensor_momentum = isinstance(momentum, torch.Tensor)
        _finish_stats[(triton.cdiv(domains, 128),)](
            partial, mean, inverse, running_mean if running_mean is not None else mean,
            running_var if running_var is not None else mean, momentum if tensor_momentum else mean,
            domains, geometry.eps, geometry.batch_stats, running_mean is not None and geometry.batch_stats and momentum is not None,
            tensor_momentum, 0.0 if tensor_momentum or momentum is None else momentum, 128, enable_fp_fusion=False)
    if output is None:
        output = torch.empty_strided(geometry.shape, output_strides(x, output_layout), device=x.device,
                                     dtype=out_dtype or x.dtype)
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
    pad = max(32 // bits, triton.next_power_of_2(group))
    pgroups = triton.next_power_of_2(groups)
    inline = candidate and (not pack or group <= 4096 and pgroups * pad <= 8192)
    if candidate:
        tensor_momentum = isinstance(momentum, torch.Tensor)
        _small_forward[(domains,)](source, scales, mean if weight is None else weight, mean if bias is None else bias,
            running_mean if running_mean is not None else mean, running_var if running_var is not None else mean,
            momentum if tensor_momentum else mean, output, mean, inverse, data, step, minimum,
            center_shift if center_shift is not None else mean,
            c, s, g, batch, elements, meta, output.stride(), weight is not None, bias is not None,
            running_mean is not None and momentum is not None, tensor_momentum,
            0.0 if tensor_momentum or momentum is None else momentum, geometry.eps,
            pack and inline, group, groups, pgroups, pad, words, bits, center_shift is not None and pack,
            triton.next_power_of_2(elements), **fused_activation, num_warps=4 if elements < 2048 else 8, enable_fp_fusion=False)
        if inline:
            if execution_info is not None:
                execution_info.update(forward_variant="small_fused", forward_launches=1, statistics_launches=0,
                    compression_launches=0, recenter_launches=0, recenter_scratch_bytes=0,
                    recenter_partial_launches=0, recenter_partial_tiles=0, recenter_finish_launches=0,
                    recenter_state_bytes=domains * 4 if center_shift is not None else 0)
            return output, mean, inverse, packed
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
    if coalesced:
        _channels_last_apply[(triton.cdiv(c, 4), groups)](
            source, scales, mean if weight is None else weight, mean if bias is None else bias,
            mean, inverse, output, data, step, minimum,
            center_shift if center_shift is not None else mean, center_errors,
            c, s, elements, group, groups, words, bits, meta, output.stride(),
            weight is not None, bias is not None, pack, recenter, block, 4,
            num_warps=8 if block >= 2048 else 4, enable_fp_fusion=False)
        launches = scratch = 0
        if recenter and groups > 1:
            launches, scratch = reduce_center(center_errors, center_shift, domains, groups)
            scratch += center_errors.numel() * center_errors.element_size()
        if execution_info is not None:
            execution_info.update(forward_variant="channels_last_tiled", channel_tile=4,
                recenter_launches=launches, recenter_scratch_bytes=scratch,
                recenter_partial_launches=int(recenter and groups > 1), recenter_partial_tiles=c * groups if recenter else 0,
                recenter_finish_launches=launches, recenter_state_bytes=c * 4 if recenter else 0)
        return output, mean, inverse, packed
    args = (source, scales, mean if weight is None else weight, mean if bias is None else bias,
            mean, inverse, output, data, step, minimum, low, high,
            c, s, g, batch, elements, groups, group, words, bits)
    _apply[(domains * groups, parts)](*args, pack, large, False, parts, weight is not None, bias is not None,
                                      meta, output.stride(), block, **correction, **fused_activation, enable_fp_fusion=False)
    if large:
        _finish_min_max[(domains * groups,)](low, high, step, minimum, parts, bits,
                                             triton.next_power_of_2(parts), enable_fp_fusion=False)
        _apply[(domains * groups, parts)](*args, True, False, True, parts, weight is not None, bias is not None,
                                          meta, output.stride(), block, **correction, **fused_activation, enable_fp_fusion=False)
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
    if activation_errors is not None and activation_errors.any():
        raise ValueError("GroupNorm/SiLU affine derivative saves require finite coefficients")
    return output, mean, inverse, packed


@torch.no_grad()
def _backward(gradient, weight, geometry, *, packed=None, x=None, mean=None, inverse=None,
             need_dx=True, need_dw=True, need_db=True, input_dtype=None, gradient_output=None, center_shift=None,
             execution_info=None, derivative_input=None, derivative_activation=None, derivative_saved=None):
    cotangent_layout = {}
    if derivative_saved is not None:
        from deepopt.opt.backends._shared.normalization_activation import (
            canonicalize_group_silu_cotangent, validate_group_silu_residuals)

        validate_group_silu_residuals(gradient, derivative_saved, geometry,
            {"enabled": packed is not None, "group_size": packed.group_size if packed is not None else None,
             "bits": packed.bits if packed is not None else None}, ("silu", "none"), gradient=True)
        if derivative_input is not None or derivative_activation is not None or gradient_output is not None:
            raise ValueError("Packed activation residual fusion cannot combine another gradient transport")
        # Standalone activation backward writes a contiguous BF16 tensor before
        # these reductions. Preserve that load specialization for expanded and
        # strided cotangents as well, and expose the additional temporary.
        gradient, cotangent_layout = canonicalize_group_silu_cotangent(gradient)
    if derivative_activation and (geometry.kind != "group" or gradient_output is not None or hasattr(gradient, "_data")):
        raise ValueError("GroupNorm derivative fusion requires ordinary dense gradients")
    info = dict(backward_scratch_bytes=cotangent_layout.get("raw_cotangent_copy_bytes", 0),
                backward_reduction_launches=0, affine_gradient_launches=0, input_gradient_launches=0,
                **cotangent_layout)
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
    derivative = dict(derivative_input=derivative_input,
        DERIVATIVE_META=input_arguments(derivative_input)[2] if derivative_input is not None else (),
        ACTIVATION=derivative_activation[0] if derivative_activation else "",
        APPROXIMATE=derivative_activation[1] if derivative_activation else "none")
    if derivative_saved is not None:
        derivative.update(derivative_saved=derivative_saved[0], derivative_step=derivative_saved[1],
            derivative_minimum=derivative_saved[2], PACKED_DERIVATIVE=True)
    info.update(activation_derivative_fused=derivative_activation[0] if derivative_activation else None,
                materialized_activation_gradient_bytes=0 if derivative_activation else None)
    if derivative_saved is not None:
        info.update(activation_derivative_fused="silu_saved_coefficient", materialized_activation_gradient_bytes=0)
    coalesced = channels_last_batch(geometry, dy_meta, group, gradient.device)
    sums = dummy
    measured_shared = (normalization_variant() == "auto" and batch and need_dx and need_dw and need_db
                       and gradient_output is None and center_shift is None and input_dtype == torch.bfloat16
                       and gradient.dtype == torch.bfloat16
                       and dy_meta[0] == x_meta[0] == 0 and gradient.is_contiguous()
                       and (x is None or x.is_contiguous())
                       and (weight is None or weight.is_contiguous())
                       and tuple(geometry.shape) in ((16, 64, 56, 56), (32, 256, 7, 7))
                       and (packed is None or packed.bits == 2 and packed.group_size == 512)
                       and measured_norm_device(gradient.device))
    shared_batch = (batch and need_dx and geometry.batch_stats
                    and (coalesced or normalization_variant() in ("candidate", "batch_shared") or measured_shared
                         or weight is None and normalization_variant() != "baseline"))
    info["backward_variant"] = "channels_last_shared" if coalesced and shared_batch else "shared_batch" if shared_batch else "separate"
    if shared_batch:
        # In BatchNorm gamma is constant within a domain. One unweighted
        # hierarchy serves dx and affine gradients, including gamma == 0.
        parts = triton.cdiv(elements, 1024)
        partial = torch.empty((2, c, parts), dtype=torch.float32, device=gradient.device)
        info["backward_scratch_bytes"] += partial.numel() * partial.element_size()
        info["backward_reduction_launches"] += 1
        if coalesced:
            _channels_last_sums[(triton.cdiv(c, 4), parts)](dy, dy_scales, data, step, minimum, source, scales,
                mu, inv, partial, c, s, elements, parts, *common, dy_meta, x_meta, True, 1024, 4,
                **correction, num_warps=4, enable_fp_fusion=False)
        else:
            _partial_affine[(c, parts)](dy, dy_scales, data, step, minimum, source, scales, mu, inv, partial,
                geometry.shape[0], c, s, g, True, elements, parts, *common, True, True, dy_meta, x_meta, 1024,
                **correction, enable_fp_fusion=False)
        sums = _reduce(partial, c, parts, 2, _merge_sums, info)
    elif need_dx and geometry.batch_stats:
        parts = triton.cdiv(elements, 1024)
        partial = torch.empty((2, domains, parts), dtype=torch.float32, device=gradient.device)
        info["backward_scratch_bytes"] += partial.numel() * partial.element_size()
        info["backward_reduction_launches"] += 1
        _partial_backward[(domains, parts)](dy, dy_scales, data, step, minimum, source, scales, mu, inv, w, partial,
            c, s, g, batch, domains, elements, parts, *common, weight is not None, dy_meta, x_meta, 1024,
            **correction, **derivative, enable_fp_fusion=False)
        sums = _reduce(partial, domains, parts, 2, _merge_sums, info)
    dx = dw = db = None
    fused = (gradient_output is not None and not gradient_output.checked and (gradient_output.scale is not None or
             gradient_output.block_shape is not None and gradient_output.block_shape[0] * gradient_output.block_shape[1] <= 16384))
    if need_dx and fused:
        info["input_gradient_launches"] += 1
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
            br, bc, current, out.observations is not None, torch.finfo(out.dtype).max, block,
            SHARED_BN=shared_batch, **correction, enable_fp_fusion=False)
        dx = out.wrap(values)
    elif need_dx and coalesced:
        info["input_gradient_launches"] += 1
        dx = torch.empty_strided(geometry.shape, output_strides(gradient, "channels"),
                                 device=gradient.device, dtype=input_dtype or gradient.dtype)
        _channels_last_dx[(triton.cdiv(c, 4), triton.cdiv(elements, 512))](
            dy, dy_scales, data, step, minimum, source, scales, mu, inv, w, sums, dx,
            c, s, elements, geometry.batch_stats, *common, weight is not None, dy_meta, x_meta,
            dx.stride(), geometry.shape[-1], 512, 4, **correction, num_warps=4, enable_fp_fusion=False)
        if gradient_output is not None:
            dx = gradient_output.convert(dx)
    elif need_dx:
        info["input_gradient_launches"] += 1
        dx = torch.empty(geometry.shape, dtype=input_dtype or gradient.dtype, device=gradient.device)
        _input_gradient[(domains, triton.cdiv(elements, 512))](
            dy, dy_scales, data, step, minimum, source, scales, mu, inv, w, sums, dx,
            c, s, g, batch, domains, elements, geometry.batch_stats, *common,
            weight is not None, dy_meta, x_meta, 512, SHARED_BN=shared_batch, **correction, **derivative, enable_fp_fusion=False)
        if gradient_output is not None:
            dx = gradient_output.convert(dx)
    if need_dw or need_db:
        if shared_batch:
            reduced = sums
        else:
            parts = triton.cdiv(geometry.shape[0] * s, 1024)
            partial = torch.empty((2, c, parts), dtype=torch.float32, device=gradient.device)
            info["backward_scratch_bytes"] += partial.numel() * partial.element_size()
            info["affine_gradient_launches"] += 1
            if coalesced:
                _channels_last_sums[(triton.cdiv(c, 4), parts)](dy, dy_scales, data, step, minimum, source, scales,
                    mu, inv, partial, c, s, elements, parts, *common, dy_meta, x_meta, need_dw, 1024, 4,
                    **correction, num_warps=4, enable_fp_fusion=False)
            else:
                _partial_affine[(c, parts)](dy, dy_scales, data, step, minimum, source, scales, mu, inv, partial,
                    geometry.shape[0], c, s, g, batch, elements, parts, *common, need_dw, need_db, dy_meta, x_meta, 1024,
                    **correction, **derivative, enable_fp_fusion=False)
            reduced = _reduce(partial, c, parts, 2, _merge_sums, info)
        dw = torch.empty(c, device=gradient.device, dtype=torch.float32) if need_dw else None
        db = torch.empty(c, device=gradient.device, dtype=torch.float32) if need_db else None
        _finish_affine[(triton.cdiv(c, 128),)](reduced, dw if need_dw else dummy, db if need_db else dummy,
                                             c, need_dw, need_db, 128)
        info["affine_gradient_launches"] += 1
    if execution_info is not None:
        execution_info.update(info)
    if gradient_output is not None:
        gradient_output.info.update(info)
    return dx, dw, db


def forward(x, *args, **kwargs):
    with torch.cuda.device(x.device):
        return _forward(x, *args, **kwargs)


def backward(gradient, *args, **kwargs):
    with torch.cuda.device(gradient.device):
        return _backward(gradient, *args, **kwargs)
