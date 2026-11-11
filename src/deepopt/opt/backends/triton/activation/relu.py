"""Launch policy for fused ReLU, precision conversion and saved masks."""

import torch
import triton

from .forward import _forward, _forward_blocks, _partial_relu_amax
from .backward import _backward
from ..quantization.fp8 import _reduce_amax, _block_scales
from deepopt.opt.nn.activation.geometry import tensor_arguments


def forward(x, geometry, *, dtype, block_shape, packed, save_mask, inplace, reuse,
            scale=None, observe=False, check_finite=False, capture_scale=False):
    with torch.cuda.device(x.device):
        return _run_forward(x, geometry, dtype=dtype, block_shape=block_shape, packed=packed,
                            save_mask=save_mask, inplace=inplace, reuse=reuse, scale=scale,
                            observe=observe, check_finite=check_finite, capture_scale=capture_scale)


def _run_forward(x, geometry, *, dtype, block_shape, packed, save_mask, inplace, reuse,
                 scale, observe, check_finite, capture_scale):
    data, inverse, meta = tensor_arguments(x)
    rows, cols = geometry.matrix_shape
    count = rows * cols
    fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    maximum = torch.finfo(dtype).max if fp8 else 0.0
    br, bc = block_shape or (max(1, rows), max(1, cols))
    sr, sc = (1, 1) if block_shape is None else (triton.cdiv(rows, br), triton.cdiv(cols, bc))
    output = data if inplace else torch.empty((rows, cols), dtype=dtype, device=x.device)
    out_inverse = (inverse if reuse else torch.empty((sr, sc), dtype=torch.float32, device=x.device)) if fp8 else None
    saved = (torch.empty((triton.cdiv(count, 32),) if packed else (count,),
                         dtype=torch.int32 if packed else torch.bool, device=x.device) if save_mask else None)
    info = {"backend": "triton", "forward_launches": int(count > 0), "scale_launches": 0,
            "scale_mode": "reuse" if reuse else "provided" if scale is not None else "current" if fp8 else "none",
            "mask": ("packed" if packed else "bool") if save_mask else "none"}
    if not count:
        return output, out_inverse, saved, None, None, info
    current = fp8 and not reuse and scale is None
    small = current and block_shape is None and count <= 4096
    width = max(32, bc) if cols % 32 == 0 and br * max(32, bc) <= 16384 else bc
    fused_blocks = current and block_shape is not None and br * width <= 16384
    block = max(32, triton.next_power_of_2(count)) if small else 1024
    programs = triton.cdiv(count, block)
    snapshot = torch.empty(2, dtype=torch.float32, device=x.device) if capture_scale else None
    observations = torch.empty(programs, dtype=torch.float32, device=x.device) if observe else None
    if fused_blocks:
        programs = triton.cdiv(rows, br) * triton.cdiv(cols, width)
        errors = torch.empty(programs, dtype=torch.int32, device=x.device) if check_finite else output
        _forward_blocks[(programs,)](
            data, inverse, output, saved if save_mask else output, out_inverse, errors,
            rows, cols, geometry.shape, geometry.layout == "channels", meta,
            br, bc, width, width // bc, sc, maximum, save_mask, packed, check_finite,
            max(32, triton.next_power_of_2(triton.cdiv(count, programs))),
            num_warps=8 if br * width >= 8192 else 4)
    else:
        if current and not small:
            parts = triton.cdiv(br * bc, 4096)
            partials = torch.empty((sr * sc, parts), dtype=torch.float32, device=x.device)
            _partial_relu_amax[(sr * sc * parts,)](
                data, inverse, partials, rows, cols, geometry.shape, geometry.layout == "channels", meta,
                br, bc, sc, parts, 4096)
            info["scale_launches"] += 1
            while parts > 1024:
                next_parts = triton.cdiv(parts, 1024)
                reduced = torch.empty((sr * sc, next_parts), dtype=torch.float32, device=x.device)
                _reduce_amax[(sr * sc * next_parts,)](partials, reduced, parts, next_parts, BLOCK=1024)
                partials, parts = reduced, next_parts
                info["scale_launches"] += 1
            scale = torch.empty((sr, sc), dtype=torch.float32, device=x.device)
            _block_scales[(sr * sc,)](partials, scale, out_inverse, parts,
                                     BLOCK=triton.next_power_of_2(parts), FP8_MAX=maximum)
            info["scale_launches"] += 1
        errors = torch.empty(programs, dtype=torch.int32, device=x.device) if check_finite else output
        _forward[(programs,)](
            data, inverse, output, saved if save_mask else output, scale if scale is not None else output,
            out_inverse if fp8 else output, errors, observations if observe else output,
            snapshot if capture_scale else output, rows, cols, geometry.shape, geometry.layout == "channels", meta,
            fp8, reuse, small, br, bc, sc, maximum, save_mask, packed, inplace,
            check_finite, observe, capture_scale, block)
    if check_finite and errors.any().item():
        raise ValueError("ReLU FP8 casting requires finite outputs and positive finite scales")
    if inplace and meta[0] > 0:
        torch.autograd.graph.increment_version(data)
    return output, out_inverse, saved, snapshot, observations, info


def backward(gradient, mask, geometry, *, packed, dtype, reuse=False, gradient_output=None):
    if gradient_output is not None:
        out = gradient_output
        if not gradient.numel():
            return out.convert(geometry.restore(torch.empty(geometry.matrix_shape, device=gradient.device, dtype=torch.float32)))
        if not out.checked and (out.scale is not None or out.block_shape is not None and out.block_shape[0] * out.block_shape[1] <= 16384
                               or gradient.numel() <= 4096):
            from .gradient import _relu_gradient
            from deepopt.opt.nn.activation.geometry import ReLUGeometry
            from math import prod

            with torch.cuda.device(gradient.device):
                target = ReLUGeometry(geometry.shape, out.layout)
                rows, cols = target.matrix_shape
                br, bc = out.block_shape or (rows, cols)
                current = out.scale is None
                block = max(32, triton.next_power_of_2(br * bc)) if current else 512
                programs = triton.cdiv(rows, br) * triton.cdiv(cols, bc) if current else triton.cdiv(rows * cols, block)
                out.allocate(rows, cols, gradient.device, programs)
                data, inverse, meta = tensor_arguments(gradient)
                values = torch.empty((rows, cols), dtype=out.dtype, device=gradient.device)
                c, s = (geometry.shape[1], prod(geometry.shape[2:])) if len(geometry.shape) >= 2 else (1, 1)
                _relu_gradient[(programs,)](
                    data, inverse, mask, values, out.scale if out.scale is not None else inverse, out.inverse,
                    out.observations if out.observations is not None else inverse, rows, cols, geometry.shape, meta,
                    out.layout == "channels", geometry.layout == "channels", c, s, packed, br, bc, current,
                    out.observations is not None, torch.finfo(out.dtype).max, block)
                return out.wrap(values)
        values = backward(gradient, mask, geometry, packed=packed, dtype=torch.float32)
        return out.convert(geometry.restore(values))
    with torch.cuda.device(gradient.device):
        data, inverse, meta = tensor_arguments(gradient)
        rows, cols = geometry.matrix_shape
        output = torch.empty((rows, cols), dtype=dtype, device=gradient.device)
        if rows * cols:
            _backward[(triton.cdiv(rows * cols, 1024),)](
                data, inverse, mask, output, rows, cols, geometry.shape, geometry.layout == "channels", meta,
                packed, reuse, dtype in (torch.float8_e4m3fn, torch.float8_e5m2), 1024)
    return output
