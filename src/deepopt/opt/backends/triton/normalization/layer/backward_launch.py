"""Bounded LayerNorm backward and planned scale-aware dx output orchestration."""

from math import prod

import torch
import triton

from .backward import _row_sums, _merge_sums, _row_backward, _input_gradient, _affine_partial, _affine_merge
from deepopt.opt.nn.activation.geometry import ReLUGeometry, tensor_arguments


def backward(gradient, weight, geometry, **options):
    with torch.cuda.device(gradient.device):
        return _backward(gradient, weight, geometry, **options)


def _backward(gradient, weight, geometry, *, packed, x, mean, inverse, need_dx, need_dw, need_db,
              input_dtype, gradient_output=None, execution_info=None, center_shift=None):
    m, k = geometry.matrix_shape
    info = {"input_gradient_launches": 0, "backward_reduction_launches": 0,
            "affine_gradient_launches": 0, "backward_workspace_dtype": None}
    if not m * k:
        dx = torch.empty(geometry.shape, device=gradient.device, dtype=input_dtype) if need_dx else None
        if dx is not None and gradient_output is not None:
            dx = gradient_output.convert(dx)
        dw = torch.zeros(geometry.normalized_shape, device=gradient.device, dtype=torch.float32) if need_dw else None
        db = torch.zeros(geometry.normalized_shape, device=gradient.device, dtype=torch.float32) if need_db else None
        if execution_info is not None:
            execution_info.update(info)
        return dx, dw, db
    dy, dy_inverse, dy_meta = tensor_arguments(gradient)
    source, source_inverse, x_meta = tensor_arguments(x) if x is not None else (dy, dy_inverse, dy_meta)
    compressed = packed is not None
    group, words, bits = (packed.group_size, packed.data.shape[1], packed.bits) if compressed else (1, 1, 2)
    data, step, minimum = (packed.data, packed.step, packed.minimum) if compressed else (dy, dy, dy)
    mu, rstd = mean if mean is not None else dy, inverse if inverse is not None else dy
    w, ws = (weight, weight.stride()) if weight is not None else (dy, ())
    base = (dy, dy_inverse, data, step, minimum, source, source_inverse, mu, rstd, w)
    meta = (m, k, geometry.shape, geometry.normalized_shape, dy_meta, x_meta, ws,
            weight is not None, compressed, group, words, bits)
    correction = dict(center_shift=center_shift, RECENTER=center_shift is not None)
    dx = dw = db = None
    if need_dx:
        info["input_gradient_launches"] += 1
        out = gradient_output
        target = ReLUGeometry(geometry.shape, out.layout if out is not None else "last")
        rows, cols = target.matrix_shape
        block = triton.next_power_of_2(k)
        br, bc = out.block_shape or (rows, cols) if out is not None else (1, 1)
        row_fused = out is not None and not out.checked and (out.scale is not None or
                    out.block_shape is None and m == 1 or out.layout == "last" and k == cols and
                    br == 1 and bc <= block and block % bc == 0)
        if k <= 8192 and (out is None or row_fused):
            if row_fused:
                out.allocate(rows, cols, gradient.device, m)
            values = torch.empty(target.matrix_shape, device=gradient.device, dtype=out.dtype if row_fused else input_dtype)
            c, s = (geometry.shape[1], prod(geometry.shape[2:])) if len(geometry.shape) >= 2 else (1, 1)
            _row_backward[(m,)](*base, values, out.scale if row_fused and out.scale is not None else dy,
                                out.inverse if row_fused else values,
                                out.observations if row_fused and out.observations is not None else values,
                                *meta, bool(row_fused), bool(row_fused and out.scale is None),
                                bool(out is not None and out.block_shape is None), cols, target.layout == "channels", c, s,
                                bc, triton.cdiv(cols, bc), bool(row_fused and out.observations is not None),
                                torch.finfo(out.dtype).max if row_fused else 0.0, block,
                                num_warps=8 if k >= 4096 else 4, **correction, enable_fp_fusion=False)
            dx = out.wrap(values) if row_fused else geometry.restore(values)
        else:
            parts = triton.cdiv(k, 4096)
            sums = torch.empty((2, m, parts), device=gradient.device, dtype=torch.float32)
            _row_sums[(m, parts)](*base, sums, *meta, parts, 4096, **correction, enable_fp_fusion=False)
            info["backward_reduction_launches"] += 1
            while parts > 1:
                nxt = triton.cdiv(parts, 128)
                reduced = torch.empty((2, m, nxt), device=gradient.device, dtype=torch.float32)
                _merge_sums[(m, nxt)](sums, reduced, m, parts, nxt, 128)
                sums, parts = reduced, nxt
                info["backward_reduction_launches"] += 1
            fused = out is not None and not out.checked and (out.scale is not None or m * k <= 4096
                    or out.block_shape is not None and out.block_shape[0] * out.block_shape[1] <= 16384)
            target = ReLUGeometry(geometry.shape, out.layout if out is not None else "last")
            rows, cols = target.matrix_shape
            br, bc = (out.block_shape or (rows, cols)) if fused else (1, 1)
            current = fused and out.scale is None
            block = max(32, triton.next_power_of_2(br * bc)) if current else 512
            programs = triton.cdiv(rows, br) * triton.cdiv(cols, bc) if current else triton.cdiv(rows * cols, block)
            if fused:
                out.allocate(rows, cols, gradient.device, programs)
            dtype = out.dtype if fused else torch.float32 if out is not None else input_dtype
            values = torch.empty((rows, cols), device=gradient.device, dtype=dtype)
            c, s = (geometry.shape[1], prod(geometry.shape[2:])) if len(geometry.shape) >= 2 else (1, 1)
            _input_gradient[(programs,)](
                *base, sums, values, out.scale if fused and out.scale is not None else dy,
                out.inverse if fused else values, out.observations if fused and out.observations is not None else values,
                *meta, rows, cols, target.layout == "channels", c, s, bool(fused), bool(current), br, bc,
                bool(fused and out.observations is not None), torch.finfo(out.dtype).max if fused else 0.0,
                block, num_warps=8 if block >= 8192 else 4, **correction, enable_fp_fusion=False)
            dx = out.wrap(values) if fused else out.convert(values, matrix=True) if out is not None else target.restore(values)
            if out is not None and not fused:
                info["backward_workspace_dtype"] = "torch.float32"
    if need_dw or need_db:
        parts = triton.cdiv(m, 128)
        partial = torch.empty((2, parts, k), device=gradient.device, dtype=torch.float32)
        _affine_partial[(triton.cdiv(k, 32), parts)](
            dy, dy_inverse, data, step, minimum, source, source_inverse, mu, rstd, partial,
            m, k, geometry.shape, dy_meta, x_meta, compressed, group, words, bits, need_dw, need_db, parts, 128, 32,
            num_warps=8, **correction, enable_fp_fusion=False)
        info["affine_gradient_launches"] += 1
        while parts > 1:
            nxt = triton.cdiv(parts, 128)
            reduced = torch.empty((2, nxt, k), device=gradient.device, dtype=torch.float32)
            _affine_merge[(triton.cdiv(k, 32), nxt)](partial, reduced, k, parts, nxt, need_dw, need_db, 128, 32,
                                                   num_warps=8, enable_fp_fusion=False)
            partial, parts = reduced, nxt
            info["affine_gradient_launches"] += 1
        dw = partial[0].reshape(geometry.normalized_shape) if need_dw else None
        db = partial[1].reshape(geometry.normalized_shape) if need_db else None
    if execution_info is not None:
        execution_info.update(info)
    if gradient_output is not None:
        gradient_output.info.update(info)
    return dx, dw, db
