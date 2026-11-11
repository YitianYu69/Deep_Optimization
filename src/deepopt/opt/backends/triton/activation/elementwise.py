"""Bounded launch orchestration for fused nonlinear activation kernels."""

from math import prod

import torch
import triton

from .elementwise_forward import _activation_forward, _statistics, _reduce_ranges
from .elementwise_backward import _activation_backward
from ..quantization.fp8 import _reduce_amax, _block_scales
from ..quantization.activation.affine import _finish_min_max
from deepopt.opt.nn.activation.geometry import ReLUGeometry, tensor_arguments
from deepopt.opt.backends.reference.activation.elementwise import allocate_saved


def forward(x, geometry, **options):
    with torch.cuda.device(x.device):
        return _forward(x, geometry, **options)


def _forward(x, geometry, *, kind, approximate, dtype, block_shape, packed, bits, group_size,
             save_mask, inplace, reuse=False, scale=None, observe=False, check_finite=False, capture_scale=False):
    data, inverse, meta = tensor_arguments(x)
    rows, cols = geometry.matrix_shape
    count = rows * cols
    fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    maximum = torch.finfo(dtype).max if fp8 else 0.0
    obr, obc = block_shape or (max(1, rows), max(1, cols))
    sr, sc = (1, 1) if block_shape is None else (triton.cdiv(rows, obr), triton.cdiv(cols, obc))
    output = data if inplace else torch.empty((rows, cols), dtype=dtype, device=x.device)
    out_inverse = (inverse if reuse else torch.empty((sr, sc), dtype=torch.float32, device=x.device)) if fp8 else None
    saved = allocate_saved(geometry, x.device, packed=packed, bits=bits, group_size=group_size,
                           block_shape=block_shape, mask=kind == "relu6") if save_mask else None
    info = {"backend": "triton", "activation": kind, "forward_launches": int(count > 0),
            "scale_launches": 0, "coefficient_launches": 0,
            "backward_launches": int(save_mask and count > 0),
            "scale_mode": "reuse" if reuse else "provided" if scale is not None else "current" if fp8 else "none",
            "saved_state": ("mask" if kind == "relu6" else "derivative") if save_mask else "none"}
    if not count:
        return output, out_inverse, saved, None, None, info

    # ReLU6 exact mask words own flat aligned ranges. Affine saved coefficients
    # instead own the configured flat groups or activation scaling rectangles.
    codec = save_mask and packed and kind != "relu6"
    tiled = block_shape is not None and (kind != "relu6" or not (save_mask and packed))
    br, bc = block_shape if tiled else (0, 0)
    group = br * bc if br else group_size if codec else 512
    groups = triton.cdiv(rows, br) * triton.cdiv(cols, bc) if br else triton.cdiv(count, group)
    block = min(4096 if group > 16384 else 16384, max(32, triton.next_power_of_2(group)))
    parts = triton.cdiv(group, block)
    precomputed = codec and parts > 1
    if precomputed:
        lo = torch.empty((groups, parts), device=x.device, dtype=torch.float32)
        hi = torch.empty_like(lo)
        _statistics[(groups, parts)](data, inverse, lo, hi, rows, cols, geometry.shape,
                                    geometry.layout == "channels", meta, kind, approximate,
                                    group, br, bc, parts, True, block, enable_fp_fusion=False)
        info["coefficient_launches"] += 1
        n = parts
        while n > 1024:
            nxt = triton.cdiv(n, 1024)
            nlo = torch.empty((groups, nxt), dtype=torch.float32, device=x.device)
            nhi = torch.empty_like(nlo)
            _reduce_ranges[(groups, nxt)](lo, hi, nlo, nhi, n, nxt, 1024)
            lo, hi, n = nlo, nhi, nxt
            info["coefficient_launches"] += 1
        _finish_min_max[(groups,)](lo, hi, saved.step, saved.minimum, n, bits,
                                  triton.next_power_of_2(n), enable_fp_fusion=False)
        info["coefficient_launches"] += 1

    # Fuse a current scale only when this program contains its entire domain.
    current = (fp8 and not reuse and scale is None and parts == 1 and
               (block_shape is None and groups == 1 or block_shape is not None and (br, bc) == block_shape))
    if fp8 and not reuse and scale is None and not current:
        n = triton.cdiv(obr * obc, 4096)
        partial = torch.empty((sr * sc, n), device=x.device, dtype=torch.float32)
        _statistics[(sr * sc, n)](data, inverse, partial, partial, rows, cols, geometry.shape,
                                  geometry.layout == "channels", meta, kind, approximate,
                                  obr * obc, obr, obc, n, False, 4096, enable_fp_fusion=False)
        info["scale_launches"] += 1
        while n > 1024:
            nxt = triton.cdiv(n, 1024)
            reduced = torch.empty((sr * sc, nxt), device=x.device, dtype=torch.float32)
            _reduce_amax[(sr * sc * nxt,)](partial, reduced, n, nxt, BLOCK=1024)
            partial, n = reduced, nxt
            info["scale_launches"] += 1
        scale = torch.empty((sr, sc), device=x.device, dtype=torch.float32)
        _block_scales[(sr * sc,)](partial, scale, out_inverse, n,
                                 BLOCK=triton.next_power_of_2(n), FP8_MAX=maximum)
        info["scale_launches"] += 1
    observations = torch.empty(groups * parts, device=x.device, dtype=torch.float32) if observe else None
    snapshot = torch.empty(2, device=x.device, dtype=torch.float32) if capture_scale else None
    errors = torch.empty(groups * parts, device=x.device, dtype=torch.int32) if check_finite else output
    words = saved.data.shape[1] if codec else 0
    _activation_forward[(groups, parts)](
        data, inverse, output, saved.data if saved else output,
        saved.step if codec else output, saved.minimum if codec else output,
        scale if scale is not None else output, out_inverse if fp8 else output,
        observations if observe else output, snapshot if capture_scale else output, errors,
        rows, cols, geometry.shape, geometry.layout == "channels", meta, kind, approximate,
        group, br, bc, words, save_mask, packed, bits, precomputed, fp8, reuse, current, obr, obc, sc,
        maximum, inplace, observe, capture_scale, check_finite, parts, block,
        num_warps=8 if block >= 8192 else 4, enable_fp_fusion=False)
    if check_finite and errors.any().item():
        raise ValueError("Activation casting/compression requires finite values and positive finite scales")
    if inplace and meta[0] > 0:
        torch.autograd.graph.increment_version(data)
    return output, out_inverse, saved, snapshot, observations, info


def backward(gradient, saved, geometry, *, dtype, gradient_output=None):
    with torch.cuda.device(gradient.device):
        out = gradient_output
        if not gradient.numel():
            empty = torch.empty(geometry.matrix_shape, device=gradient.device, dtype=dtype)
            return out.convert(geometry.restore(empty)) if out is not None else empty
        fused = out is not None and not out.checked and (
            out.scale is not None or gradient.numel() <= 4096
            or out.block_shape is not None and out.block_shape[0] * out.block_shape[1] <= 16384)
        if out is not None and not fused:
            dense = backward(gradient, saved, geometry, dtype=torch.float32)
            return out.convert(geometry.restore(dense))
        target = ReLUGeometry(geometry.shape, out.layout) if fused else geometry
        rows, cols = target.matrix_shape
        obr, obc = (out.block_shape or (rows, cols)) if fused else (1, 1)
        current = fused and out.scale is None
        block = max(32, triton.next_power_of_2(obr * obc)) if current else 512
        programs = triton.cdiv(rows, obr) * triton.cdiv(cols, obc) if current else triton.cdiv(rows * cols, block)
        if fused:
            out.allocate(rows, cols, gradient.device, programs)
        output = torch.empty((rows, cols), device=gradient.device, dtype=out.dtype if fused else dtype)
        data, inverse, meta = tensor_arguments(gradient)
        c, s = (geometry.shape[1], prod(geometry.shape[2:])) if len(geometry.shape) >= 2 else (1, 1)
        br, bc = saved.block_shape or (0, 0)
        words = saved.data.shape[1] if saved.packed and not saved.mask else 0
        _activation_backward[(programs,)](
            data, inverse, saved.data, saved.step if saved.step is not None else output,
            saved.minimum if saved.minimum is not None else output, output,
            out.scale if fused and out.scale is not None else inverse,
            out.inverse if fused else output, out.observations if fused and out.observations is not None else output,
            rows, cols, geometry.shape, meta, target.layout == "channels", geometry.layout == "channels",
            c, s, geometry.matrix_shape[1], saved.group_size, br, bc, words, saved.bits, saved.mask, saved.packed,
            bool(fused), bool(current), obr, obc, bool(fused and out.observations is not None),
            torch.finfo(out.dtype).max if fused else 0.0, block,
            num_warps=8 if block >= 8192 else 4, enable_fp_fusion=False)
        return out.wrap(output) if fused else output
