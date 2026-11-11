"""Shared bounded launch metadata and forward/gradient output conversion."""

from math import prod

import torch
import triton

from .output import _cast
from .....quantization.quantizer import quantize_fp8


def kernel_geometry(g):
    family = {"max": 0, "average": 1, "adaptive_max": 2, "adaptive_average": 3,
              "fractional_max": 4, "power": 5, "unpool": 6}[g.family]
    return (family, g.input_spatial, g.output_spatial, g.kernel, g.stride, g.padding,
            g.dilation, g.channels, g.rank, g.power, g.include_pad, g.divisor)


def candidate_count(g):
    if g.family.startswith("adaptive"):
        sizes = [min(o, triton.cdiv(o, i) + 1) for i, o in zip(g.input_spatial, g.output_spatial)]
    elif g.family == "fractional_max":
        sizes = [min(o, k + 4) for o, k in zip(g.output_spatial, g.kernel)]
    else:
        sizes = [min(o, triton.cdiv((k - 1) * d, s) + 1) for o, k, d, s in zip(g.output_spatial, g.kernel, g.dilation, g.stride)]
    return max(1, prod(sizes))


def can_inline(rows, cols, blocks, scale, checked, program_block=32):
    if checked:
        return False
    if scale is not None:
        return True
    if rows * cols <= program_block and (blocks is None or blocks[0] >= rows and blocks[1] >= cols):
        return True
    return (blocks is not None and blocks[0] * blocks[1] <= program_block
            and program_block % (blocks[0] * blocks[1]) == 0
            and (blocks[0] == 1 and cols % blocks[1] == 0 or blocks[1] == cols))


def output_storage(shape, dtype, blocks, scale, observe, capture_scale, checked, device, *, force_workspace=False, program_block=32):
    rows, cols = shape
    fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    inline = not fp8 or can_inline(rows, cols, blocks, scale, checked, program_block) and not force_workspace
    output = torch.empty(shape, device=device, dtype=dtype if inline else torch.float32)
    inverse = observations = snapshot = None
    br, bc = blocks or (max(1, rows), max(1, cols))
    if inline and fp8:
        inverse = torch.empty((triton.cdiv(rows, br), triton.cdiv(cols, bc)), device=device, dtype=torch.float32)
        observations = torch.empty(triton.cdiv(rows * cols, program_block), device=device, dtype=torch.float32) if observe else None
        snapshot = torch.empty(2, device=device, dtype=torch.float32) if capture_scale else None
    return output, inverse, observations, snapshot, inline, br, bc


def finish_output(values, dtype, blocks, scale, observe, capture_scale, checked, info):
    if scale is None:
        q = quantize_fp8(values, fp8_dtype=dtype, block_shape=blocks, check_finite=checked)
        info["output_cast_launches"] += 1
        rows, cols = values.shape
        br, bc = blocks or (rows, cols)
        elements = min(rows, br) * min(cols, bc)
        if elements > (4096 if blocks is None else 16384):
            groups = triton.cdiv(rows, br) * triton.cdiv(cols, bc)
            parts = triton.cdiv(elements, 4096)
            info["scale_launches"] += 2
            info["scratch_bytes"] += groups * parts * 4 + groups * 4
            while parts > 1024:
                parts = triton.cdiv(parts, 1024)
                info["scale_launches"] += 1
                info["scratch_bytes"] += groups * parts * 4
        info["workspace_dtype"] = "torch.float32"
        return q.data, q.scale_inv, None, None
    rows, cols = values.shape
    br, bc = blocks or (rows, cols)
    output = torch.empty_like(values, dtype=dtype)
    inverse = torch.empty_like(scale)
    programs = triton.cdiv(values.numel(), 256)
    observations = torch.empty(programs, device=values.device, dtype=torch.float32) if observe else None
    snapshot = torch.empty(2, device=values.device, dtype=torch.float32) if capture_scale else None
    errors = torch.empty(programs, device=values.device, dtype=torch.int32) if checked else values
    _cast[(programs,)](values, output, scale, inverse, observations if observe else values,
                       snapshot if capture_scale else values, errors, rows, cols, br, bc,
                       observe, capture_scale, checked, torch.finfo(dtype).max, 256, enable_fp_fusion=False)
    if checked and errors.any().item():
        raise ValueError("Pooling FP8 output requires finite values and positive finite scales")
    info["output_cast_launches"] += 1
    info["workspace_dtype"] = "torch.float32"
    return output, inverse, snapshot, observations
