"""Pool input-owned gradient launch and planned FP8 epilogue ownership."""

from math import prod

import torch
import triton

from .window.backward import _backward as _input_gradient
from .common.launch import kernel_geometry, candidate_count, can_inline
from ....nn.activation.geometry import ReLUGeometry, tensor_arguments
from ....quantization.indices import PackedIndices
from ....quantization.activation import PackedActivation


def backward(gradient, g, **options):
    with torch.cuda.device(gradient.device):
        return _backward(gradient, g, **options)


def _backward(gradient, g, *, routing, coefficient, factor, samples, coefficient_layout,
              input_dtype, gradient_output=None, execution_info=None):
    out = gradient_output
    target = ReLUGeometry(g.shape, out.layout if out is not None else "last")
    rows, cols = target.matrix_shape
    if not prod(g.shape):
        values = torch.empty(target.matrix_shape, device=gradient.device, dtype=input_dtype)
        return out.convert(values, matrix=True) if out is not None else target.restore(values)
    dy, dy_inverse, dy_meta = tensor_arguments(gradient)
    route = routing.data if isinstance(routing, PackedIndices) else routing if routing is not None else dy
    route_bits = routing.bits if isinstance(routing, PackedIndices) else -1
    a = coefficient.data if isinstance(coefficient, PackedActivation) else coefficient if coefficient is not None else dy
    step = coefficient.step if isinstance(coefficient, PackedActivation) else dy
    low = coefficient.minimum if isinstance(coefficient, PackedActivation) else dy
    compressed = isinstance(coefficient, PackedActivation)
    if compressed:
        br_a, bc_a = coefficient.block_shape or (0, 0)
        group, words, bits = br_a * bc_a if br_a else coefficient.group_size, coefficient.data.shape[1], coefficient.bits
    else:
        br_a = bc_a = 0
        group, words, bits = 1, 1, 2
    fused = out is not None and can_inline(rows, cols, out.block_shape, out.scale, out.checked)
    if fused:
        out.allocate(rows, cols, gradient.device, triton.cdiv(rows * cols, 32))
    dtype = out.dtype if fused else torch.float32 if out is not None else input_dtype
    values = torch.empty(target.matrix_shape, device=gradient.device, dtype=dtype)
    br, bc = (out.block_shape or (rows, cols)) if out is not None else (1, 1)
    candidates = candidate_count(g)
    _input_gradient[(triton.cdiv(values.numel(), 32),)](
        dy, dy_inverse, route, a, step, low, factor if factor is not None else dy,
        samples.contiguous() if samples is not None else dy, values,
        out.scale if fused and out.scale is not None else dy, out.inverse if fused else dy,
        out.observations if fused and out.observations is not None else dy, dy, dy,
        g.shape, g.output_shape, dy_meta, kernel_geometry(g), target.layout == "channels", g.layout == "channels",
        values.numel(), candidates, route_bits, coefficient_layout == "channels",
        ReLUGeometry(g.shape, coefficient_layout).matrix_shape[1], compressed, br_a, bc_a, group, words, bits,
        rows, cols, br, bc, fused, fused and out.scale is None,
        fused and out.observations is not None, False, torch.finfo(out.dtype).max if fused else 0.0,
        min(128, triton.next_power_of_2(candidates)), num_warps=4, enable_fp_fusion=False)
    info = dict(backward_launches=1, backward_workspace_dtype="torch.float32" if out is not None and not fused else None,
                deterministic_strategy="input_owned_gather")
    if execution_info is not None:
        execution_info.update(info)
    if out is not None:
        out.info.update(info)
        return out.wrap(values) if fused else out.convert(values, matrix=True)
    return target.restore(values)
