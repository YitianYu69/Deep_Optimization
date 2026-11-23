"""Invocation-local SwiGLU arithmetic and three distinct flat affine saves."""

from math import prod

import torch
import triton
import triton.language as tl

from .math import _evaluate
from .elementwise_backward import _coefficient
from ..quantization.activation.affine import _affine_quantize_pack


@triton.jit
def _save(values, valid, data, steps, minima, group, BLOCK: tl.constexpr):
    words, step, minimum = _affine_quantize_pack(values, valid, 2, BLOCK)
    word = tl.arange(0, BLOCK // 16)
    tl.store(data + group * (BLOCK // 16) + word, words)
    tl.store(steps + group, step)
    tl.store(minima + group, minimum)


@triton.jit
def _swiglu_forward(x, up, output, derivative, dstep, dmin, gate, gstep, gmin,
                    saved_up, ustep, umin, errors, N: tl.constexpr, PRODUCT: tl.constexpr,
                    CHECK: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    index = group * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    original = tl.load(x + index, valid, other=0).to(tl.float32)
    activated, coefficient = _evaluate(original, "silu", "none")
    rounded_gate = activated.to(tl.bfloat16).to(tl.float32)
    _save(coefficient, valid, derivative, dstep, dmin, group, BLOCK)
    _save(rounded_gate, valid, gate, gstep, gmin, group, BLOCK)
    if PRODUCT:
        raw_up = tl.load(up + index, valid, other=0).to(tl.float32)
        _save(raw_up, valid, saved_up, ustep, umin, group, BLOCK)
        result = (rounded_gate * raw_up).to(tl.bfloat16)
    else:
        result = rounded_gate.to(tl.bfloat16)
    tl.store(output + index, result, valid)
    if CHECK:
        bad = valid & (~(tl.abs(coefficient) < float("inf")) | ~(tl.abs(rounded_gate) < float("inf")))
        if PRODUCT:
            bad = bad | (valid & ~(tl.abs(raw_up) < float("inf")))
        tl.atomic_max(errors, tl.max(bad.to(tl.int32), 0))


@triton.jit
def _swiglu_backward(dy, derivative, dstep, dmin, gate, gstep, gmin, up, ustep, umin,
                     dx, dup, N: tl.constexpr, NEED_X: tl.constexpr, NEED_UP: tl.constexpr,
                     SHAPE: tl.constexpr, STRIDES: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    address = tl.zeros((BLOCK,), tl.int64)
    remaining = index
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        address += (remaining % SHAPE[axis]) * STRIDES[axis]
        remaining = remaining // SHAPE[axis]
    gradient = tl.load(dy + address, valid, other=0).to(tl.float32)
    if NEED_X:
        coefficient = _coefficient(derivative, dstep, dmin, index, valid, 1, 1024, 0, 0, 64, 2, False, True)
        rounded_up = _coefficient(up, ustep, umin, index, valid, 1, 1024, 0, 0, 64, 2, False, True)
        rounded_up = rounded_up.to(tl.bfloat16).to(tl.float32)
        product_adjoint = (gradient * rounded_up).to(tl.bfloat16).to(tl.float32)
        tl.store(dx + index, (product_adjoint * coefficient).to(tl.bfloat16), valid)
    if NEED_UP:
        rounded_gate = _coefficient(gate, gstep, gmin, index, valid, 1, 1024, 0, 0, 64, 2, False, True)
        rounded_gate = rounded_gate.to(tl.bfloat16).to(tl.float32)
        tl.store(dup + index, (gradient * rounded_gate).to(tl.bfloat16), valid)


@triton.jit
def _swiglu_backward_panel(dy, derivative, dstep, dmin, gate, gstep, gmin, up, ustep, umin,
                           dx, dup, start_element, element_count, N: tl.constexpr,
                           SHAPE: tl.constexpr, STRIDES: tl.constexpr, BLOCK: tl.constexpr):
    # Codec addressing is local to this three-triplet window. Cotangent reads
    # and final dense writes retain the full logical tensor's int64 offsets.
    local = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    index = start_element.to(tl.int64) + local
    valid = (local < element_count) & (index < N)
    address = tl.zeros((BLOCK,), tl.int64)
    remaining = index
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        address += (remaining % SHAPE[axis]) * STRIDES[axis]
        remaining = remaining // SHAPE[axis]
    gradient = tl.load(dy + address, valid, other=0).to(tl.float32)
    coefficient = _coefficient(derivative, dstep, dmin, local, valid, 1, 1024, 0, 0, 64, 2, False, True)
    rounded_up = _coefficient(up, ustep, umin, local, valid, 1, 1024, 0, 0, 64, 2, False, True)
    rounded_up = rounded_up.to(tl.bfloat16).to(tl.float32)
    product_adjoint = (gradient * rounded_up).to(tl.bfloat16).to(tl.float32)
    tl.store(dx + index, (product_adjoint * coefficient).to(tl.bfloat16), valid)
    rounded_gate = _coefficient(gate, gstep, gmin, local, valid, 1, 1024, 0, 0, 64, 2, False, True)
    rounded_gate = rounded_gate.to(tl.bfloat16).to(tl.float32)
    tl.store(dup + index, (gradient * rounded_gate).to(tl.bfloat16), valid)


def _tensor(value, shape, dtype, device, *, strided=False):
    if (type(value) is not torch.Tensor or value.device != device or value.device.type != "cuda"
            or value.dtype != dtype or tuple(value.shape) != tuple(shape)
            or not strided and not value.is_contiguous()
            or strided and any(stride < 0 for stride in value.stride())):
        raise ValueError("SwiGLU fusion operand violates its captured CUDA dense metadata")


def forward(x, up, *, shape, product, check_finite=False):
    """Seven activation/pack outputs or ten complete training outputs."""
    _tensor(x, shape, torch.bfloat16, x.device)
    if product:
        _tensor(up, shape, torch.bfloat16, x.device)
    count, groups = prod(shape), triton.cdiv(prod(shape), 1024)
    if count <= 0:
        raise ValueError("SwiGLU fusion requires positive geometry")
    def allocate():
        return (torch.empty((groups, 64), dtype=torch.int32, device=x.device),
                torch.empty((groups,), dtype=torch.float32, device=x.device),
                torch.empty((groups,), dtype=torch.float32, device=x.device))
    with torch.cuda.device(x.device):
        output = torch.empty(shape, dtype=torch.bfloat16, device=x.device)
        derivative, gate = allocate(), allocate()
        saved_up = allocate() if product else (output, output, output)
        errors = torch.zeros((), dtype=torch.int32, device=x.device) if check_finite else output
        _swiglu_forward[(groups,)](x, up if product else x, output, *derivative, *gate, *saved_up,
            errors, count, product, check_finite, 1024, num_warps=4, enable_fp_fusion=False)
        if check_finite and errors.item():
            raise ValueError("SwiGLU affine saves require finite values")
    return (output, *derivative, *gate, *(saved_up if product else ()))


def backward(gradient, *saved, shape, need_x=True, need_up=True):
    """Decode and evaluate both adjoints in one kernel, with no dense decode."""
    _tensor(gradient, shape, torch.bfloat16, gradient.device, strided=True)
    groups = triton.cdiv(prod(shape), 1024)
    if len(saved) != 9 or not (need_x or need_up):
        raise ValueError("SwiGLU fusion requires its nine saved leaves and an output")
    for index, value in enumerate(saved):
        _tensor(value, (groups, 64) if index % 3 == 0 else (groups,),
                torch.int32 if index % 3 == 0 else torch.float32, gradient.device)
    with torch.cuda.device(gradient.device):
        dx = torch.empty(shape, dtype=torch.bfloat16, device=gradient.device) if need_x else gradient
        dup = torch.empty(shape, dtype=torch.bfloat16, device=gradient.device) if need_up else gradient
        _swiglu_backward[(triton.cdiv(prod(shape), 1024),)](gradient, *saved, dx, dup, prod(shape),
            need_x, need_up, shape, tuple(gradient.stride()), 1024, num_warps=4, enable_fp_fusion=False)
    return tuple(value for value, needed in ((dx, need_x), (dup, need_up)) if needed)


def allocate_panel_outputs(gradient, *, shape):
    """Only the unavoidable two final gradients are full-size allocations."""
    _tensor(gradient, shape, torch.bfloat16, gradient.device, strided=True)
    with torch.cuda.device(gradient.device):
        return tuple(torch.empty(shape, dtype=torch.bfloat16, device=gradient.device) for _ in range(2))


def backward_panel(gradient, outputs, triplets, start_element, element_count, *, shape):
    """Consume three group-aligned packed windows without a whole restore.

    The transport executor owns the windows until this launch completes. This
    function neither copies tensors nor allocates a packed/dense coefficient.
    """
    _tensor(gradient, shape, torch.bfloat16, gradient.device, strided=True)
    count = prod(shape)
    if (type(start_element) is not int or start_element < 0 or start_element % 1024
            or type(element_count) is not int or element_count < 1 or start_element + element_count > count
            or element_count % 1024 and start_element + element_count != count):
        raise ValueError("SwiGLU panel must contain aligned complete groups or the final tail")
    if (not isinstance(outputs, tuple) or len(outputs) != 2 or not isinstance(triplets, tuple)
            or len(triplets) != 3 or any(not isinstance(triplet, tuple) or len(triplet) != 3 for triplet in triplets)):
        raise ValueError("SwiGLU panel requires two final outputs and three ordered codec triplets")
    groups = triton.cdiv(element_count, 1024)
    for output in outputs:
        _tensor(output, shape, torch.bfloat16, gradient.device)
    saved = tuple(value for triplet in triplets for value in triplet)
    for index, value in enumerate(saved):
        _tensor(value, (groups, 64) if index % 3 == 0 else (groups,),
                torch.int32 if index % 3 == 0 else torch.float32, gradient.device)
    # Producer/private component identities are checked again at runtime; a
    # caller cannot donate an operand to either final gradient destination.
    addresses = [value.untyped_storage().data_ptr() for value in (*outputs, *saved)]
    if len(set(addresses)) != len(addresses) or gradient.untyped_storage().data_ptr() in addresses[:2]:
        raise ValueError("SwiGLU panel outputs and codec components require independent storage")
    with torch.cuda.device(gradient.device):
        _swiglu_backward_panel[(groups,)](gradient, *saved, *outputs, start_element, element_count,
            count, shape, tuple(gradient.stride()), 1024, num_warps=4, enable_fp_fusion=False)
