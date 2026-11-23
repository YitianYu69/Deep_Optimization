"""Storage-policy model exp with register FP32 coefficients and BF16 stores.

The compressed residual encodes exp(BF16 input) before output rounding, using
the ordinary affine bits/group-1024 codec. Dense backward recomputes that same
coefficient from an immutable BF16 input snapshot. No global FP32 coefficient
or derivative tensor is allocated. CUDA availability is not qualification.
"""
from math import prod

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..quantization.activation.affine import _affine_quantize_pack
from .elementwise_backward import _coefficient


@triton.jit
def _logical_address(index, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    address = index * 0
    remaining = index
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        address += remaining % SHAPE[axis] * STRIDES[axis]
        remaining = remaining // SHAPE[axis]
    return address


@triton.jit
def _storage_exp_forward(x, output, saved, steps, minima, N: tl.constexpr,
                         SHAPE: tl.constexpr, STRIDES: tl.constexpr, PACKED: tl.constexpr,
                         BITS: tl.constexpr, GROUP: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    index = group * GROUP + tl.arange(0, GROUP)
    valid = index < N
    value = tl.load(x + _logical_address(index, SHAPE, STRIDES), valid, other=0).to(tl.float32)
    coefficient = libdevice.exp(value)
    tl.store(output + index, coefficient.to(output.dtype.element_ty), valid)
    if PACKED:
        words, step, minimum = _affine_quantize_pack(coefficient, valid, BITS, GROUP)
        offsets = tl.arange(0, GROUP // (32 // BITS))
        tl.store(saved + group * (GROUP // (32 // BITS)) + offsets, words)
        tl.store(steps + group, step)
        tl.store(minima + group, minimum)
    else:
        tl.store(saved + index, value.to(saved.dtype.element_ty), valid)


@triton.jit
def _storage_exp_backward(gradient, saved, steps, minima, output, N: tl.constexpr,
                          SHAPE: tl.constexpr, STRIDES: tl.constexpr, PACKED: tl.constexpr,
                          BITS: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N
    dy = tl.load(gradient + _logical_address(index, SHAPE, STRIDES), valid, other=0).to(tl.float32)
    if PACKED:
        coefficient = _coefficient(saved, steps, minima, index, valid, N, GROUP,
                                   0, 0, GROUP // (32 // BITS), BITS, False, True)
    else:
        value = tl.load(saved + index, valid, other=0).to(tl.float32)
        coefficient = libdevice.exp(value)
    tl.store(output + index, (dy * coefficient).to(output.dtype.element_ty), valid)


def _validate(tensor, *, bits, group_size):
    if tensor.device.type != 'cuda' or tensor.dtype != torch.bfloat16 or tensor.layout != torch.strided:
        raise ValueError('Storage exp requires dense BF16 CUDA operands')
    if type(bits) is not int or bits not in (1, 2, 4, 8):
        raise ValueError('Storage exp bits must be 1, 2, 4, or 8')
    if type(group_size) is not int or group_size != 1024:
        raise ValueError('Storage exp compression qualification requires group_size=1024')
    if tensor.is_neg() or tensor.is_conj():
        raise ValueError('Storage exp pointer kernels require resolved negative/conjugate views')


@torch.no_grad()
def forward(x, *, packed=False, bits=2, group_size=1024):
    _validate(x, bits=bits, group_size=group_size)
    with torch.cuda.device(x.device):
        output = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
        n = x.numel()
        groups = triton.cdiv(n, group_size)
        if packed:
            data = torch.empty((groups, group_size // (32 // bits)), device=x.device, dtype=torch.int32)
            steps = torch.empty(groups, device=x.device, dtype=torch.float32)
            minima = torch.empty_like(steps)
            residual = (data, steps, minima)
        else:
            data = torch.empty_like(output)
            steps = minima = output
            residual = (data,)
        if n:
            _storage_exp_forward[(groups,)](x, output, data, steps, minima, n, tuple(x.shape),
                tuple(x.stride()), packed, bits, group_size, enable_fp_fusion=False)
        return output, *residual


@torch.no_grad()
def backward(gradient, *saved, shape, packed=False, bits=2, group_size=1024):
    _validate(gradient, bits=bits, group_size=group_size)
    if tuple(gradient.shape) != tuple(shape) or len(saved) != (3 if packed else 1):
        raise ValueError('Storage exp gradient or residual cardinality differs from captured ABI')
    if any(t.device != gradient.device for t in saved):
        raise ValueError('Storage exp residual operands must share a device')
    if any(t.is_neg() or t.is_conj() for t in saved):
        raise ValueError('Storage exp residuals require resolved negative/conjugate views')
    n = prod(shape)
    if packed:
        groups = triton.cdiv(n, group_size)
        data, steps, minima = saved
        if (data.dtype != torch.int32 or tuple(data.shape) != (groups, group_size // (32 // bits))
                or any(t.dtype != torch.float32 or tuple(t.shape) != (groups,) for t in (steps, minima))
                or not all(t.is_contiguous() for t in saved)):
            raise ValueError('Storage exp packed residual ABI mismatch')
    else:
        data, = saved
        if data.dtype != torch.bfloat16 or tuple(data.shape) != tuple(shape) or not data.is_contiguous():
            raise ValueError('Storage exp exact-input residual ABI mismatch')
        steps = minima = gradient
    with torch.cuda.device(gradient.device):
        output = torch.empty(shape, device=gradient.device, dtype=torch.bfloat16)
        if n:
            _storage_exp_backward[(triton.cdiv(n, 512),)](gradient, data, steps, minima, output, n,
                tuple(shape), tuple(gradient.stride()), packed, bits, group_size, 512, enable_fp_fusion=False)
        return output
