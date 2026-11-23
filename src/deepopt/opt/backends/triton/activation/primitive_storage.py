"""Direct output stores for mixed BF16/FP32 model addition."""

from math import prod

import torch
import triton
import triton.language as tl


@triton.jit
def _add(A, B, O, N: tl.constexpr, SHAPE: tl.constexpr, AS: tl.constexpr, BS: tl.constexpr,
         SCALAR_A: tl.constexpr, SCALAR_B: tl.constexpr, AV: tl.constexpr, BV: tl.constexpr,
         BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ao, bo = tl.full((BLOCK,), 0, tl.int64), tl.full((BLOCK,), 0, tl.int64)
    remaining = index
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coordinate = remaining % SHAPE[axis]
        remaining = remaining // SHAPE[axis]
        ao += coordinate * AS[axis]
        bo += coordinate * BS[axis]
    a = AV if SCALAR_A else tl.load(A + ao, index < N, 0).to(tl.float32)
    b = BV if SCALAR_B else tl.load(B + bo, index < N, 0).to(tl.float32)
    tl.store(O + index, a + b, index < N)


def add(input, other, dtype):
    tensors = tuple(t for t in (input, other) if isinstance(t, torch.Tensor))
    shape = tuple(torch.broadcast_shapes(*(t.shape for t in tensors)))
    output = torch.empty(shape, device=tensors[0].device, dtype=dtype)
    if not output.numel():
        return output
    def strides(value):
        if not isinstance(value, torch.Tensor):
            return (0,) * len(shape)
        padding = len(shape) - value.ndim
        return (0,) * padding + tuple(0 if size == 1 else stride for size, stride in zip(value.shape, value.stride()))
    a_scalar, b_scalar = not isinstance(input, torch.Tensor), not isinstance(other, torch.Tensor)
    _add[(triton.cdiv(prod(shape), 256),)](
        output if a_scalar else input, output if b_scalar else other, output, prod(shape), shape,
        strides(input), strides(other), a_scalar, b_scalar, input if a_scalar else 0., other if b_scalar else 0.,
        256, enable_fp_fusion=False)
    return output
