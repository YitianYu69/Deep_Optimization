"""Padding with a first-order adjoint that retains geometry, never input values."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from torch._subclasses.fake_tensor import FakeTensor

from deepopt.kernel_policy import kernel_backward, kernel_forward


@dataclass(frozen=True)
class PaddingGeometry:
    shape: tuple[int, ...]
    pads: tuple[int, ...]
    mode: str
    dtype: torch.dtype
    memory_format: torch.memory_format = torch.contiguous_format

    @classmethod
    def from_tensor(cls, input, pad, mode):
        from torch._prims_common import suggest_memory_format

        return cls(tuple(input.shape), tuple(pad), mode, input.dtype, suggest_memory_format(input))


def padding_adjoint(gradient, geometry):
    """Apply the exact padding adjoint using only a shape/layout descriptor.

    ATen reflection/replication kernels inspect the input geometry, but never
    read its values. An expanded scalar supplies that ABI without retaining or
    allocating a source-shaped value tensor. Allocation qualification of the
    selected native kernel remains separate from this saved-state contract.
    """
    shape, pads, mode = geometry.shape, geometry.pads, geometry.mode
    if not any(pads):
        # Nonempty zero-width tuples are also identity maps. Avoid invoking
        # native scatter adjoints whose strict CUDA policy can reject them.
        result = gradient.clone()
    elif mode == "constant":
        result = F.pad(gradient, tuple(-value for value in pads))
    elif mode in ("reflect", "replicate"):
        prefix = "reflection" if mode == "reflect" else "replication"
        function = getattr(torch.ops.aten, f"{prefix}_pad{len(pads) // 2}d_backward").default
        descriptor = gradient.new_empty(()).expand(shape)
        result = function(gradient, descriptor, list(pads))
    elif mode == "circular":
        result = gradient
        # Native circular padding crops negative pads before wrapping. Fold
        # axes in reverse order so the adjoint also handles corners correctly.
        for index in range(len(pads) // 2 - 1, -1, -1):
            axis = len(shape) - index - 1
            left, right = pads[2 * index:2 * index + 2]
            size = shape[axis]
            start, stop = max(-left, 0), size - max(-right, 0)
            count, positive_left, positive_right = stop - start, max(left, 0), max(right, 0)
            folded = result.narrow(axis, positive_left, count).clone()
            # Reverse native's right-copy then left-copy, preserving each
            # BF16 addition boundary when both wraps reach the same cell.
            if positive_right:
                folded.narrow(axis, 0, positive_right).add_(result.narrow(axis, positive_left + count, positive_right))
            if positive_left:
                folded.narrow(axis, count - positive_left, positive_left).add_(result.narrow(axis, 0, positive_left))
            if start or stop != size:
                restored_shape = list(folded.shape)
                restored_shape[axis] = size
                restored = result.new_zeros(restored_shape)
                restored.narrow(axis, start, count).copy_(folded)
                folded = restored
            result = folded
    else:
        raise ValueError(f"Unsupported padding mode {mode!r}")
    return result.to(geometry.dtype).contiguous(memory_format=geometry.memory_format)


class _PaddingFunction(torch.autograd.Function):
    @staticmethod
    @kernel_forward
    def forward(ctx, input, pad, mode, value):
        # Validate all native argument constraints before constructing state.
        output = F.pad(input, pad, mode, value)
        ctx.geometry = PaddingGeometry.from_tensor(input, pad, mode)
        return output

    @staticmethod
    @once_differentiable
    @kernel_backward
    def backward(ctx, gradient):
        from .._autocast import _suspend_fp8_interception

        with _suspend_fp8_interception(), torch.autocast(gradient.device.type, enabled=False):
            return padding_adjoint(gradient, ctx.geometry), None, None, None


def padding(input, pad, mode="constant", value=None):
    """Keep native forward semantics and save metadata only for first order AD."""
    if (type(input) not in (torch.Tensor, FakeTensor) or input.layout != torch.strided or input.is_nested
            or input.device.type not in ("cpu", "cuda") or not input.is_floating_point()
            or not torch.is_grad_enabled() or not input.requires_grad):
        return F.pad(input, pad, mode, value)
    return _PaddingFunction.apply(input, pad, mode, value)
