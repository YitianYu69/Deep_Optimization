"""Metadata guards for compiler-owned normalization output destinations."""

import torch


def validate_normalization_output(output, x, weight, bias, geometry, *, dtype,
                                  output_layout=None, activation=None, preactivation=None):
    """Accept a non-overlapping strided destination without allocating/copying.

    The compiler owns the destination allocation. This private executor hook
    does not grant permission to overwrite caller/model storage. Inputs and
    affine parameters must never alias the destination, including disjoint
    views of their backing storage.
    """
    if (geometry.kind != "group" or output_layout is not None
            or activation is not None or preactivation is not None):
        raise ValueError("Normalization output destinations require unactivated dense GroupNorm")
    if not isinstance(output, torch.Tensor) or output.layout != torch.strided or output.is_nested:
        raise TypeError("Normalization output destination must be a strided tensor")
    if (tuple(output.shape) != tuple(geometry.shape) or output.dtype != dtype
            or output.device != x.device):
        raise ValueError("Normalization output destination shape, dtype or device does not match")
    if output.requires_grad or output.grad_fn is not None:
        raise ValueError("Normalization output destination must be compiler-owned non-differentiable storage")
    # A channel slice has gaps between batches and is intentionally not dense.
    # Prove positive-stride, non-overlapping indexing rather than requiring
    # is_contiguous(), which would reject the intended destination view.
    span = 1
    for stride, size in sorted((stride, size) for size, stride in zip(output.shape, output.stride()) if size > 1):
        if stride < span:
            raise ValueError("Normalization output destination has overlapping strides")
        span += (size - 1) * stride
    if any(stride < 0 for stride in output.stride()):
        raise ValueError("Normalization output destination requires nonnegative strides")
    end = output.storage_offset() + (span if output.numel() else 0)
    if end * output.element_size() > output.untyped_storage().nbytes():
        raise ValueError("Normalization output destination exceeds its backing storage")
    for source in (x, weight, bias):
        if isinstance(source, torch.Tensor) and torch._C._overlaps(output, source):
            raise ValueError("Normalization output destination aliases an input or affine parameter")
    return output
