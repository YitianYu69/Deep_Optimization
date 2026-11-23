"""Strict runtime contract of the first GroupNorm/SiLU saved-state composite."""

from math import prod

import torch


def canonicalize_group_silu_cotangent(source):
    """Match the original activation VJP's contiguous BF16 reduction input.

    Zero/broadcast and spatially strided inputs otherwise specialize Triton's
    reduction loads differently from the materialized native VJP. This is an
    exact layout copy of the *upstream* cotangent, not a decoded coefficient or
    activation-adjoint allocation. Contiguous inputs keep their storage.
    Device/geometry guards belong to validate_group_silu_residuals; this layout
    operation can also be tested independently on CPU.
    """
    if type(source) is not torch.Tensor or source.layout != torch.strided or source.dtype != torch.bfloat16:
        raise ValueError("GroupNorm/SiLU cotangent layout preparation requires dense BF16")
    original_stride = tuple(source.stride())
    prepared = source.contiguous()
    copied = prepared is not source
    return prepared, {"upstream_cotangent_stride": original_stride,
        "reduction_cotangent_stride": tuple(prepared.stride()),
        "raw_cotangent_copy_bytes": source.numel() * source.element_size() if copied else 0,
        "cotangent_layout_policy": "contiguous_before_normalization_reductions"}


def validate_group_silu_residuals(source, saved, geometry, compression, activation, *, gradient=False):
    from .normalization import normalization_variant

    if (geometry.kind != "group" or not geometry.batch_stats or len(geometry.shape) != 4
            or geometry.elements <= 4096 or geometry.elements % 1024
            or compression.get("enabled") is not True or compression.get("group_size") != 1024
            or compression.get("bits") != 2 or activation != ("silu", "none")
            or normalization_variant() not in ("auto", "baseline")):
        raise ValueError("GroupNorm/SiLU residual fusion requires aligned flat1024 two-bit baseline geometry")
    if (type(source) is not torch.Tensor or source.device.type != "cuda"
            or source.dtype != torch.bfloat16 or tuple(source.shape) != tuple(geometry.shape)
            or source.layout != torch.strided or not gradient and not source.is_contiguous()):
        raise ValueError("GroupNorm/SiLU residual fusion requires CUDA BF16 contiguous NCHW operands")
    if not isinstance(saved, (tuple, list)) or len(saved) != 3:
        raise ValueError("GroupNorm/SiLU residual fusion requires three declared activation states")
    groups = prod(geometry.shape) // 1024
    for value, shape, dtype in zip(saved, ((groups, 64), (groups,), (groups,)),
                                    (torch.int32, torch.float32, torch.float32)):
        if (type(value) is not torch.Tensor or tuple(value.shape) != shape or value.dtype != dtype
                or value.device != source.device or value.layout != torch.strided or not value.is_contiguous()):
            raise ValueError("GroupNorm/SiLU activation residual metadata differs from its declared ABI")
    if len({value.untyped_storage().data_ptr() for value in saved}) != 3 or any(
            value.untyped_storage().data_ptr() == source.untyped_storage().data_ptr() for value in saved):
        raise ValueError("GroupNorm/SiLU activation residuals must have distinct storage")
    return tuple(saved)
