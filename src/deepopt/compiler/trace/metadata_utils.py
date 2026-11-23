from __future__ import annotations

from dataclasses import replace
from numbers import Number
from typing import Any

import torch

from ..ir import TensorMeta, TensorValue
from .tracer import SymbolicTensor


def clone_meta(meta: TensorMeta, *, requires_grad: bool | None = None) -> TensorMeta:
    return meta if requires_grad is None else replace(meta, requires_grad=requires_grad)


def broadcast_shape(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
    result: list[int] = []
    for offset in range(1, max(len(lhs), len(rhs)) + 1):
        left = lhs[-offset] if offset <= len(lhs) else 1
        right = rhs[-offset] if offset <= len(rhs) else 1
        if left == right:
            result.append(left)
        elif left == 1:
            result.append(right)
        elif right == 1:
            result.append(left)
        else:
            raise ValueError(f"Shapes are not broadcastable: {lhs} and {rhs}")
    return tuple(reversed(result))


def contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [0] * len(shape)
    running_stride = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = running_stride
        running_stride *= max(shape[index], 1)
    return tuple(strides)


def _tensor_meta(value: Any) -> TensorMeta | None:
    if isinstance(value, SymbolicTensor):
        value = value.value
    if isinstance(value, TensorValue):
        if value.meta is None:
            raise ValueError(f"TensorValue {value.name!r} has no TensorMeta")
        return value.meta
    return None


def _binary_tensor_metas(
    lhs: Any,
    rhs: Any,
) -> tuple[TensorMeta | None, TensorMeta | None]:
    lhs_meta = _tensor_meta(lhs)
    rhs_meta = _tensor_meta(rhs)
    if lhs_meta is None and rhs_meta is None:
        raise ValueError(
            "At least one operand of a tensor operation must be a TensorValue"
        )
    if lhs_meta is None and not isinstance(lhs, Number):
        raise TypeError(f"Unsupported lhs operand type: {type(lhs).__name__}")
    if rhs_meta is None and not isinstance(rhs, Number):
        raise TypeError(f"Unsupported rhs operand type: {type(rhs).__name__}")
    return lhs_meta, rhs_meta


def _scalar_result_dtype(tensor_dtype: torch.dtype, scalar: Number) -> torch.dtype:
    # Match PyTorch's wrapped-scalar promotion behavior without allocating on
    # the tensor's accelerator device.
    return torch.result_type(torch.empty((), dtype=tensor_dtype), scalar)


def infer_binary_meta(args: tuple[Any, ...], kwargs: dict[str, Any]) -> TensorMeta:
    if len(args) < 2:
        raise ValueError("Binary op requires at least two arguments")
    lhs, rhs = args[:2]
    lhs_meta, rhs_meta = _binary_tensor_metas(lhs, rhs)

    if lhs_meta is None:
        assert rhs_meta is not None
        shape = rhs_meta.shape
        dtype = _scalar_result_dtype(rhs_meta.dtype, lhs)
        device = rhs_meta.device
        requires_grad = rhs_meta.requires_grad
    elif rhs_meta is None:
        shape = lhs_meta.shape
        dtype = _scalar_result_dtype(lhs_meta.dtype, rhs)
        device = lhs_meta.device
        requires_grad = lhs_meta.requires_grad
    else:
        shape = broadcast_shape(lhs_meta.shape, rhs_meta.shape)
        # Zero-dimensional floating tensors follow wrapped-scalar promotion:
        # e.g. BF16 feature * FP32 scalar remains BF16.
        dtype = torch.result_type(torch.empty(lhs_meta.shape and (1,) or (), dtype=lhs_meta.dtype),
                                  torch.empty(rhs_meta.shape and (1,) or (), dtype=rhs_meta.dtype))
        if (
            lhs_meta.device is not None
            and rhs_meta.device is not None
            and torch.device(lhs_meta.device) != torch.device(rhs_meta.device)
        ):
            raise ValueError(
                "Pointwise operands are on different devices: "
                f"{lhs_meta.device} and {rhs_meta.device}"
            )
        device = lhs_meta.device if lhs_meta.device is not None else rhs_meta.device
        requires_grad = lhs_meta.requires_grad or rhs_meta.requires_grad

    return TensorMeta(
        shape=shape,
        dtype=dtype,
        device=device,
        stride=contiguous_stride(shape),
        requires_grad=requires_grad,
    )


def infer_unary_meta(args: tuple[Any, ...], kwargs: dict[str, Any]) -> TensorMeta:
    if not args:
        raise ValueError("Unary op requires one positional argument")
    meta = _tensor_meta(args[0])
    if meta is None:
        raise TypeError("Unary tensor op requires a TensorValue")
    return TensorMeta(
        shape=meta.shape,
        dtype=meta.dtype,
        device=meta.device,
        stride=contiguous_stride(meta.shape),
        requires_grad=meta.requires_grad,
    )


def infer_sum_to_size_meta(args: tuple[Any, ...], kwargs: dict[str, Any]) -> TensorMeta:
    if len(args) < 2:
        raise ValueError("sum_to_size expects (tensor, size)")
    source_meta = _tensor_meta(args[0])
    if source_meta is None:
        raise TypeError("sum_to_size source must be a TensorValue")
    size = tuple(int(item) for item in args[1])
    return TensorMeta(
        shape=size,
        dtype=source_meta.dtype,
        device=source_meta.device,
        stride=contiguous_stride(size),
        requires_grad=source_meta.requires_grad,
    )


def infer_first_tensor_meta(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> TensorMeta:
    for arg in args:
        meta = _tensor_meta(arg)
        if meta is not None:
            return TensorMeta(
                shape=meta.shape,
                dtype=meta.dtype,
                device=meta.device,
                stride=contiguous_stride(meta.shape),
                requires_grad=meta.requires_grad,
            )
    raise TypeError("Expected at least one TensorValue argument")
