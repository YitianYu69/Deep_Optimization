"""Multiplication VJPs with per-invocation coefficient and policy snapshots.

The affine codec is a saved-state policy: forward values and PyTorch's scalar
promotion are unchanged. Products round in the operand-promoted dtype before
``sum_to_size``; packed streaming preserves that rounding before FP32 reduction.
"""

from dataclasses import asdict
from numbers import Real

import torch
from torch.autograd.function import once_differentiable
from torch._subclasses.fake_tensor import FakeTensor

from deepopt.config import activation_compression_config_from_dict
from deepopt._activation_memory import invocation_activation_memory, validate_activation_memory_execution
from deepopt.kernel_policy import current_kernel_policy, kernel_policy_scope
from ..backends._shared.packed_multiply import multiply as packed_multiply
from ..quantization.activation import (
    PackedActivation, _dimensions, dequantize_unpack_activation, quantize_pack_activation,
)


def multiplication_support(input, other):
    """Return a deliberate native-fallback reason, or None for ordinary floats."""
    tensors = tuple(value for value in (input, other) if isinstance(value, torch.Tensor))
    if not tensors:
        return "multiplication requires a tensor operand"
    if any(type(value) not in (torch.Tensor, torch.nn.Parameter, FakeTensor) for value in tensors):
        return "tensor subclasses retain their native multiplication dispatch"
    if any(value.is_nested or value.layout != torch.strided for value in tensors):
        return "multiplication compression requires dense strided tensors"
    if any(value.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64)
           for value in tensors):
        return "complex, integer and FP8 multiplication retain native dispatch"
    if any(value.device.type not in ("cpu", "cuda") for value in tensors):
        return "multiplication compression supports CPU and CUDA"
    # A CPU zero-dimensional coefficient is allowed in a CUDA pointwise op.
    device = next((value.device for value in tensors if value.ndim), tensors[0].device)
    if any(value.device != device and not (value.ndim == 0 and value.device.type == "cpu")
           for value in tensors):
        return "multiplication operands must share a device, except CPU scalars"
    if any(not isinstance(value, (torch.Tensor, Real)) for value in (input, other)):
        return "unsupported multiplication scalar type"
    return None


def _pack_coefficient(value, compression):
    """Keep small coefficients and model parameters exact, including AMP scalars."""
    if (not compression["enabled"] or value.dtype != torch.bfloat16
            or isinstance(value, torch.nn.Parameter) or value.numel() < compression["group_size"]):
        return False
    # Codec geometry is a logical row-major matrix, independent of source strides.
    matrix_shape = (1, value.numel()) if value.ndim < 2 else (value.numel() // value.shape[-1], value.shape[-1])
    groups, _, words = _dimensions(matrix_shape, compression["group_size"], None, compression["bits"])
    dense_elements = value.numel()
    if any(size > 1 and stride == 0 for size, stride in zip(value.shape, value.stride())):
        dense_elements = min(dense_elements, 1 + sum((size - 1) * stride
                                                    for size, stride in zip(value.shape, value.stride())))
    return groups * (words * 4 + 8) < dense_elements * value.element_size()


def _policy_tensor(values):
    tensors = tuple(value for value in values if isinstance(value, torch.Tensor))
    return next((value for value in tensors if value.ndim), tensors[0])


class MultiplicationFunction(torch.autograd.Function):
    """Save only coefficients required by the actual input gradient mask."""

    @staticmethod
    def forward(ctx, input, other, compression, memory, kernel_policy, execution_info):
        from .._autocast import _suspend_fp8_interception

        with _suspend_fp8_interception():
            output = torch.mul(input, other)
            ctx.memory, ctx.kernel_policy = memory, kernel_policy
            ctx.product_dtype, ctx.execution_info = output.dtype, execution_info
            ctx.targets = tuple((tuple(value.shape), value.dtype, value.device) if isinstance(value, torch.Tensor) else None
                                for value in (input, other))
            ctx.need = ctx.needs_input_grad[:2]
            saved, descriptors, cached = [], [], {}
            for needed, coefficient in zip(ctx.need, (other, input)):
                if not needed:
                    descriptors.append(None)
                elif not isinstance(coefficient, torch.Tensor):
                    descriptors.append(("scalar", coefficient))
                else:
                    descriptor = cached.get(id(coefficient))
                    if descriptor is None:
                        offset = len(saved)
                        if _pack_coefficient(coefficient, compression):
                            shape = tuple(coefficient.shape)
                            matrix = coefficient.reshape((1, coefficient.numel()) if coefficient.ndim < 2
                                                         else (-1, shape[-1]))
                            packed = quantize_pack_activation(matrix, compression["group_size"], bits=compression["bits"])
                            saved.extend((packed.data, packed.step, packed.minimum))
                            descriptor = ("packed", offset, shape, coefficient.dtype, packed.shape,
                                          packed.group_size, packed.bits)
                        else:
                            saved.append(coefficient)
                            descriptor = ("exact", offset)
                        cached[id(coefficient)] = descriptor
                    descriptors.append(descriptor)
            ctx.descriptors = tuple(descriptors)
            ctx.save_for_backward(*saved)
            if execution_info is not None:
                execution_info.update(gradient_mask=tuple(ctx.need),
                                      coefficient_states=tuple(item[0] if item else "unused" for item in descriptors),
                                      saved_state_bytes=sum(value.numel() * value.element_size() for value in saved),
                                      rounding_domain="operand_dtype", activation_memory=memory.as_dict(),
                                      kernel_policy=kernel_policy.as_dict())
            return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        from .._autocast import _suspend_fp8_interception

        with _suspend_fp8_interception(), kernel_policy_scope(ctx.kernel_policy, role="compiler.backward"):
            saved, restored, gradients, routes = ctx.saved_tensors, {}, [], []
            for descriptor, target in zip(ctx.descriptors, ctx.targets):
                if descriptor is None:
                    gradients.append(None)
                    routes.append({"route": "unused", "scratch_bytes": 0})
                    continue
                if descriptor[0] == "scalar":
                    coefficient = descriptor[1]
                elif descriptor[0] == "exact":
                    coefficient = saved[descriptor[1]]
                else:
                    _, offset, shape, dtype, matrix_shape, group_size, bits = descriptor
                    if ctx.memory.multiply_backward == "packed_streaming":
                        target_shape, target_dtype, device = target
                        info = {} if ctx.execution_info is not None else None
                        options = dict(logical_shape=shape, logical_dtype=dtype, matrix_shape=matrix_shape,
                                       group_size=group_size, bits=bits, traversal="logical_row_major")
                        result = packed_multiply(gradient, *saved[offset:offset + 3], options=options,
                            shape=tuple(gradient.shape), product_dtype=ctx.product_dtype,
                            target_shape=target_shape, target_dtype=target_dtype,
                            scratch_bytes=ctx.memory.reduction_scratch_bytes,
                            kernel_policy=ctx.kernel_policy, execution_info=info,
                            allow_materialized=gradient.device.type != "cuda")
                        gradients.append(result.to(device=device))
                        routes.append(info)
                        continue
                    coefficient = restored.get(offset)
                    if coefficient is None:
                        packed = PackedActivation(*saved[offset:offset + 3], matrix_shape, group_size, bits=bits)
                        coefficient = dequantize_unpack_activation(packed, dtype=dtype, row_alignment=1).reshape(shape)
                        restored[offset] = coefficient
                shape, dtype, device = target
                product = torch.mul(gradient, coefficient)
                result = product.sum_to_size(shape).to(device=device, dtype=dtype)
                gradients.append(result)
                if ctx.execution_info is not None:
                    routes.append(dict(route="materialized", selected=False,
                        rejection_reason=("memory_policy_" + ctx.memory.multiply_backward
                                          if descriptor[0] == "packed" else "exact_coefficient"),
                        decoded_full_temporary_bytes=(coefficient.numel() * coefficient.element_size()
                                                      if descriptor[0] == "packed" else 0),
                        product_full_temporary_bytes=product.numel() * product.element_size(),
                        scratch_bytes=0, output_bytes=result.numel() * result.element_size()))
            if ctx.execution_info is not None:
                ctx.execution_info["multiply_backward"] = tuple(routes)
            return *gradients, None, None, None, None


def multiply(input, other, *, plan=None, execution_info=None):
    """Use native forward values and optional BF16 saved-coefficient compression.

    Packed coefficients are immutable forward snapshots. Later mutations through
    an alias do not invalidate them. Exact coefficients retain PyTorch saved
    tensor version checks; no original activation is kept for a packed check.
    """
    from .._autocast import _get_activation_compression, _suspend_fp8_interception

    with _suspend_fp8_interception():
        reason = multiplication_support(input, other)
        if reason is not None:
            if execution_info is not None:
                execution_info.update(native_fallback=reason, saved_state_bytes=0)
            return torch.mul(input, other)
        tensor = _policy_tensor((input, other))
        checked = None
        if plan is not None:
            from .functional import _prepare_linear_plan

            checked = _prepare_linear_plan(plan)
            compression = (checked.get("fp8_training") or checked)["activation_compression"]
        else:
            compression = _get_activation_compression(tensor.device.type) or {}
        compression = asdict(activation_compression_config_from_dict(compression))
        memory = invocation_activation_memory(checked, tensor.device.type)
        policy = (getattr(checked, "kernel_policy", None) if checked is not None else None) or current_kernel_policy()
        validate_activation_memory_execution(memory, kernel_policy=policy)
        if (not torch.is_grad_enabled() or not any(isinstance(value, torch.Tensor) and value.requires_grad
                                                 for value in (input, other))):
            if execution_info is not None:
                execution_info.update(gradient_mask=(False, False), coefficient_states=("unused", "unused"),
                                      saved_state_bytes=0, rounding_domain="operand_dtype")
            return torch.mul(input, other)
        return MultiplicationFunction.apply(input, other, compression, memory, policy, execution_info)
