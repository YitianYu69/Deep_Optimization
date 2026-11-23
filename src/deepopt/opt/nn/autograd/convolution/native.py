"""Native convolutions with independently compressed saved source values."""

from deepopt.kernel_policy import kernel_forward, kernel_backward

from contextlib import nullcontext

import torch
from torch._prims_common import suggest_memory_format
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ....backends.native.convolution import (
    native_input_bias_gradients,
    native_convolution,
    native_weight_gradient,
)
from ....quantization.activation import (PackedActivation, _quantize_pack_strided_activation,
    _dequantize_unpack_spatial_activation,quantize_pack_activation,dequantize_unpack_activation)
from ....backends._shared.activation_codec import convolution_codec
from ...convolution import source_matrix,restore_matrix
from ...convolution.geometry import _contiguous, zero_synthetic_padding


def _memory_format(x):
    # Singleton channels can satisfy both contiguity predicates. Use native
    # stride preference rather than silently changing the channel stride.
    return suggest_memory_format(x)


class _NativeConvolutionFunction(Function):
    @staticmethod
    @kernel_forward
    def forward(ctx, x, weight, bias, geometry, compression):
        # Native CPU depthwise convolution has an inference-only Winograd path.
        # Keep training algorithm selection while preventing an inner autograd
        # graph from saving full activations or invoking saved-tensor hooks.
        # Casts below Autograd have no grad_fn and must not enter the caller's
        # AMP parameter cache, where later native calls expect differentiable
        # casts. Preserve the caller's AMP dtype/enabled policy independently.
        amp = (torch.autocast(x.device.type, enabled=torch.is_autocast_enabled(x.device.type),
                              dtype=torch.get_autocast_dtype(x.device.type), cache_enabled=False)
               if x.device.type in ("cpu", "cuda") else nullcontext())
        with amp, torch.enable_grad(), torch._C._AutoDispatchBelowAutograd():
            output = native_convolution(x, weight, bias, geometry)
        need_dx, need_dw = ctx.needs_input_grad[:2]
        ctx.codec=convolution_codec()
        packed = None
        if need_dw and x.numel():
            # Compress the source values that native AMP actually consumed;
            # retain each input position once, without an unfolded workspace.
            packed=(quantize_pack_activation(source_matrix(x).to(output.dtype),compression["group_size"],bits=compression["bits"])
                    if ctx.codec=="materialized" else _quantize_pack_strided_activation(x.movedim(1,-1),
                        compression["group_size"],bits=compression["bits"],source_dtype=output.dtype))
        ctx.save_for_backward(
            packed.data if packed is not None else None,
            packed.step if packed is not None else None,
            packed.minimum if packed is not None else None,
            weight if need_dx else None,
        )
        ctx.geometry = geometry
        ctx.group_size, ctx.bits = compression["group_size"], compression["bits"]
        ctx.input_dtype, ctx.weight_dtype = x.dtype, weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.compute_dtype, ctx.device_type = output.dtype, x.device.type
        ctx.input_memory_format, ctx.weight_memory_format = _memory_format(x), _memory_format(weight)
        return output

    @staticmethod
    @once_differentiable
    @kernel_backward
    def backward(ctx, grad_output):
        data, step, minimum, weight = ctx.saved_tensors
        need_dx, need_dw, need_db = ctx.needs_input_grad[:3]
        geometry = ctx.geometry
        disabled_autocast = (torch.autocast(ctx.device_type, enabled=False)
                             if ctx.device_type in ("cpu", "cuda") else nullcontext())
        dx = dw = db = None
        with disabled_autocast:
            grad = grad_output.to(ctx.compute_dtype)
            if need_dx or need_db:
                dx, _, db = native_input_bias_gradients(
                    grad, weight.to(ctx.compute_dtype) if need_dx else None, geometry,
                    (need_dx, False, need_db), input_memory_format=ctx.input_memory_format,
                    weight_memory_format=ctx.weight_memory_format,
                )
                if need_dx:
                    dx = _contiguous(dx.to(ctx.input_dtype), ctx.input_memory_format)
                if need_db:
                    db = db.to(ctx.bias_dtype)
            if need_dw:
                if data is None:
                    activation = grad.new_empty(geometry.input_shape)
                else:
                    packed = PackedActivation(data, step, minimum, geometry.input_matrix_shape,
                                              ctx.group_size, bits=ctx.bits)
                    activation=(restore_matrix(dequantize_unpack_activation(packed,dtype=ctx.compute_dtype),
                                    geometry.input_shape,ctx.input_memory_format) if ctx.codec=="materialized" else
                                _dequantize_unpack_spatial_activation(packed, geometry.input_shape,
                                    dtype=ctx.compute_dtype, memory_format=ctx.input_memory_format))
                    zero_synthetic_padding(activation, geometry)
                dw = native_weight_gradient(activation, grad, geometry,
                                            weight_memory_format=ctx.weight_memory_format)
                dw = _contiguous(dw.to(ctx.weight_dtype), ctx.weight_memory_format)
        return dx, dw, db, None, None
