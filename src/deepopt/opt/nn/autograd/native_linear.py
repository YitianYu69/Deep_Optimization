"""Native linear autograd with optional packed affine saved activation storage."""

from contextlib import nullcontext

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ...quantization.activation import PackedActivation, quantize_pack_activation, dequantize_unpack_activation


class _NativeLinearFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, group_size, backward_enabled, bits=2):
        # The functional entry point suspends DeepOpt dispatch before calling
        # this native operator, while retaining PyTorch's autocast policy.
        output = torch._C._nn.linear(x, weight, bias)
        need_dx, need_dw = (backward_enabled and need for need in ctx.needs_input_grad[:2])
        packed = None
        if need_dw:
            # Native autocast rounds inputs before the GEMM. Compress those
            # same values, including when the caller keeps FP32 master tensors.
            packed = quantize_pack_activation(x.reshape(-1, weight.shape[1]).to(output.dtype), group_size, bits=bits)
        ctx.save_for_backward(
            packed.data if packed is not None else None,
            packed.step if packed is not None else None,
            packed.minimum if packed is not None else None,
            weight if need_dx else None,
        )
        ctx.input_shape, ctx.input_dtype = tuple(x.shape), x.dtype
        ctx.weight_shape, ctx.weight_dtype = tuple(weight.shape), weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.compute_dtype, ctx.group_size = output.dtype, group_size
        ctx.compression_bits = bits
        ctx.device_type = x.device.type
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        data, step, minimum, weight = ctx.saved_tensors
        need_dx, need_dw, need_db = ctx.needs_input_grad[:3]
        grad = grad_output.reshape(-1, ctx.weight_shape[0]).to(ctx.compute_dtype)
        disabled_autocast = (torch.autocast(ctx.device_type, enabled=False)
                             if ctx.device_type in ("cpu", "cuda") else nullcontext())
        dx = dw = db = None
        with disabled_autocast:
            if need_dx:
                dx = (grad @ weight.to(ctx.compute_dtype)).reshape(ctx.input_shape).to(ctx.input_dtype)
            if need_dw:
                packed = PackedActivation(data, step, minimum,
                                          (grad.shape[0], ctx.weight_shape[1]), ctx.group_size,
                                          bits=ctx.compression_bits)
                activation = dequantize_unpack_activation(packed, dtype=ctx.compute_dtype)
                dw = (grad.T @ activation).to(ctx.weight_dtype)
            if need_db:
                # Native AMP computes the bias gradient in the projection's
                # dtype before promoting it to an FP32 master parameter.
                db = grad.sum(dim=0).to(ctx.bias_dtype)
        return (dx, dw, db, None, None, None)[:len(ctx.needs_input_grad)]
