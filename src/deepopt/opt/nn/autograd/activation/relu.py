"""ReLU autograd saving only an exact boolean or one-bit mask."""

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ....backends.activation import executor
from ....quantization.activation_tensor import FP8Activation
from ....precision.gradient import capture_gradient
from ....quantization.gradient import enabled
from ...._execution import backward_finite_check_policy


class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, x, geometry, options, scaling_state, state_training, backward_enabled, execution_info):
        compute = executor(x)
        ctx.gradient_slot = capture_gradient(x)
        ctx.gradient_check = backward_finite_check_policy()
        arguments = dict(options, save_mask=backward_enabled)
        if scaling_state is not None and options["dtype"] in (torch.float8_e4m3fn, torch.float8_e5m2) and not options["reuse"]:
            result = scaling_state.relu_forward(x, geometry, compute, arguments, training=state_training)
        else:
            result = compute.forward(x, geometry, **arguments)
        output, inverse, mask, _, _, info = result
        if execution_info is not None:
            execution_info.update(info, input="fp8" if isinstance(x, FP8Activation) else str(x.dtype),
                                  output="fp8" if inverse is not None else str(options["dtype"]),
                                  compression=options["packed"] and backward_enabled,
                                  saved_mask_bytes=mask.numel() * mask.element_size() if mask is not None else 0,
                                  backward_launches=int(info["backend"] == "triton" and backward_enabled and x.numel() > 0))
        ctx.geometry, ctx.packed, ctx.input_dtype = geometry, options["packed"], x.dtype
        ctx.input_fp8 = isinstance(x, FP8Activation)
        if backward_enabled:
            ctx.save_for_backward(mask)
        if options["inplace"]:
            ctx.mark_dirty(x)
            return x
        if inverse is not None:
            logical_dtype = x.dtype if x.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) else torch.bfloat16
            return FP8Activation(output, inverse, geometry.shape, options["block_shape"], geometry.layout, logical_dtype)
        return geometry.restore(output)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        mask, = ctx.saved_tensors
        slot = ctx.gradient_slot
        if enabled(slot):
            with slot.output(ctx.geometry.shape, ctx.input_dtype, check_finite=ctx.gradient_check) as output:
                compatible = (isinstance(gradient, FP8Activation) and output.scale is None
                              and gradient._data.dtype == output.dtype and gradient._block_shape == output.block_shape
                              and (gradient._activation_layout == output.layout or output.block_shape is None)
                              and output.layout == ctx.geometry.layout)
                if compatible:
                    values = executor(gradient).backward(gradient, mask, ctx.geometry, packed=ctx.packed,
                                                         dtype=output.dtype, reuse=True)
                    output.inverse = gradient._scale_inv
                    output.info["scale_mode"] = "reuse"
                    dx = output.wrap(values)
                else:
                    dx = executor(gradient).backward(gradient, mask, ctx.geometry, packed=ctx.packed,
                                                      dtype=ctx.input_dtype, gradient_output=output)
            return dx, None, None, None, None, None, None
        # A scale-preserving FP8 upstream gradient can stay FP8 through gating.
        reuse = (slot is None and ctx.input_fp8 and isinstance(gradient, FP8Activation)
                 and (gradient._activation_layout == ctx.geometry.layout or gradient._block_shape is None))
        dtype = gradient._data.dtype if reuse else ctx.input_dtype
        output = executor(gradient).backward(gradient, mask, ctx.geometry, packed=ctx.packed, dtype=dtype, reuse=reuse)
        if reuse:
            dx = FP8Activation(output, gradient._scale_inv, ctx.geometry.shape, gradient._block_shape,
                               ctx.geometry.layout, ctx.input_dtype)
        else:
            dx = ctx.geometry.restore(output)
        return dx, None, None, None, None, None, None
