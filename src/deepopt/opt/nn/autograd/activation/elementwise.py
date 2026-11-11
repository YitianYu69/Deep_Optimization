"""First-order activation autograd saving derivatives or exact ReLU6 masks."""

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ...activation.state import SavedActivation
from ....backends.activation import executor
from ....quantization.activation_tensor import FP8Activation
from ....precision.gradient import capture_gradient
from ....quantization.gradient import enabled
from ...._execution import backward_finite_check_policy


class ActivationFunction(Function):
    @staticmethod
    def forward(ctx, x, geometry, options, scaling_state, state_training, backward_enabled, execution_info):
        compute = executor(x, options["kind"])
        ctx.gradient_slot = capture_gradient(x)
        ctx.gradient_check = backward_finite_check_policy()
        arguments = dict(options, save_mask=backward_enabled)
        if scaling_state is not None and not options["reuse"] and options["dtype"] in (torch.float8_e4m3fn, torch.float8_e5m2):
            result = scaling_state.activation_forward(x, geometry, compute, arguments, training=state_training)
        else:
            result = compute.forward(x, geometry, **arguments)
        output, inverse, saved, _, _, info = result
        if execution_info is not None:
            execution_info.update(info, input="fp8" if isinstance(x, FP8Activation) else str(x.dtype),
                                  output="fp8" if inverse is not None else str(options["dtype"]),
                                  compression=options["packed"] and backward_enabled,
                                  saved_state_bytes=sum(t.numel() * t.element_size() for t in saved.tensors) if saved else 0)
        ctx.geometry, ctx.input_dtype, ctx.kind = geometry, x.dtype, options["kind"]
        if backward_enabled:
            ctx.saved_metadata = saved.metadata
            ctx.save_for_backward(*saved.tensors)
        if options["inplace"]:
            ctx.mark_dirty(x)
            return x
        if inverse is not None:
            logical = x.dtype if x.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) else torch.bfloat16
            return FP8Activation(output, inverse, geometry.shape, options["block_shape"], geometry.layout, logical)
        return geometry.restore(output)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        saved = SavedActivation.restore(ctx.saved_tensors, ctx.saved_metadata)
        compute = executor(gradient, ctx.kind)
        slot = ctx.gradient_slot
        if enabled(slot):
            with slot.output(ctx.geometry.shape, ctx.input_dtype, check_finite=ctx.gradient_check) as output:
                dx = compute.backward(gradient, saved, ctx.geometry, dtype=ctx.input_dtype, gradient_output=output)
        else:
            dx = ctx.geometry.restore(compute.backward(gradient, saved, ctx.geometry, dtype=ctx.input_dtype))
        return dx, None, None, None, None, None, None
