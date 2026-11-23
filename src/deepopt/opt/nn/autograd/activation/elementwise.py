"""First-order activation autograd saving derivatives or exact ReLU6 masks."""

from deepopt.kernel_policy import kernel_forward, kernel_backward

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ...activation.state import SavedActivation
from ....backends.activation import executor
from ....quantization.activation_tensor import FP8Activation
from ....precision.gradient import capture_gradient
from ....quantization.gradient import enabled
from ...._execution import backward_finite_check_policy
from .._output import restore_fresh_output
from ....backends._shared.activation_codec import activation_layout_policy, saved_activation_layout


@kernel_forward
def activation_forward(ctx, x, geometry, options, scaling_state, state_training, backward_enabled, execution_info):
    ctx.activation_kernel_policy = activation_layout_policy()
    from ....backends._shared.activation_codec import activation_pointwise_tile, gelu_saved_state, silu_saved_state
    ctx.pointwise_tile = activation_pointwise_tile()
    compute = executor(x, options["kind"])
    ctx.gradient_slot = capture_gradient(x)
    ctx.gradient_check = backward_finite_check_policy()
    from deepopt._bf16_policy import resolve_bf16_policy

    ctx.bf16_policy = resolve_bf16_policy(options.get("bf16_policy"))
    if options["inplace"] and backward_enabled:
        base = getattr(x, "_base", None)
        if x.requires_grad and x.is_leaf or base is not None and base.requires_grad and base.is_leaf:
            raise RuntimeError("a leaf Variable that requires grad, or its view, is being used in an in-place operation")
    arguments = {key: value for key, value in options.items() if key != "bf16_policy"}
    if ctx.bf16_policy.storage_enabled and options["inplace"]:
        arguments["dtype"] = x.dtype
    arguments["save_mask"] = backward_enabled
    from ....backends._shared.producer import allocate_output_pack
    output_pack = allocate_output_pack(x, geometry, kind=options["kind"], dtype=options["dtype"],
        block_shape=options["block_shape"], compression={"enabled": options["packed"], "bits": options["bits"],
        "group_size": options["group_size"]}, enabled=backward_enabled and not options["inplace"], delayed=scaling_state is not None)
    arguments["output_pack"] = output_pack
    state = silu_saved_state() if options["kind"] == "silu" else gelu_saved_state()
    storage = ctx.bf16_policy.storage_enabled
    recompute = storage or state == "recompute" or (state == "auto" and options["dtype"] == torch.float8_e4m3fn)
    arguments["recompute"] = (recompute and options["kind"] in ("gelu", "silu")
                              and backward_enabled and not options["packed"] and not options["inplace"]
                              and scaling_state is None and not isinstance(x, FP8Activation)
                              and x.dtype == torch.bfloat16 and geometry.layout == "last"
                              and (x.is_contiguous() or options["kind"] == "silu" and x.ndim == 4
                                   and x.is_contiguous(memory_format=torch.channels_last)))
    # A copy owns only this logical invocation's values, preserves later alias
    # mutation semantics and does not retain a larger view backing allocation.
    # The CUDA executor handles arbitrary logical layouts via a contiguous copy.
    if storage and options["kind"] in ("silu", "gelu", "sigmoid", "tanh") and backward_enabled and not options["packed"]:
        if options["inplace"]:
            # Keep the write destination and the private backward snapshot
            # distinct; mark_dirty and the public result still name x.
            arguments["recompute_input"] = x.clone(memory_format=torch.contiguous_format)
        else:
            x = x.clone(memory_format=torch.contiguous_format)
        arguments["recompute"] = True
    if scaling_state is not None and not options["reuse"] and options["dtype"] in (torch.float8_e4m3fn, torch.float8_e5m2):
        result = scaling_state.activation_forward(x, geometry, compute, arguments, training=state_training)
    else:
        result = compute.forward(x, geometry, **arguments)
    output, inverse, saved, _, _, info = result
    if execution_info is not None:
        execution_info.update(info, input="fp8" if isinstance(x, FP8Activation) else str(x.dtype),
                              input_storage_dtype=str(getattr(x, "_data", x).dtype), output_storage_dtype=str(output.dtype),
                              output="fp8" if inverse is not None else str(output.dtype),
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
        return FP8Activation(output, inverse, geometry.shape, options["block_shape"], geometry.layout, logical,
                             linear_packed=output_pack if info.get("producer_pack_ready", info.get("producer_pack_fused")) else None)
    return restore_fresh_output(output, geometry.restore)


@once_differentiable
@saved_activation_layout
@kernel_backward
def activation_backward(ctx, gradient):
    saved = SavedActivation.restore(ctx.saved_tensors, ctx.saved_metadata)
    compute = executor(gradient, ctx.kind)
    slot = ctx.gradient_slot
    if enabled(slot):
        with slot.output(ctx.geometry.shape, ctx.input_dtype, check_finite=ctx.gradient_check) as output:
            dx = compute.backward(gradient, saved, ctx.geometry, dtype=ctx.input_dtype, gradient_output=output,
                                  pointwise_tile=ctx.pointwise_tile)
    else:
        dx = ctx.geometry.restore(compute.backward(gradient, saved, ctx.geometry, dtype=ctx.input_dtype,
                                                   pointwise_tile=ctx.pointwise_tile))
    return dx, None, None, None, None, None, None



class ActivationFunction(Function):
    """Eager autograd boundary over shared explicit saved-state executors."""

    forward = staticmethod(activation_forward)
    backward = staticmethod(activation_backward)
