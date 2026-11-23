"""LayerNorm autograd saving packed xhat or original representation/statistics."""

from deepopt.kernel_policy import kernel_forward, kernel_backward

from importlib import import_module
from contextlib import nullcontext

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch._subclasses.fake_tensor import FakeTensor

from ....precision.gradient import capture_gradient
from ....quantization.gradient import enabled
from ....quantization.activation import PackedActivation
from ....quantization.activation_tensor import FP8Activation
from ...._execution import backward_finite_check_policy
from ....backends._shared.normalization import normalization_variant, normalization_tuning, saved_normalization_variant
from .._output import restore_fresh_output
from deepopt.kernel_policy import kernel_backend


def executor(x, backend):
    storage = x._data if isinstance(x, FP8Activation) else x
    family = (kernel_backend("normalization", backend=backend)
              if backend != "reference" and x.device.type == "cuda" and storage.dtype != torch.float64
              and not isinstance(storage, FakeTensor) else "reference")
    if family != "reference":
        from ....backends.dispatch import backend_module
        return backend_module("normalization", "normalization.layer", backend=family)
    return import_module(f"deepopt.opt.backends.{family}.normalization.layer")


@kernel_forward
def layer_norm_forward(ctx, x, weight, bias, geometry, options, scaling_state, state_training, execution_info):
    from ...normalization.policy import capture_normalization_policy, normalization_policy_receipt

    ctx.bf16_policy = options.get("bf16_policy") or capture_normalization_policy(x)
    ctx.normalization_variant = normalization_variant()
    ctx.normalization_tuning = normalization_tuning()
    ctx.gradient_slot = capture_gradient(x)
    ctx.gradient_check = backward_finite_check_policy()
    need_dx, need_dw, _ = (options["backward_enabled"] and value for value in ctx.needs_input_grad[:3])
    need_h = (need_dx or need_dw) and x.numel() > 0
    compression = dict(options["compression"])
    compression["enabled"] &= need_h
    if x.numel():
        compression["group_size"] = min(compression["group_size"], geometry.elements)
    # Compressed LayerNorm always restores both moments. The generic
    # recenter option remains independent for BatchNorm and GroupNorm.
    center_shift = (torch.empty(geometry.rows, device=x.device, dtype=torch.float32)
                    if compression["enabled"] else None)
    moment_gain = torch.empty_like(center_shift) if center_shift is not None else None
    ctx.backend = options["backend"]
    ctx.execution_info = execution_info
    compute = executor(x, ctx.backend)
    arguments = {**{k: v for k, v in options.items() if k not in ("logical_dtype", "backend", "backward_enabled", "bf16_policy")},
                 "compression": compression, "weight": weight, "bias": bias,
                 "center_shift": center_shift, "moment_gain": moment_gain}
    from ....backends._shared.producer import allocate_output_pack
    output_pack = allocate_output_pack(x, geometry, kind="layer_norm", dtype=options["dtype"],
        block_shape=options["block_shape"], compression=options["compression"],
        enabled=options["backward_enabled"], delayed=scaling_state is not None)
    arguments["output_pack"] = output_pack
    if scaling_state is not None and options["dtype"] in (torch.float8_e4m3fn, torch.float8_e5m2):
        result = scaling_state.activation_forward(x, geometry, compute, arguments, training=state_training)
    else:
        result = compute.forward(x, geometry, **arguments)
    output, out_inverse, state, _, _, info = result
    original = original_scale = minimum = None
    ctx.input_metadata = None
    if need_h:
        if state.packed is not None:
            original, original_scale, minimum = state.packed.data, state.packed.step, state.packed.minimum
        elif isinstance(x, FP8Activation):
            original, original_scale = x._data, x._scale_inv
            ctx.input_metadata = (x._logical_shape, x._block_shape, x._activation_layout, x._logical_dtype, x._logical_strides)
        else:
            original = x
    saved = (original, original_scale, minimum,
             state.inverse if need_dx or center_shift is not None or need_h and state.packed is None else None,
             state.mean if need_h and state.packed is None else None, weight if need_dx else None,
             center_shift, moment_gain)
    ctx.save_for_backward(*saved)
    ctx.geometry, ctx.compression, ctx.input_dtype = geometry, compression, x.dtype
    ctx.weight_dtype, ctx.bias_dtype = weight.dtype if weight is not None else None, bias.dtype if bias is not None else None
    if execution_info is not None:
        execution_info.update(info, **normalization_policy_receipt(ctx.bf16_policy, x, options["dtype"], ctx.backend))
        execution_info.update(compression_target="x_hat", compression=compression["enabled"],
                              norm_xhat_recenter=center_shift is not None,
                              layer_norm_xhat_moment_correction=moment_gain is not None,
                              moment_state_bytes=moment_gain.numel() * moment_gain.element_size() if moment_gain is not None else 0,
                              recenter_state_bytes=center_shift.numel() * center_shift.element_size() if center_shift is not None else 0,
                              recenter_rstd_bytes=state.inverse.numel() * state.inverse.element_size() if center_shift is not None else 0,
                              recenter_extra_rstd_bytes=state.inverse.numel() * state.inverse.element_size() if center_shift is not None and not need_dx else 0,
                              input="fp8" if isinstance(x, FP8Activation) else str(x.dtype),
                              input_storage_dtype=str(getattr(x, "_data", x).dtype), output_storage_dtype=str(output.dtype),
                              output="fp8" if out_inverse is not None else str(options["dtype"]),
                              realized_compression_group_size=compression["group_size"] if compression["enabled"] else None,
                              saved_state_bytes=sum(t.numel() * t.element_size() for t in saved if t is not None))
    if out_inverse is not None:
        return FP8Activation(output, out_inverse, geometry.shape, options["block_shape"], geometry.layout,
                             options["logical_dtype"], linear_packed=output_pack if info.get("producer_pack_ready", info.get("producer_pack_fused")) else None)
    return restore_fresh_output(output, geometry.transport.restore)


@once_differentiable
@saved_normalization_variant
@kernel_backward
def layer_norm_backward(ctx, gradient):
    data, step, minimum, inverse, mean, weight, center_shift, moment_gain = ctx.saved_tensors
    packed = original = None
    if minimum is not None:
        group = ctx.compression["group_size"]
        packed = PackedActivation(data, step, minimum, ctx.geometry.matrix_shape, group, (1, group), ctx.compression["bits"])
    elif data is not None:
        original = data if ctx.input_metadata is None else FP8Activation(data, step, *ctx.input_metadata)
    args = dict(packed=packed, x=original, mean=mean, inverse=inverse,
                center_shift=center_shift, moment_gain=moment_gain,
                need_dx=ctx.needs_input_grad[0], need_dw=ctx.needs_input_grad[1], need_db=ctx.needs_input_grad[2],
                input_dtype=ctx.input_dtype, execution_info=ctx.execution_info)
    with torch.autocast(gradient.device.type, enabled=False) if gradient.device.type != "meta" else nullcontext():
        if ctx.needs_input_grad[0] and enabled(ctx.gradient_slot):
            with ctx.gradient_slot.output(ctx.geometry.shape, ctx.input_dtype, check_finite=ctx.gradient_check) as out:
                dx, dw, db = executor(gradient, ctx.backend).backward(gradient, weight, ctx.geometry, gradient_output=out, **args)
        else:
            dx, dw, db = executor(gradient, ctx.backend).backward(gradient, weight, ctx.geometry, **args)
    return dx, dw.to(ctx.weight_dtype) if dw is not None else None, db.to(ctx.bias_dtype) if db is not None else None, None, None, None, None, None



class LayerNormFunction(Function):
    """Eager autograd boundary over shared explicit saved-state executors."""

    forward = staticmethod(layer_norm_forward)
    backward = staticmethod(layer_norm_backward)
