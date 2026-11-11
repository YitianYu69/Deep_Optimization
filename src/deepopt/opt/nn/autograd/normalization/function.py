"""Shared BatchNorm2d/GroupNorm autograd and saved-xhat lifetime."""

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ....backends.normalization import executor
from ....quantization.activation import PackedActivation
from ....quantization.activation_tensor import FP8Activation, activation_matrix
from ....quantization.quantizer import quantize_fp8
from ....precision.gradient import capture_gradient
from ....quantization.gradient import enabled
from ...._execution import backward_finite_check_policy, finite_checks_enabled


class NormalizationFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, geometry, plan, output_layout,
                scaling_state, state_training, momentum, backward_enabled, execution_info):
        training = plan.get("fp8_training")
        ctx.gradient_slot = capture_gradient(x)
        ctx.gradient_check = backward_finite_check_policy()
        compression = dict(training["activation_compression"] if training else plan["activation_compression"])
        need_dx, need_dw = ctx.needs_input_grad[:2]
        need_h = backward_enabled and (need_dw or need_dx and geometry.batch_stats)
        compression["enabled"] &= need_h
        if compression["enabled"]:
            compression["group_size"] = min(compression["group_size"], geometry.elements)
        center_shift = (torch.empty(geometry.domains, device=x.device, dtype=torch.float32)
                        if compression["enabled"] and compression.get("norm_xhat_recenter", False) else None)
        backend = training["backend"] if training else "triton" if x.device.type == "cuda" else "reference"
        compute = executor(x, backend)
        logical_dtype = x.dtype if x.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) else torch.bfloat16
        if geometry.kind == "group" and x.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
            logical_dtype = torch.float32
        if training is not None and training["out_dtype"] is not None:
            logical_dtype = training["out_dtype"]
        output, mean, inverse, packed = compute.forward(
            x, weight, bias, geometry, compression, running_mean=running_mean, running_var=running_var,
            momentum=momentum, out_dtype=torch.float32 if output_layout is not None else logical_dtype,
            output_layout=output_layout, center_shift=center_shift, execution_info=execution_info)
        if (center_shift is not None and finite_checks_enabled()
                and not isinstance(getattr(x, "_data", x), FakeTensor)):
            if not (torch.isfinite(inverse) & (inverse > 0) & torch.isfinite(center_shift)).all():
                raise ValueError("Normalization xhat correction requires positive finite rstd and finite codec error")
        if execution_info is not None:
            execution_info.update(norm_xhat_recenter=center_shift is not None,
                                  recenter_rstd_bytes=inverse.numel() * inverse.element_size() if center_shift is not None else 0,
                                  recenter_extra_rstd_bytes=0, saved_state_bytes=0)
        if backward_enabled:
            original, original_scale = None, None
            ctx.input_metadata = None
            if need_h and packed is None:
                if isinstance(x, FP8Activation):
                    original, original_scale = x._data, x._scale_inv
                    ctx.input_metadata = (x._logical_shape, x._block_shape, x._activation_layout,
                                          x._logical_dtype, x._logical_strides)
                else:
                    original = x
            saved = (
                packed.data if packed is not None else original,
                packed.step if packed is not None else original_scale,
                packed.minimum if packed is not None else None,
                inverse if need_h or need_dx else None,
                mean if need_h and packed is None else None,
                weight if need_dx else None,
                center_shift,
            )
            ctx.save_for_backward(*saved)
            if execution_info is not None:
                execution_info["saved_state_bytes"] = sum(t.numel() * t.element_size() for t in saved if t is not None)
            ctx.geometry, ctx.compression, ctx.backend = geometry, compression, backend
            ctx.input_dtype = x.dtype
            ctx.weight_dtype = weight.dtype if weight is not None else None
            ctx.bias_dtype = bias.dtype if bias is not None else None
        if output_layout is not None:
            matrix = activation_matrix(output, output_layout)
            q = (scaling_state.quantize(matrix, "activation", training=state_training)[0]
                 if scaling_state is not None else
                 quantize_fp8(matrix, fp8_dtype=training["fp8_dtypes"]["activation"],
                              block_shape=plan["block_shapes"]["activation"]))
            return FP8Activation(q.data, q.scale_inv, geometry.shape, q.block_shape, output_layout,
                                 logical_dtype, output.stride())
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        data, step, minimum, inverse, mean, weight, center_shift = ctx.saved_tensors
        original = packed = None
        if ctx.compression["enabled"]:
            packed = PackedActivation(data, step, minimum, ctx.geometry.matrix_shape,
                                      ctx.compression["group_size"], (1, ctx.compression["group_size"]),
                                      ctx.compression["bits"])
        elif data is not None:
            original = (data if ctx.input_metadata is None else FP8Activation(data, step, *ctx.input_metadata))
        compute = executor(gradient, ctx.backend)
        with torch.autocast(gradient.device.type, enabled=False):
            arguments = dict(packed=packed, x=original, mean=mean, inverse=inverse, center_shift=center_shift,
                             need_dx=ctx.needs_input_grad[0], need_dw=ctx.needs_input_grad[1], need_db=ctx.needs_input_grad[2])
            if ctx.needs_input_grad[0] and enabled(ctx.gradient_slot):
                with ctx.gradient_slot.output(ctx.geometry.shape, ctx.input_dtype, check_finite=ctx.gradient_check) as output:
                    dx, dw, db = compute.backward(gradient, weight, ctx.geometry, input_dtype=torch.float32,
                                                  gradient_output=output, **arguments)
            else:
                dx, dw, db = compute.backward(gradient, weight, ctx.geometry, input_dtype=ctx.input_dtype, **arguments)
        return (dx if isinstance(dx, FP8Activation) else dx.to(ctx.input_dtype) if dx is not None else None,
                dw.to(ctx.weight_dtype) if dw is not None else None,
                db.to(ctx.bias_dtype) if db is not None else None,
                None, None, None, None, None, None, None, None, None, None)
