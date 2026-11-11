"""Plan-driven transposed compute with independent activation compression."""

from contextlib import nullcontext

import torch
from torch.nn import functional as F

from .geometry import ConvTransposeGeometry



def convolution_execution(geometry, plan):
    """Describe executor eligibility without reading tensor values."""
    if "fp8" in plan:
        return {"forward": "native", "backward": "native", "pointwise": False,
                "compression": plan["activation_compression"]["enabled"], "fallback_reason": None}
    training, blocks = plan["fp8_training"], plan["block_shapes"]
    pointwise = geometry.pointwise and all(block is None for block in blocks.values())
    backend = training["backend"]
    requested = backend if training["backward_backend"] == "inherit" else training["backward_backend"]
    mode = "pointwise_gemm" if pointwise else "implicit_fp8"
    if backend == "reference":
        mode = "reference"
    elif any(block is not None for block in blocks.values()):
        mode = "implicit_blockwise_fp8"
    reason = None
    backward = requested
    if not pointwise and requested in ("native_fast", "native_precise"):
        backward = "triton"
        reason = "Native scaled GEMM applies to pointwise convolution; spatial gradients use Triton"
    return {"forward": mode, "backward": backward, "pointwise": pointwise,
            "compression": training["activation_compression"]["enabled"], "fallback_reason": reason}


def _normalize(rank, x, weight, stride, padding, output_padding, groups, dilation):
    unbatched = x.ndim == rank + 1
    if unbatched:
        from ...quantization.activation_tensor import FP8Activation

        x = (FP8Activation(x._data, x._scale_inv, (1, *x.shape), x._block_shape, "channels", x.dtype)
             if isinstance(x, FP8Activation) else x.unsqueeze(0))
    geometry = ConvTransposeGeometry.from_tensors(x, weight, stride, padding, output_padding, groups, dilation)
    return x, geometry, unbatched


def convolution_transpose(rank, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1,
                *, plan=None, weight_cache=None, scaling_state=None, state_training=True, execution_info=None, output_slot=None):
    from ..functional import _prepare_linear_plan, _prepare_fp8_plan
    from ..._autocast import _get_activation_compression, _get_bf16_compression, _get_fp8_plan

    native = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[rank - 1]
    from ...quantization.activation_tensor import FP8Activation, dequantize_activation

    if isinstance(input, FP8Activation) and input._activation_layout != "channels":
        input = dequantize_activation(input)
    explicit_native = isinstance(plan, dict) and "fp8" in plan
    compression = _get_activation_compression(input.device.type) if isinstance(input, torch.Tensor) else None
    active = _get_fp8_plan(input.device.type) if isinstance(input, torch.Tensor) else None
    if explicit_native:
        checked = _prepare_linear_plan(plan)
        compression, fp8 = checked["activation_compression"], False
    elif active is False or active is None and plan is None:
        checked, fp8 = None, False
        compression = (_get_bf16_compression(input) if isinstance(input, torch.Tensor) else None)
    else:
        if active is not None:
            if compression is None and isinstance(plan, dict):
                compression = plan.get("fp8_training", {}).get("activation_compression")
            plan = active
        checked = _prepare_fp8_plan(plan)
        if compression is not None and checked["fp8_training"]["activation_compression"] != compression:
            checked = _prepare_fp8_plan({
                "fp8_training": {**checked["fp8_training"], "activation_compression": compression},
                "block_shapes": dict(checked["block_shapes"]),
            })
        compression, fp8 = checked["fp8_training"]["activation_compression"], True
    if fp8 and any(isinstance(tensor, torch.Tensor) and tensor.is_complex()
                   for tensor in (input, weight, bias)):
        if execution_info is not None:
            execution_info.clear()
            execution_info.update(forward="native", backward="native", pointwise=False,
                                  compression=False, fallback_reason="FP8 convolution requires real operands")
        return native(input, weight, bias, stride, padding, output_padding, groups, dilation)
    if not fp8:
        input = dequantize_activation(input)
        if (compression is None or not compression["enabled"] or not torch.is_grad_enabled()
                or not isinstance(weight, torch.Tensor) or not weight.requires_grad
                or not isinstance(input, torch.Tensor) or not input.numel()
                or input.dtype not in (torch.float32, torch.bfloat16, torch.float16)
                or weight.dtype not in (torch.float32, torch.bfloat16, torch.float16)):
            if execution_info is not None:
                execution_info.clear()
                execution_info.update(forward="native", backward="native", pointwise=False,
                                      compression=False, fallback_reason=None)
            return native(input, weight, bias, stride, padding, output_padding, groups, dilation)
    tensors = (("input", input), ("weight", weight)) + ((("bias", bias),) if bias is not None else ())
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or tensor.is_nested or tensor.layout != torch.strided:
            raise TypeError(f"convolution {name} must be a dense tensor")
        if tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError(f"convolution {name} must be FP32, BF16 or FP16")
        if tensor.device != input.device:
            raise ValueError("convolution tensors must share a device")
    if weight.ndim != rank + 2 or input.ndim not in (rank + 1, rank + 2):
        raise ValueError(f"conv{rank}d input and weight have incorrect ranks")
    if bias is not None and tuple(bias.shape) != (weight.shape[1] * groups,):
        raise ValueError("convolution bias must have shape (out_channels,)")
    if input.device.type not in ("cpu", "cuda"):
        raise ValueError("Optimized convolution supports CPU and CUDA")
    if not input.numel():
        return native(input, weight, bias, stride, padding, output_padding, groups, dilation)
    x, geometry, unbatched = _normalize(rank, input, weight, stride, padding, output_padding, groups, dilation)
    if execution_info is not None:
        execution_info.clear()
        execution_info.update(convolution_execution(geometry, checked if fp8 else {
            "fp8": False, "activation_compression": compression}))
    from ..autograd.convolution_transpose.native import _NativeConvTransposeFunction

    if not fp8:
        output = _NativeConvTransposeFunction.apply(x, weight, bias, geometry, compression)
    else:
        training = checked["fp8_training"]
        if training["backend"] == "triton" and input.device.type != "cuda":
            raise ValueError("The Triton convolution backend requires CUDA")
        if training["scaling_recipe"] == "delayed":
            if scaling_state is None:
                raise ValueError("Delayed scaling requires a prepared convolution adapter or FX call site")
            if scaling_state.recipe_signature != checked.delayed_signature:
                raise ValueError("Delayed-scaling state does not match the active convolution recipe")
        else:
            scaling_state = None
        from ..autograd.convolution_transpose.fp8 import _FP8ConvTransposeFunction

        scope = nullcontext()
        if scaling_state is not None:
            from ...scaling.delayed import fp8_scaling_scope

            scope = fp8_scaling_scope()
        out_dtype = training["out_dtype"]
        if out_dtype is None:
            out_dtype = (torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type)
                         else x.dtype)
        if unbatched:
            # The public unbatched view has a different logical FP8 shape.
            output_slot = None
        with scope:
            output = _FP8ConvTransposeFunction.apply(
                x, weight, bias, geometry, checked, out_dtype,
                torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, bias) if t is not None),
                weight_cache, scaling_state, state_training and torch.is_grad_enabled(), output_slot)
    if execution_info is not None:
        execution_info["output"] = "fp8" if isinstance(output, FP8Activation) else str(output.dtype)
        if output_slot is not None:
            execution_info["output_conversion"] = dict(output_slot.last_execution or {})
        elif unbatched and fp8:
            execution_info["output_boundary"] = "unbatched output uses native logical layout"
    return output.squeeze(0) if unbatched else output
