"""LayerNorm argument, precision and independent xhat compression policies."""

from math import isfinite

import torch
from torch.nn import functional as F

from .layer_geometry import LayerNormGeometry, normalized_shape_tuple
from ...quantization.activation_tensor import FP8Activation


FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, *, plan=None,
               output_dtype=None, output_layout=None, output_scale=None, planned_layout=None,
               scaling_state=None, state_training=True, execution_info=None):
    from ..functional import _prepare_linear_plan
    from ..._autocast import _get_fp8_plan, _get_activation_compression
    from ..._execution import finite_checks_enabled

    if not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.is_nested:
        raise TypeError("LayerNorm input must be a dense tensor")
    if x.device.type not in ("cpu", "cuda", "meta") or x.dtype not in (*FLOAT_DTYPES, *FP8_DTYPES):
        raise TypeError("LayerNorm requires a floating-point CPU, CUDA or meta tensor")
    suffix = normalized_shape_tuple(normalized_shape)
    if type(eps) not in (int, float) or not isfinite(eps):
        raise ValueError("LayerNorm eps must be a finite number")
    active = _get_fp8_plan(x.device.type)
    compression = _get_activation_compression(x.device.type)
    if isinstance(plan, dict) and "fp8" in plan:
        selected = plan
    elif active is False:
        selected = {"fp8": False, "activation_compression": compression or {}}
    else:
        selected = active if active is not None else plan
        if selected is not None and compression is not None:
            selected = {**selected, "fp8_training": {**selected["fp8_training"], "activation_compression": compression}}
    checked = _prepare_linear_plan(selected or {"fp8": False, "activation_compression": compression or {}})
    training = checked.get("fp8_training")
    compression = (training or checked)["activation_compression"]
    native_dtype = x.dtype if x.dtype not in FP8_DTYPES else torch.bfloat16
    amp = x.device.type == "cuda" and native_dtype != torch.float64 and torch.is_autocast_enabled("cuda")
    logical_dtype = torch.float32 if amp and native_dtype != torch.float64 else native_dtype
    params = [v for v in (weight, bias) if v is not None]
    for value in params:
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.is_nested:
            raise TypeError("LayerNorm affine values must be dense tensors")
        if value.device != x.device or tuple(value.shape) != suffix:
            raise ValueError("LayerNorm affine values must share the input device and exact normalized_shape")
        if value.dtype not in FLOAT_DTYPES:
            raise TypeError("LayerNorm affine values require native floating-point dtypes")
    if not amp and (len({v.dtype for v in params}) > 1 or any(
        v.dtype != native_dtype and not (native_dtype in (torch.bfloat16, torch.float16) and v.dtype == torch.float32)
        for v in params
    )):
        raise RuntimeError("LayerNorm mixed dtype: affine values must match input dtype or supported FP32 mixed precision")
    configured = training["out_dtype"] if training else None
    dtype = output_dtype if output_dtype is not None else configured or (
        training["fp8_dtypes"]["activation"] if training and planned_layout is not None else logical_dtype)
    if dtype not in (*FLOAT_DTYPES, *FP8_DTYPES):
        raise ValueError("LayerNorm output_dtype must be a supported floating-point dtype")
    fp8 = dtype in FP8_DTYPES
    layout = (output_layout or planned_layout or (x._activation_layout if isinstance(x, FP8Activation) else "last")) if fp8 else "last"
    geometry = LayerNormGeometry(tuple(x.shape), suffix, eps, layout)
    blocks = checked["block_shapes"]["activation"] if training else None
    if fp8 and blocks is not None and x.device.type == "cuda" and any(v & (v - 1) for v in blocks):
        raise ValueError("CUDA LayerNorm FP8 scale blocks must have power-of-two dimensions")
    if output_scale is not None:
        rows, cols = geometry.transport.matrix_shape
        scale_shape = (1, 1) if blocks is None else ((rows + blocks[0] - 1) // blocks[0], (cols + blocks[1] - 1) // blocks[1])
        if not fp8 or not isinstance(output_scale, torch.Tensor) or output_scale.dtype != torch.float32:
            raise TypeError("LayerNorm output_scale requires FP8 output and FP32 multipliers")
        if output_scale.device != x.device or tuple(output_scale.shape) != scale_shape or not output_scale.is_contiguous():
            raise ValueError("LayerNorm output_scale must match the output scale grid and device")
        if output_scale.requires_grad:
            raise ValueError("LayerNorm output_scale is non-differentiable metadata")
    if fp8 and training and training["scaling_recipe"] == "delayed" and output_scale is None and scaling_state is None:
        raise ValueError("Delayed LayerNorm FP8 output requires a prepared adapter or explicit output_scale")
    if fp8 and scaling_state is not None and output_scale is None and dtype != scaling_state.dtypes["activation"]:
        raise ValueError("LayerNorm output format must match its prepared delayed scales; use matching settings or explicit output_scale")
    if not training and not compression["enabled"] and not isinstance(x, FP8Activation) and x.dtype not in FP8_DTYPES and output_dtype is None and output_scale is None:
        return F.layer_norm(x, suffix, weight, bias, eps)
    from ..autograd.normalization.layer import LayerNormFunction

    backward = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, weight, bias))
    need_xhat = backward and (x.requires_grad or weight is not None and weight.requires_grad)
    options = {"dtype": dtype, "logical_dtype": logical_dtype, "block_shape": blocks,
               "backward_enabled": backward,
               "compression": compression, "scale": output_scale,
               "backend": training["backend"] if training else "triton" if x.device.type == "cuda" else "reference",
               "check_finite": finite_checks_enabled() and (fp8 or compression["enabled"] and need_xhat)}
    return LayerNormFunction.apply(x, weight, bias, geometry, options, scaling_state if output_scale is None else None,
                                   state_training and torch.is_grad_enabled(), execution_info)
