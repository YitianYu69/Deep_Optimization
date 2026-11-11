"""LayerNorm oracle with pre-affine normalized-activation packing."""

import torch
from torch._subclasses.fake_tensor import FakeTensor

from deepopt.opt.nn.normalization.layer_state import LayerNormState
from deepopt.opt.nn.normalization.recenter import record_center_shift, corrected_xhat
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_tensor import FP8Activation
from deepopt.opt.quantization.quantizer import quantize_fp8


def dense(value):
    if isinstance(value, FP8Activation):
        return value.dequantize(torch.float32)
    return value if value.dtype == torch.float64 else value.float()


def forward(x, geometry, *, weight, bias, dtype, block_shape, compression, check_finite,
            scale=None, observe=False, capture_scale=False, center_shift=None):
    fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    matrix = geometry.matrix(dense(x))
    fake = isinstance(getattr(x, "_data", x), FakeTensor) or x.device.type == "meta"
    mean, inverse = torch.empty(geometry.rows, dtype=matrix.dtype, device=x.device), None
    packed = snapshot = observations = out_inverse = None
    info = {"backend": "reference", "forward_launches": 0, "statistics_launches": 0,
            "scale_launches": 0, "compression_launches": 0, "workspace_dtype": None,
            "scale_mode": "provided" if scale is not None else "current" if fp8 else "none"}
    if matrix.numel():
        variance, mean = torch.var_mean(matrix, dim=1, correction=0)
        inverse = torch.rsqrt(variance + geometry.eps)
        h = (matrix - mean[:, None]) * inverse[:, None]
        if compression["enabled"]:
            if check_finite and not fake and (not torch.isfinite(h).all() or
                    center_shift is not None and not (torch.isfinite(inverse) & (inverse > 0)).all()):
                raise ValueError("LayerNorm xhat compression requires finite normalized values")
            group = min(compression["group_size"], geometry.elements)
            packed = quantize_pack_activation(h, group, (1, group), bits=compression["bits"])
            record_center_shift(packed, h, center_shift)
        values = h
        if weight is not None:
            values = values * dense(weight).reshape(1, -1)
        if bias is not None:
            values = values + dense(bias).reshape(1, -1)
        values = geometry.transport.matrix(geometry.restore(values))
    else:
        inverse = torch.empty_like(mean)
        values = torch.empty(geometry.transport.matrix_shape, dtype=matrix.dtype, device=x.device)
    rows, cols = geometry.transport.matrix_shape
    if fp8:
        if fake or not values.numel():
            shape = (1, 1) if block_shape is None else ((rows + block_shape[0] - 1) // block_shape[0],
                                                       (cols + block_shape[1] - 1) // block_shape[1])
            output = torch.empty_like(values, dtype=dtype)
            out_inverse = torch.ones(shape, device=x.device, dtype=torch.float32)
        elif scale is None:
            q = quantize_fp8(values.float(), fp8_dtype=dtype, block_shape=block_shape, check_finite=check_finite)
            output, out_inverse = q.data, q.scale_inv
        else:
            if check_finite and not fake and (not torch.isfinite(values).all() or not torch.isfinite(scale).all() or (scale <= 0).any()):
                raise ValueError("LayerNorm FP8 output requires finite values and positive finite scales")
            expanded = scale if block_shape is None else scale.repeat_interleave(block_shape[0], 0).repeat_interleave(block_shape[1], 1)[:rows, :cols]
            output = (values * expanded).clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max).to(dtype)
            out_inverse = scale.reciprocal().clone()
            snapshot = torch.stack((scale.flatten()[0], scale.flatten()[0].reciprocal())) if capture_scale else None
            observations = values.abs().amax().reshape(1) if observe else None
    else:
        output = values.to(dtype)
    info.update(recenter_launches=0, recenter_scratch_bytes=0,
                recenter_partial_launches=0, recenter_partial_tiles=0, recenter_finish_launches=0,
                recenter_state_bytes=0 if center_shift is None else center_shift.numel() * center_shift.element_size())
    return output, out_inverse, LayerNormState(mean, inverse, packed), snapshot, observations, info


def backward(gradient, weight, geometry, *, packed, x, mean, inverse, need_dx, need_dw, need_db,
             input_dtype, gradient_output=None, execution_info=None, center_shift=None):
    dy = geometry.matrix(dense(gradient))
    h = None
    if dy.numel() and (need_dx or need_dw):
        h = (dequantize_unpack_activation(packed, dtype=torch.float32) if packed is not None else
             (geometry.matrix(dense(x)) - mean[:, None]) * inverse[:, None])
        h = corrected_xhat(h, inverse, center_shift)
    dw = ((dy * h).sum(0).reshape(geometry.normalized_shape) if dy.numel() else
          torch.zeros(geometry.normalized_shape, device=gradient.device, dtype=dy.dtype)) if need_dw else None
    db = dy.sum(0).reshape(geometry.normalized_shape) if need_db else None
    dx = None
    if need_dx:
        if dy.numel():
            u = dy * dense(weight).reshape(1, -1) if weight is not None else dy
            dx = (u - u.mean(1, keepdim=True) - h * (u * h).mean(1, keepdim=True)) * inverse[:, None]
        else:
            dx = torch.empty_like(dy)
        dx = geometry.restore(dx)
        dx = gradient_output.convert(dx) if gradient_output is not None else dx.to(input_dtype)
    if execution_info is not None:
        execution_info.update(input_gradient_launches=0, backward_reduction_launches=0, affine_gradient_launches=0,
                              backward_workspace_dtype=None)
    return dx, dw, db
