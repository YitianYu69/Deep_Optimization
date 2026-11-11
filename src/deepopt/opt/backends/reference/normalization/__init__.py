"""FP32 normalization oracle, packing normalized values before affine."""

import torch

from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_tensor import FP8Activation
from deepopt.opt.nn.normalization.geometry import output_strides
from deepopt.opt.nn.normalization.recenter import record_center_shift, corrected_xhat


def dense_input(x):
    return x.dequantize(torch.float32) if isinstance(x, FP8Activation) else x.float()


def forward(x, weight, bias, geometry, compression, *, running_mean=None, running_var=None, momentum=None,
            out_dtype=None, output_layout=None, center_shift=None, execution_info=None):
    values = geometry.matrix(dense_input(x))
    if geometry.batch_stats:
        variance, mean = torch.var_mean(values, dim=1, correction=0)
        inverse = torch.rsqrt(variance + geometry.eps)
        if running_mean is not None and momentum is not None:
            with torch.no_grad():
                running_mean.copy_(running_mean.float() + momentum * (mean - running_mean.float()))
                unbiased = variance * (geometry.elements / (geometry.elements - 1))
                running_var.copy_(running_var.float() + momentum * (unbiased - running_var.float()))
    else:
        mean = running_mean.float().clone()
        inverse = torch.rsqrt(running_var.float() + geometry.eps)
    xhat = (values - mean[:, None]) * inverse[:, None]
    output = geometry.restore(xhat)
    affine_shape = (1, geometry.channels) + (1,) * (len(geometry.shape) - 2)
    if weight is not None:
        output = output * weight.float().reshape(affine_shape)
    if bias is not None:
        output = output + bias.float().reshape(affine_shape)
    packed = (quantize_pack_activation(xhat, compression["group_size"], (1, compression["group_size"]), bits=compression["bits"])
              if compression["enabled"] else None)
    record_center_shift(packed, xhat, center_shift)
    dtype = out_dtype or (x.dtype if x.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) else torch.bfloat16)
    result = torch.empty_strided(geometry.shape, output_strides(x, output_layout), dtype=dtype, device=x.device)
    result.copy_(output)
    if execution_info is not None:
        execution_info.update(recenter_launches=0, recenter_scratch_bytes=0,
                              recenter_partial_launches=0, recenter_partial_tiles=0, recenter_finish_launches=0,
                              recenter_state_bytes=0 if center_shift is None else center_shift.numel() * 4)
    return result, mean, inverse, packed


def backward(gradient, weight, geometry, *, packed=None, x=None, mean=None, inverse=None,
             need_dx=True, need_dw=True, need_db=True, input_dtype=None, gradient_output=None, center_shift=None):
    grad = dense_input(gradient)
    h = None
    if need_dw or need_dx and geometry.batch_stats:
        h = (dequantize_unpack_activation(packed, torch.float32) if packed is not None else
             (geometry.matrix(dense_input(x)) - mean[:, None]) * inverse[:, None])
        h = corrected_xhat(h, inverse, center_shift)
    spatial_dims = (0, *range(2, len(geometry.shape)))
    dw = (grad * geometry.restore(h)).sum(spatial_dims) if need_dw else None
    db = grad.sum(spatial_dims) if need_db else None
    dx = None
    if need_dx:
        if weight is not None:
            grad = grad * weight.float().reshape((1, geometry.channels) + (1,) * (len(geometry.shape) - 2))
        u = geometry.matrix(grad)
        if geometry.batch_stats:
            u = u - u.mean(1, keepdim=True) - h * (u * h).mean(1, keepdim=True)
        dx = geometry.restore(u * inverse[:, None])
        if gradient_output is not None:
            dx = gradient_output.convert(dx)
    return dx, dw, db
