"""FP32 normalization oracle, packing normalized values before affine."""

import torch
from torch._subclasses.fake_tensor import FakeTensor

from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation
from deepopt.opt.quantization.activation_tensor import FP8Activation
from deepopt.opt.nn.normalization.geometry import output_strides
from deepopt.opt.nn.normalization.recenter import record_center_shift, corrected_xhat


def dense_input(x):
    return x.dequantize(torch.float32) if isinstance(x, FP8Activation) else x.float()


def forward(x, weight, bias, geometry, compression, *, running_mean=None, running_var=None, momentum=None,
            out_dtype=None, output_layout=None, center_shift=None, execution_info=None,
            activation=None, preactivation=None, output=None):
    destination = output
    if destination is not None:
        from deepopt.opt.backends._shared.normalization_output import validate_normalization_output

        validate_normalization_output(destination, x, weight, bias, geometry, dtype=out_dtype or x.dtype,
            output_layout=output_layout, activation=activation, preactivation=preactivation)
    if isinstance(getattr(x, "_data", x), FakeTensor) or x.device.type == "meta":
        mean = torch.empty(geometry.domains, device=x.device, dtype=torch.float32)
        inverse = torch.empty_like(mean)
        packed = (quantize_pack_activation(torch.empty(geometry.matrix_shape, device=x.device),
                    compression["group_size"], (1, compression["group_size"]), bits=compression["bits"])
                  if compression["enabled"] else None)
        dtype = out_dtype or (x.dtype if x.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) else torch.bfloat16)
        result = (destination if destination is not None else
                  torch.empty_strided(geometry.shape, output_strides(x, output_layout), dtype=dtype, device=x.device))
        if execution_info is not None:
            execution_info.update(recenter_launches=0, recenter_scratch_bytes=0, recenter_partial_launches=0,
                                  recenter_partial_tiles=0, recenter_finish_launches=0,
                                  recenter_state_bytes=0 if center_shift is None else center_shift.numel() * 4)
        return result, mean, inverse, packed
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
    result = (destination if destination is not None else
              torch.empty_strided(geometry.shape, output_strides(x, output_layout), dtype=dtype, device=x.device))
    from .activation import apply
    result.copy_(output)
    if activation is not None:
        result.copy_(apply(result, activation, preactivation))
    if execution_info is not None:
        execution_info.update(recenter_launches=0, recenter_scratch_bytes=0,
                              recenter_partial_launches=0, recenter_partial_tiles=0, recenter_finish_launches=0,
                              recenter_state_bytes=0 if center_shift is None else center_shift.numel() * 4)
    return result, mean, inverse, packed


def backward(gradient, weight, geometry, *, packed=None, x=None, mean=None, inverse=None,
             need_dx=True, need_dw=True, need_db=True, input_dtype=None, gradient_output=None, center_shift=None,
             execution_info=None, derivative_input=None, derivative_activation=None):
    if isinstance(getattr(gradient, "_data", gradient), FakeTensor) or gradient.device.type == "meta":
        dx = torch.empty(geometry.shape, device=gradient.device, dtype=input_dtype or gradient.dtype) if need_dx else None
        if dx is not None and gradient_output is not None:
            dx = gradient_output.convert(dx)
        dw = torch.empty(geometry.channels, device=gradient.device, dtype=torch.float32) if need_dw else None
        db = torch.empty(geometry.channels, device=gradient.device, dtype=torch.float32) if need_db else None
        return dx, dw, db
    from .activation import adjust
    gradient = adjust(gradient, derivative_input, derivative_activation)
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
        elif input_dtype is not None:
            dx = dx.to(input_dtype)
    return dx, dw, db
