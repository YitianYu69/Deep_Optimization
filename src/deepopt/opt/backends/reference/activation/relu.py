"""ReLU reference oracle and allocation-only FakeTensor/meta execution."""

import torch
from torch._subclasses.fake_tensor import FakeTensor

from deepopt.opt.quantization.activation_tensor import FP8Activation
from deepopt.opt.quantization.boolean import pack_boolean_mask, unpack_boolean_mask, PackedBooleanMask
from deepopt.opt.quantization.quantizer import quantize_fp8


def forward(x, geometry, *, dtype, block_shape, packed, save_mask, inplace, reuse,
            scale=None, observe=False, check_finite=False, capture_scale=False):
    rows, cols = geometry.matrix_shape
    is_fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    scale_shape = ((1, 1) if block_shape is None else
                   ((rows + block_shape[0] - 1) // block_shape[0], (cols + block_shape[1] - 1) // block_shape[1]))
    storage = x._data if isinstance(x, FP8Activation) else x
    fake = isinstance(storage, FakeTensor) or x.device.type == "meta"
    info = {"forward_launches": 0, "scale_launches": 0, "mask": ("packed" if packed else "bool") if save_mask else "none",
            "scale_mode": "reuse" if reuse else "provided" if scale is not None else "current" if is_fp8 else "none",
            "backend": "reference"}
    if fake or not x.numel():
        output = storage if inplace else torch.empty((rows, cols), dtype=dtype, device=x.device)
        inverse = (x._scale_inv if reuse else torch.empty(scale_shape, dtype=torch.float32, device=x.device)) if is_fp8 else None
        mask = (torch.empty(((x.numel() + 31) // 32,) if packed else (x.numel(),),
                            dtype=torch.int32 if packed else torch.bool, device=x.device) if save_mask else None)
        return output, inverse, mask, None, None, info
    dense = geometry.matrix(x.dequantize(torch.float32) if isinstance(x, FP8Activation) else
                            x if x.dtype == torch.float64 else x.float())
    # Preserve NaNs and signed zeros; threshold backward passes gradients for NaNs.
    values = torch.where(dense < 0, 0.0, dense)
    predicate = ~(dense <= 0)
    mask = (pack_boolean_mask(predicate).data if packed else predicate.reshape(-1).contiguous()) if save_mask else None
    inverse = snapshot = observations = None
    if reuse:
        # Geometry can change for tensorwise scales, so restore logical axes first.
        from deepopt.opt.quantization.activation_tensor import restore_activation

        raw = geometry.matrix(restore_activation(x._data, tuple(x.shape), x._activation_layout))
        output, inverse = torch.where(dense < 0, 0.0, raw.float()).to(dtype), x._scale_inv
    elif is_fp8 and scale is None:
        q = quantize_fp8(values.float(), fp8_dtype=dtype, block_shape=block_shape, check_finite=check_finite)
        output, inverse = q.data, q.scale_inv
    elif is_fp8:
        if check_finite and (not torch.isfinite(values).all() or not torch.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError("ReLU FP8 casting requires finite outputs and positive finite scales")
        snapshot = torch.stack((scale.reshape(-1)[0], scale.reshape(-1)[0].reciprocal())) if capture_scale else None
        inverse = scale.reciprocal().clone()
        expanded = (scale if block_shape is None else
                    scale.repeat_interleave(block_shape[0], 0).repeat_interleave(block_shape[1], 1)[:rows, :cols])
        maximum = torch.finfo(dtype).max
        output = (values * expanded).clamp(-maximum, maximum).to(dtype)
        observations = values.abs().amax().reshape(1) if observe else None
    else:
        output = values.to(dtype)
    if inplace:
        if isinstance(x, FP8Activation):
            x._data.copy_(output)
        else:
            x.copy_(geometry.restore(output))
        output = x._data if isinstance(x, FP8Activation) else x
    return output, inverse, mask, snapshot, observations, info


def backward(gradient, mask, geometry, *, packed, dtype, reuse=False, gradient_output=None):
    storage = gradient._data if isinstance(gradient, FP8Activation) else gradient
    if isinstance(storage, FakeTensor) or gradient.device.type == "meta" or not gradient.numel():
        result = torch.empty(geometry.matrix_shape, dtype=dtype, device=gradient.device)
        return gradient_output.convert(geometry.restore(result)) if gradient_output is not None else result
    predicate = (unpack_boolean_mask(PackedBooleanMask(mask, geometry.matrix_shape)) if packed else
                 mask.reshape(geometry.matrix_shape))
    if reuse:
        from deepopt.opt.quantization.activation_tensor import restore_activation

        values = geometry.matrix(restore_activation(gradient._data, tuple(gradient.shape), gradient._activation_layout)).float()
    else:
        values = geometry.matrix(gradient.dequantize(torch.float32) if isinstance(gradient, FP8Activation) else
                                 gradient if gradient.dtype == torch.float64 else gradient.float())
    result = torch.where(predicate, values, 0.0)
    if gradient_output is not None:
        return gradient_output.convert(geometry.restore(result))
    return result.to(dtype)
