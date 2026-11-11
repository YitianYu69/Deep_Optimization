"""Reference activation execution; saved inputs are preprocessed coefficients."""

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .math import evaluate
from deepopt.opt.nn.activation.state import SavedActivation
from deepopt.opt.quantization.activation import (
    PackedActivation, _dimensions, quantize_pack_activation, dequantize_unpack_activation,
)
from deepopt.opt.quantization.activation_tensor import FP8Activation
from deepopt.opt.quantization.boolean import pack_boolean_mask, unpack_boolean_mask, PackedBooleanMask
from deepopt.opt.quantization.quantizer import quantize_fp8


def allocate_saved(geometry, device, *, packed, bits, group_size, block_shape, mask, dtype=torch.float32):
    count = geometry.matrix_shape[0] * geometry.matrix_shape[1]
    if mask:
        return SavedActivation(torch.empty(((count + 31) // 32,) if packed else (count,), device=device,
                                           dtype=torch.int32 if packed else torch.bool), mask=True, packed=packed)
    if not packed:
        return SavedActivation(torch.empty(geometry.matrix_shape, device=device, dtype=dtype))
    groups, _, words = _dimensions(geometry.matrix_shape, group_size, block_shape, bits)
    return SavedActivation(torch.empty((groups, words), dtype=torch.int32, device=device),
                           torch.empty(groups, dtype=torch.float32, device=device),
                           torch.empty(groups, dtype=torch.float32, device=device),
                           group_size, block_shape, bits, packed=True)


def forward(x, geometry, *, kind, approximate, dtype, block_shape, packed, bits, group_size,
            save_mask, inplace, reuse=False, scale=None, observe=False, check_finite=False, capture_scale=False):
    rows, cols = geometry.matrix_shape
    fp8 = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    storage = x._data if isinstance(x, FP8Activation) else x
    fake = isinstance(storage, FakeTensor) or x.device.type == "meta"
    info = {"backend": "reference", "activation": kind, "forward_launches": 0, "scale_launches": 0,
            "coefficient_launches": 0, "backward_launches": 0,
            "scale_mode": "provided" if scale is not None else "current" if fp8 else "none",
            "saved_state": ("mask" if kind == "relu6" else "derivative") if save_mask else "none"}
    if fake or not x.numel():
        output = storage if inplace else torch.empty((rows, cols), dtype=dtype, device=x.device)
        shape = (1, 1) if block_shape is None else ((rows + block_shape[0] - 1) // block_shape[0],
                                                   (cols + block_shape[1] - 1) // block_shape[1])
        inverse = torch.empty(shape, dtype=torch.float32, device=x.device) if fp8 else None
        saved = allocate_saved(geometry, x.device, packed=packed, bits=bits, group_size=group_size,
                               block_shape=block_shape, mask=kind == "relu6",
                               dtype=torch.float64 if x.dtype == torch.float64 else torch.float32) if save_mask else None
        return output, inverse, saved, None, None, info
    dense = geometry.matrix(x.dequantize(torch.float32) if isinstance(x, FP8Activation) else
                            x if x.dtype == torch.float64 else x.float())
    values, derivative = evaluate(dense, kind, approximate)
    saved = None
    if save_mask:
        if kind == "relu6":
            saved = SavedActivation(pack_boolean_mask(derivative).data if packed else derivative.flatten(),
                                    mask=True, packed=packed)
        elif packed:
            if check_finite and not torch.isfinite(derivative).all():
                raise ValueError("Activation derivative compression requires finite coefficients")
            q = quantize_pack_activation(derivative, group_size, block_shape, bits=bits)
            saved = SavedActivation(q.data, q.step, q.minimum, group_size, block_shape, bits, packed=True)
        else:
            saved = SavedActivation(derivative)
    inverse = snapshot = observations = None
    if reuse:
        inverse = x._scale_inv
        expanded = inverse if block_shape is None else inverse.repeat_interleave(block_shape[0], 0).repeat_interleave(block_shape[1], 1)[:rows, :cols]
        output = (values / expanded).to(dtype)
        info["scale_mode"] = "reuse"
    elif fp8 and scale is None:
        q = quantize_fp8(values.float(), fp8_dtype=dtype, block_shape=block_shape, check_finite=check_finite)
        output, inverse = q.data, q.scale_inv
    elif fp8:
        if check_finite and (not torch.isfinite(values).all() or not torch.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError("Activation FP8 casting requires finite outputs and positive finite scales")
        inverse = scale.reciprocal().clone()
        snapshot = torch.stack((scale.flatten()[0], scale.flatten()[0].reciprocal())) if capture_scale else None
        expanded = scale if block_shape is None else scale.repeat_interleave(block_shape[0], 0).repeat_interleave(block_shape[1], 1)[:rows, :cols]
        output = (values * expanded).clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max).to(dtype)
        observations = values.abs().amax().reshape(1) if observe else None
    else:
        output = values.to(dtype)
    if inplace:
        if isinstance(x, FP8Activation):
            x._data.copy_(output)
            output = x._data
        else:
            x.copy_(geometry.restore(output))
            output = x
    return output, inverse, saved, snapshot, observations, info


def backward(gradient, saved, geometry, *, dtype, gradient_output=None):
    storage = gradient._data if isinstance(gradient, FP8Activation) else gradient
    if isinstance(storage, FakeTensor) or gradient.device.type == "meta" or not gradient.numel():
        result = torch.empty(geometry.matrix_shape, device=gradient.device, dtype=dtype)
    else:
        if saved.mask:
            coefficient = (unpack_boolean_mask(PackedBooleanMask(saved.data, geometry.matrix_shape))
                           if saved.packed else saved.data.reshape(geometry.matrix_shape))
        elif saved.packed:
            q = PackedActivation(saved.data, saved.step, saved.minimum, geometry.matrix_shape,
                                 saved.group_size, saved.block_shape, saved.bits)
            coefficient = dequantize_unpack_activation(q, dtype=torch.float32)
        else:
            coefficient = saved.data
        dy = geometry.matrix(gradient.dequantize(torch.float32) if isinstance(gradient, FP8Activation) else
                             gradient if gradient.dtype == torch.float64 else gradient.float())
        result = torch.where(coefficient, dy, 0.0) if saved.mask else dy * coefficient
    return gradient_output.convert(geometry.restore(result)) if gradient_output is not None else result.to(dtype)
