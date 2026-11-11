"""Reference output conversion matching the shared scale-aware ABI."""

import torch

from deepopt.opt.quantization.quantizer import quantize_fp8


def convert(values, dtype, block_shape, scale, observe, capture_scale, check_finite, fake):
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return values.to(dtype), None, None, None
    rows, cols = values.shape
    if fake or not values.numel():
        grid = (1, 1) if block_shape is None else ((rows + block_shape[0] - 1) // block_shape[0],
                                                  (cols + block_shape[1] - 1) // block_shape[1])
        return torch.empty_like(values, dtype=dtype), torch.ones(grid, device=values.device, dtype=torch.float32), None, None
    if scale is None:
        q = quantize_fp8(values.float(), fp8_dtype=dtype, block_shape=block_shape, check_finite=check_finite)
        return q.data, q.scale_inv, None, None
    if check_finite and (not torch.isfinite(values).all() or not torch.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Pooling FP8 output requires finite values and positive finite scales")
    expanded = scale if block_shape is None else scale.repeat_interleave(block_shape[0], 0).repeat_interleave(block_shape[1], 1)[:rows, :cols]
    result = (values * expanded).clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max).to(dtype)
    snapshot = torch.stack((scale.flatten()[0], scale.flatten()[0].reciprocal())) if capture_scale else None
    observations = values.abs().amax().reshape(1) if observe else None
    return result, scale.reciprocal().clone(), snapshot, observations
