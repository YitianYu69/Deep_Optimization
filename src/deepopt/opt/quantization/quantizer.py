"""
Reference FP8 quantization using current amax scaling
"""

import torch

from ..scaling_strategy.amax import AmaxScaling
from .tensor import QuantizedTensor


@torch.no_grad()
def quantize_fp8(
    x: torch.Tensor,
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    block_shape: tuple[int, int] | None = None
):
    """
    Quantize a non-empty 2D matrix with tensor-wise or block-wise scales.
    """

    scaling = AmaxScaling(block_shape=block_shape).compute(
        x, fp8_dtype=fp8_dtype
    )

    x_fp32 = x.float()
    if scaling.block_shape is None:
        scaled = x_fp32 * scaling.scale
    else:
        block_rows, block_cols = scaling.block_shape
        rows, cols = x.shape

        expanded_scales = (
            scaling.scale
            .repeat_interleave(block_rows, dim=0)
            .repeat_interleave(block_cols, dim=1)
        )

        scaled = x_fp32 * expanded_scales[:rows, :cols]
    
    fp8_max = torch.finfo(fp8_dtype).max
    data = (
        scaled
        .clamp(min=-fp8_max, max=fp8_max)
        .to(fp8_dtype)
        .contiguous()
    )

    return QuantizedTensor(
        data=data,
        scale_inv=scaling.scale_inv,
        block_shape=scaling.block_shape
    )