"""
Current amax scaling for 2D tensor.
"""

from dataclasses import dataclass 


import torch
import torch.nn.functional as F 



@dataclass(frozen=True)
class ScalingResult:
    # Quantization: q = cast_fp8(x * scale)
    scale: torch.Tensor

    # Reconstruction: x_hat = q.float() * scale_inv
    scale_inv: torch.Tensor 

    # None mean tensor-wise; otherwise (block_rows, block_cols)
    block_shape: tuple[int, int] | None



@dataclass(frozen=True)
class AmaxScaling:
    block_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.block_shape is None:
            return

        if (not isinstance(self.block_shape, tuple) or len(self.block_shape) != 2
            or any(type(size) is not int or size <= 0 for size in self.block_shape)):
            raise ValueError("block_shape must be None or a tuple of two positive integers")

    @torch.no_grad()
    def compute(self, x: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e4m3fn) -> ScalingResult:
        if x.ndim != 2:
            raise ValueError("AmaxScaling excepts a 2D tensor")
        if x.numel() == 0:
            raise ValueError("AmaxScaling does not accept empty tensors")
        if x.dtype not in (torch.float32, torch.bfloat16, torch.float16, torch.int32):
            raise TypeError("Input must be FP32, BF16, FP16, or Int32")
        if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("Unsupported FP8 dtype")

        x_fp32 = x.float()
        if not torch.isfinite(x_fp32).all():
            raise ValueError("AmaxScaling requires finite inputs")
        
        abs_x = x_fp32.abs()
        if self.block_shape is None:
            amax = abs_x.amax().reshape(1, 1)
        else:
            block_rows, block_cols = self.block_shape
            rows, cols = x.shape

            pad_rows = (-rows) % block_rows
            pad_cols = (-cols) % block_cols

            padded = F.pad(
                abs_x,
                (0, pad_cols, 0, pad_rows),
                value=0.0
            )

            num_row_blocks = padded.shape[0] // block_rows
            num_col_blocks = padded.shape[1] // block_cols

            blocks = padded.reshape(
                num_row_blocks,
                block_rows,
                num_col_blocks,
                block_cols
            )

            amax = blocks.amax(dim=(1, 3))
        
        fp8_max = torch.finfo(fp8_dtype).max

        safe_amax = amax.clamp_min(1e-12)
        scale = fp8_max / safe_amax
        scale = torch.where(
            amax == 0,
            torch.ones_like(scale),
            scale
        )

        return ScalingResult(
            scale=scale,
            scale_inv=scale.reciprocal(),
            block_shape=self.block_shape
        )