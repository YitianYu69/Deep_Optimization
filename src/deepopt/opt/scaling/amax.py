"""
Current amax scaling for 2D tensor.
"""

from dataclasses import dataclass 

import torch
import torch.nn.functional as F 
from torch._subclasses.fake_tensor import FakeTensor

from .._execution import finite_checks_enabled


def is_not_power_of_2(n: int) -> bool:
    return n <= 0 or (n & (n - 1)) != 0


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
    def compute(self, x: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e4m3fn,
                *, check_finite: bool | None = None) -> ScalingResult:
        if check_finite is None:
            check_finite = finite_checks_enabled()
        self._validate_input(x, fp8_dtype, check_finite)

        fp8_max = torch.finfo(fp8_dtype).max
        if x.device.type == "cpu" and not isinstance(x, FakeTensor):
            # FP32 tensors belong only to the CPU reference implementation.
            x_fp32 = x.float()
            if check_finite and not torch.isfinite(x_fp32).all():
                raise ValueError("AmaxScaling requires finite inputs")
            return self._compute_cpu(x_fp32, fp8_max)
        if isinstance(x, FakeTensor):
            rows, cols = x.shape
            shape = ((1, 1) if self.block_shape is None else
                     tuple((size + block - 1) // block for size, block in zip((rows, cols), self.block_shape)))
            scale = torch.empty(shape, device=x.device, dtype=torch.float32)
            scale_inv = torch.empty_like(scale)
        else:
            from deepopt.opt.backends.triton.quantization.fp8 import amax_scale

            scale, scale_inv = amax_scale(x, fp8_max, self.block_shape, check_finite)
        return ScalingResult(scale=scale, scale_inv=scale_inv, block_shape=self.block_shape)

    def _validate_input(self, x, fp8_dtype, check_finite):
        """Share structural checks with quantization without scanning payloads."""
        if type(check_finite) is not bool:
            raise TypeError("check_finite must be a boolean")
        if x.ndim != 2:
            raise ValueError("AmaxScaling excepts a 2D tensor")
        if x.numel() == 0:
            raise ValueError("AmaxScaling does not accept empty tensors")
        if x.dtype not in (torch.float32, torch.bfloat16, torch.float16, torch.int32):
            raise TypeError("Input must be FP32, BF16, FP16, or Int32")
        if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("Unsupported FP8 dtype")
        if x.device.type not in ("cpu", "cuda"):
            raise ValueError("AmaxScaling supports CPU and CUDA tensors")
        if x.device.type == "cuda" and self.block_shape is not None:
            # These restrictions belong to the fused CUDA kernel, not the CPU
            # reference implementation used for numerical testing.
            if all(size == 1 for size in self.block_shape):
                raise ValueError("block_shape must not be (1, 1)")
            if any(is_not_power_of_2(size) for size in self.block_shape):
                raise ValueError("block_shape must be powers of two")

    def _compute_cpu(self, x_fp32: torch.Tensor, fp8_max: float) -> ScalingResult:
        """PyTorch reference scaling, including strided inputs and edge blocks."""
        abs_x = x_fp32.abs()
        if self.block_shape is None:
            amax = abs_x.amax().reshape(1, 1)
        else:
            block_rows, block_cols = self.block_shape
            rows, cols = x_fp32.shape
            padded = F.pad(abs_x, (0, (-cols) % block_cols, 0, (-rows) % block_rows))
            amax = padded.reshape(
                (rows + block_rows - 1) // block_rows, block_rows,
                (cols + block_cols - 1) // block_cols, block_cols,
            ).amax(dim=(1, 3))
        scale = fp8_max / amax.clamp_min(1e-12)
        scale = torch.where(amax == 0, torch.ones_like(scale), scale)
        return ScalingResult(scale=scale, scale_inv=scale.reciprocal(), block_shape=self.block_shape)
