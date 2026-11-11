"""
FP8 values and the metadata needed to reconstruct them
"""

from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensor


@dataclass(frozen=True)
class QuantizedTensor:
    data: torch.Tensor 

    # FP32 reconstruction scale
    # tensor-wise: [1, 1]
    # block-wise: [ceil(rows / block_rows), ceil(cols / block_cols)]
    scale_inv: torch.Tensor 

    block_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("QuantizedTensor data must be 2D")
        if self.data.numel() == 0:
            raise ValueError("QuantizedTensor data must be non-empty")
        if self.data.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise TypeError("QuantizedTensor data must have an FP8 dtype")
        if self.scale_inv.dtype != torch.float32:
            raise TypeError("scale_inv must have dtype torch.float32")
        if self.scale_inv.device != self.data.device:
            raise ValueError("data and scale_inv must be on the same device")
        if self.block_shape is None:
            expected_shape = (1, 1)
        else:
            if (not isinstance(self.block_shape, tuple) or len(self.block_shape) != 2
                or any(type(size) is not int or size <= 0 for size in self.block_shape)):
                raise ValueError("block_shape must be None or a tuple of two positive integers")
        
            rows, cols = self.data.shape
            block_rows, block_cols = self.block_shape

            expected_shape = (
                (rows + block_rows - 1) // block_rows,
                (cols + block_cols - 1) // block_cols
            )

        if tuple(self.scale_inv.shape) != expected_shape:
            raise ValueError(
                f"scale_inv must have shape {expected_shape}",
                f"got {tuple(self.scale_inv.shape)}"
            )

    @torch.no_grad()
    def dequantize(self) -> torch.Tensor:
        """
        Reconstruct an FP32 matrix for reference computations.
        """

        if self.data.device.type == "cuda":
            if isinstance(self.data, FakeTensor):
                return torch.empty(self.data.shape, device=self.data.device, dtype=torch.float32)
            from deepopt.opt.backends.triton.quantization.fp8 import dequantize

            return dequantize(self.data, self.scale_inv, self.block_shape)

        values = self.data.float()

        if self.block_shape is None:
            return values * self.scale_inv
        
        block_rows, block_cols = self.block_shape
        rows, cols = self.data.shape

        expanded_scales = (
            self.scale_inv
            .repeat_interleave(block_rows, dim=0)
            .repeat_interleave(block_cols, dim=1)
        )
        return values * expanded_scales[:rows, :cols]

    @torch.no_grad()
    def transpose(self, *, contiguous: bool = True) -> "QuantizedTensor":
        """Transpose scales and values; backends can pack a view directly."""
        if type(contiguous) is not bool:
            raise TypeError("contiguous must be a boolean")
        if self.block_shape is None:
            transposed_block_shape = None
        else:
            block_rows, block_cols = self.block_shape
            transposed_block_shape = (block_cols, block_rows)

        values, scales = self.data.T, self.scale_inv.T
        return QuantizedTensor(
            data=values.contiguous() if contiguous else values,
            scale_inv=scales.contiguous() if contiguous else scales,
            block_shape=transposed_block_shape
        )
