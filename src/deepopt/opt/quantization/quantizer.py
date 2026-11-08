"""
FP8 quantization with fused CUDA execution and a PyTorch CPU reference.
"""

import torch
from torch._subclasses.fake_tensor import FakeTensor

from ..scaling_strategy.amax import AmaxScaling
from .tensor import QuantizedTensor
from .._execution import finite_checks_enabled


@torch.no_grad()
def quantize_fp8(
    x: torch.Tensor,
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    block_shape: tuple[int, int] | None = None,
    check_finite: bool | None = None,
):
    """
    Quantize a non-empty 2D matrix with tensor-wise or block-wise scales.
    """

    check_finite = finite_checks_enabled() if check_finite is None else check_finite
    strategy = AmaxScaling(block_shape=block_shape)
    if x.device.type == "cuda":
        strategy._validate_input(x, fp8_dtype, check_finite)
        if isinstance(x, FakeTensor):
            # Metadata probes need the fused operation's outputs, not a trace
            # of the old FP32/expanded-scale intermediates or a real GPU launch.
            rows, cols = x.shape
            shape = ((1, 1) if block_shape is None else
                     ((rows + block_shape[0] - 1) // block_shape[0],
                      (cols + block_shape[1] - 1) // block_shape[1]))
            data = torch.empty(x.shape, device=x.device, dtype=fp8_dtype)
            scale_inv = torch.empty(shape, device=x.device, dtype=torch.float32)
        else:
            from ..backends.triton_quantization import quantize

            data, scale_inv = quantize(x, fp8_dtype, block_shape, check_finite)
        return QuantizedTensor(data=data, scale_inv=scale_inv, block_shape=block_shape)

    scaling = strategy.compute(x, fp8_dtype=fp8_dtype, check_finite=check_finite)

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


@torch.no_grad()
def _quantize_tensorwise_pair(x, *, fp8_dtype=torch.float8_e4m3fn, block_shape=None, check_finite=None):
    """Internal training layout pair; neither output changes the scaling recipe."""
    if block_shape is not None:
        raise ValueError("Paired quantization requires tensorwise scaling")
    check_finite = finite_checks_enabled() if check_finite is None else check_finite
    if x.device.type != "cuda":
        normal = quantize_fp8(x, fp8_dtype=fp8_dtype, check_finite=check_finite)
        return normal, normal.transpose()
    AmaxScaling()._validate_input(x, fp8_dtype, check_finite)
    if isinstance(x, FakeTensor):
        m, n = x.shape
        normal = torch.empty_strided((m, n), (((n + 15) // 16) * 16, 1), device=x.device, dtype=fp8_dtype)
        transposed = torch.empty_strided((n, m), (((m + 15) // 16) * 16, 1), device=x.device, dtype=fp8_dtype)
        inverse = torch.empty((1, 1), device=x.device, dtype=torch.float32)
    else:
        from ..backends.triton_quantization import quantize_tensorwise_pair

        normal, inverse, transposed = quantize_tensorwise_pair(x, fp8_dtype, check_finite)
    return QuantizedTensor(normal, inverse), QuantizedTensor(transposed, inverse)
