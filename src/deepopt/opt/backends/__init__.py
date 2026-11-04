"""
Explicit backend selection for FP8 GEMM
"""

import torch

from ..quantization.tensor import QuantizedTensor


def gemm_nt(
    a: QuantizedTensor,
    b: QuantizedTensor,
    *,
    backend: str,
    out_dtype: torch.dtype = torch.float32,
    bias: torch.Tensor | None = None,
    num_stages: int = 3,
) -> torch.Tensor:

    if type(num_stages) is not int or num_stages not in (1, 2, 3, 4):
        raise ValueError("num_stages must be 1, 2, 3, or 4")

    if backend == "reference":
        from .reference import gemm_nt as implementation

        return implementation(a, b, out_dtype=out_dtype, bias=bias)
    
    if backend == "triton":
        from .triton_fp8 import gemm_nt as implementation

        return implementation(a, b, out_dtype=out_dtype, bias=bias, num_stages=num_stages)
    
    raise ValueError("backend must be 'reference' or 'triton'")