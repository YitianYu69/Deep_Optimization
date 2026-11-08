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
    num_stages: int | str = "auto",
    max_num_imprecise_acc: int = 32,
) -> torch.Tensor:

    if num_stages != "auto" and (type(num_stages) is not int or num_stages not in (1, 2, 3, 4, 5)):
        raise ValueError("num_stages must be 'auto', 1, 2, 3, 4, or 5")

    if (type(max_num_imprecise_acc) is not int or not 32 <= max_num_imprecise_acc <= 2147483616
            or max_num_imprecise_acc % 32):
        raise ValueError("max_num_imprecise_acc must be a multiple of 32 up to 2147483616")

    if backend == "reference":
        from .reference import gemm_nt as implementation

        return implementation(a, b, out_dtype=out_dtype, bias=bias)
    
    if backend == "triton":
        from .triton_fp8 import gemm_nt as implementation

        return implementation(a, b, out_dtype=out_dtype, bias=bias, num_stages=num_stages,
                              max_num_imprecise_acc=max_num_imprecise_acc)
    
    raise ValueError("backend must be 'reference' or 'triton'")
