"""
Explicit backend selection for FP8 GEMM
"""

import torch
from os.path import join as _join

# Legacy import files share their canonical implementation's module object.
__path__.append(_join(__path__[0], "_legacy"))
del _join

from ..quantization.tensor import QuantizedTensor


def gemm_nt(
    a: QuantizedTensor,
    b: QuantizedTensor,
    *,
    backend: str,
    out_dtype: torch.dtype = torch.float32,
    bias: torch.Tensor | None = None,
    num_stages: int | str = "auto",
    gradient_output=None,
) -> torch.Tensor:

    if num_stages != "auto" and (type(num_stages) is not int or num_stages not in (1, 2, 3, 4, 5)):
        raise ValueError("num_stages must be 'auto', 1, 2, 3, 4, or 5")

    if backend == "reference":
        from deepopt.opt.backends.reference.gemm import gemm_nt as implementation

        result = implementation(a, b, out_dtype=out_dtype, bias=bias)
        return gradient_output.convert(result, matrix=True) if gradient_output is not None else result
    
    if backend == "triton":
        from deepopt.opt.backends.triton.gemm.fp8 import gemm_nt as implementation

        return implementation(a, b, out_dtype=out_dtype, bias=bias, num_stages=num_stages,
                              **({"gradient_output": gradient_output} if gradient_output is not None else {}))
    if backend in ("native_fast", "native_precise"):
        if gradient_output is not None:
            raise ValueError("The selected native GEMM backend cannot emit scaled FP8 gradients")
        from deepopt.opt.backends.native.fp8 import gemm_nt as implementation

        return implementation(a, b, out_dtype=out_dtype, bias=bias, num_stages=num_stages,
                              use_fast_accum=backend == "native_fast")
    
    raise ValueError("backend must be reference, triton, native_fast, or native_precise")
