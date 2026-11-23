"""
Explicit backend selection for FP8 GEMM
"""

import torch
from os.path import join as _join
from torch._subclasses.fake_tensor import FakeTensor

# Legacy import files share their canonical implementation's module object.
__path__.append(_join(__path__[0], "_legacy"))
del _join

from ..quantization.tensor import QuantizedTensor
from ._shared.gemm_diagnostics import observe_gemm, record_gemm_launch
from deepopt.kernel_policy import kernel_backend, kernel_role


def _gemm_role(*args, **kwargs):
    role = kwargs.get("diagnostic_role", "unspecified")
    return role if "." in role else f"linear.{role}"


@kernel_role(_gemm_role)
@record_gemm_launch
def gemm_nt(
    a: QuantizedTensor,
    b: QuantizedTensor,
    *,
    backend: str,
    out_dtype: torch.dtype = torch.float32,
    bias: torch.Tensor | None = None,
    num_stages: int | str = "auto",
    autotune: bool = True,
    gradient_output=None,
    diagnostic_role="unspecified",
    diagnostic_preparation=None,
) -> torch.Tensor:

    if type(autotune) is not bool:
        raise ValueError("autotune must be a boolean")
    if num_stages != "auto" and (type(num_stages) is not int or num_stages not in (1, 2, 3, 4, 5)):
        raise ValueError("num_stages must be 'auto', 1, 2, 3, 4, or 5")

    if backend in ("triton", "tilelang", "mixed", "inherit"):
        role = diagnostic_role if "." in diagnostic_role else f"linear.{diagnostic_role}"
        backend = kernel_backend("gemm", backend=backend, role=role)

    if backend in ("triton", "tilelang", "native_fast", "native_precise") and isinstance(a.data, FakeTensor):
        return _fake_gemm_nt(a, b, backend=backend, out_dtype=out_dtype, bias=bias,
                             gradient_output=gradient_output)

    if backend == "reference":
        from deepopt.opt.backends.reference.gemm import gemm_nt as implementation

        result = implementation(a, b, out_dtype=out_dtype, bias=bias)
        observe_gemm(a, b, backend, result, diagnostic_role, diagnostic_preparation,bias=bias)
        return gradient_output.convert(result, matrix=True) if gradient_output is not None else result

    if backend in ("triton", "tilelang"):
        from .dispatch import backend_module
        module = backend_module("gemm", "gemm.fp8", backend=backend,
                                role=f"linear.{diagnostic_role}" if "." not in diagnostic_role else diagnostic_role)
        implementation = module.gemm_nt

        result = implementation(a, b, out_dtype=out_dtype, bias=bias, num_stages=num_stages, autotune=autotune,
                                **({"gradient_output": gradient_output} if gradient_output is not None else {}))
        effective = getattr(module, "effective_backend", backend)
        if hasattr(module, "fallback_reason"):
            observe_gemm(a, b, effective, result, diagnostic_role, diagnostic_preparation, bias=bias,
                         routing={"requested_backend": backend, "effective_backend": effective,
                                  "fallback_reason": module.fallback_reason})
        else:
            observe_gemm(a, b, effective, result, diagnostic_role, diagnostic_preparation,bias=bias)
        return result
    if backend in ("native_fast", "native_precise"):
        if gradient_output is not None:
            raise ValueError("The selected native GEMM backend cannot emit scaled FP8 gradients")
        from deepopt.opt.backends.native.fp8 import gemm_nt as implementation

        from deepopt.kernel_policy import current_kernel_policy, kernel_policy_scope
        with kernel_policy_scope(current_kernel_policy(), role=f"linear.{diagnostic_role}"):
            result = implementation(a, b, out_dtype=out_dtype, bias=bias, num_stages=num_stages, autotune=autotune,
                                    use_fast_accum=backend == "native_fast")
        observe_gemm(a, b, backend, result, diagnostic_role, diagnostic_preparation,bias=bias)
        return result

    raise ValueError("backend must be reference, triton, tilelang, mixed, native_fast, or native_precise")


def _fake_gemm_nt(a, b, *, backend, out_dtype, bias, gradient_output):
    """Infer the executor's output without addresses, device queries or launches."""
    if gradient_output is not None:
        if backend not in ("triton", "tilelang"):
            raise ValueError("The selected native GEMM backend cannot emit scaled FP8 gradients")
        # Fake gradient outputs use the same unfused FP32 conversion contract
        # as the real executor when GradientOutput.can_fuse returns False.
        out_dtype = torch.float32
    if a.data.device != b.data.device:
        raise ValueError("Operands must be on the same device")
    if a.data.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("This backend requires an NVIDIA CUDA GPU")
    if a.data.shape[1] != b.data.shape[1]:
        raise ValueError("Operands must have the same reduction dimension")
    if out_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Output dtype must be FP32, BF16, or FP16")
    rows, cols = a.data.shape[0], b.data.shape[0]
    if bias is not None:
        if bias.device != a.data.device or tuple(bias.shape) != (cols,):
            raise ValueError("Bias must have shape [N] on the operands' device")
        if bias.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError("Bias must be FP32, BF16, or FP16")

    uses_custom = backend in ("triton", "tilelang")
    if not uses_custom:
        from .native.fp8 import unsupported_reason

        uses_custom = unsupported_reason(a, b, bias=bias, out_dtype=out_dtype) is not None
    # TMA stores pad rows to 16 bytes. Preserve the physical row stride so
    # downstream view/reshape decisions agree with actual CUDA execution.
    alignment = 4 if out_dtype == torch.float32 else 8
    stride = ((cols + alignment - 1) // alignment) * alignment if uses_custom else cols
    result = torch.empty_strided((rows, cols), (stride, 1), device=a.data.device, dtype=out_dtype)
    return gradient_output.convert(result, matrix=True) if gradient_output is not None else result
