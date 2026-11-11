"""Tensorwise native scaled GEMM over the existing, immutable FP8 operands."""

import torch
from torch._subclasses.fake_tensor import FakeTensor

from deepopt.opt.backends._shared.cuda import device_metadata


def unsupported_reason(a, b):
    """Only metadata is inspected; never read scales or copy operand values."""
    if a.block_shape is not None or b.block_shape is not None:
        return "native GEMM requires tensorwise scales"
    _, k = a.data.shape
    n, reduction = b.data.shape
    if k != reduction:
        raise ValueError("Operands must have the same reduction dimension")
    if k % 16 or n % 16:
        return "native GEMM requires K and N divisible by 16"
    if a.data.dtype == b.data.dtype == torch.float8_e5m2:
        return "native GEMM does not support two E5M2 operands"
    for operand in (a, b):
        data = operand.data
        if (data.stride(1) != 1 or data.stride(0) < data.shape[1] or data.stride(0) % 16
                or (not isinstance(data, FakeTensor) and data.data_ptr() % 16)):
            return "native GEMM requires aligned row-major operand storage"
    return None


@torch.no_grad()
def gemm_nt(a, b, *, out_dtype=torch.float32, bias=None, num_stages="auto", use_fast_accum=False):
    if bias is not None:
        raise ValueError("The native FP8 executor is bias-free; use it for backward products")
    if a.data.device != b.data.device:
        raise ValueError("Operands must be on the same device")
    if a.data.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Native FP8 backward requires an NVIDIA CUDA GPU")
    if out_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Output dtype must be FP32, BF16, or FP16")
    capability, _ = device_metadata(a.data.device.index)
    if capability < 90:
        raise ValueError("Native FP8 backward requires compute capability >= 9")
    if unsupported_reason(a, b) is not None:
        # Eligibility is deterministic and part of the numerical recipe. Do
        # not catch execution failures and silently switch numerical policy.
        from deepopt.opt.backends.triton.gemm.fp8 import gemm_nt as triton_gemm

        return triton_gemm(a, b, out_dtype=out_dtype, num_stages=num_stages)
    with torch.cuda.device(a.data.device), torch.autocast("cuda", enabled=False):
        result = torch._scaled_mm(a.data, b.data.T, scale_a=a.scale_inv, scale_b=b.scale_inv,
                                  out_dtype=out_dtype, use_fast_accum=use_fast_accum)
        # PyTorch 2.4 returns (output, amax); newer versions return output.
        return result[0] if isinstance(result, tuple) else result
