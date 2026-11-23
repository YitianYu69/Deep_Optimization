"""Tensorwise native scaled GEMM over the existing, immutable FP8 operands."""

import torch
from torch._subclasses.fake_tensor import FakeTensor

from deepopt.opt.backends._shared.cuda import device_metadata


def row_major_eligible(data):
    return (data.stride(1) == 1 and data.stride(0) >= data.shape[1] and data.stride(0) % 16 == 0
            and (isinstance(data, FakeTensor) or data.data_ptr() % 16 == 0))


def bias_strategy(bias,out_dtype):
    if bias is None:
        return "none"
    if bias.dtype==out_dtype and out_dtype in (torch.bfloat16,torch.float16) and bias.is_contiguous():
        return "native_fused"
    if bias.dtype in (torch.float32,torch.bfloat16,torch.float16) and bias.is_contiguous():
        return "fp32_product_fused_bias_cast"
    return "unsupported"


def unsupported_reason(a, b, *, bias=None, out_dtype=torch.float32):
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
        if not row_major_eligible(data):
            return "native GEMM requires aligned row-major operand storage"
    if bias_strategy(bias,out_dtype)=="unsupported":
        return "native GEMM bias requires contiguous FP32, BF16 or FP16 storage"
    return None


@torch.no_grad()
def gemm_nt(a, b, *, out_dtype=torch.float32, bias=None, num_stages="auto", use_fast_accum=False, autotune=True):
    if type(autotune) is not bool:
        raise ValueError("autotune must be a boolean")
    if bias is not None and (bias.ndim!=1 or bias.shape[0]!=b.data.shape[0] or bias.device!=a.data.device):
        raise ValueError("Bias must be a vector matching output columns on the operand device")
    if a.data.device != b.data.device:
        raise ValueError("Operands must be on the same device")
    if a.data.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Native FP8 backward requires an NVIDIA CUDA GPU")
    if out_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Output dtype must be FP32, BF16, or FP16")
    capability, _ = device_metadata(a.data.device.index)
    if capability < 90:
        raise ValueError("Native FP8 backward requires compute capability >= 9")
    if unsupported_reason(a, b,bias=bias,out_dtype=out_dtype) is not None:
        # Eligibility is deterministic and part of the numerical recipe. Do
        # not catch execution failures and silently switch numerical policy.
        from deepopt.opt.backends.dispatch import backend_module
        custom_gemm = backend_module("gemm", "gemm.fp8").gemm_nt

        return custom_gemm(a, b, out_dtype=out_dtype, bias=bias, num_stages=num_stages, autotune=autotune)
    strategy=bias_strategy(bias,out_dtype)
    with torch.cuda.device(a.data.device), torch.autocast("cuda", enabled=False):
        options={"bias":bias} if strategy=="native_fused" else {}
        result = torch._scaled_mm(a.data, b.data.T, scale_a=a.scale_inv, scale_b=b.scale_inv,
                                  out_dtype=torch.float32 if strategy=="fp32_product_fused_bias_cast" else out_dtype,
                                  use_fast_accum=use_fast_accum,**options)
        # PyTorch 2.4 returns (output, amax); newer versions return output.
        result=result[0] if isinstance(result,tuple) else result
        from deepopt.opt.backends._shared.gemm_diagnostics import note_gemm_launch, gemm_launch_is_recording
        if gemm_launch_is_recording():
            note_gemm_launch(kind="native_scaled_mm", use_fast_accum=use_fast_accum,
                             bias_strategy=strategy, autotune=autotune)
        if strategy=="fp32_product_fused_bias_cast":
            if isinstance(result,FakeTensor):
                return torch.empty(result.shape,device=result.device,dtype=out_dtype)
            from deepopt.opt.backends.dispatch import backend_module
            bias_output = backend_module("gemm", "gemm.epilogue").bias_output
            result=bias_output(result,bias,out_dtype)
        return result
