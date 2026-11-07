"""Real-hardware prerequisites shared by FP8 CUDA tests."""

import importlib.util
import os

import torch


def cuda_fp8_skip_reason():
    if not torch.cuda.is_available():
        return "Real CUDA hardware is required for Triton FP8 validation"
    if torch.version.hip is not None:
        return "Triton TMA FP8 requires an NVIDIA GPU"
    if torch.cuda.get_device_capability()[0] < 9:
        return "Triton TMA FP8 requires NVIDIA compute capability >= 9"
    if importlib.util.find_spec("triton") is None:
        return "Triton is not installed"
    return ""


FP8_CUDA_SKIP_REASON = cuda_fp8_skip_reason()
if FP8_CUDA_SKIP_REASON and os.environ.get("DEEPOPT_REQUIRE_CUDA_FP8") == "1":
    raise RuntimeError(FP8_CUDA_SKIP_REASON)


def quantize_gemm_operand(x, *, fp8_dtype=torch.float8_e4m3fn, block_shape=None):
    """Keep arbitrary-grid GEMM coverage separate from fused quantizer limits.

    CUDA quantization requires power-of-two grids. Build other scale layouts
    with the CPU reference solely for testing the GEMM backend's metadata API.
    Supported CUDA grids still exercise the real fused scaling kernels.
    """
    from deepopt.opt.quantization.quantizer import quantize_fp8
    from deepopt.opt.quantization.tensor import QuantizedTensor

    if (x.device.type == "cuda" and block_shape is not None
            and (block_shape == (1, 1) or any(size & (size - 1) for size in block_shape))):
        operand = quantize_fp8(x.cpu(), fp8_dtype=fp8_dtype, block_shape=block_shape)
        return QuantizedTensor(operand.data.to(x.device), operand.scale_inv.to(x.device), block_shape)
    return quantize_fp8(x, fp8_dtype=fp8_dtype, block_shape=block_shape)
