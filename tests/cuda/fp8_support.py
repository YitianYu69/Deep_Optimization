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
