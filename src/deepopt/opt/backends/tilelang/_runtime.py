"""Lazy TileLang runtime shared by the correctness-first CUDA kernels.

Importing this module never imports TileLang or initializes CUDA.  Compiled
factories must include the CUDA device in their cache key: target selection is
performed under that device's context, rather than implicitly using device 0.
"""

from functools import lru_cache
from importlib import import_module

import torch


@lru_cache(maxsize=1)
def runtime():
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm the TileLang backend before starting CUDA graph capture")
    try:
        tilelang = import_module("tilelang")
        language = import_module("tilelang.cuda.language")
    except ImportError as error:
        raise RuntimeError(
            "The TileLang kernel backend requires the optional TileLang dependency; "
            "install this checkout with python -m pip install '.[tilelang]' in a compatible CUDA environment"
        ) from error
    return tilelang, language


require_tilelang = runtime


def dtype_name(value):
    return str(value).removeprefix("torch.")


dtype = dtype_name


def require_cuda(*values):
    from torch._subclasses.fake_tensor import FakeTensor

    if torch.version.hip is not None:
        raise ValueError("The TileLang backend currently requires NVIDIA CUDA")
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if not tensors or any(value.device.type != "cuda" or isinstance(value, FakeTensor) for value in tensors):
        raise ValueError("TileLang kernels require real CUDA tensors")
    if any(value.device != tensors[0].device for value in tensors):
        raise ValueError("TileLang operands must share a CUDA device")
    return tensors[0].device


def storage_view(value):
    """Expose an input's addressable storage without a layout conversion."""
    if any(stride < 0 for stride in value.stride()):
        raise ValueError("TileLang kernels require nonnegative tensor strides")
    span = 0 if not value.numel() else 1 + sum((size - 1) * stride for size, stride in zip(value.shape, value.stride()))
    return value.as_strided((span,), (1,))


def compile_kernel(function, device, *, pass_configs=None, compile_flags=None):
    if torch.version.hip is not None:
        raise ValueError("The TileLang backend currently requires NVIDIA CUDA")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm this TileLang kernel specialization before starting CUDA graph capture")
        tilelang, _ = runtime()
        major, minor = torch.cuda.get_device_capability(device)
        # Upstream appends the architecture-specific suffix needed by Hopper
        # WGMMA and later matrix instructions (e.g. sm_90a, not sm_90).
        architecture = import_module("tilelang.contrib.nvcc").get_target_arch((major, minor))
        return tilelang.compile(
            function,
            target={"kind": "cuda", "arch": f"sm_{architecture}"},
            execution_backend="tvm_ffi",
            # Match the explicitly ordered arithmetic in the existing codecs.
            compile_flags=["--fmad=false", "--prec-div=true", "--prec-sqrt=true"] + list(compile_flags or ()),
            pass_configs=pass_configs,
        )


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def device_key(device):
    return str(torch.device(device))


def matrix_storage(value):
    if value.ndim != 2:
        raise ValueError("Expected a matrix")
    return storage_view(value), tuple(value.stride())


def auxiliary_module(family, module):
    """Honor Mixed routes when an operation expands into separate launches.

    Direct low-level TileLang calls outside Mixed remain TileLang calls; the
    default public policy must not redirect those calls back to Triton.
    """
    from deepopt.kernel_policy import current_kernel_policy

    if current_kernel_policy().backend == "mixed":
        from deepopt.opt.backends.dispatch import backend_module
        return backend_module(family, module)
    return import_module(f"deepopt.opt.backends.tilelang.{module}")
