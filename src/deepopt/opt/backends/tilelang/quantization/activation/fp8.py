"""FP8 conversion and affine saving using the same rounded FP8 values."""

from .affine import quantize_pack_affine
from ..fp8 import amax_scale, cast_scaled


def quantize_pack_fp8_activation(x, dtype, block_shape, group_size, check_finite, *, bits=2, row_alignment=16):
    import torch
    from deepopt.opt.quantization.activation import PackedActivation

    from ..._runtime import auxiliary_module
    from deepopt.kernel_policy import current_kernel_policy
    if type(row_alignment) is not int or row_alignment not in (1, 16):
        raise ValueError("row_alignment must be 1 or 16")
    if current_kernel_policy().backend == "mixed":
        normal, inverse = auxiliary_module("quantization", "quantization.fp8").quantize(x, dtype, block_shape, check_finite)
        if row_alignment == 16:
            from ...gemm.fp8 import _pack_fp8_rows
            normal = _pack_fp8_rows(normal, 16)[:, :x.shape[1]]
    else:
        scale, inverse = amax_scale(x, torch.finfo(dtype).max, block_shape, check_finite)
        normal = cast_scaled(x, scale, dtype, block_shape, row_alignment=row_alignment)
    packed, step, minimum = quantize_pack_affine(normal, group_size, block_shape, bits=bits)
    codec = PackedActivation(packed, step, minimum, tuple(x.shape), group_size, block_shape, bits)
    return normal, inverse, codec


def quantize_pack_delayed_fp8_activation(x, scale, dtype, group_size, *, observe, check_finite,
                                         bits=2, row_alignment=16):
    from deepopt.opt.quantization.activation import PackedActivation
    from ..._runtime import require_cuda
    from ...scaling.reduction import block_amax
    import torch

    require_cuda(x, scale)
    if type(row_alignment) is not int or row_alignment not in (1, 16):
        raise ValueError("row_alignment must be 1 or 16")
    # Capture the scale once so conversion and saved metadata use one generation.
    from ..._runtime import auxiliary_module
    from deepopt.kernel_policy import current_kernel_policy
    if row_alignment == 16 or current_kernel_policy().backend == "mixed":
        conversion = auxiliary_module("quantization", "quantization.delayed").convert
        normal, _, snapshot, observations, _ = conversion(
            x, scale, dtype, normal=True, transposed=False, bias=False,
            observe=observe, check_finite=check_finite, autotune=False)
        if row_alignment == 1:
            from ...gemm.fp8 import _copy_rows
            from ..._runtime import storage_view, dtype_name, device_key
            compact = torch.empty(x.shape, device=x.device, dtype=dtype)
            source = storage_view(normal)
            with torch.cuda.device(x.device):
                _copy_rows(*x.shape, *normal.stride(), source.numel(), x.shape[1], dtype_name(dtype), device_key(x.device))(
                    source, compact.view(-1))
            normal = compact
    else:
        if type(row_alignment) is not int or row_alignment != 1:
            raise ValueError("row_alignment must be 1 or 16")
        from ..delayed import _snapshot
        from ..._runtime import device_key
        if check_finite:
            amax_scale(x, torch.finfo(dtype).max, None, True)
        snapshot = torch.empty(2, device=x.device, dtype=torch.float32)
        _snapshot(device_key(x.device))(scale.reshape(1), snapshot)
        normal = cast_scaled(x, snapshot[:1], dtype)
        observations = block_amax(x, 32, 32, minimum=0).reshape(-1) if observe else None
    packed, step, minimum = quantize_pack_affine(normal, group_size, bits=bits)
    codec = PackedActivation(packed, step, minimum, tuple(x.shape), group_size, None, bits)
    return normal, codec, snapshot, observations
