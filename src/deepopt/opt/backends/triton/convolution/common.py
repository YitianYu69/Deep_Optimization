"""Shared in-register loads and validation for implicit FP8 convolution."""

import torch
import triton
import triton.language as tl

from deepopt.opt.backends._shared.cuda import device_metadata
from .launch import geometry_arguments, operand_arguments, launch_options, needs_int64
from .scales import scale_chunk_length


_scale_chunk_length = triton.jit(scale_chunk_length)


@triton.jit
def _load_operand(data, rows, cols, mask, RS: tl.constexpr, CS: tl.constexpr):
    """Keep multiplication operands in their original FP8 format."""
    return tl.load(data + rows * RS + cols * CS, mask=mask, other=0.0)


@triton.jit
def _load_scale(scales, rows, cols, mask, BR: tl.constexpr, BC: tl.constexpr,
                SRS: tl.constexpr, SCS: tl.constexpr):
    return tl.load(scales + rows // BR * SRS + cols // BC * SCS, mask=mask, other=0.0)


@triton.jit
def _coordinates(offset, H: tl.constexpr, W: tl.constexpr):
    return offset // (H * W), offset // W % H, offset % W


@triton.jit
def _tile_coordinates(M: tl.constexpr, N: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    pid = tl.program_id(0)
    group = pid // (tiles_m * tiles_n)
    tile = pid % (tiles_m * tiles_n)
    return group, tile // tiles_n, tile % tiles_n


def validate(a, b, out_dtype):
    if a.data.device != b.data.device:
        raise ValueError("Convolution operands must share a device")
    if a.data.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Triton FP8 convolution requires an NVIDIA CUDA GPU")
    capability, _ = device_metadata(a.data.device.index)
    if capability < 90:
        raise ValueError("Triton FP8 convolution requires compute capability >= 9")
    if out_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Convolution output dtype must be FP32, BF16, or FP16")


def arguments(a, b, geometry, num_stages, *, extra_tensors=()):
    tensorwise = a.scale_inv.numel() == 1 and b.scale_inv.numel() == 1
    return {
        **geometry_arguments(geometry),
        **operand_arguments("A", a), **operand_arguments("B", b),
        "TENSORWISE": tensorwise,
        "INDEX64": needs_int64(a, b, geometry, extra_tensors=extra_tensors),
        **launch_options(num_stages, tensorwise=tensorwise),
    }


def grid(m, n, groups):
    return lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]) * groups,
    )
