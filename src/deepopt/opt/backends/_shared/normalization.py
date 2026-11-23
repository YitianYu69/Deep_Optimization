"""Scoped normalization variants for measured rollout; no device/state cache."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from functools import lru_cache


_variant = ContextVar("deepopt_normalization_variant", default="auto")
_tuning = ContextVar("deepopt_normalization_tuning", default=(4, 512))
NORMALIZATION_VARIANTS = ("auto", "baseline", "candidate", "batch_shared", "small_forward", "layer_fused", "layer_tiled", "channels_last")


def channels_last_batch(geometry, meta, group, device):
    """A coalesced BatchNorm domain; payload grouping stays per channel."""
    if geometry.kind != "batch" or len(geometry.shape) != 4 or geometry.channels < 4 or not geometry.elements or group > 2048:
        return False
    channel_storage = meta[0] == 1 or (meta[0] == 0 and meta[1][1] == 1 and meta[1][-1] == geometry.channels)
    if not channel_storage:
        return False
    variant = normalization_variant()
    # Promote this candidate into auto only after complete-model H100 timing.
    return variant in ("candidate", "channels_last")


@lru_cache(maxsize=32)
def _measured_h100(index):
    import torch
    from .cuda import device_metadata

    return device_metadata(index) == (90, 132) and "H100" in torch.cuda.get_device_name(index)


def measured_norm_device(device):
    import torch
    return device.type == "cuda" and torch.cuda.is_available() and _measured_h100(device.index)


def measured_layer_variant(rows, width, input_dtype, packed, recenter, device):
    """V3 native inputs and v4's measured FP32 residual/FP8 consumer boundary."""
    if str(input_dtype) == "torch.float32" and (rows, width) == (100864, 768):
        eligible = (packed is None and not recenter or packed is not None and recenter
                    and packed.bits == 2 and packed.group_size == 768)
        return "layer_tiled" if eligible and measured_norm_device(device) else None
    if str(input_dtype) != "torch.bfloat16" or recenter or (rows, width) not in ((100864, 768), (6304, 3072)):
        return None
    if packed is not None and (packed.bits != 2 or packed.group_size != 512):
        return None
    if not measured_norm_device(device):
        return None
    return "layer_fused" if width == 3072 and packed is not None else "layer_tiled"


def measured_group_forward(geometry, input_dtype, output_dtype, compression, recenter, device):
    if (geometry.kind != "group" or tuple(geometry.shape) != (16, 128, 7, 7) or geometry.groups != 32
            or str(input_dtype) != "torch.bfloat16" or str(output_dtype) != "torch.float32" or recenter):
        return False
    # Autograd clips the requested 512-element group to this 196-element
    # normalization domain before calling the executor.
    if compression["enabled"] and (compression["bits"] != 2 or compression["group_size"] != geometry.elements):
        return False
    return measured_norm_device(device)


def layer_backward_programs(rows, width, row_chunk, max_programs, *, num_sms=132, scratch_bytes=8 * 2**20):
    """Bound row work and affine scratch without the old ceil(rows/128) cap."""
    target = max_programs or 4 * num_sms
    capacity = scratch_bytes // (2 * width * 4)
    programs = max(1, min(target, (rows + row_chunk - 1) // row_chunk, capacity))
    # Account for every affine-merge allocation as well as the initial planes.
    while programs > 1:
        partials, total = programs, programs
        while partials > 1:
            partials = (partials + 127) // 128
            total += partials
        if total <= capacity:
            break
        programs -= max(1, total - capacity)
    return programs


def normalization_variant():
    return _variant.get()


def normalization_tuning():
    return _tuning.get()


def saved_normalization_variant(function):
    @wraps(function)
    def call(ctx, *args):
        variant = ctx.normalization_variant
        tuning = ctx.normalization_tuning
        if normalization_variant() == variant and normalization_tuning() == tuning:
            return function(ctx, *args)
        with normalization_kernel_scope(variant, row_chunk=tuning[0], max_programs=tuning[1]):
            return function(ctx, *args)
    return call


@contextmanager
def normalization_kernel_scope(variant="auto", *, row_chunk=4, max_programs=512):
    """Select independent norm experiments without changing precision policy.

    Candidate enables register-intensive fusion prototypes pending target-GPU
    acceptance. Auto keeps the general fallback for those paths.
    """
    if variant not in NORMALIZATION_VARIANTS:
        raise ValueError(f"normalization variant must be one of {NORMALIZATION_VARIANTS}")
    if type(row_chunk) is not int or row_chunk not in (1, 2, 4, 8, 16) or type(max_programs) is not int or not 0 <= max_programs <= 4096:
        raise ValueError("row_chunk must be 1/2/4/8/16 and max_programs must be 0..4096 (0 selects by SM count)")
    token = _variant.set(variant)
    tuning_token = _tuning.set((row_chunk, max_programs))
    try:
        yield
    finally:
        _tuning.reset(tuning_token)
        _variant.reset(token)
