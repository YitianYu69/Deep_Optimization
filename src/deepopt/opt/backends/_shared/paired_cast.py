"""Bounded, pointer-free selection of tensorwise cast/transpose tiles."""

from collections import OrderedDict
from math import isfinite


PAIRED_CAST_BASELINE = (32, 4)
# TorchAO's eager float8 prototype also emits both layouts in one tiled cast.
# Retain our existing tile and measure a small set of warp/tile alternatives.
# https://github.com/pytorch/ao/blob/566d15ea721da0d305270b4c552c59970ec89304/torchao/prototype/float8nocompile/kernels/fp8_dynamic_tensorwise.py
PAIRED_CAST_CANDIDATES = ((32, 4), (32, 1), (64, 4), (64, 8), (128, 4))
MAX_PAIRED_CAST_CACHE = 256
_paired_cast_cache = OrderedDict()


def paired_cast_key(shape, strides, source_dtype, target_dtype, device, capability,
                    sms, transpose_only, shared_statistics, input_alignment=0):
    """Shape and storage geometry both affect a fused transpose's winner."""
    return (tuple(shape), tuple(strides), str(source_dtype), str(target_dtype),
            device, capability, sms, bool(transpose_only), bool(shared_statistics), input_alignment)


def select_paired_cast(key, benchmark, *, autotune, capturing):
    """Never benchmark during capture; cache no tensors, closures or timings."""
    if not autotune:
        return PAIRED_CAST_BASELINE
    if key in _paired_cast_cache:
        config = _paired_cast_cache.pop(key)
        _paired_cast_cache[key] = config
        return config
    # Small casts cannot amortize a cold search and have little transpose work.
    rows, cols = key[0]
    if capturing or rows * cols < 4096 or min(rows, cols) < 16:
        return PAIRED_CAST_BASELINE
    winner, minimum = PAIRED_CAST_BASELINE, float("inf")
    for config in PAIRED_CAST_CANDIDATES:
        elapsed = benchmark(config)
        if isfinite(elapsed) and elapsed >= 0 and elapsed < minimum:
            winner, minimum = config, elapsed
    if not isfinite(minimum):
        raise RuntimeError("No tensorwise paired-cast candidate completed")
    _paired_cast_cache[key] = winner
    while len(_paired_cast_cache) > MAX_PAIRED_CAST_CACHE:
        _paired_cast_cache.popitem(last=False)
    return winner


def clear_paired_cast_cache():
    _paired_cast_cache.clear()
