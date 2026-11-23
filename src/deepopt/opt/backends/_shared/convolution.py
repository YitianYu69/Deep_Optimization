"""Bounded convolution launch selection; cached entries never retain tensors."""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import logging
import math


_variant = ContextVar("deepopt_convolution_kernel_variant", default="auto")
_recording = ContextVar("deepopt_convolution_launch_recording", default=None)
_choices = OrderedDict()
_LIMIT = 256
_LAUNCH_VERSION = 2
_log = logging.getLogger(__name__)


def depthwise_eligible(geometry):
    return (geometry.groups == geometry.in_channels == geometry.out_channels and geometry.kernel_volume > 1
            and 4 * geometry.in_channels * geometry.kernel_volume <= 32 * 1024 * 1024)


def convolution_kernel_variant():
    return _variant.get()


def capture_convolution_recording(ctx):
    ctx.convolution_kernel_policy = _variant.get()
    ctx.convolution_launch_recording = _recording.get()


def restore_convolution_recording(function):
    @wraps(function)
    def call(ctx, *args):
        policy = getattr(ctx, "convolution_kernel_policy", "auto")
        rows = getattr(ctx, "convolution_launch_recording", None)
        if policy == _variant.get() and rows is _recording.get():
            return function(ctx, *args)
        token, record_token = _variant.set(policy), _recording.set(rows)
        try:
            return function(ctx, *args)
        finally:
            _recording.reset(record_token)
            _variant.reset(token)
    return call


@contextmanager
def convolution_kernel_scope(variant="auto"):
    if variant not in ("auto", "baseline", "tuned"):
        raise ValueError("convolution variant must be auto, baseline or tuned")
    token = _variant.set(variant)
    try:
        yield
    finally:
        _variant.reset(token)


@contextmanager
def record_convolution_launches():
    rows = []
    token = _recording.set(rows)
    try:
        yield rows
    finally:
        _recording.reset(token)


def launch_candidates(base, geometry, role, *, num_stages=None):
    """Bounded FP8 choices; new families stay opt-in under ``tuned``.

    ``auto`` retains the established search. Explicit stage counts are never
    overwritten. The expanded search is deliberately not a Cartesian product;
    a conservative shared-memory estimate prunes large pipeline candidates.
    """
    if role not in ("forward", "dx", "dw"):
        raise ValueError("convolution role must be forward, dx or dw")
    initial = {**base, "CHANNEL_FAST": False, "GROUP_M": 1}
    if role != "dw":
        initial["TAP_MAJOR"] = False
    result = [initial]
    if not base["TENSORWISE"] or geometry.groups != 1:
        return result
    m = geometry.out_channels_per_group if role == "dw" else geometry.batch_size * (
        geometry.input_volume if role == "dx" else geometry.output_volume)
    n = (geometry.in_channels_per_group * geometry.kernel_volume if role == "dw" else
         geometry.in_channels_per_group if role == "dx" else geometry.out_channels_per_group)
    k = geometry.batch_size * geometry.output_volume if role == "dw" else geometry.kernel_volume * (
        geometry.out_channels_per_group if role == "dx" else geometry.in_channels_per_group)
    result.append({**initial, "CHANNEL_FAST": True})
    for bm,bn,bk,warps in ((64,64,64,4),(128,64,64,4),(128,128,64,8),(64,64,128,4)):
        if n < bn // 2 or m < bm // 2 or k < bk // 2:
            continue
        result.append({**initial,"BLOCK_M":bm,"BLOCK_N":bn,"BLOCK_K":bk,"num_warps":warps,"CHANNEL_FAST":True})
    if convolution_kernel_variant() != "tuned":
        return result

    # dW reduces batch/output positions instead of channels and needs a
    # different tile balance. This is a bounded list, not a Cartesian sweep.
    shapes = {
        "forward": ((64,128,64,4), (128,64,128,4), (64,256,64,8)),
        "dx": ((64,128,64,4), (128,64,128,4), (128,128,128,8)),
        "dw": ((64,32,128,4), (64,128,64,4), (128,64,128,8)),
    }[role]

    def append(config):
        # FP8 operands use one byte. Include staging for layout conversion;
        # the compiler's actual resource checks remain authoritative.
        bm, bn, bk = (config[name] for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        estimated_shared = (bm + bn) * bk * config["num_stages"] + bm * bn * 2
        if estimated_shared <= 128 * 1024 and config not in result:
            result.append(config)

    for bm,bn,bk,warps in shapes:
        if n >= bn // 2 and m >= bm // 2 and k >= bk // 2:
            append({**initial, "BLOCK_M":bm, "BLOCK_N":bn, "BLOCK_K":bk,
                    "num_warps":warps, "CHANNEL_FAST":True, "GROUP_M":4})
    for group_m in (4, 8):
        append({**initial, "CHANNEL_FAST":True, "GROUP_M":group_m})
    if role != "dw" and geometry.kernel_volume > 1:
        append({**initial, "TAP_MAJOR":True})
        append({**initial, "BLOCK_N":64, "TAP_MAJOR":True, "GROUP_M":4})
    if num_stages == "auto":
        representatives = [result[1], result[-1]]
        for config in representatives:
            for stages in (3, 4):
                append({**config, "num_stages":stages})
    return result


def _resource_exhausted(error):
    """Only compiler resource exhaustion is a safe candidate rejection."""
    try:
        from triton.runtime.errors import OutOfResources
    except ImportError:
        return False
    return isinstance(error, OutOfResources)


def select_convolution_launch(key, candidates, launch, *, autotune=True, capturing=None, benchmark=None):
    """Tune before capture, then directly launch a pointer-free cached choice.

    The callback closes over this invocation's operands only. It is never
    stored. Cold capture uses a baseline without caching it as a tuned winner.
    Disabled autotuning never consults a previous tuned choice.
    """
    if not candidates:
        raise ValueError("convolution requires a baseline launch candidate")
    policy = convolution_kernel_variant()
    rejected = []
    if not autotune or policy == "baseline" or len(candidates) == 1:
        selected = candidates[0]
    else:
        fingerprint = tuple(tuple(sorted(candidate.items())) for candidate in candidates)
        cache_key = (_LAUNCH_VERSION, policy, key, fingerprint)
        selected = _choices.get(cache_key)
        if selected is None:
            if capturing is None:
                import torch
                capturing = torch.cuda.is_current_stream_capturing()
            if capturing:
                selected = candidates[0]
            else:
                if benchmark is None:
                    from triton.testing import do_bench_cudagraph
                    benchmark = lambda call: do_bench_cudagraph(call, rep=5)
                scores = []
                for index, candidate in enumerate(candidates):
                    try:
                        launch(candidate)  # Compile/allocate before timing.
                        score = float(benchmark(lambda candidate=candidate: launch(candidate)))
                    except Exception as error:
                        if index == 0 or not _resource_exhausted(error):
                            raise
                        rejected.append({"candidate":index, "reason":"compiler_resources"})
                        _log.debug("Rejected convolution launch %s: %s", candidate, error)
                        continue
                    if not math.isfinite(score) or score < 0:
                        raise ValueError("convolution benchmark must return a finite nonnegative latency")
                    scores.append((score, index))
                selected = candidates[min(scores)[1]]
                _choices[cache_key] = dict(selected)
                if len(_choices) > _LIMIT:
                    _choices.popitem(last=False)
        else:
            _choices.move_to_end(cache_key)
    recording = _recording.get()
    if recording is not None:
        recording.append({"role":key[0],"variant":policy,
                          "tile":{name:selected[name] for name in ("BLOCK_M","BLOCK_N","BLOCK_K","BLOCK","BLOCK_R","BLOCK_C") if name in selected},
                          "num_warps":selected["num_warps"],"num_stages":selected.get("num_stages",1),
                          "split_k":selected.get("SPLIT_K",selected.get("SPLITS",1)),
                          "channel_fast":selected.get("CHANNEL_FAST",False),
                          "indexing_family":"tap_major" if selected.get("TAP_MAJOR",False) else "flattened",
                          "group_m":selected.get("GROUP_M",1),
                          "rejected_candidates":rejected})
    return dict(selected)


def convolution_launch_key(role, a, b, geometry, arguments, output_dtype, *, bias=None):
    """Include source/scale strides and geometry; no tensor identity or storage."""
    from .cuda import device_metadata
    index = a.data.device.index
    constants = tuple(sorted((k,v) for k,v in arguments.items()
                             if k not in ("BLOCK_M","BLOCK_N","BLOCK_K","num_warps")))
    operands = tuple((str(q.data.dtype),tuple(q.data.shape),tuple(q.data.stride()),
                      tuple(q.scale_inv.shape),tuple(q.scale_inv.stride())) for q in (a,b))
    return (role,index,device_metadata(index),constants,operands,str(output_dtype),
            None if bias is None else (str(bias.dtype),tuple(bias.stride())))
