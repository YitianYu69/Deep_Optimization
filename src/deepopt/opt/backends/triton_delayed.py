"""Single-read delayed conversion and ordered bias/metadata reductions."""

import inspect

import torch
import triton
import triton.language as tl
from ._delayed_launch import CONVERSION_KEYS, conversion_cache_key, scratch_tile, observation_rows
from ._gemm_launch import selected_config, launch_selected


_cache = {"cache_results": True} if "cache_results" in inspect.signature(triton.autotune).parameters else {}


@triton.autotune(configs=[triton.Config({"BM": 32, "BN": 32}, num_warps=4),
                          triton.Config({"BM": 64, "BN": 64}, num_warps=8),
                          triton.Config({"BM": 32, "BN": 128}, num_warps=4)],
                 key=list(CONVERSION_KEYS), **_cache)
@triton.jit
def _convert(x, scale_ptr, normal, transpose, snapshot, amax, db, errors,
             M, N, SM, SN, PAD_M: tl.constexpr, PAD_N: tl.constexpr,
             NORMAL: tl.constexpr, TRANSPOSE: tl.constexpr, BIAS: tl.constexpr,
             OBSERVE: tl.constexpr, CHECK: tl.constexpr, DEVICE: tl.constexpr,
             MAXIMUM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rows, cols = pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN)
    values = tl.load(x + rows[:, None] * SM + cols[None, :] * SN,
                     mask=(rows[:, None] < M) & (cols[None, :] < N), other=0.).to(tl.float32)
    scale = tl.load(scale_ptr)
    scaled = tl.minimum(tl.maximum(values * scale, -MAXIMUM), MAXIMUM)
    # Triton exposes neither isnan nor isinf in triton.language. IEEE
    # comparisons preserve NaN handling without a CUDA-only libdevice import.
    converted = tl.where(values != values, float("nan"), scaled).to(normal.dtype.element_ty,
                                                                                fp_downcast_rounding="rtne")
    if NORMAL:
        tl.store(normal + rows[:, None] * PAD_N + cols[None, :], converted,
                 mask=(rows[:, None] < M) & (cols[None, :] < PAD_N))
    if TRANSPOSE:
        tl.store(transpose + cols[:, None] * PAD_M + rows[None, :], converted.T,
                 mask=(cols[:, None] < N) & (rows[None, :] < PAD_M))
    if pm == 0 and pn == 0:
        tl.store(snapshot, scale)
        tl.store(snapshot + 1, tl.div_rn(1., scale))
    if OBSERVE:
        tl.store(amax + pm * tl.cdiv(N, BN) + pn, tl.max(tl.max(tl.abs(values), 0), 0))
    if BIAS:
        tl.store(db + pm * N + cols, tl.sum(values, 0), mask=cols < N)
    if CHECK:
        invalid = tl.max(tl.max((~(tl.abs(values) < float("inf"))).to(tl.int32), 0), 0)
        tl.atomic_or(errors, invalid, sem="relaxed")


@triton.jit
def _sum_bias(source, output, ROWS, N, CHUNK: tl.constexpr, COLS: tl.constexpr):
    group = tl.program_id(0)
    rows = group * CHUNK + tl.arange(0, CHUNK)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    values = tl.load(source + rows[:, None] * N + cols[None, :],
                     mask=(rows[:, None] < ROWS) & (cols[None, :] < N), other=0.)
    tl.store(output + group * N + cols, tl.sum(values, 0), mask=cols < N)


def _finish_bias(partials, rows, n):
    while rows > 1:
        groups = triton.cdiv(rows, 256)
        output = torch.empty((groups, n), dtype=torch.float32, device=partials.device)
        _sum_bias[(groups, triton.cdiv(n, 32))](partials, output, rows, n,
                                               CHUNK=min(256, triton.next_power_of_2(rows)), COLS=32)
        partials, rows = output, groups
    return partials[:1].view(n)


def convert(x, scale, dtype, *, normal, transposed, bias, observe, check_finite):
    m, n = x.shape
    pm, pn = triton.cdiv(m, 16) * 16, triton.cdiv(n, 16) * 16
    with torch.cuda.device(x.device):
        key = conversion_cache_key(x, scale, dtype, normal=normal, transposed=transposed, bias=bias,
                                   observe=observe, check_finite=check_finite)
        bm, bn = scratch_tile(_convert, key)
        snapshot = torch.empty(2, device=x.device, dtype=torch.float32)
        values = torch.empty((m, pn), device=x.device, dtype=dtype) if normal else None
        transpose = torch.empty((n, pm), device=x.device, dtype=dtype) if transposed else None
        dummy = values if normal else transpose
        amax = torch.empty(triton.cdiv(m, bm) * triton.cdiv(n, bn), device=x.device, dtype=torch.float32) if observe else snapshot
        db = torch.empty((triton.cdiv(m, bm), n), device=x.device, dtype=torch.float32) if bias else snapshot
        errors = torch.zeros((), device=x.device, dtype=torch.int32) if check_finite else snapshot
        args = (x, scale, dummy, transpose if transposed else dummy, snapshot, amax, db, errors, m, n, *x.stride())
        kwargs = dict(PAD_M=pm, PAD_N=pn, NORMAL=normal, TRANSPOSE=transposed, BIAS=bias,
                      OBSERVE=observe, CHECK=check_finite, DEVICE=x.device.index, MAXIMUM=torch.finfo(dtype).max)
        selected = selected_config(_convert, key)
        if selected is None:
            _convert[lambda meta: (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]))](*args, **kwargs)
            selected = selected_config(_convert, key)
            if selected is None:
                raise RuntimeError("Triton converter winner is missing from its per-key autotune cache")
        else:
            launch_selected(_convert, selected, args, kwargs,
                             (triton.cdiv(m, selected.kwargs["BM"]), triton.cdiv(n, selected.kwargs["BN"])))
        bm, bn = (selected.kwargs[name] for name in ("BM", "BN"))
        if check_finite and errors.item():
            raise ValueError("Delayed scaling requires finite inputs")
        return (values[:, :n] if normal else None, transpose[:, :m] if transposed else None, snapshot,
                amax[:triton.cdiv(m, bm) * triton.cdiv(n, bn)] if observe else None,
                _finish_bias(db, triton.cdiv(m, bm), n) if bias else None)


@triton.jit
def _observations(metadata, packed, COUNT, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    low, high = 0, COUNT
    # Prefix ends map one flat program id to exactly one valid chunk. Small
    # weight observations no longer inherit the largest activation's grid.
    while low < high:
        middle = (low + high) // 2
        end = tl.load(metadata + middle * 4 + 3)
        after = pid >= end
        low = tl.where(after, middle + 1, low)
        high = tl.where(after, high, middle)
    observation = low
    first = tl.load(metadata + (observation - 1) * 4 + 3, mask=observation > 0, other=0)
    pointer = tl.load(metadata + observation * 4).to(tl.pointer_type(tl.float32))
    size = tl.load(metadata + observation * 4 + 1)
    group = tl.load(metadata + observation * 4 + 2)
    offsets = (pid - first) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(pointer + offsets, mask=offsets < size, other=0.)
    kind = tl.max(tl.where(values != values, 2, tl.where(tl.abs(values) == float("inf"), 1, 0)), 0)
    maximum = tl.max(tl.where(tl.abs(values) < float("inf"), values, 0.), 0)
    tl.atomic_max(packed + group, maximum, sem="relaxed")
    tl.atomic_max(packed + GROUPS + group, 1., sem="relaxed")
    tl.atomic_max(packed + 2 * GROUPS + group, kind.to(tl.float32), sem="relaxed")


def reduce_observations(observations, groups, device, cache=None):
    packed = torch.zeros((3, groups), device=device, dtype=torch.float32)
    if observations:
        rows, programs = observation_rows(observations)
        key = ("observations", tuple(rows), device)
        metadata = (cache.table(key, lambda: rows, device) if cache is not None else
                    torch.tensor(rows, device=device, dtype=torch.int64))
        for _, value in observations:
            value.record_stream(torch.cuda.current_stream(device))
        _observations[(programs,)](metadata, packed, len(observations), groups, BLOCK=1024)
    return packed


@triton.jit
def _update(metadata, packed, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    if tl.load(packed + GROUPS + group) > 0:
        scales = tl.load(metadata + group * 8).to(tl.pointer_type(tl.float32))
        history = tl.load(metadata + group * 8 + 1).to(tl.pointer_type(tl.float32))
        steps = tl.load(metadata + group * 8 + 2).to(tl.pointer_type(tl.int64))
        generations = tl.load(metadata + group * 8 + 3).to(tl.pointer_type(tl.int64))
        length = tl.load(metadata + group * 8 + 4)
        margin = tl.load(metadata + group * 8 + 5).to(tl.float32)
        recent = tl.load(metadata + group * 8 + 6)
        maximum = tl.load(metadata + group * 8 + 7).to(tl.float32)
        kind = tl.load(packed + 2 * GROUPS + group)
        observed = tl.where(kind == 2, float("nan"),
                            tl.where(kind == 1, float("inf"), tl.load(packed + group)))
        count = tl.load(steps)
        offsets = tl.arange(0, BLOCK)
        slot = count % length
        values = tl.load(history + offsets, mask=offsets < length, other=0.)
        values = tl.where(offsets == slot, observed, values)
        amax = tl.where(recent != 0, observed, tl.max(values, 0))
        candidate = tl.div_rn(maximum, amax) * tl.exp2(-margin)
        previous = tl.load(scales)
        valid = (amax > 0) & (amax < float("inf")) & (candidate > 0) & (candidate < float("inf"))
        tl.store(scales, tl.where(valid, candidate, previous))
        tl.store(history + slot, observed)
        tl.store(steps, count + 1)
        tl.store(generations, tl.load(generations) + 1)


def update_states(states, packed, cache=None):
    if cache is None:
        from ._delayed_launch import MetadataCache

        cache = MetadataCache()
    metadata = cache.state_table(states, packed.device)
    _update[(len(states) * 3,)](metadata, packed, len(states) * 3,
                               BLOCK=triton.next_power_of_2(max(state.options.history_length for state in states)))
