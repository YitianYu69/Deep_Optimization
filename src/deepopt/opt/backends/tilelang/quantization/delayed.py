"""Delayed conversion and ordered, explicit-tensor scale-state updates."""

from functools import lru_cache

import torch

from .._runtime import compile_kernel, device_key, require_cuda, runtime, storage_view
from ..gradient.reduction import sum_columns
from ..scaling.reduction import block_amax
from .fp8 import amax_scale, cast_scaled


@lru_cache(maxsize=32)
def _snapshot(device):
    _, T = runtime()

    @T.prim_func
    def kernel(S: T.Tensor((1,), "float32"), Y: T.Tensor((2,), "float32")):
        with T.Kernel(1, threads=32):
            for lane in T.Parallel(1):
                Y[0] = S[0]
                Y[1] = T.ieee_fdiv(T.float32(1), S[0])

    return compile_kernel(kernel, device)


def convert(x, scale, dtype, *, normal, transposed, bias, observe, check_finite, autotune=True):
    if type(autotune) is not bool:
        raise ValueError("autotune must be a boolean")
    device = require_cuda(x, scale)
    if check_finite:
        try:
            amax_scale(x, torch.finfo(dtype).max, None, True)
        except ValueError as error:
            raise ValueError("Delayed scaling requires finite inputs") from error
    snapshot = torch.empty(2, device=device, dtype=torch.float32)
    with torch.cuda.device(device):
        _snapshot(device_key(device))(scale.reshape(1), snapshot)
    values = cast_scaled(x, snapshot[:1], dtype, row_alignment=16) if normal else None
    transpose = cast_scaled(x, snapshot[:1], dtype, transpose=True, row_alignment=16) if transposed else None
    observations = block_amax(x, 32, 32, minimum=0).reshape(-1) if observe else None
    db = sum_columns(x) if bias else None
    return values, transpose, snapshot, observations, db


@lru_cache(maxsize=128)
def _observe(count, span, groups, group, device):
    _, T = runtime()

    @T.prim_func
    def kernel(X: T.Tensor((span,), "float32"), P: T.Tensor((3 * groups,), "float32")):
        with T.Kernel(1, threads=32):
            for lane in T.Parallel(1):
                maximum = T.alloc_local((1,), "float32")
                kind = T.alloc_local((1,), "float32")
                maximum[0] = P[group]
                kind[0] = P[2 * groups + group]
                for index in T.serial(count):
                    value = X[index]
                    if value != value:
                        kind[0] = 2.0
                    elif T.abs(value) == T.infinity("float32"):
                        kind[0] = T.max(kind[0], 1.0)
                    else:
                        maximum[0] = T.max(maximum[0], value)
                P[group] = maximum[0]
                P[groups + group] = 1.0
                P[2 * groups + group] = kind[0]

    return compile_kernel(kernel, device)


def reduce_observations(observations, groups, device, cache=None):
    packed = torch.zeros((3, groups), device=device, dtype=torch.float32)
    for group, values in observations:
        if values.numel() < 1:
            raise ValueError("Amax observations must be nonempty")
        require_cuda(values, packed)
        if not values.is_contiguous():
            raise ValueError("Amax observations must be contiguous")
        source = storage_view(values)
        with torch.cuda.device(device):
            _observe(values.numel(), source.numel(), groups, group, device_key(device))(source, packed.view(-1))
        values.record_stream(torch.cuda.current_stream(device))
    return packed


@lru_cache(maxsize=128)
def _update(length, groups, group, role, margin, recent, maximum, device):
    _, T = runtime()

    @T.prim_func
    def kernel(S: T.Tensor((3,), "float32"), H: T.Tensor((3 * length,), "float32"),
               U: T.Tensor((3,), "int64"), G: T.Tensor((3,), "int64"),
               P: T.Tensor((3 * groups,), "float32")):
        with T.Kernel(1, threads=32):
            for lane in T.Parallel(1):
                if P[groups + group] > 0:
                    observed = T.if_then_else(P[2 * groups + group] == 2, T.float32(float("nan")),
                                              T.if_then_else(P[2 * groups + group] == 1,
                                                             T.infinity("float32"), P[group]))
                    slot = U[role] % length
                    H[role * length + slot] = observed
                    amax = T.alloc_local((1,), "float32")
                    amax[0] = observed
                    if not recent:
                        amax[0] = 0.0
                        for offset in T.serial(length):
                            value = H[role * length + offset]
                            amax[0] = T.if_then_else(amax[0] != amax[0], amax[0],
                                T.if_then_else(value != value, value, T.max(amax[0], value)))
                    candidate = T.ieee_fdiv(T.float32(maximum), amax[0]) * T.exp2(T.float32(-margin))
                    if amax[0] > 0 and amax[0] < T.infinity("float32") and candidate > 0 and candidate < T.infinity("float32"):
                        S[role] = candidate
                    U[role] = U[role] + T.int64(1)
                    G[role] = G[role] + T.int64(1)

    return compile_kernel(kernel, device)


def update_states(states, packed, cache=None):
    groups = len(states) * 3
    for index, state in enumerate(states):
        device = require_cuda(state.scales, state.history, state.updates, state.generations, packed)
        for role, name in enumerate(("activation", "weight", "grad")):
            with torch.cuda.device(device):
                _update(state.options.history_length, groups, index * 3 + role, role,
                        state.options.margin, state.options.amax_algorithm == "most_recent",
                        float(torch.finfo(state.dtypes[name]).max), device_key(device))(
                            state.scales, state.history.view(-1), state.updates, state.generations, packed.view(-1))
