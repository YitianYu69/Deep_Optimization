"""Completed input-owned pooling gradients with inline saved-state decoding."""

import triton
import triton.language as tl

from ..common.addressing import _logical, _matrix_index, _coordinates, _flatten, _load_value
from ..common.windows import _window, _divisor, _candidates
from ..common.output import _store_value
from ...quantization.activation.indices import _unpack_codes


@triton.jit
def _load_coefficient(data, step, minimum, logical, valid, SHAPE: tl.constexpr,
                       CHANNELS: tl.constexpr, COLS: tl.constexpr, COMPRESSED: tl.constexpr,
                       BR: tl.constexpr, BC: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr):
    index = _matrix_index(logical, SHAPE, CHANNELS)
    if not COMPRESSED:
        return tl.load(data + index, valid, other=0).to(tl.float32)
    if BR:
        row, col = index // COLS, index % COLS
        group = row // BR * triton.cdiv(COLS, BC) + col // BC
        offset = row % BR * BC + col % BC
    else:
        group, offset = index // GROUP, index % GROUP
    word = tl.load(data + group * WORDS + offset // (32 // BITS), valid, other=0)
    code = ((word >> (offset % (32 // BITS) * BITS)) & ((1 << BITS) - 1)).to(tl.float32)
    low, scale = tl.load(minimum + group, valid, other=0), tl.load(step + group, valid, other=0)
    value = code * scale + low
    if BITS == 1:
        value = tl.where(scale < 0, (low * 0.5 + code * -scale) * 2, value)
    return value


@triton.jit
def _backward(dy, dy_inverse, routing, coefficient, steps, minima, factor, samples,
               output, scales, inverse, observations, snapshot, errors,
               SHAPE: tl.constexpr, OUT_SHAPE: tl.constexpr, DY_META: tl.constexpr, GEO: tl.constexpr,
               CHANNELS: tl.constexpr, ROUTE_CHANNELS: tl.constexpr, TOTAL: tl.constexpr,
               CANDIDATES: tl.constexpr, ROUTE_BITS: tl.constexpr,
               COEFF_CHANNELS: tl.constexpr, COEFF_COLS: tl.constexpr, COMPRESSED: tl.constexpr,
               CBR: tl.constexpr, CBC: tl.constexpr, CGROUP: tl.constexpr, CWORDS: tl.constexpr, CBITS: tl.constexpr,
               ROWS: tl.constexpr, COLS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
               FP8: tl.constexpr, CURRENT: tl.constexpr, OBSERVE: tl.constexpr, CHECK: tl.constexpr,
               MAXIMUM: tl.constexpr, WB: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32)
    valid = index < TOTAL
    logical = _logical(index, SHAPE, CHANNELS)
    i_spatial: tl.constexpr = GEO[1][0] * GEO[1][1] * GEO[1][2]
    o_spatial: tl.constexpr = GEO[2][0] * GEO[2][1] * GEO[2][2]
    nc, position = logical // i_spatial, logical % i_spatial
    if GEO[0] == 6:
        route_index = _matrix_index(logical, SHAPE, ROUTE_CHANNELS)
        route = _unpack_codes(routing, route_index, valid, ROUTE_BITS) if ROUTE_BITS >= 0 else tl.load(routing + route_index, valid, other=0)
        value = _load_value(dy, dy_inverse, nc * o_spatial + route, valid & (route < o_spatial), OUT_SHAPE, DY_META)
    else:
        positions = _coordinates(position, GEO[1])
        lows, highs = _candidates(nc, positions, samples, valid, GEO)
        extents = (highs[0] - lows[0], highs[1] - lows[1], highs[2] - lows[2])
        result = tl.full((32,), 0.0, tl.float32)
        maximum: tl.constexpr = GEO[0] in (0, 2, 4) or GEO[0] == 5 and (GEO[9] == float("inf") or GEO[9] == -float("inf"))
        step: tl.constexpr = (1, 1, 1) if GEO[0] in (2, 3, 4) else GEO[6]
        for part in range(triton.cdiv(CANDIDATES, WB)):
            local = part * WB + tl.arange(0, WB)
            eh = tl.maximum(extents[1][:, None], 1)
            ew = tl.maximum(extents[2][:, None], 1)
            z = lows[0][:, None] + local[None, :] // (eh * ew)
            y = lows[1][:, None] + local[None, :] // ew % eh
            x = lows[2][:, None] + local[None, :] % ew
            mask = valid[:, None] & (local[None, :] < (extents[0] * extents[1] * extents[2])[:, None])
            starts, sizes = _window(nc[:, None], (z, y, x), samples, mask, GEO)
            rel = ()
            for axis in tl.static_range(3):
                delta = positions[axis][:, None] - starts[axis]
                mask = mask & (delta >= 0) & (delta < sizes[axis] * step[axis]) & (delta % step[axis] == 0)
                rel += (delta // step[axis],)
            o_logical = nc[:, None] * o_spatial + _flatten(z, y, x, GEO[2])
            o_index = _matrix_index(o_logical, OUT_SHAPE, ROUTE_CHANNELS)
            gradient = _load_value(dy, dy_inverse, o_logical, mask, OUT_SHAPE, DY_META)
            if maximum:
                route = _unpack_codes(routing, o_index, mask, ROUTE_BITS) if ROUTE_BITS >= 0 else tl.load(routing + o_index, mask, other=0)
                if GEO[0] == 5:
                    sign = (route % 3 == 1).to(tl.float32) - (route % 3 == 2).to(tl.float32)
                    gradient *= sign
                    route = route // 3
                code = (rel[0] * sizes[1] + rel[1]) * sizes[2] + rel[2]
                gradient = tl.where(mask & (route == code), gradient, 0.0)
            elif GEO[0] == 5:
                gradient *= tl.load(factor + o_index, mask, other=0)
            else:
                divisor = _divisor(starts, sizes, GEO)
                gradient = tl.where(mask, gradient / divisor, 0.0)
            result += tl.sum(tl.where(mask, gradient, 0.0), 1)
        if GEO[0] == 5 and not maximum:
            a = _load_coefficient(coefficient, steps, minima, logical, valid, SHAPE, COEFF_CHANNELS,
                                  COEFF_COLS, COMPRESSED, CBR, CBC, CGROUP, CWORDS, CBITS)
            result *= a
        value = result
    _store_value(value, output, scales, inverse, observations, snapshot, errors, index, valid,
                 ROWS, COLS, BR, BC, FP8, CURRENT, OBSERVE, False, CHECK, MAXIMUM, 32)
