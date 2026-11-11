"""Shared regular/adaptive/fractional window reductions and fused saved state."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ..common.addressing import _logical, _matrix_index, _coordinates, _flatten, _load_value
from ..common.windows import _window, _divisor
from ..common.output import _store_value
from ...quantization.activation.indices import _pack_codes
from ...quantization.activation.affine import _affine_quantize_pack, _affine_step, _affine_quantize_pack_with_metadata


@triton.jit
def _forward(x, x_inverse, samples, output, public, routing, factor,
              scales, inverse, observations, snapshot, errors,
              SHAPE: tl.constexpr, OUT_SHAPE: tl.constexpr, X_META: tl.constexpr, GEO: tl.constexpr,
              CHANNELS: tl.constexpr, TOTAL: tl.constexpr, WINDOW: tl.constexpr,
              SAVE: tl.constexpr, PUBLIC: tl.constexpr, PACK: tl.constexpr, BITS: tl.constexpr, WORD_BLOCK: tl.constexpr,
              ROWS: tl.constexpr, COLS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
              FP8: tl.constexpr, CURRENT: tl.constexpr, OBSERVE: tl.constexpr, SNAPSHOT: tl.constexpr,
              CHECK: tl.constexpr, MAXIMUM: tl.constexpr, WB: tl.constexpr,
              coefficient=None, coefficient_step=None, coefficient_minimum=None,
              INLINE_COEFF: tl.constexpr = False, CGROUP: tl.constexpr = 1,
              CBITS: tl.constexpr = 2, CWORDS: tl.constexpr = 1, CBLOCK: tl.constexpr = 1, BLOCK: tl.constexpr = 32,
              BLOCK_COEFF: tl.constexpr = False):
    pid = tl.program_id(0).to(tl.int64)
    index = pid * BLOCK + tl.arange(0, BLOCK)
    valid = index < TOTAL
    logical = _logical(index, OUT_SHAPE, CHANNELS)
    spatial: tl.constexpr = GEO[2][0] * GEO[2][1] * GEO[2][2]
    input_spatial: tl.constexpr = GEO[1][0] * GEO[1][1] * GEO[1][2]
    nc = logical // spatial
    coordinates = _coordinates(logical % spatial, GEO[2])
    starts, extents = _window(nc, coordinates, samples, valid, GEO)
    step: tl.constexpr = (1, 1, 1) if GEO[0] in (2, 3, 4) else GEO[6]
    maximum: tl.constexpr = GEO[0] in (0, 2, 4) or GEO[0] == 5 and (GEO[9] == float("inf") or GEO[9] == -float("inf"))
    initial = ()
    for axis in tl.static_range(3):
        initial += (tl.where(starts[axis] < 0, starts[axis] + ((-starts[axis] + step[axis] - 1) // step[axis]) * step[axis], starts[axis]),)
    best_index = _flatten(initial[0], initial[1], initial[2], GEO[1])
    best_code = ((initial[0] - starts[0]) // step[0] * extents[1] +
                 (initial[1] - starts[1]) // step[1]) * extents[2] + (initial[2] - starts[2]) // step[2]
    best_code = tl.minimum(best_code, WINDOW - 1)
    best_value = tl.full((BLOCK,), -float("inf"), tl.float32)
    best_nan = tl.full((BLOCK,), False, tl.int1)
    seen = tl.full((BLOCK,), False, tl.int1)
    best_sign = tl.full((BLOCK,), 0, tl.int64)
    summed = tl.full((BLOCK,), 0.0, tl.float32)
    if BLOCK_COEFF:
        raw_tile = tl.full((BLOCK, WB), 0.0, tl.float32)
        raw_valid = tl.full((BLOCK, WB), False, tl.int1)
        raw_position = tl.full((BLOCK, WB), 0, tl.int64)
    for part in range(triton.cdiv(WINDOW, WB)):
        local = (part * WB + tl.arange(0, WB)).to(tl.int64)
        e_h, e_w = extents[1][:, None], extents[2][:, None]
        z = starts[0][:, None] + local[None, :] // (e_h * e_w) * step[0]
        y = starts[1][:, None] + local[None, :] // e_w % e_h * step[1]
        xx = starts[2][:, None] + local[None, :] % e_w * step[2]
        mask = (valid[:, None] & (local[None, :] < (extents[0] * extents[1] * extents[2])[:, None]) &
                (z >= 0) & (z < GEO[1][0]) & (y >= 0) & (y < GEO[1][1]) & (xx >= 0) & (xx < GEO[1][2]))
        position = _flatten(z, y, xx, GEO[1])
        raw = _load_value(x, x_inverse, nc[:, None] * input_spatial + position, mask, SHAPE, X_META)
        if BLOCK_COEFF:
            raw_tile, raw_valid, raw_position = raw, mask, position
        if maximum:
            value = (tl.abs(raw) if GEO[9] > 0 else -tl.abs(raw)) if GEO[0] == 5 else raw
            nan = mask & (value != value)
            nan_code = tl.max(tl.where(nan, local[None, :], -1), 1)
            candidate = tl.max(tl.where(mask & ~nan, value, -float("inf")), 1)
            code = tl.min(tl.where(mask & (value == candidate[:, None]), local[None, :], 9223372036854775807), 1)
            code = tl.where(nan_code >= 0, nan_code, code)
            selected_lane = tl.minimum(tl.maximum(code - part * WB, 0), WB - 1)
            candidate = tl.reshape(tl.gather(value, selected_lane[:, None], 1), (BLOCK,))
            chosen = mask & (local[None, :] == code[:, None])
            arg = tl.sum(tl.where(chosen, position, 0), 1)
            sign = tl.sum(tl.where(chosen, (raw > 0).to(tl.int64) + (raw < 0).to(tl.int64) * 2, 0), 1)
            present = tl.sum(mask.to(tl.int32), 1) > 0
            replace = present & (tl.where(nan_code >= 0, ~best_nan | (arg > best_index),
                                ~best_nan & (~seen | (candidate > best_value) | ((candidate == best_value) & (arg < best_index)))))
            best_value = tl.where(replace, candidate, best_value)
            best_index, best_code = tl.where(replace, arg, best_index), tl.where(replace, code, best_code)
            best_sign = tl.where(replace, sign, best_sign)
            best_nan, seen = best_nan | (nan_code >= 0), seen | present
        else:
            values = libdevice.pow(raw, GEO[9]) if GEO[0] == 5 else raw
            summed += tl.sum(tl.where(mask, values, 0.0), 1)
    if maximum:
        value = -best_value if GEO[0] == 5 and GEO[9] < 0 else best_value
        if PUBLIC:
            tl.store(public + index, best_index, valid)
        if SAVE:
            codes = best_code * 3 + best_sign if GEO[0] == 5 else best_code
            codes = tl.where(valid, codes, 0)
            if PACK:
                _pack_codes(codes, routing, pid, BITS, WORD_BLOCK, TOTAL)
            else:
                tl.store(routing + index, codes, valid)
    else:
        divisor = _divisor(starts, extents, GEO)
        value = summed / divisor
        if GEO[0] == 5:
            z_value = value
            base = value * (GEO[3][0] * GEO[3][1] * GEO[3][2])
            value = libdevice.pow(base, 1.0 / GEO[9])
            if SAVE:
                sign = (z_value > 0).to(tl.float32) - (z_value < 0).to(tl.float32)
                upstream = libdevice.pow(base, 1.0 / GEO[9] - 1.0) * (GEO[3][0] * GEO[3][1] * GEO[3][2] / GEO[9]) * sign
                derivative = tl.where(tl.abs(z_value) <= 0, 0.0, upstream) * sign / divisor
                tl.store(factor + index, derivative, valid)
    if BLOCK_COEFF:
        a = tl.trans(GEO[9] * libdevice.pow(raw_tile, GEO[9] - 1.0))
        a_valid = tl.trans(raw_valid)
        low_a = tl.min(tl.where(a_valid, a, float("inf")), 1)
        high_a = tl.max(tl.where(a_valid, a, -float("inf")), 1)
        scale_a = _affine_step(low_a, high_a, CBITS)
        flat_scale = tl.reshape(tl.broadcast_to(scale_a[:, None], (WB, BLOCK)), (WB * BLOCK,))
        flat_low = tl.reshape(tl.broadcast_to(low_a[:, None], (WB, BLOCK)), (WB * BLOCK,))
        words = _affine_quantize_pack_with_metadata(tl.reshape(a, (WB * BLOCK,)), tl.reshape(a_valid, (WB * BLOCK,)),
                                                    flat_scale, flat_low, CBITS, WB * BLOCK)
        words = tl.reshape(words, (WB, CWORDS))
        source_index = _matrix_index(nc[:, None] * input_spatial + raw_position, SHAPE, True)
        source_group = source_index // SHAPE[1] * triton.cdiv(SHAPE[1], 32) + source_index % SHAPE[1] // 32
        group_id = tl.min(source_group, 0)
        group_valid = tl.sum(raw_valid.to(tl.int32), 0) > 0
        word = tl.arange(0, CWORDS)
        tl.store(coefficient + group_id[:, None] * CWORDS + word[None, :], words, group_valid[:, None])
        tl.store(coefficient_step + group_id, scale_a, group_valid)
        tl.store(coefficient_minimum + group_id, low_a, group_valid)
    elif INLINE_COEFF:
        input_index = pid * CGROUP + tl.arange(0, CBLOCK)
        input_valid = input_index < TOTAL * (GEO[3][0] * GEO[3][1] * GEO[3][2])
        input_valid = input_valid & (tl.arange(0, CBLOCK) < CGROUP)
        a_input = _load_value(x, x_inverse, input_index, input_valid, SHAPE, X_META)
        a = GEO[9] * libdevice.pow(a_input, GEO[9] - 1.0)
        words, scale_a, low_a = _affine_quantize_pack(a, input_valid, CBITS, CBLOCK)
        word = tl.arange(0, CBLOCK // (32 // CBITS))
        tl.store(coefficient + pid * CWORDS + word, words, word < CWORDS)
        tl.store(coefficient_step + pid, scale_a)
        tl.store(coefficient_minimum + pid, low_a)
    _store_value(value, output, scales, inverse, observations, snapshot, errors, index, valid,
                 ROWS, COLS, BR, BC, FP8, CURRENT, OBSERVE, SNAPSHOT, CHECK, MAXIMUM, BLOCK)
