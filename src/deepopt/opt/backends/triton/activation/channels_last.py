"""Physical channel traversal with unchanged logical NCHW derivative groups."""

import triton
import triton.language as tl

from .math import _evaluate
from .elementwise_backward import _coefficient
from ..quantization.activation.affine import _affine_step, _affine_quantize_pack, _affine_quantize_pack_with_metadata


@triton.jit
def _channels_last_activation(x, output, saved, steps, minima, errors,
                              BATCH: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
                              GROUP: tl.constexpr, GROUPS: tl.constexpr, WORDS: tl.constexpr,
                              PERIOD: tl.constexpr, CHANNEL_PROGRAMS: tl.constexpr,
                              KIND: tl.constexpr, APPROXIMATE: tl.constexpr,
                              SAVE: tl.constexpr, PACKED: tl.constexpr, BITS: tl.constexpr,
                              CHECK: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr,
                              input_inverse=None, scale=None, output_inverse=None,
                              INPUT_FP8: tl.constexpr = False, OUTPUT_FP8: tl.constexpr = False,
                              MAXIMUM: tl.constexpr = 0.0):
    pid = tl.program_id(0).to(tl.int64)
    batch = pid // (PERIOD * CHANNEL_PROGRAMS)
    phase = pid // CHANNEL_PROGRAMS % PERIOD
    channel = phase + (pid % CHANNEL_PROGRAMS * BLOCK_C + tl.arange(0, BLOCK_C)) * PERIOD
    logical_channel = batch * C + channel
    group = tl.cdiv(logical_channel * S, GROUP) + tl.program_id(1).to(tl.int64)
    valid_group = ((batch < BATCH) & (channel < C) & (group < GROUPS)
                   & (group < tl.cdiv((logical_channel + 1) * S, GROUP)))
    offset = tl.arange(0, BLOCK_S)
    logical = group[None, :] * GROUP + offset[:, None]
    valid = valid_group[None, :] & (offset[:, None] < GROUP) & (logical < BATCH * C * S)
    address = (logical // (C * S) * S + logical % S) * C + logical // S % C
    value = tl.load(x + address, valid, other=0.).to(tl.float32)
    if INPUT_FP8:
        value *= tl.load(input_inverse)
    if KIND == "relu":
        y = tl.where(value < 0., 0., value)
        derivative = (~(value <= 0.)).to(tl.float32)
    else:
        y, derivative = _evaluate(value, KIND, APPROXIMATE)
    if OUTPUT_FP8:
        multiplier = tl.load(scale)
        result = tl.minimum(tl.maximum(y * multiplier, -MAXIMUM), MAXIMUM)
        tl.store(output + logical, result.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
        if pid == 0 and tl.program_id(1) == 0:
            tl.store(output_inverse, tl.div_rn(1.0, multiplier))
    else:
        tl.store(output + address, y, valid)
    if SAVE:
        if PACKED:
            if KIND == "relu":
                codes = tl.reshape(tl.trans(valid & (derivative != 0)).to(tl.int32), (BLOCK_C, BLOCK_S // 32, 32))
                words = tl.sum(codes << tl.arange(0, 32)[None, None, :], 2)
            else:
                low = tl.min(tl.where(valid, derivative, float("inf")), 0)
                high = tl.max(tl.where(valid, derivative, -float("inf")), 0)
                step = _affine_step(low, high, BITS)
                words = _affine_quantize_pack_with_metadata(tl.trans(derivative), tl.trans(valid),
                    step[:, None], low[:, None], BITS, BLOCK_S, BATCH=BLOCK_C)
                tl.store(steps + group, step, valid_group)
                tl.store(minima + group, low, valid_group)
            word = tl.arange(0, BLOCK_S // (32 // BITS))
            word_valid = word[None, :] < WORDS
            if KIND == "relu":
                word_valid = word_valid & (group[:, None] * WORDS + word[None, :] < tl.cdiv(BATCH * C * S, 32))
            tl.store(saved + group[:, None] * WORDS + word[None, :], words,
                     valid_group[:, None] & word_valid)
        else:
            tl.store(saved + logical, derivative, valid)
    if CHECK:
        bad = valid & ~(tl.abs(derivative) < float("inf"))
        if OUTPUT_FP8:
            bad = bad | (valid & (~(tl.abs(y) < float("inf"))
                         | (multiplier <= 0) | ~(tl.abs(multiplier) < float("inf"))))
        tl.store(errors + group, tl.max(bad.to(tl.int32), 0), valid_group)


@triton.jit
def _channels_last_group_activation(x, output, saved, steps, minima, errors,
                                    C: tl.constexpr, S: tl.constexpr, TOTAL: tl.constexpr,
                                    GROUP: tl.constexpr, WORDS: tl.constexpr,
                                    KIND: tl.constexpr, APPROXIMATE: tl.constexpr,
                                    SAVE: tl.constexpr, PACKED: tl.constexpr, BITS: tl.constexpr,
                                    CHECK: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr,
                                    BLOCK_GROUP: tl.constexpr, input_inverse=None, scale=None,
                                    output_inverse=None, INPUT_FP8: tl.constexpr = False,
                                    OUTPUT_FP8: tl.constexpr = False, MAXIMUM: tl.constexpr = 0.0):
    # One owner per logical NCHW group, including groups crossing channels
    # and batches. Only the load traversal changes: channel lanes are dense
    # in physical NHWC storage. Codes are locally gathered into codec order.
    group = tl.program_id(0).to(tl.int64)
    channel_start = group * GROUP // S
    channel = channel_start + tl.arange(0, BLOCK_C)
    spatial = tl.arange(0, BLOCK_S)
    logical = channel[None, :] * S + spatial[:, None]
    valid = ((spatial[:, None] < S) & (logical >= group * GROUP)
             & (logical < (group + 1) * GROUP) & (logical < TOTAL))
    address = (channel[None, :] // C * S + spatial[:, None]) * C + channel[None, :] % C
    value = tl.load(x + address, valid, other=0.).to(tl.float32)
    if INPUT_FP8:
        value *= tl.load(input_inverse)
    if KIND == "relu":
        y = tl.where(value < 0., 0., value)
        derivative = (~(value <= 0.)).to(tl.float32)
    else:
        y, derivative = _evaluate(value, KIND, APPROXIMATE)
    if OUTPUT_FP8:
        multiplier = tl.load(scale)
        result = tl.minimum(tl.maximum(y * multiplier, -MAXIMUM), MAXIMUM)
        tl.store(output + logical, result.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
        if group == 0:
            tl.store(output_inverse, tl.div_rn(1.0, multiplier))
    else:
        tl.store(output + address, y, valid)
    offset = tl.arange(0, BLOCK_GROUP)
    index = group * GROUP + offset
    local_index = (index % S * BLOCK_C + index // S - channel_start).to(tl.int32)
    coefficient = tl.gather(tl.reshape(derivative, (BLOCK_S * BLOCK_C,)), local_index, 0)
    keep = (offset < GROUP) & (index < TOTAL)
    if SAVE:
        if PACKED:
            if KIND == "relu":
                codes = tl.reshape((keep & (coefficient != 0)).to(tl.int32), (BLOCK_GROUP // 32, 32))
                words = tl.sum(codes << tl.arange(0, 32)[None, :], 1)
                destination = group * WORDS + tl.arange(0, BLOCK_GROUP // 32)
                tl.store(saved + destination, words, destination < tl.cdiv(TOTAL, 32))
            else:
                words, step, minimum = _affine_quantize_pack(coefficient, keep, BITS, BLOCK_GROUP)
                tl.store(saved + group * WORDS + tl.arange(0, BLOCK_GROUP // (32 // BITS)), words)
                tl.store(steps + group, step)
                tl.store(minima + group, minimum)
        else:
            tl.store(saved + index, coefficient, keep)
    if CHECK:
        bad = keep & ~(tl.abs(coefficient) < float("inf"))
        if OUTPUT_FP8:
            y_bad = valid & (~(tl.abs(y) < float("inf"))
                            | (multiplier <= 0) | ~(tl.abs(multiplier) < float("inf")))
            tl.store(errors + group, tl.maximum(tl.max(bad.to(tl.int32), 0),
                     tl.max(tl.max(y_bad.to(tl.int32), 0), 0)))
        else:
            tl.store(errors + group, tl.max(bad.to(tl.int32), 0))


@triton.jit
def _channels_last_activation_dx(dy, saved, steps, minima, output,
                                 C: tl.constexpr, S: tl.constexpr, TOTAL: tl.constexpr,
                                 SAVED_COLS: tl.constexpr, GROUP: tl.constexpr, WORDS: tl.constexpr,
                                 BITS: tl.constexpr, MASK: tl.constexpr, PACKED: tl.constexpr, BLOCK: tl.constexpr):
    physical = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = physical < TOTAL
    logical = (physical // (C * S) * C + physical % C) * S + physical // C % S
    coefficient = _coefficient(saved, steps, minima, logical, valid, SAVED_COLS, GROUP,
                               0, 0, WORDS, BITS, MASK, PACKED)
    gradient = tl.load(dy + physical, valid, other=0.).to(tl.float32)
    value = tl.where(coefficient != 0, gradient, 0.) if MASK else gradient * coefficient
    tl.store(output + physical, value, valid)
