"""Output-scale epilogue shared by pooling values and completed gradients."""

import triton
import triton.language as tl

from ...quantization.fp8 import _scale


@triton.jit
def _store_value(value, output, scales, inverse, observations, snapshot, errors, index, valid,
                  ROWS: tl.constexpr, COLS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
                  FP8: tl.constexpr, CURRENT: tl.constexpr, OBSERVE: tl.constexpr, SNAPSHOT: tl.constexpr,
                  CHECK: tl.constexpr, MAXIMUM: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    if FP8:
        if CURRENT:
            if BR >= ROWS and BC >= COLS:
                multiplier = _scale(tl.max(tl.where(valid, tl.abs(value), 0.0), 0), MAXIMUM)
                tl.store(inverse, tl.div_rn(1.0, multiplier))
            else:
                groups = tl.reshape(tl.where(valid, tl.abs(value), 0.0), (BLOCK // (BR * BC), BR * BC))
                local = _scale(tl.max(groups, 1), MAXIMUM)
                multiplier = tl.reshape(tl.broadcast_to(local[:, None], (BLOCK // (BR * BC), BR * BC)), (BLOCK,))
                group_index = pid * (BLOCK // (BR * BC)) + tl.arange(0, BLOCK // (BR * BC))
                tl.store(inverse + group_index, tl.div_rn(1.0, local), group_index < triton.cdiv(ROWS, BR) * triton.cdiv(COLS, BC))
        else:
            row, col = index // COLS, index % COLS
            scale_index = row // BR * triton.cdiv(COLS, BC) + col // BC
            multiplier = tl.load(scales + scale_index, valid, other=1)
            tl.store(inverse + scale_index, tl.div_rn(1.0, multiplier), valid & (row % BR == 0) & (col % BC == 0))
        result = tl.minimum(tl.maximum(value * multiplier, -MAXIMUM), MAXIMUM)
        tl.store(output + index, result.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
        if OBSERVE:
            tl.store(observations + pid, tl.max(tl.where(valid, tl.abs(value), 0.0), 0))
        if SNAPSHOT:
            if pid == 0:
                scale = tl.load(scales)
                tl.store(snapshot, scale)
                tl.store(snapshot + 1, tl.div_rn(1.0, scale))
        if CHECK:
            bad = valid & (~(tl.abs(value) < float("inf")) | ~(tl.abs(multiplier) < float("inf")) | (multiplier <= 0))
            tl.store(errors + pid, tl.max(bad.to(tl.int32), 0))
    else:
        tl.store(output + index, value.to(output.dtype.element_ty), valid)


@triton.jit
def _cast(input, output, scales, inverse, observations, snapshot, errors,
           ROWS: tl.constexpr, COLS: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr,
           OBSERVE: tl.constexpr, SNAPSHOT: tl.constexpr, CHECK: tl.constexpr, MAXIMUM: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = index < ROWS * COLS
    value = tl.load(input + index, valid, other=0).to(tl.float32)
    _store_value(value, output, scales, inverse, observations, snapshot, errors, index, valid,
                 ROWS, COLS, BR, BC, True, False, OBSERVE, SNAPSHOT, CHECK, MAXIMUM, BLOCK)
