"""Affine activation quantization/packing and helpers shared by fused FP8 casts.

Codes occupy the low bits first in int32 words. Metadata is FP32 so it can
represent BF16 activations without imposing FP16's much smaller range.
"""

import torch
import triton
import triton.language as tl


def _bits_consts(bits: int, group_size: int):
    if bits not in (1, 2, 4, 8):
        raise ValueError("bits must be one of 1, 2, 4, or 8")
    if type(group_size) is not int or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    vpw = 32 // bits
    return vpw, triton.cdiv(group_size, vpw), (1 << bits) - 1


@triton.jit
def _affine_step(minimum, maximum, BITS: tl.constexpr):
    QMAX: tl.constexpr = (1 << BITS) - 1
    span = maximum - minimum
    if BITS == 1:
        # A negative half-step represents an overflowing positive full step.
        return tl.where(span == float("inf"), -(maximum * 0.5 - minimum * 0.5), span)
    return tl.where(span == float("inf"),
                    tl.div_rn(maximum, QMAX) - tl.div_rn(minimum, QMAX),
                    tl.div_rn(span, QMAX))


@triton.jit
def _affine_quantize_pack_with_metadata(values, mask, step, minimum, BITS: tl.constexpr, BLOCK: tl.constexpr):
    """Quantize using supplied affine metadata and pack the rounded codes."""
    QMAX: tl.constexpr = (1 << BITS) - 1
    VPW: tl.constexpr = 32 // BITS
    difference = values - minimum
    normalized = tl.where(difference == float("inf"),
                          tl.div_rn(values, step) - tl.div_rn(minimum, step),
                          tl.div_rn(difference, step))
    if BITS == 1:
        normalized = tl.where(step < 0, tl.div_rn(values * 0.5 - minimum * 0.5, -step), normalized)
        scaled = tl.where(step != 0, normalized, 0.0)
    else:
        scaled = tl.where(step > 0, normalized, 0.0)
    scaled = tl.minimum(tl.maximum(scaled, 0.0), QMAX)
    lower = tl.floor(scaled)
    fraction = scaled - lower
    rounded = lower.to(tl.int32) + (
        (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    ).to(tl.int32)
    codes = tl.where(mask, rounded, 0)
    codes = tl.reshape(codes, (BLOCK // VPW, VPW))
    return tl.sum(codes << (tl.arange(0, VPW) * BITS)[None, :], axis=1)


@triton.jit
def _affine_quantize_pack(values, mask, BITS: tl.constexpr, BLOCK: tl.constexpr):
    minimum = tl.min(tl.where(mask, values, float("inf")), axis=0)
    maximum = tl.max(tl.where(mask, values, float("-inf")), axis=0)
    step = _affine_step(minimum, maximum, BITS)
    return _affine_quantize_pack_with_metadata(values, mask, step, minimum, BITS, BLOCK), step, minimum


@triton.jit
def _coordinates(group, offsets, ROWS: tl.constexpr, COLS: tl.constexpr,
                 GROUP: tl.constexpr, BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr):
    if BLOCK_ROWS > 0:
        column_groups: tl.constexpr = triton.cdiv(COLS, BLOCK_COLS)
        rows = group // column_groups * BLOCK_ROWS + offsets // BLOCK_COLS
        cols = group % column_groups * BLOCK_COLS + offsets % BLOCK_COLS
        valid = (offsets < GROUP) & (rows < ROWS) & (cols < COLS)
    else:
        positions = group * GROUP + offsets
        rows, cols = positions // COLS, positions % COLS
        valid = (offsets < GROUP) & (positions < ROWS * COLS)
    return rows, cols, valid


@triton.jit
def _quantize_pack_groups(x_ptr, packed_ptr, step_ptr, minimum_ptr,
                 STRIDE0: tl.constexpr, STRIDE1: tl.constexpr,
                 ROWS: tl.constexpr, COLS: tl.constexpr, GROUP: tl.constexpr,
                 BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                 WORDS: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    rows, cols, valid = _coordinates(group, offsets, ROWS, COLS, GROUP, BLOCK_ROWS, BLOCK_COLS)
    values = tl.load(x_ptr + rows * STRIDE0 + cols * STRIDE1, valid, other=0).to(tl.float32)
    words, step, minimum = _affine_quantize_pack(values, valid, BITS, BLOCK)
    word_offsets = tl.arange(0, BLOCK // (32 // BITS))
    tl.store(packed_ptr + group * WORDS + word_offsets, words, word_offsets < WORDS)
    tl.store(step_ptr + group, step)
    tl.store(minimum_ptr + group, minimum)


@triton.jit
def _partial_min_max(x_ptr, partial_min_ptr, partial_max_ptr,
                     STRIDE0: tl.constexpr, STRIDE1: tl.constexpr,
                     ROWS: tl.constexpr, COLS: tl.constexpr, GROUP: tl.constexpr,
                     BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                     PARTS: tl.constexpr, CHUNK: tl.constexpr):
    group, part = tl.program_id(0).to(tl.int64), tl.program_id(1)
    offsets = part * CHUNK + tl.arange(0, CHUNK)
    rows, cols, valid = _coordinates(group, offsets, ROWS, COLS, GROUP, BLOCK_ROWS, BLOCK_COLS)
    values = tl.load(x_ptr + rows * STRIDE0 + cols * STRIDE1, valid, other=0).to(tl.float32)
    tl.store(partial_min_ptr + group * PARTS + part, tl.min(tl.where(valid, values, float("inf"))))
    tl.store(partial_max_ptr + group * PARTS + part, tl.max(tl.where(valid, values, float("-inf"))))


@triton.jit
def _finish_min_max(partial_min_ptr, partial_max_ptr, step_ptr, minimum_ptr,
                    PARTS: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    minimum = tl.min(tl.load(partial_min_ptr + group * PARTS + offsets, offsets < PARTS, other=float("inf")))
    maximum = tl.max(tl.load(partial_max_ptr + group * PARTS + offsets, offsets < PARTS, other=float("-inf")))
    tl.store(minimum_ptr + group, minimum)
    tl.store(step_ptr + group, _affine_step(minimum, maximum, BITS))


@triton.jit
def _quantize_pack_chunks(x_ptr, packed_ptr, step_ptr, minimum_ptr,
                 STRIDE0: tl.constexpr, STRIDE1: tl.constexpr,
                 ROWS: tl.constexpr, COLS: tl.constexpr, GROUP: tl.constexpr,
                 BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                 WORDS: tl.constexpr, BITS: tl.constexpr, CHUNK: tl.constexpr):
    group, part = tl.program_id(0).to(tl.int64), tl.program_id(1)
    offsets = part * CHUNK + tl.arange(0, CHUNK)
    rows, cols, valid = _coordinates(group, offsets, ROWS, COLS, GROUP, BLOCK_ROWS, BLOCK_COLS)
    values = tl.load(x_ptr + rows * STRIDE0 + cols * STRIDE1, valid, other=0).to(tl.float32)
    words = _affine_quantize_pack_with_metadata(values, valid, tl.load(step_ptr + group), tl.load(minimum_ptr + group), BITS, CHUNK)
    word_offsets = part * (CHUNK // (32 // BITS)) + tl.arange(0, CHUNK // (32 // BITS))
    tl.store(packed_ptr + group * WORDS + word_offsets, words, word_offsets < WORDS)


@triton.jit
def _dequantize_unpack_groups(packed_ptr, step_ptr, minimum_ptr, output_ptr,
                   STRIDE0: tl.constexpr, STRIDE1: tl.constexpr,
                   ROWS: tl.constexpr, COLS: tl.constexpr, GROUP: tl.constexpr,
                   BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                   WORDS: tl.constexpr, BITS: tl.constexpr, TRANSPOSE: tl.constexpr,
                   OUTPUT_MAX: tl.constexpr,
                   CHUNK: tl.constexpr):
    group, part = tl.program_id(0).to(tl.int64), tl.program_id(1)
    VPW: tl.constexpr = 32 // BITS
    word_offsets = part * (CHUNK // VPW) + tl.arange(0, CHUNK // VPW)
    words = tl.load(packed_ptr + group * WORDS + word_offsets, word_offsets < WORDS, other=0)
    codes = (words[:, None] >> (tl.arange(0, VPW) * BITS)[None, :]) & ((1 << BITS) - 1)
    codes = tl.reshape(codes, (CHUNK,)).to(tl.float32)
    step, minimum = tl.load(step_ptr + group), tl.load(minimum_ptr + group)
    product = codes * step
    values = tl.where(product == float("inf"),
                      (codes * (step * 0.5) + minimum * 0.5) * 2.0,
                      product + minimum)
    if BITS == 1:
        values = tl.where(step < 0, (minimum * 0.5 + codes * -step) * 2.0, values)
    if OUTPUT_MAX > 0:
        values = tl.minimum(tl.maximum(values, -OUTPUT_MAX), OUTPUT_MAX)
    offsets = part * CHUNK + tl.arange(0, CHUNK)
    rows, cols, valid = _coordinates(group, offsets, ROWS, COLS, GROUP, BLOCK_ROWS, BLOCK_COLS)
    if TRANSPOSE:
        destination = cols * STRIDE0 + rows * STRIDE1
    else:
        destination = rows * STRIDE0 + cols * STRIDE1
    tl.store(output_ptr + destination, values.to(output_ptr.dtype.element_ty), valid)


def quantize_pack_affine(x, group_size=512, block_shape=None, *, bits=2):
    """Affine-quantize/pack a CUDA matrix; return words, steps and minima."""
    group = group_size if block_shape is None else block_shape[0] * block_shape[1]
    vpw, words, _ = _bits_consts(bits, group)
    rows, cols = x.shape
    block_rows, block_cols = block_shape or (0, 0)
    groups = (triton.cdiv(rows * cols, group) if block_shape is None
              else triton.cdiv(rows, block_rows) * triton.cdiv(cols, block_cols))
    packed = torch.empty((groups, words), dtype=torch.int32, device=x.device)
    step = torch.empty(groups, dtype=torch.float32, device=x.device)
    minimum = torch.empty_like(step)
    args = (x.stride(0), x.stride(1), rows, cols, group, block_rows, block_cols)
    if group <= 16384:
        _quantize_pack_groups[(groups,)](x, packed, step, minimum, *args, words, bits,
                                max(vpw, triton.next_power_of_2(group)), enable_fp_fusion=False)
    else:
        chunk = 4096
        parts = triton.cdiv(group, chunk)
        partial_min = torch.empty((groups, parts), dtype=torch.float32, device=x.device)
        partial_max = torch.empty_like(partial_min)
        _partial_min_max[(groups, parts)](x, partial_min, partial_max, *args, parts, chunk)
        _finish_min_max[(groups,)](partial_min, partial_max, step, minimum, parts, bits,
                                   triton.next_power_of_2(parts), enable_fp_fusion=False)
        _quantize_pack_chunks[(groups, parts)](x, packed, step, minimum, *args, words, bits, chunk,
                                      enable_fp_fusion=False)
    return packed, step, minimum


def dequantize_unpack_affine(packed, step, minimum, shape, group_size=512, block_shape=None,
                            *, dtype=torch.bfloat16, transpose=False, bits=2):
    """Unpack codes and dequantize into the requested dtype and orientation."""
    group = group_size if block_shape is None else block_shape[0] * block_shape[1]
    vpw, words, _ = _bits_consts(bits, group)
    rows, cols = shape
    output_shape = (cols, rows) if transpose else (rows, cols)
    stride = (triton.cdiv(output_shape[1], 16) * 16, 1)
    output = torch.empty_strided(output_shape, stride, dtype=dtype, device=packed.device)
    block_rows, block_cols = block_shape or (0, 0)
    chunk = min(4096, max(vpw, triton.next_power_of_2(group)))
    output_max = torch.finfo(dtype).max if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 0.0
    _dequantize_unpack_groups[(packed.shape[0], triton.cdiv(group, chunk))](
        packed, step, minimum, output, *output.stride(), rows, cols, group,
        block_rows, block_cols, words, bits, transpose, output_max, chunk,
        enable_fp_fusion=False)
    return output


def affine_min_max_quantize_pack(x: torch.Tensor, bits: int, group_size):
    """Legacy signature, now supporting tails and FP32 affine metadata."""
    if x.numel() == 0:
        raise ValueError("x must be non-empty")
    packed, step, minimum = quantize_pack_affine(x.reshape(1, -1), group_size, bits=bits)
    return packed, step, minimum, x.shape


def affine_min_max_dequantize_unpack(packed, scalar, min_x, bits, group_size, ori_shape,
                                   output_dtype=torch.float16):
    count = 1
    for dimension in ori_shape:
        count *= dimension
    output = dequantize_unpack_affine(packed, scalar, min_x, (1, count), group_size,
                           dtype=output_dtype, bits=bits)
    return output.reshape(ori_shape)
