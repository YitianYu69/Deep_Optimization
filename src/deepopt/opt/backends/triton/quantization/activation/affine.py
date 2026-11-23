"""Affine activation quantization/packing and helpers shared by fused FP8 casts.

Codes occupy the low bits first in int32 words. Metadata is FP32 so it can
represent BF16 activations without imposing FP16's much smaller range.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from math import prod

from ...._shared.operand_device import operand_device_guard


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
def _affine_quantize_pack_with_metadata(values, mask, step, minimum, BITS: tl.constexpr, BLOCK: tl.constexpr,
                                      BATCH: tl.constexpr = 0, FINITE_RANGE: tl.constexpr = False):
    """Quantize using supplied affine metadata and pack the rounded codes."""
    QMAX: tl.constexpr = (1 << BITS) - 1
    VPW: tl.constexpr = 32 // BITS
    difference = values - minimum
    if FINITE_RANGE:
        # FP8 extrema and their difference fit FP32. Keep the precise divide,
        # avoiding the two overflow-recovery divides needed by BF16 inputs.
        normalized = tl.div_rn(difference, step)
    else:
        normalized = tl.where(difference == float("inf"),
                              tl.div_rn(values, step) - tl.div_rn(minimum, step),
                              tl.div_rn(difference, step))
    if BITS == 1:
        if not FINITE_RANGE:
            normalized = tl.where(step < 0, tl.div_rn(values * 0.5 - minimum * 0.5, -step), normalized)
        scaled = tl.where(step != 0, normalized, 0.0)
    else:
        scaled = tl.where(step > 0, normalized, 0.0)
    scaled = tl.minimum(tl.maximum(scaled, 0.0), QMAX)
    if FINITE_RANGE:
        # __nv_float2int_rn is round-to-nearest-even, the existing code policy.
        rounded = libdevice.float2int_rn(scaled)
    else:
        lower = tl.floor(scaled)
        fraction = scaled - lower
        rounded = lower.to(tl.int32) + (
            (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
        ).to(tl.int32)
    codes = tl.where(mask, rounded, 0)
    if BATCH:
        codes = tl.reshape(codes, (BATCH, BLOCK // VPW, VPW))
        words = tl.sum(codes << (tl.arange(0, VPW) * BITS)[None, None, :], axis=2)
    else:
        codes = tl.reshape(codes, (BLOCK // VPW, VPW))
        words = tl.sum(codes << (tl.arange(0, VPW) * BITS)[None, :], axis=1)
    return words


@triton.jit
def _affine_quantize_pack(values, mask, BITS: tl.constexpr, BLOCK: tl.constexpr, FINITE_RANGE: tl.constexpr = False):
    minimum = tl.min(tl.where(mask, values, float("inf")), axis=0)
    maximum = tl.max(tl.where(mask, values, float("-inf")), axis=0)
    step = _affine_step(minimum, maximum, BITS)
    return _affine_quantize_pack_with_metadata(values, mask, step, minimum, BITS, BLOCK, FINITE_RANGE=FINITE_RANGE), step, minimum


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
def _source_address(rows, cols, COLS: tl.constexpr, STRIDE0: tl.constexpr, STRIDE1: tl.constexpr,
                    SOURCE_SHAPE: tl.constexpr, SOURCE_STRIDES: tl.constexpr, SOURCE_NARROW: tl.constexpr = False):
    index_type: tl.constexpr = tl.int32 if SOURCE_NARROW else tl.int64
    rows, cols = rows.to(index_type), cols.to(index_type)
    if len(SOURCE_SHAPE):
        logical = rows * COLS + cols
        address = logical * 0
        for axis in tl.static_range(len(SOURCE_SHAPE) - 1, -1, -1):
            address += (logical % SOURCE_SHAPE[axis]) * SOURCE_STRIDES[axis]
            logical = logical // SOURCE_SHAPE[axis]
    else:
        address = rows * STRIDE0 + cols * STRIDE1
    return address.to(index_type)


@triton.jit
def _source_values(x_ptr, address, valid, SOURCE_DTYPE: tl.constexpr = None):
    values = tl.load(x_ptr + address, valid, other=0.0)
    # Native autocast consumes rounded BF16/FP16 inputs. Round before finding
    # group extrema, without allocating a dense cast or layout conversion.
    if SOURCE_DTYPE is not None:
        values = values.to(SOURCE_DTYPE)
    return values.to(tl.float32)


@triton.jit
def _quantize_pack_spatial_coalesced(x_ptr,packed_ptr,step_ptr,minimum_ptr,
                                    ROWS:tl.constexpr,COLS:tl.constexpr,SPATIAL:tl.constexpr,
                                    GROUP:tl.constexpr,WORDS:tl.constexpr,BITS:tl.constexpr,
                                    GROUPS:tl.constexpr,BATCH:tl.constexpr,SOURCE_DTYPE:tl.constexpr=None):
    # NCHW input: load several adjacent groups in physical spatial order,
    # then reorder the register tile into the unchanged channel-row payload.
    row=tl.program_id(0).to(tl.int64)*(BATCH*GROUP//COLS)+tl.arange(0,BATCH*GROUP//COLS)
    channel=tl.arange(0,COLS)
    address=(row[None,:]//SPATIAL)*(COLS*SPATIAL)+channel[:,None]*SPATIAL+row[None,:]%SPATIAL
    valid=tl.broadcast_to(row[None,:]<ROWS,(COLS,BATCH*GROUP//COLS))
    values=_source_values(x_ptr,address,valid,SOURCE_DTYPE)
    values=tl.reshape(values.T,(BATCH,GROUP))
    mask=tl.reshape(valid.T,(BATCH,GROUP))
    minimum=tl.min(tl.where(mask,values,float("inf")),1)
    maximum=tl.max(tl.where(mask,values,float("-inf")),1)
    step=_affine_step(minimum,maximum,BITS)
    words=_affine_quantize_pack_with_metadata(values,mask,step[:,None],minimum[:,None],BITS,GROUP,BATCH=BATCH)
    group=tl.program_id(0).to(tl.int64)*BATCH+tl.arange(0,BATCH)
    word=tl.arange(0,WORDS)
    tl.store(packed_ptr+group[:,None]*WORDS+word[None,:],words,group[:,None]<GROUPS)
    tl.store(step_ptr+group,step,group<GROUPS)
    tl.store(minimum_ptr+group,minimum,group<GROUPS)


@triton.jit
def _quantize_pack_qkv_groups(q, k, v, packed, steps, minima,
                              ROWS: tl.constexpr, COLS: tl.constexpr, GROUP: tl.constexpr,
                              GROUPS: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                              BLOCK: tl.constexpr, SOURCE_SHAPE: tl.constexpr,
                              SOURCE_STRIDES: tl.constexpr, SOURCE_NARROW: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    role = tl.arange(0, 4)[:, None]
    offsets = tl.arange(0, BLOCK)[None, :]
    rows, cols, valid = _coordinates(group, offsets, ROWS, COLS, GROUP, 0, 0)
    address = _source_address(rows, cols, COLS, 0, 0, SOURCE_SHAPE, SOURCE_STRIDES, SOURCE_NARROW)
    source = tl.where(role == 0, q, tl.where(role == 1, k, v))
    valid = valid & (role < 3)
    values = tl.load(source + address, valid, other=0).to(tl.float32)
    low = tl.min(tl.where(valid, values, float("inf")), 1)
    high = tl.max(tl.where(valid, values, float("-inf")), 1)
    step = _affine_step(low, high, BITS)
    words = _affine_quantize_pack_with_metadata(values, valid, step[:, None], low[:, None], BITS, BLOCK, BATCH=4)
    word = tl.arange(0, BLOCK // (32 // BITS))[None, :]
    tl.store(packed + role * (GROUPS * WORDS) + group * WORDS + word, words, (role < 3) & (word < WORDS))
    index = tl.arange(0, 4) * GROUPS + group
    tl.store(steps + index, step, tl.arange(0, 4) < 3)
    tl.store(minima + index, low, tl.arange(0, 4) < 3)


@triton.jit
def _quantize_pack_groups(x_ptr, packed_ptr, step_ptr, minimum_ptr,
                 STRIDE0: tl.constexpr, STRIDE1: tl.constexpr,
                 ROWS: tl.constexpr, COLS: tl.constexpr, GROUP: tl.constexpr,
                 BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                 WORDS: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr,
                 SOURCE_SHAPE: tl.constexpr = (), SOURCE_STRIDES: tl.constexpr = (), SOURCE_NARROW: tl.constexpr = False,
                 SOURCE_DTYPE: tl.constexpr = None, FP8_SOURCE: tl.constexpr = False):
    group = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    rows, cols, valid = _coordinates(group, offsets, ROWS, COLS, GROUP, BLOCK_ROWS, BLOCK_COLS)
    address = _source_address(rows, cols, COLS, STRIDE0, STRIDE1, SOURCE_SHAPE, SOURCE_STRIDES, SOURCE_NARROW)
    values = _source_values(x_ptr, address, valid, SOURCE_DTYPE)
    words, step, minimum = _affine_quantize_pack(values, valid, BITS, BLOCK, FINITE_RANGE=FP8_SOURCE)
    word_offsets = tl.arange(0, BLOCK // (32 // BITS))
    tl.store(packed_ptr + group * WORDS + word_offsets, words, word_offsets < WORDS)
    tl.store(step_ptr + group, step)
    tl.store(minimum_ptr + group, minimum)


@triton.jit
def _quantize_pack_batched(x_ptr, packed_ptr, step_ptr, minimum_ptr,
                          STRIDE0: tl.constexpr, STRIDE1: tl.constexpr,
                          ROWS: tl.constexpr, COLS: tl.constexpr, GROUP: tl.constexpr,
                          BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                          WORDS: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr,
                          GROUPS: tl.constexpr, GROUPS_PER_PROGRAM: tl.constexpr,
                          SOURCE_SHAPE: tl.constexpr = (), SOURCE_STRIDES: tl.constexpr = (), SOURCE_NARROW: tl.constexpr = False,
                          SOURCE_DTYPE: tl.constexpr = None, FP8_SOURCE: tl.constexpr = False):
    groups = tl.program_id(0).to(tl.int64) * GROUPS_PER_PROGRAM + tl.arange(0, GROUPS_PER_PROGRAM)
    offset = tl.arange(0, BLOCK)
    row, col, valid = _coordinates(groups[:, None], offset[None, :], ROWS, COLS,
                                    GROUP, BLOCK_ROWS, BLOCK_COLS)
    valid = valid & (groups[:, None] < GROUPS)
    address = _source_address(row, col, COLS, STRIDE0, STRIDE1, SOURCE_SHAPE, SOURCE_STRIDES, SOURCE_NARROW)
    value = _source_values(x_ptr, address, valid, SOURCE_DTYPE)
    lo = tl.min(tl.where(valid, value, float("inf")), 1)
    hi = tl.max(tl.where(valid, value, float("-inf")), 1)
    step = _affine_step(lo, hi, BITS)
    # Pack the complete batched tile. Each row has its own metadata; no
    # compression groups are merged and masked tail groups never store.
    VPW: tl.constexpr = 32 // BITS
    words = _affine_quantize_pack_with_metadata(value, valid, step[:, None], lo[:, None], BITS, BLOCK,
                                                BATCH=GROUPS_PER_PROGRAM, FINITE_RANGE=FP8_SOURCE)
    word = tl.arange(0, BLOCK // VPW)
    tl.store(packed_ptr + groups[:, None] * WORDS + word[None, :], words,
             (groups[:, None] < GROUPS) & (word[None, :] < WORDS))
    tl.store(step_ptr + groups, step, groups < GROUPS)
    tl.store(minimum_ptr + groups, lo, groups < GROUPS)


@triton.jit
def _quantize_pack_serial(x_ptr, packed_ptr, step_ptr, minimum_ptr,
                          STRIDE0: tl.constexpr, STRIDE1: tl.constexpr,
                          ROWS: tl.constexpr, COLS: tl.constexpr, GROUP: tl.constexpr,
                          BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                          WORDS: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr,
                          GROUPS: tl.constexpr, GROUPS_PER_PROGRAM: tl.constexpr,
                          SOURCE_SHAPE: tl.constexpr = (), SOURCE_STRIDES: tl.constexpr = (), SOURCE_NARROW: tl.constexpr = False,
                          SOURCE_DTYPE: tl.constexpr = None, FP8_SOURCE: tl.constexpr = False):
    offsets = tl.arange(0, BLOCK)
    words_at = tl.arange(0, BLOCK // (32 // BITS))
    # Reuse one group's live arrays rather than retaining a wide batched tile.
    for local in range(GROUPS_PER_PROGRAM):
        group = tl.program_id(0).to(tl.int64) * GROUPS_PER_PROGRAM + local
        row, col, valid = _coordinates(group, offsets, ROWS, COLS, GROUP, BLOCK_ROWS, BLOCK_COLS)
        valid = valid & (group < GROUPS)
        address = _source_address(row, col, COLS, STRIDE0, STRIDE1, SOURCE_SHAPE, SOURCE_STRIDES, SOURCE_NARROW)
        values = _source_values(x_ptr, address, valid, SOURCE_DTYPE)
        words, step, minimum = _affine_quantize_pack(values, valid, BITS, BLOCK, FINITE_RANGE=FP8_SOURCE)
        tl.store(packed_ptr + group * WORDS + words_at, words, (group < GROUPS) & (words_at < WORDS))
        tl.store(step_ptr + group, step, group < GROUPS)
        tl.store(minimum_ptr + group, minimum, group < GROUPS)


@triton.jit
def _partial_min_max(x_ptr, partial_min_ptr, partial_max_ptr,
                     STRIDE0: tl.constexpr, STRIDE1: tl.constexpr,
                     ROWS: tl.constexpr, COLS: tl.constexpr, GROUP: tl.constexpr,
                     BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                     PARTS: tl.constexpr, CHUNK: tl.constexpr,
                     SOURCE_SHAPE: tl.constexpr = (), SOURCE_STRIDES: tl.constexpr = (), SOURCE_NARROW: tl.constexpr = False,
                     SOURCE_DTYPE: tl.constexpr = None):
    group, part = tl.program_id(0).to(tl.int64), tl.program_id(1)
    offsets = part * CHUNK + tl.arange(0, CHUNK)
    rows, cols, valid = _coordinates(group, offsets, ROWS, COLS, GROUP, BLOCK_ROWS, BLOCK_COLS)
    address = _source_address(rows, cols, COLS, STRIDE0, STRIDE1, SOURCE_SHAPE, SOURCE_STRIDES, SOURCE_NARROW)
    values = _source_values(x_ptr, address, valid, SOURCE_DTYPE)
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
                 WORDS: tl.constexpr, BITS: tl.constexpr, CHUNK: tl.constexpr,
                 SOURCE_SHAPE: tl.constexpr = (), SOURCE_STRIDES: tl.constexpr = (), SOURCE_NARROW: tl.constexpr = False,
                 SOURCE_DTYPE: tl.constexpr = None):
    group, part = tl.program_id(0).to(tl.int64), tl.program_id(1)
    offsets = part * CHUNK + tl.arange(0, CHUNK)
    rows, cols, valid = _coordinates(group, offsets, ROWS, COLS, GROUP, BLOCK_ROWS, BLOCK_COLS)
    address = _source_address(rows, cols, COLS, STRIDE0, STRIDE1, SOURCE_SHAPE, SOURCE_STRIDES, SOURCE_NARROW)
    values = _source_values(x_ptr, address, valid, SOURCE_DTYPE)
    words = _affine_quantize_pack_with_metadata(values, valid, tl.load(step_ptr + group), tl.load(minimum_ptr + group), BITS, CHUNK)
    word_offsets = part * (CHUNK // (32 // BITS)) + tl.arange(0, CHUNK // (32 // BITS))
    tl.store(packed_ptr + group * WORDS + word_offsets, words, word_offsets < WORDS)


@triton.jit
def _affine_decode(codes, step, minimum, BITS: tl.constexpr, OUTPUT_MAX: tl.constexpr):
    # Keep the same operation order in the grouped and tiled paths. In
    # particular, the negative one-bit step encodes an overflowing full span.
    product = codes * step
    values = tl.where(product == float("inf"),
                      (codes * (step * 0.5) + minimum * 0.5) * 2.0,
                      product + minimum)
    if BITS == 1:
        values = tl.where(step < 0, (minimum * 0.5 + codes * -step) * 2.0, values)
    if OUTPUT_MAX > 0:
        values = tl.minimum(tl.maximum(values, -OUTPUT_MAX), OUTPUT_MAX)
    return values


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
    values = _affine_decode(codes, tl.load(step_ptr + group), tl.load(minimum_ptr + group), BITS, OUTPUT_MAX)
    offsets = part * CHUNK + tl.arange(0, CHUNK)
    rows, cols, valid = _coordinates(group, offsets, ROWS, COLS, GROUP, BLOCK_ROWS, BLOCK_COLS)
    if TRANSPOSE:
        destination = cols * STRIDE0 + rows * STRIDE1
    else:
        destination = rows * STRIDE0 + cols * STRIDE1
    tl.store(output_ptr + destination, values.to(output_ptr.dtype.element_ty), valid)


@triton.jit
def _dequantize_unpack_transpose(packed_ptr, step_ptr, minimum_ptr, output_ptr,
                   STRIDE0: tl.constexpr, STRIDE1: tl.constexpr,
                   ROWS: tl.constexpr, COLS: tl.constexpr, GROUP: tl.constexpr,
                   BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                   WORDS: tl.constexpr, BITS: tl.constexpr, OUTPUT_MAX: tl.constexpr,
                   TILE_ROWS: tl.constexpr, TILE_COLS: tl.constexpr, INDEX64: tl.constexpr = True):
    # Gather a matrix tile across many compression groups. A single flat group
    # contains only 1-2 source rows in ViT, making its transposed stores scatter
    # across cache sectors. The 2D tile permits a coalesced output layout.
    index_type: tl.constexpr = tl.int64 if INDEX64 else tl.int32
    rows = tl.program_id(0).to(index_type) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    cols = tl.program_id(1).to(index_type) * TILE_COLS + tl.arange(0, TILE_COLS)
    r, c = rows[:, None], cols[None, :]
    valid = (r < ROWS) & (c < COLS)
    if BLOCK_ROWS > 0:
        group = (r // BLOCK_ROWS) * triton.cdiv(COLS, BLOCK_COLS) + c // BLOCK_COLS
        within = (r % BLOCK_ROWS) * BLOCK_COLS + c % BLOCK_COLS
    else:
        position = r * COLS + c
        group, within = position // GROUP, position % GROUP
    VPW: tl.constexpr = 32 // BITS
    words = tl.load(packed_ptr + group * WORDS + within // VPW, valid, other=0)
    codes = ((words >> ((within % VPW) * BITS)) & ((1 << BITS) - 1)).to(tl.float32)
    step = tl.load(step_ptr + group, valid, other=0.0)
    minimum = tl.load(minimum_ptr + group, valid, other=0.0)
    values = _affine_decode(codes, step, minimum, BITS, OUTPUT_MAX)
    destination = cols[:, None] * STRIDE0 + rows[None, :] * STRIDE1
    tl.store(output_ptr + destination, tl.trans(values).to(output_ptr.dtype.element_ty), tl.trans(valid))


@triton.jit
def _dequantize_unpack_spatial(packed_ptr, step_ptr, minimum_ptr, output_ptr,
                              ELEMENTS: tl.constexpr, CHANNELS: tl.constexpr, SPATIAL: tl.constexpr,
                              GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                              CHANNELS_LAST: tl.constexpr, OUTPUT_MAX: tl.constexpr, BLOCK: tl.constexpr):
    # Iterate in the destination's physical order for contiguous stores. The
    # payload keeps the existing channel-row grouping in both memory formats.
    position = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = position < ELEMENTS
    if CHANNELS_LAST:
        logical = position
    else:
        channel = (position // SPATIAL) % CHANNELS
        row = (position // (SPATIAL * CHANNELS)) * SPATIAL + position % SPATIAL
        logical = row * CHANNELS + channel
    group, within = logical // GROUP, logical % GROUP
    VPW: tl.constexpr = 32 // BITS
    words = tl.load(packed_ptr + group * WORDS + within // VPW, valid, other=0)
    codes = ((words >> ((within % VPW) * BITS)) & ((1 << BITS) - 1)).to(tl.float32)
    step = tl.load(step_ptr + group, valid, other=0.0)
    minimum = tl.load(minimum_ptr + group, valid, other=0.0)
    values = _affine_decode(codes, step, minimum, BITS, OUTPUT_MAX)
    tl.store(output_ptr + position, values.to(output_ptr.dtype.element_ty), valid)


def _transpose_tiles(rows, cols, *, dtype=None, bits=None, group_size=None, block_shape=None, device=None):
    """Measured Hopper case; retain the general policy for unmeasured layouts."""
    if (rows == 100864 and cols in (768, 3072) and dtype == torch.float8_e4m3fn
            and bits == 2 and group_size == 1024 and block_shape is None
            and device is not None and device.type == "cuda"):
        from deepopt.opt.backends._shared.cuda import device_metadata

        index = device.index if device.index is not None else torch.cuda.current_device()
        if device_metadata(index)[0] == 90:
            return 64, 64
    return (32, 64) if cols >= 64 else (32, 32)


@operand_device_guard
def quantize_pack_affine(x, group_size=512, block_shape=None, *, bits=2,
                         implementation=None, groups_per_program=None, logical_shape=None, num_warps=None,
                         source_dtype=None):
    """Affine-quantize/pack a CUDA matrix; return words, steps and minima."""
    group = group_size if block_shape is None else block_shape[0] * block_shape[1]
    vpw, words, _ = _bits_consts(bits, group)
    rows, cols = tuple(logical_shape) if logical_shape is not None else x.shape
    if logical_shape is not None and (block_shape is not None or rows * cols != x.numel()):
        raise ValueError("Strided logical packing requires matching flat groups")
    logical_contiguous=logical_shape is not None and x.is_contiguous()
    source = ({"SOURCE_SHAPE": tuple(x.shape), "SOURCE_STRIDES": tuple(x.stride())}
              if logical_shape is not None and x.ndim != 2 and not logical_contiguous else {})
    if source:
        from deepopt.opt.backends._shared.activation_codec import narrow_source_indices
        source["SOURCE_NARROW"] = narrow_source_indices(tuple(x.shape), tuple(x.stride()))
    if source_dtype is not None and source_dtype != x.dtype:
        names = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32",
                 torch.float64: "float64"}
        if source_dtype not in names:
            raise ValueError("source_dtype must be a native convolution floating dtype")
        source["SOURCE_DTYPE"] = getattr(tl, names[source_dtype])
    block_rows, block_cols = block_shape or (0, 0)
    groups = (triton.cdiv(rows * cols, group) if block_shape is None
              else triton.cdiv(rows, block_rows) * triton.cdiv(cols, block_cols))
    packed = torch.empty((groups, words), dtype=torch.int32, device=x.device)
    step = torch.empty(groups, dtype=torch.float32, device=x.device)
    minimum = torch.empty_like(step)
    strides=(cols,1) if logical_contiguous else x.stride()[:2]
    args = (*strides, rows, cols, group, block_rows, block_cols)
    if implementation is None or groups_per_program is None:
        from deepopt.opt.backends._shared.activation_codec import activation_packing

        policy, batch = activation_packing()
        implementation = policy if implementation is None else implementation
        groups_per_program = batch if groups_per_program is None else groups_per_program
    if num_warps is None:
        from deepopt.opt.backends._shared.activation_codec import activation_pack_warps
        num_warps = activation_pack_warps()
    from deepopt.opt.backends._shared.activation_codec import select_pack_warps
    num_warps = select_pack_warps(num_warps, (rows, cols), x.dtype, bits, group, block_shape,
                                  x.device, contiguous=x.is_contiguous(), strided=bool(source),
                                  grouped=implementation in ("auto","grouped"))
    if implementation not in ("auto", "grouped", "batched", "serial"):
        raise ValueError("implementation must be auto, grouped, batched, or serial")
    if type(groups_per_program) is not int or groups_per_program not in (1, 2, 4, 8):
        raise ValueError("groups_per_program must be 1, 2, 4, or 8")
    from deepopt.opt.backends._shared.activation_codec import observe_pack
    spatial=prod(x.shape[1:-1]) if logical_shape is not None and x.ndim>=3 else 0
    nchw=(spatial and cols<=512 and cols&(cols-1)==0 and group&(group-1)==0 and 32<=group<=2048 and group%cols==0
          and x.stride(-1)==spatial and x.stride(0)==cols*spatial
          and all(x.stride(axis)==prod(x.shape[axis+1:-1]) for axis in range(1,x.ndim-1)))
    coalesced=bool(nchw and implementation in ("auto","grouped"))
    batch=min(8,max(4,32//(group//cols))) if coalesced else groups_per_program
    actual = "spatial_coalesced" if coalesced else implementation if implementation in ("batched","serial") and group <= 2048 else "grouped" if group <= 16384 else "chunked"
    observe_pack(x,(rows,cols),group,bits,block_shape,actual,
                 batch if actual in ("batched","serial","spatial_coalesced") else 1,num_warps,source)
    if coalesced:
        _quantize_pack_spatial_coalesced[(triton.cdiv(groups,batch),)](x,packed,step,minimum,rows,cols,spatial,
            group,words,bits,groups,batch,source.get("SOURCE_DTYPE"),num_warps=num_warps,enable_fp_fusion=False)
    elif implementation in ("batched", "serial") and group <= 2048:
        kernel = _quantize_pack_serial if implementation == "serial" else _quantize_pack_batched
        kernel[(triton.cdiv(groups, groups_per_program),)](
            x, packed, step, minimum, *args, words, bits, max(vpw, triton.next_power_of_2(group)),
            groups, groups_per_program, num_warps=num_warps, enable_fp_fusion=False,
            FP8_SOURCE=x.dtype in (torch.float8_e4m3fn,torch.float8_e5m2), **source)
    elif group <= 16384:
        _quantize_pack_groups[(groups,)](x, packed, step, minimum, *args, words, bits,
                                max(vpw, triton.next_power_of_2(group)), num_warps=num_warps,
                                enable_fp_fusion=False, FP8_SOURCE=x.dtype in (torch.float8_e4m3fn,torch.float8_e5m2), **source)
    else:
        chunk = 4096
        parts = triton.cdiv(group, chunk)
        partial_min = torch.empty((groups, parts), dtype=torch.float32, device=x.device)
        partial_max = torch.empty_like(partial_min)
        _partial_min_max[(groups, parts)](x, partial_min, partial_max, *args, parts, chunk, **source)
        _finish_min_max[(groups,)](partial_min, partial_max, step, minimum, parts, bits,
                                   triton.next_power_of_2(parts), enable_fp_fusion=False)
        _quantize_pack_chunks[(groups, parts)](x, packed, step, minimum, *args, words, bits, chunk,
                                      enable_fp_fusion=False, **source)
    return packed, step, minimum


@operand_device_guard
def quantize_pack_strided(x, group_size=512, *, bits=2, source_dtype=None):
    """Read the original N-D strides without materializing a logical matrix."""
    return quantize_pack_affine(x, group_size, bits=bits,
                                logical_shape=(prod(x.shape[:-1]), x.shape[-1]), source_dtype=source_dtype)


@operand_device_guard
def quantize_pack_qkv(inputs, group_size=512, *, bits=4):
    """Single launch sharing source-index arithmetic across three equal layouts."""
    from deepopt.opt.backends._shared.activation_codec import (
        activation_pack_warps, narrow_source_indices, observe_pack,
    )
    q, k, v = inputs
    vpw, words, _ = _bits_consts(bits, group_size)
    rows, cols = prod(q.shape[:-1]), q.shape[-1]
    groups = triton.cdiv(rows * cols, group_size)
    packed = torch.empty((3, groups, words), dtype=torch.int32, device=q.device)
    steps = torch.empty((3, groups), dtype=torch.float32, device=q.device)
    minima = torch.empty_like(steps)
    warps = activation_pack_warps()
    warps = 4 if warps == "auto" else warps
    source = {"SOURCE_SHAPE": tuple(q.shape), "SOURCE_STRIDES": tuple(q.stride()),
              "SOURCE_NARROW": narrow_source_indices(tuple(q.shape), tuple(q.stride()))}
    _quantize_pack_qkv_groups[(groups,)](q, k, v, packed, steps, minima, rows, cols, group_size,
        groups, words, bits, max(vpw, triton.next_power_of_2(group_size)),
        num_warps=warps, enable_fp_fusion=False, **source)
    observe_pack(q, (rows, cols), group_size, bits, None, "attention_joint", 3, warps, source)
    return packed, steps, minima


@operand_device_guard
def dequantize_unpack_spatial(packed, step, minimum, shape, group_size, *, dtype, bits, memory_format):
    """Decode channel-row groups directly to a fresh native convolution input."""
    _, words, _ = _bits_consts(bits, group_size)
    output = torch.empty(shape, dtype=dtype, device=packed.device, memory_format=memory_format)
    channels_last = memory_format in (torch.channels_last, torch.channels_last_3d)
    maximum = torch.finfo(dtype).max if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 0.0
    _dequantize_unpack_spatial[(triton.cdiv(output.numel(), 512),)](
        packed, step, minimum, output, output.numel(), shape[1], prod(shape[2:]),
        group_size, words, bits, channels_last, maximum, 512, enable_fp_fusion=False)
    return output


@operand_device_guard
def dequantize_unpack_affine(packed, step, minimum, shape, group_size=512, block_shape=None,
                            *, dtype=torch.bfloat16, transpose=False, bits=2, implementation="auto",
                            tile_shape=None, num_warps=4, row_alignment=16):
    """Unpack codes and dequantize into the requested dtype and orientation."""
    group = group_size if block_shape is None else block_shape[0] * block_shape[1]
    vpw, words, _ = _bits_consts(bits, group)
    rows, cols = shape
    if implementation not in ("auto", "grouped", "tiled", "copy"):
        raise ValueError("implementation must be auto, grouped, tiled, or copy")
    if implementation in ("tiled", "copy") and not transpose:
        raise ValueError("tiled/copy controls require transpose=True")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    if tile_shape is not None and (len(tile_shape) != 2 or any(type(x) is not int or x not in (16, 32, 64, 128) for x in tile_shape)
                                   or tile_shape[0] * tile_shape[1] > 8192):
        raise ValueError("tile_shape requires two power-of-two dimensions 16..128 and at most 8192 elements")
    output_shape = (cols, rows) if transpose else (rows, cols)
    if type(row_alignment) is not int or row_alignment not in (1, 16):
        raise ValueError("row_alignment must be 1 or 16")
    stride = (triton.cdiv(output_shape[1], row_alignment) * row_alignment, 1)
    output = torch.empty_strided(output_shape, stride, dtype=dtype, device=packed.device)
    if implementation == "copy":
        dense = dequantize_unpack_affine(packed, step, minimum, shape, group_size, block_shape,
                                        dtype=dtype, bits=bits, implementation="grouped", row_alignment=row_alignment)
        output.copy_(dense.T)
        return output
    block_rows, block_cols = block_shape or (0, 0)
    chunk = min(4096, max(vpw, triton.next_power_of_2(group)))
    output_max = torch.finfo(dtype).max if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 0.0
    tiled = implementation == "tiled" or (implementation == "auto" and transpose and min(rows, cols) >= 32
                                           and rows * cols >= 4096)
    if tiled:
        tile_rows, tile_cols = tile_shape or _transpose_tiles(
            rows, cols, dtype=dtype, bits=bits, group_size=group_size, block_shape=block_shape, device=packed.device)
        padded_rows, padded_cols = triton.cdiv(rows,tile_rows)*tile_rows, triton.cdiv(cols,tile_cols)*tile_cols
        narrow = (block_shape is None and max(padded_rows*cols+padded_cols, packed.numel(),
                  (padded_cols-1)*output.stride(0)+(padded_rows-1)*output.stride(1)) < 2**31)
        _dequantize_unpack_transpose[(triton.cdiv(rows, tile_rows), triton.cdiv(cols, tile_cols))](
            packed, step, minimum, output, *output.stride(), rows, cols, group,
            block_rows, block_cols, words, bits, output_max, tile_rows, tile_cols,
            INDEX64=not narrow, num_warps=num_warps, enable_fp_fusion=False)
        return output
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
