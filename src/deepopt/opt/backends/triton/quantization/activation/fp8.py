"""FP8 conversion with packed affine saving; large groups use bounded reductions."""

import torch
import triton
import triton.language as tl

from deepopt.opt.backends.triton.quantization.activation.affine import _affine_quantize_pack, _affine_quantize_pack_with_metadata, _affine_step
from deepopt.opt.backends.triton.quantization.fp8 import _run_scaling
from deepopt.opt.quantization.activation import PackedActivation, _validate_bits


@triton.jit
def _coordinates(group, offsets, M, N, GROUP_SIZE: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, GRID_N: tl.constexpr,
                 TENSORWISE: tl.constexpr):
    if TENSORWISE:
        flat = group * GROUP_SIZE + offsets
        rows, cols = flat // N, flat % N
        valid = (offsets < GROUP_SIZE) & (flat < M * N)
    else:
        rows = (group // GRID_N) * BM + offsets // BN
        cols = (group % GRID_N) * BN + offsets % BN
        valid = (offsets < BM * BN) & (rows < M) & (cols < N)
    return rows, cols, valid


@triton.jit
def _cast_values(x, scale, MAXIMUM: tl.constexpr, DTYPE: tl.constexpr):
    scaled = tl.minimum(tl.maximum(x * scale, -MAXIMUM), MAXIMUM)
    # Match delayed conversion's handling of unchecked NaNs as well.
    scaled = tl.where(x != x, float("nan"), scaled)
    return scaled.to(DTYPE, fp_downcast_rounding="rtne")


@triton.jit
def _cast_quantize_pack(x, output, scales, packed, step, minimum, snapshot, observations, errors,
               M, N, SM, SN, PAD_N: tl.constexpr, GROUP_SIZE: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, GRID_N: tl.constexpr,
               TENSORWISE: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr,
               MAXIMUM: tl.constexpr, SNAPSHOT: tl.constexpr,
               OBSERVE: tl.constexpr, CHECK: tl.constexpr, BITS: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    rows, cols, valid = _coordinates(group, offsets, M, N, GROUP_SIZE, BM, BN, GRID_N, TENSORWISE)
    values = tl.load(x + rows * SM + cols * SN, mask=valid, other=0.).to(tl.float32)
    scale = tl.load(scales + (0 if TENSORWISE else group))
    converted = _cast_values(values, scale, MAXIMUM, output.dtype.element_ty)
    tl.store(output + rows * PAD_N + cols, converted, mask=valid)
    words, delta, low = _affine_quantize_pack(converted.to(tl.float32), valid, BITS, BLOCK)
    word_offsets = tl.arange(0, BLOCK // (32 // BITS))
    tl.store(packed + group * WORDS + word_offsets, words, mask=word_offsets < WORDS)
    tl.store(step + group, delta)
    tl.store(minimum + group, low)
    if SNAPSHOT:
        if group == 0:
            tl.store(snapshot, scale)
            tl.store(snapshot + 1, tl.div_rn(1., scale))
    if OBSERVE:
        tl.store(observations + group, tl.max(tl.abs(values), 0))
    if CHECK:
        tl.store(errors + group, tl.max((~(tl.abs(values) < float("inf"))).to(tl.int32), 0))


@triton.jit
def _partial_minmax(x, scales, partial_min, partial_max, snapshot, observations, errors,
                    M, N, SM, SN, GROUP_SIZE: tl.constexpr, BM: tl.constexpr,
                    BN: tl.constexpr, GRID_N: tl.constexpr, TENSORWISE: tl.constexpr,
                    CHUNKS: tl.constexpr, BLOCK: tl.constexpr, MAXIMUM: tl.constexpr,
                    DTYPE: tl.constexpr, SNAPSHOT: tl.constexpr,
                    OBSERVE: tl.constexpr, CHECK: tl.constexpr):
    program = tl.program_id(0).to(tl.int64)
    group, chunk = program // CHUNKS, program % CHUNKS
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    rows, cols, valid = _coordinates(group, offsets, M, N, GROUP_SIZE, BM, BN, GRID_N, TENSORWISE)
    values = tl.load(x + rows * SM + cols * SN, mask=valid, other=0.).to(tl.float32)
    scale = tl.load(scales + (0 if TENSORWISE else group))
    converted = _cast_values(values, scale, MAXIMUM, DTYPE).to(tl.float32)
    tl.store(partial_min + program, tl.min(tl.where(valid, converted, float("inf")), 0))
    tl.store(partial_max + program, tl.max(tl.where(valid, converted, -float("inf")), 0))
    if SNAPSHOT:
        if program == 0:
            tl.store(snapshot, scale)
            tl.store(snapshot + 1, tl.div_rn(1., scale))
    if OBSERVE:
        tl.store(observations + program, tl.max(tl.abs(values), 0))
    if CHECK:
        tl.store(errors + program, tl.max((~(tl.abs(values) < float("inf"))).to(tl.int32), 0))


@triton.jit
def _reduce_minmax(low, high, out_low, out_high, COUNT, NEXT: tl.constexpr, BLOCK: tl.constexpr):
    program = tl.program_id(0).to(tl.int64)
    group, chunk = program // NEXT, program % NEXT
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    minima = tl.load(low + group * COUNT + offsets, mask=offsets < COUNT, other=float("inf"))
    maxima = tl.load(high + group * COUNT + offsets, mask=offsets < COUNT, other=-float("inf"))
    tl.store(out_low + group * NEXT + chunk, tl.min(minima, 0))
    tl.store(out_high + group * NEXT + chunk, tl.max(maxima, 0))


@triton.jit
def _finish_minmax(low, high, step, minimum, COUNT, BLOCK: tl.constexpr, BITS: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    minima = tl.load(low + group * COUNT + offsets, mask=offsets < COUNT, other=float("inf"))
    maxima = tl.load(high + group * COUNT + offsets, mask=offsets < COUNT, other=-float("inf"))
    first, last = tl.min(minima, 0), tl.max(maxima, 0)
    tl.store(minimum + group, first)
    tl.store(step + group, _affine_step(first, last, BITS))


@triton.jit
def _cast_quantize_pack_chunks(x, output, scales, packed, step, minimum, M, N, SM, SN,
                      PAD_N: tl.constexpr, GROUP_SIZE: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, GRID_N: tl.constexpr,
                      TENSORWISE: tl.constexpr, CHUNKS: tl.constexpr,
                      WORDS: tl.constexpr, BLOCK: tl.constexpr, MAXIMUM: tl.constexpr,
                      BITS: tl.constexpr):
    program = tl.program_id(0).to(tl.int64)
    group, chunk = program // CHUNKS, program % CHUNKS
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    rows, cols, valid = _coordinates(group, offsets, M, N, GROUP_SIZE, BM, BN, GRID_N, TENSORWISE)
    values = tl.load(x + rows * SM + cols * SN, mask=valid, other=0.).to(tl.float32)
    scale = tl.load(scales + (0 if TENSORWISE else group))
    converted = _cast_values(values, scale, MAXIMUM, output.dtype.element_ty)
    tl.store(output + rows * PAD_N + cols, converted, mask=valid)
    words = _affine_quantize_pack_with_metadata(converted.to(tl.float32), valid, tl.load(step + group),
                           tl.load(minimum + group), BITS, BLOCK)
    word_offsets = chunk * (BLOCK // (32 // BITS)) + tl.arange(0, BLOCK // (32 // BITS))
    tl.store(packed + group * WORDS + word_offsets, words, mask=word_offsets < WORDS)


def _quantize_pack(x, scale, dtype, block_shape, group_size, *, observe=False, check_finite=False, snapshot=False, bits=2):
    _validate_bits(bits)
    m, n = x.shape
    bm, bn = block_shape or (1, group_size)
    elements = bm * bn
    groups = (triton.cdiv(m, bm) * triton.cdiv(n, bn) if block_shape else
              triton.cdiv(m * n, group_size))
    words = triton.cdiv(elements, 32 // bits)
    padded_n = triton.cdiv(n, 16) * 16
    normal = torch.empty((m, padded_n), device=x.device, dtype=dtype)
    packed = torch.empty((groups, words), device=x.device, dtype=torch.int32)
    step = torch.empty(groups, device=x.device, dtype=torch.float32)
    minimum = torch.empty_like(step)
    saved_scale = torch.empty(2, device=x.device, dtype=torch.float32) if snapshot else scale
    chunks = triton.cdiv(elements, 4096) if elements > 16384 else 1
    observations = torch.empty(groups * chunks, device=x.device, dtype=torch.float32) if observe else step
    errors = torch.empty(groups * chunks, device=x.device, dtype=torch.int32) if check_finite else packed
    common = dict(GROUP_SIZE=group_size, BM=bm, BN=bn, GRID_N=triton.cdiv(n, bn),
                  TENSORWISE=block_shape is None, MAXIMUM=torch.finfo(dtype).max)
    if chunks == 1:
        block = max(32 // bits, triton.next_power_of_2(elements))
        _cast_quantize_pack[(groups,)](
            x, normal, scale, packed, step, minimum, saved_scale, observations, errors,
            m, n, *x.stride(), PAD_N=padded_n, WORDS=words, BLOCK=block,
            SNAPSHOT=snapshot, OBSERVE=observe, CHECK=check_finite, BITS=bits, **common,
            num_warps=8 if block >= 8192 else 4, enable_fp_fusion=False)
    else:
        low = torch.empty((groups, chunks), device=x.device, dtype=torch.float32)
        high = torch.empty_like(low)
        # A pointer's element type supplies the FP8 format without a device cast.
        tl_dtype = tl.float8e4nv if dtype == torch.float8_e4m3fn else tl.float8e5
        _partial_minmax[(groups * chunks,)](
            x, scale, low, high, saved_scale, observations, errors, m, n, *x.stride(),
            CHUNKS=chunks, BLOCK=4096, DTYPE=tl_dtype, SNAPSHOT=snapshot,
            OBSERVE=observe, CHECK=check_finite, **common, enable_fp_fusion=False)
        count = chunks
        while count > 1024:
            next_count = triton.cdiv(count, 1024)
            next_low = torch.empty((groups, next_count), device=x.device, dtype=torch.float32)
            next_high = torch.empty_like(next_low)
            _reduce_minmax[(groups * next_count,)](low, high, next_low, next_high, count,
                                                  NEXT=next_count, BLOCK=1024)
            low, high, count = next_low, next_high, next_count
        _finish_minmax[(groups,)](low, high, step, minimum, count,
                                 BLOCK=triton.next_power_of_2(count), BITS=bits)
        _cast_quantize_pack_chunks[(groups * chunks,)](
            x, normal, scale, packed, step, minimum, m, n, *x.stride(), PAD_N=padded_n,
            CHUNKS=chunks, WORDS=words, BLOCK=4096, BITS=bits, **common, enable_fp_fusion=False)
    if check_finite and errors.any().item():
        raise ValueError("Delayed scaling requires finite inputs")
    codec = PackedActivation(packed, step, minimum, tuple(x.shape), group_size, block_shape, bits)
    return normal[:, :n], codec, saved_scale, observations if observe else None


def quantize_pack_fp8_activation(x, dtype, block_shape, group_size, check_finite, *, bits=2):
    """Retain current scaling reductions; fuse FP8 casting and affine quantize/pack."""
    with torch.cuda.device(x.device):
        scale, inverse = _run_scaling(x, torch.finfo(dtype).max, block_shape, check_finite)
        values, codec, _, _ = _quantize_pack(x, scale, dtype, block_shape, group_size, bits=bits)
    return values, inverse, codec


def quantize_pack_delayed_fp8_activation(x, scale, dtype, group_size, *, observe, check_finite, bits=2):
    """Fuse FP8 casting and affine quantize/pack with a snapshot of the scale."""
    with torch.cuda.device(x.device):
        return _quantize_pack(x, scale, dtype, None, group_size, observe=observe,
                              check_finite=check_finite, snapshot=True, bits=bits)
