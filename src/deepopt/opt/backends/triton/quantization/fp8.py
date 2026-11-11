"""Fused CUDA FP8 scaling, quantization and reconstruction from strided inputs."""

import torch
import triton
import triton.language as tl


# Algorithm thresholds, not supported block-size limits. Larger groups use
# bounded partial reductions before fused scaling/conversion.
_MAX_SINGLE_PASS_BLOCK_ELEMENTS = 16384
_MAX_SINGLE_PASS_TENSOR_ELEMENTS = 4096


@triton.jit
def _scale(amax, FP8_MAX: tl.constexpr):
    return tl.where(amax == 0, 1.0, tl.div_rn(FP8_MAX, tl.maximum(amax, 1e-12)))


@triton.jit
def _fused_blocks(x_ptr, out_ptr, inv_ptr, errors_ptr, M, N, STRIDE_M, STRIDE_N,
                  NUM_COL_BLOCKS: tl.constexpr, NUM_BLOCKS: tl.constexpr,
                  BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                  ELEMENTS: tl.constexpr, GROUPS: tl.constexpr,
                  FP8_MAX: tl.constexpr, CHECK_FINITE: tl.constexpr,
                  QUANTIZE: tl.constexpr, CONTIGUOUS: tl.constexpr):

    blocks = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    elements = tl.arange(0, ELEMENTS)

    rows = (blocks // NUM_COL_BLOCKS)[:, None] * BLOCK_ROWS + (elements // BLOCK_COLS)[None, :]
    cols = (blocks % NUM_COL_BLOCKS)[:, None] * BLOCK_COLS + (elements % BLOCK_COLS)[None, :]

    mask = ((blocks[:, None] < NUM_BLOCKS) & (rows < M) & (cols < N)
            & (elements[None, :] < BLOCK_ROWS * BLOCK_COLS))

    if CONTIGUOUS:
        addresses = rows * N + cols
    else:
        addresses = rows * STRIDE_M + cols * STRIDE_N
    values = tl.load(x_ptr + addresses, mask=mask, other=0.0).to(tl.float32)

    amax = tl.max(tl.abs(values), axis=1)
    scale = _scale(amax, FP8_MAX)
    if QUANTIZE:
        scaled = tl.minimum(tl.maximum(values * scale[:, None], -FP8_MAX), FP8_MAX)
        quantized = scaled.to(out_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(out_ptr + rows * N + cols, quantized, mask=mask)
    else:
        tl.store(out_ptr + blocks, scale, mask=blocks < NUM_BLOCKS)
    tl.store(inv_ptr + blocks, tl.div_rn(1.0, scale), mask=blocks < NUM_BLOCKS)

    if CHECK_FINITE:
        bad = tl.max(tl.max((~(tl.abs(values) < float("inf"))).to(tl.int32), axis=1), axis=0)
        tl.store(errors_ptr + tl.program_id(0), bad)


@triton.jit
def _partial_amax(x_ptr, partials_ptr, errors_ptr, M, N, STRIDE_M, STRIDE_N,
                  BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                  NUM_COL_BLOCKS: tl.constexpr, CHUNKS: tl.constexpr, BLOCK: tl.constexpr,
                  CONTIGUOUS: tl.constexpr, TENSORWISE: tl.constexpr, CHECK_FINITE: tl.constexpr):
    scale_block = tl.program_id(0) // CHUNKS
    chunk = tl.program_id(0) % CHUNKS
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)

    rows = (scale_block // NUM_COL_BLOCKS) * BLOCK_ROWS + offsets // BLOCK_COLS
    cols = (scale_block % NUM_COL_BLOCKS) * BLOCK_COLS + offsets % BLOCK_COLS

    if CONTIGUOUS and TENSORWISE:
        addresses = offsets
    elif CONTIGUOUS:
        addresses = rows * N + cols
    else:
        addresses = rows * STRIDE_M + cols * STRIDE_N
    if TENSORWISE:
        mask = offsets < M * N
    else:
        mask = (rows < M) & (cols < N) & (offsets < BLOCK_ROWS * BLOCK_COLS)
    values = tl.load(x_ptr + addresses, mask=mask, other=0.0).to(tl.float32)

    tl.store(partials_ptr + scale_block * CHUNKS + chunk, tl.max(tl.abs(values), axis=0))
    if CHECK_FINITE:
        bad = tl.max((~(tl.abs(values) < float("inf"))).to(tl.int32), axis=0)
        tl.store(errors_ptr + tl.program_id(0), bad)


@triton.jit
def _reduce_amax(partials_ptr, reduced_ptr, COUNT, OUTPUT_COUNT, BLOCK: tl.constexpr):
    scale_block = tl.program_id(0) // OUTPUT_COUNT
    chunk = tl.program_id(0) % OUTPUT_COUNT
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(partials_ptr + scale_block * COUNT + offsets,
                     mask=offsets < COUNT, other=0.0)
    tl.store(reduced_ptr + scale_block * OUTPUT_COUNT + chunk, tl.max(values, axis=0))


@triton.jit
def _block_scales(partials_ptr, scale_ptr, inv_ptr, COUNT,
                  BLOCK: tl.constexpr, FP8_MAX: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(partials_ptr + tl.program_id(0) * COUNT + offsets,
                     mask=offsets < COUNT, other=0.0)
    scale = _scale(tl.max(values, axis=0), FP8_MAX)
    tl.store(scale_ptr + tl.program_id(0), scale)
    tl.store(inv_ptr + tl.program_id(0), tl.div_rn(1.0, scale))


@triton.jit
def _cast_scaled(x_ptr, q_ptr, scale_ptr, errors_ptr, N, NUMEL, STRIDE_M, STRIDE_N,
                 BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr, NUM_COL_BLOCKS: tl.constexpr,
                 TENSORWISE: tl.constexpr, BLOCK: tl.constexpr,
                 FP8_MAX: tl.constexpr, CHECK_FINITE: tl.constexpr, CONTIGUOUS: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CONTIGUOUS:
        addresses = offsets
    else:
        addresses = (offsets // N) * STRIDE_M + (offsets % N) * STRIDE_N
    values = tl.load(x_ptr + addresses,
                     mask=offsets < NUMEL, other=0.0).to(tl.float32)
    if TENSORWISE:
        scale = tl.load(scale_ptr)
    else:
        scale_offsets = ((offsets // N) // BLOCK_ROWS) * NUM_COL_BLOCKS + (offsets % N) // BLOCK_COLS
        scale = tl.load(scale_ptr + scale_offsets, mask=offsets < NUMEL, other=1.0)
    scaled = tl.minimum(tl.maximum(values * scale, -FP8_MAX), FP8_MAX)
    tl.store(q_ptr + offsets, scaled.to(q_ptr.dtype.element_ty, fp_downcast_rounding="rtne"),
             mask=offsets < NUMEL)
    if CHECK_FINITE:
        bad = tl.max((~(tl.abs(values) < float("inf"))).to(tl.int32), axis=0)
        tl.store(errors_ptr + tl.program_id(0), bad)


def quantize(x, fp8_dtype, block_shape, check_finite):
    """Inputs are structurally validated by the public quantization wrapper."""
    with torch.cuda.device(x.device):
        return _run_scaling(x, torch.finfo(fp8_dtype).max, block_shape, check_finite, fp8_dtype)


def amax_scale(x, fp8_max, block_shape, check_finite):
    """Compute both scales using the same reductions as fused conversion."""
    with torch.cuda.device(x.device):
        return _run_scaling(x, fp8_max, block_shape, check_finite)


def _run_scaling(x, fp8_max, block_shape, check_finite, fp8_dtype=None):
    m, n = x.shape
    numel = m * n
    quantizing = fp8_dtype is not None
    # All layout selection belongs to the launcher; kernels read original
    # strides directly when the contiguous address fast path does not apply.
    contiguous = x.is_contiguous()
    scale_shape = ((1, 1) if block_shape is None else
                   (triton.cdiv(m, block_shape[0]), triton.cdiv(n, block_shape[1])))
    output = torch.empty((m, n) if quantizing else scale_shape, device=x.device,
                         dtype=fp8_dtype if quantizing else torch.float32)
    inverse = torch.empty(scale_shape, device=x.device, dtype=torch.float32)
    # If a configured block exceeds a dimension there is only one block along
    # that axis. Clipping preserves its scale grid while bounding padded work.
    bm, bn = (min(block_shape[0], m), min(block_shape[1], n)) if block_shape else (m, n)
    blocks = scale_shape[0] * scale_shape[1]
    limit = _MAX_SINGLE_PASS_TENSOR_ELEMENTS if block_shape is None else _MAX_SINGLE_PASS_BLOCK_ELEMENTS
    if bm * bn <= limit:
        elements = triton.next_power_of_2(bm * bn)
        groups = min(32, max(1, 1024 // elements))
        programs = triton.cdiv(blocks, groups)
        errors = torch.empty((programs,), device=x.device, dtype=torch.int32) if check_finite else inverse

        _fused_blocks[(programs,)](
            x, output, inverse, errors, m, n, *x.stride(),
            NUM_COL_BLOCKS=scale_shape[1], NUM_BLOCKS=blocks,
            BLOCK_ROWS=bm, BLOCK_COLS=bn, ELEMENTS=elements, GROUPS=groups,
            FP8_MAX=fp8_max, CHECK_FINITE=check_finite,
            QUANTIZE=quantizing, CONTIGUOUS=contiguous,
            num_warps=8 if elements >= 8192 else 4,
        )
    else:
        # Large scale groups need their complete amax before conversion.
        # Each reduction stage is bounded; no atomics or grid-wide barrier.
        block = 4096
        count = triton.cdiv(bm * bn, block)
        partials = torch.empty((blocks, count), device=x.device, dtype=torch.float32)
        partial_checks = check_finite and not quantizing
        errors = (torch.empty((blocks * count,), device=x.device, dtype=torch.int32)
                  if partial_checks else inverse)
        # Flatten partials into the X grid to avoid CUDA's much smaller Y limit.
        _partial_amax[(blocks * count,)](
            x, partials, errors, m, n, *x.stride(), BLOCK_ROWS=bm, BLOCK_COLS=bn,
            NUM_COL_BLOCKS=scale_shape[1], CHUNKS=count, BLOCK=block,
            CONTIGUOUS=contiguous, TENSORWISE=block_shape is None, CHECK_FINITE=partial_checks)
        while count > 1024:
            next_count = triton.cdiv(count, 1024)
            reduced = torch.empty((blocks, next_count), device=x.device, dtype=torch.float32)
            _reduce_amax[(blocks * next_count,)](partials, reduced, count, next_count, BLOCK=1024)
            partials, count = reduced, next_count
        scale = torch.empty(scale_shape, device=x.device, dtype=torch.float32) if quantizing else output
        _block_scales[(blocks,)](partials, scale, inverse, count,
                                 BLOCK=triton.next_power_of_2(count), FP8_MAX=fp8_max)
        if quantizing:
            programs = triton.cdiv(numel, block)
            errors = torch.empty((programs,), device=x.device, dtype=torch.int32) if check_finite else inverse
            _cast_scaled[(programs,)](
                x, output, scale, errors, n, numel, *x.stride(), BLOCK_ROWS=bm, BLOCK_COLS=bn,
                NUM_COL_BLOCKS=scale_shape[1], TENSORWISE=block_shape is None, BLOCK=block,
                FP8_MAX=fp8_max, CHECK_FINITE=check_finite, CONTIGUOUS=contiguous)
    if check_finite and errors.any().item():
        raise ValueError("AmaxScaling requires finite inputs")
    return output, inverse


@triton.jit
def _dequantize(values_ptr, scales_ptr, output_ptr, M, N, STRIDE_M, STRIDE_N,
                 SCALE_STRIDE_M, SCALE_STRIDE_N, BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr,
                 TENSORWISE: tl.constexpr, CONTIGUOUS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows, cols = offsets // N, offsets % N
    mask = offsets < M * N
    if CONTIGUOUS:
        addresses = offsets
    else:
        addresses = rows * STRIDE_M + cols * STRIDE_N
    values = tl.load(values_ptr + addresses, mask=mask, other=0.0).to(tl.float32)
    if TENSORWISE:
        scales = tl.load(scales_ptr)
    else:
        addresses = (rows // BLOCK_ROWS) * SCALE_STRIDE_M + (cols // BLOCK_COLS) * SCALE_STRIDE_N
        scales = tl.load(scales_ptr + addresses, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, values * scales, mask=mask)


def dequantize(values, inverse, block_shape):
    """Reconstruct FP32 directly, without an FP32 copy or expanded scale grid."""
    with torch.cuda.device(values.device):
        m, n = values.shape
        output = torch.empty((m, n), device=values.device, dtype=torch.float32)
        bm, bn = block_shape or (m, n)
        _dequantize[(triton.cdiv(m * n, 1024),)](
            values, inverse, output, m, n, *values.stride(), *inverse.stride(),
            BLOCK_ROWS=bm, BLOCK_COLS=bn, TENSORWISE=block_shape is None,
            CONTIGUOUS=values.is_contiguous(), BLOCK=1024)
        return output


@triton.jit
def _cast_tensorwise_pair(x_ptr, normal_ptr, transpose_ptr, scale_ptr, M, N,
                          STRIDE_M, STRIDE_N, PADDED_M: tl.constexpr, PADDED_N: tl.constexpr,
                          N_TILES: tl.constexpr, FP8_MAX: tl.constexpr, BLOCK: tl.constexpr,
                          WRITE_NORMAL: tl.constexpr):
    rows = (tl.program_id(0) // N_TILES) * BLOCK + tl.arange(0, BLOCK)
    cols = (tl.program_id(0) % N_TILES) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x_ptr + rows[:, None] * STRIDE_M + cols[None, :] * STRIDE_N,
                     mask=(rows[:, None] < M) & (cols[None, :] < N), other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr)
    values = tl.minimum(tl.maximum(values * scale, -FP8_MAX), FP8_MAX)
    converted = values.to(normal_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    if WRITE_NORMAL:
        tl.store(normal_ptr + rows[:, None] * PADDED_N + cols[None, :], converted,
                 mask=(rows[:, None] < M) & (cols[None, :] < PADDED_N))
    tl.store(transpose_ptr + cols[:, None] * PADDED_M + rows[None, :], converted.T,
             mask=(cols[:, None] < N) & (rows[None, :] < PADDED_M))


def quantize_tensorwise_pair(x, fp8_dtype, check_finite, *, transpose_only=False):
    """Emit both TMA-ready orientations without a later FP8 transpose/pack."""
    with torch.cuda.device(x.device):
        maximum = torch.finfo(fp8_dtype).max
        scale, inverse = _run_scaling(x, maximum, None, check_finite)
        m, n = x.shape
        padded_m, padded_n = triton.cdiv(m, 16) * 16, triton.cdiv(n, 16) * 16
        normal = None if transpose_only else torch.empty((m, padded_n), device=x.device, dtype=fp8_dtype)
        transposed = torch.empty((n, padded_m), device=x.device, dtype=fp8_dtype)
        tiles_n = triton.cdiv(n, 32)
        _cast_tensorwise_pair[(triton.cdiv(m, 32) * tiles_n,)](
            x, transposed if transpose_only else normal, transposed, scale, m, n, *x.stride(),
            PADDED_M=padded_m, PADDED_N=padded_n, WRITE_NORMAL=not transpose_only,
            N_TILES=tiles_n, FP8_MAX=maximum, BLOCK=32, num_warps=4)
        return None if transpose_only else normal[:, :n], inverse, transposed[:, :m]
