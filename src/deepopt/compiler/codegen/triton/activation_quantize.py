"""Experimental fused GELU/SiLU statistics, FP8 conversion and affine packing.

Current tensorwise scaling retains the global reduction boundary. The producer
is evaluated in-register in statistics and conversion kernels, eliminating its
materialized activation. Qualify these templates on the target GPU before
enabling them; recomputation can cost more than the saved memory traffic.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from deepopt.opt.backends.triton.quantization.fp8 import _reduce_amax, _block_scales
from deepopt.opt.backends.triton.quantization.activation.affine import _affine_quantize_pack
from deepopt.opt.quantization.tensor import QuantizedTensor
from deepopt.opt.quantization.activation import PackedActivation
from deepopt.opt._execution import finite_checks_enabled


@triton.jit
def _activate(value, KIND: tl.constexpr, APPROXIMATE: tl.constexpr, DTYPE: tl.constexpr):
    value = value.to(tl.float32)
    if KIND == "gelu":
        if APPROXIMATE == "tanh":
            result = 0.5 * value * (1.0 + libdevice.tanh(
                0.7978845608028654 * (value + 0.044715 * value * value * value)))
        else:
            result = 0.5 * value * (1.0 + libdevice.erf(value * 0.7071067811865476))
    else:
        result = value * (1.0 / (1.0 + tl.exp(-value)))
    # Match the native activation's materialized logical output before amax.
    return result.to(DTYPE).to(tl.float32)


@triton.jit
def _activation_amax(x, partial, errors, COUNT: tl.constexpr, KIND: tl.constexpr,
                     APPROXIMATE: tl.constexpr, CHECK: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + offsets, offsets < COUNT, other=0.0)
    activated = _activate(value, KIND, APPROXIMATE, x.dtype.element_ty)
    tl.store(partial + tl.program_id(0), tl.max(tl.abs(activated), 0))
    if CHECK:
        bad = (offsets < COUNT) & (~(tl.abs(activated) < float("inf")))
        tl.store(errors + tl.program_id(0), tl.max(bad.to(tl.int32), 0))


@triton.jit
def _activation_cast(x, normal, transposed, scale, M: tl.constexpr, N: tl.constexpr,
                     PAD_M: tl.constexpr, PAD_N: tl.constexpr, TILES_N: tl.constexpr,
                     KIND: tl.constexpr, APPROXIMATE: tl.constexpr,
                     PAIR: tl.constexpr, MAXIMUM: tl.constexpr, BLOCK: tl.constexpr):
    program = tl.program_id(0).to(tl.int64)
    rows = (program // TILES_N) * BLOCK + tl.arange(0, BLOCK)
    cols = (program % TILES_N) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + rows[:, None] * N + cols[None, :],
                    (rows[:, None] < M) & (cols[None, :] < N), other=0.0)
    activated = _activate(value, KIND, APPROXIMATE, x.dtype.element_ty)
    converted = tl.minimum(tl.maximum(activated * tl.load(scale), -MAXIMUM), MAXIMUM)
    converted = converted.to(normal.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(normal + rows[:, None] * PAD_N + cols[None, :], converted,
             (rows[:, None] < M) & (cols[None, :] < PAD_N))
    if PAIR:
        tl.store(transposed + cols[:, None] * PAD_M + rows[None, :], converted.T,
                 (cols[:, None] < N) & (rows[None, :] < PAD_M))


@triton.jit
def _activation_pack(x, normal, scale, payload, steps, minima,
                     COUNT: tl.constexpr, N: tl.constexpr, PAD_N: tl.constexpr,
                     GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                     KIND: tl.constexpr, APPROXIMATE: tl.constexpr,
                     MAXIMUM: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    offsets = group * GROUP + tl.arange(0, BLOCK)
    valid = (tl.arange(0, BLOCK) < GROUP) & (offsets < COUNT)
    value = tl.load(x + offsets, valid, other=0.0)
    activated = _activate(value, KIND, APPROXIMATE, x.dtype.element_ty)
    scaled = tl.minimum(tl.maximum(activated * tl.load(scale), -MAXIMUM), MAXIMUM)
    scaled = tl.where(activated != activated, float("nan"), scaled)
    converted = scaled.to(normal.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(normal + (offsets // N) * PAD_N + offsets % N, converted, valid)
    words, step, minimum = _affine_quantize_pack(converted.to(tl.float32), valid, BITS, BLOCK)
    indices = tl.arange(0, BLOCK // (32 // BITS))
    tl.store(payload + group * WORDS + indices, words, indices < WORDS)
    tl.store(steps + group, step)
    tl.store(minima + group, minimum)


def prepare(x, *, activation, fp8_dtype, paired, compression):
    if not x.is_contiguous() or x.ndim != 2:
        raise ValueError("Fused activation preparation requires a contiguous matrix")
    kind, approximate = activation
    if kind not in ("gelu", "silu") or approximate not in ("none", "tanh"):
        raise ValueError("Unsupported fused activation")
    checked = finite_checks_enabled()
    rows, cols = x.shape
    count = triton.cdiv(x.numel(), 4096)
    with torch.cuda.device(x.device):
        partial = torch.empty(count, device=x.device, dtype=torch.float32)
        errors = torch.empty(count, device=x.device, dtype=torch.int32) if checked else partial
        _activation_amax[(count,)](
            x, partial, errors, x.numel(), kind, approximate, checked, 4096,
            enable_fp_fusion=False,
        )
        remaining = count
        while remaining > 1024:
            next_count = triton.cdiv(remaining, 1024)
            reduced = torch.empty(next_count, device=x.device, dtype=torch.float32)
            _reduce_amax[(next_count,)](partial, reduced, remaining, next_count, BLOCK=1024)
            partial, remaining = reduced, next_count
        scale = torch.empty((1, 1), device=x.device, dtype=torch.float32)
        inverse = torch.empty_like(scale)
        maximum = torch.finfo(fp8_dtype).max
        _block_scales[(1,)](
            partial, scale, inverse, remaining,
            BLOCK=triton.next_power_of_2(remaining), FP8_MAX=maximum,
        )
        if checked and errors.any().item():
            raise ValueError("AmaxScaling requires finite inputs")
        padded_rows, padded_cols = triton.cdiv(rows, 16) * 16, triton.cdiv(cols, 16) * 16
        normal = torch.empty((rows, padded_cols), device=x.device, dtype=fp8_dtype)
        packed = None
        transposed = None
        if compression is not None:
            group, bits = compression["group_size"], compression["bits"]
            if not 0 < group <= 16384:
                raise ValueError("Fused packing supports groups up to 16384")
            groups = triton.cdiv(x.numel(), group)
            words = triton.cdiv(group, 32 // bits)
            payload = torch.empty((groups, words), device=x.device, dtype=torch.int32)
            steps = torch.empty(groups, device=x.device, dtype=torch.float32)
            minima = torch.empty_like(steps)
            block = max(32 // bits, triton.next_power_of_2(group))
            _activation_pack[(groups,)](
                x, normal, scale, payload, steps, minima,
                x.numel(), cols, padded_cols, group, words, bits,
                kind, approximate, maximum, block,
                num_warps=8 if block >= 8192 else 4, enable_fp_fusion=False,
            )
            packed = PackedActivation(payload, steps, minima, tuple(x.shape), group, None, bits)
        else:
            transposed = (
                torch.empty((cols, padded_rows), device=x.device, dtype=fp8_dtype)
                if paired else inverse
            )
            _activation_cast[(triton.cdiv(rows, 32) * triton.cdiv(cols, 32),)](
                x, normal, transposed, scale, rows, cols, padded_rows, padded_cols,
                triton.cdiv(cols, 32), kind, approximate, paired, maximum, 32,
                num_warps=4, enable_fp_fusion=False,
            )
            transposed = QuantizedTensor(transposed[:, :rows], inverse) if paired else None
    return QuantizedTensor(normal[:, :cols], inverse), transposed, packed
