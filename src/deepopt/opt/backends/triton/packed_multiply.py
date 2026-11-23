"""Affine decode/product and bounded broadcast-gradient reductions."""

from math import prod

import torch
import triton
import triton.language as tl

from .quantization.activation.affine import _affine_decode
from .._shared.packed_multiply import contiguous_strides, plan_packed_multiply


@triton.jit
def _addresses(index, SHAPE: tl.constexpr, LOGICAL_STRIDES: tl.constexpr, STRIDES: tl.constexpr):
    # Promote before division/stride multiplication, including each input to
    # this helper. A logical VAE activation can exceed 2**32 elements.
    index = index.to(tl.int64)
    address = tl.full(index.shape, 0, tl.int64)
    for axis in tl.static_range(len(SHAPE)):
        address += index // LOGICAL_STRIDES[axis] % SHAPE[axis] * STRIDES[axis]
    return address


@triton.jit
def _product(index, G, Data, Step, Minimum, SHAPE: tl.constexpr, LOGICAL_STRIDES: tl.constexpr,
             GS: tl.constexpr, CS: tl.constexpr, GROUP: tl.constexpr,
             WORDS: tl.constexpr, BITS: tl.constexpr,
             COEFF_BF16: tl.constexpr, PRODUCT_BF16: tl.constexpr, valid):
    coeff_index = _addresses(index, SHAPE, LOGICAL_STRIDES, CS)
    group, within = coeff_index // GROUP, coeff_index % GROUP
    word = tl.load(Data + group * WORDS + within // (32 // BITS), valid, other=0).to(tl.uint32)
    codes = ((word >> (within % (32 // BITS) * BITS)) & ((1 << BITS) - 1)).to(tl.float32)
    decoded = _affine_decode(codes, tl.load(Step + group, valid, other=0),
                            tl.load(Minimum + group, valid, other=0), BITS, 0.0)
    if COEFF_BF16:
        decoded = decoded.to(tl.bfloat16).to(tl.float32)
    upstream = tl.load(G + _addresses(index, SHAPE, LOGICAL_STRIDES, GS), valid, other=0).to(tl.float32)
    value = upstream * decoded
    if PRODUCT_BF16:
        value = value.to(tl.bfloat16).to(tl.float32)
    return value


@triton.jit
def _packed_multiply(G, Data, Step, Minimum, Y, COUNT: tl.constexpr,
                     SHAPE: tl.constexpr, LOGICAL_STRIDES: tl.constexpr, GS: tl.constexpr, CS: tl.constexpr,
                     GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                     COEFF_BF16: tl.constexpr, PRODUCT_BF16: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    value = _product(index, G, Data, Step, Minimum, SHAPE, LOGICAL_STRIDES, GS, CS, GROUP, WORDS, BITS,
                     COEFF_BF16, PRODUCT_BF16, index < COUNT)
    tl.store(Y + index, value, index < COUNT)


@triton.jit
def _packed_multiply_partial(G, Data, Step, Minimum, Partial,
                             SHAPE: tl.constexpr, LOGICAL_STRIDES: tl.constexpr, GS: tl.constexpr, CS: tl.constexpr,
                             TARGET: tl.constexpr, TARGET_STRIDES: tl.constexpr,
                             REDUCTION: tl.constexpr, REDUCTION_STRIDES: tl.constexpr,
                             OUTPUTS: tl.constexpr, REDUCED: tl.constexpr,
                             OUTPUT_OFFSET, TILE_OUTPUTS: tl.constexpr,
                             PARTS: tl.constexpr, LOOPS: tl.constexpr,
                             GROUP: tl.constexpr, WORDS: tl.constexpr, BITS: tl.constexpr,
                             COEFF_BF16: tl.constexpr, PRODUCT_BF16: tl.constexpr,
                             R: tl.constexpr, O: tl.constexpr):
    local_outputs = tl.program_id(0).to(tl.int64) * O + tl.arange(0, O).to(tl.int64)
    outputs = OUTPUT_OFFSET.to(tl.int64) + local_outputs
    shard = tl.program_id(1).to(tl.int64)
    # TARGET and REDUCTION divide the logical product axes. Their contiguous
    # indices map directly into the product's original row-major traversal.
    # Division strides are precomputed on the host. Reassigning a constexpr
    # inside a static loop is rejected by Triton 3.9 for multi-axis shapes.
    kept = _addresses(outputs, TARGET, TARGET_STRIDES, LOGICAL_STRIDES)
    partial = tl.full((O,), 0, tl.float32)
    for iteration in range(LOOPS):
        # A grid-stride traversal caps the number of shards even for scalar
        # reductions across billions of rounded products.
        reduced = ((iteration.to(tl.int64) * PARTS + shard) * R
                   + tl.arange(0, R).to(tl.int64))
        logical = (kept[None, :] + _addresses(reduced, REDUCTION,
                   REDUCTION_STRIDES, LOGICAL_STRIDES)[:, None])
        valid = ((outputs[None, :] < OUTPUTS) & (local_outputs[None, :] < TILE_OUTPUTS)
                 & (reduced[:, None] < REDUCED))
        value = _product(logical, G, Data, Step, Minimum, SHAPE, LOGICAL_STRIDES, GS, CS, GROUP, WORDS, BITS,
                         COEFF_BF16, PRODUCT_BF16, valid)
        partial += tl.sum(tl.where(valid, value, 0.0), axis=0)
    tl.store(Partial + shard * TILE_OUTPUTS + local_outputs, partial,
             (outputs < OUTPUTS) & (local_outputs < TILE_OUTPUTS))


@triton.jit
def _packed_multiply_finish(Partial, Y, OUTPUTS: tl.constexpr, PARTS: tl.constexpr,
                            OUTPUT_OFFSET, TILE_OUTPUTS: tl.constexpr,
                            PRODUCT_BF16: tl.constexpr, R: tl.constexpr, O: tl.constexpr):
    local_outputs = tl.program_id(0).to(tl.int64) * O + tl.arange(0, O).to(tl.int64)
    outputs = OUTPUT_OFFSET.to(tl.int64) + local_outputs
    reduced = tl.arange(0, R).to(tl.int64)
    value = tl.load(Partial + reduced[:, None] * TILE_OUTPUTS + local_outputs[None, :],
                    (reduced[:, None] < PARTS) & (local_outputs[None, :] < TILE_OUTPUTS)
                    & (outputs[None, :] < OUTPUTS), other=0)
    total = tl.sum(value, axis=0)
    # sum_to_size retains the product dtype before a distinct target cast
    # (notably the exact FP32 scalar residual parameter).
    if PRODUCT_BF16:
        total = total.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + outputs, total, (outputs < OUTPUTS) & (local_outputs < TILE_OUTPUTS))


def broadcast_strides(shape, strides, output_shape):
    padded_shape = (1,) * (len(output_shape) - len(shape)) + tuple(shape)
    padded_stride = (0,) * (len(output_shape) - len(shape)) + tuple(strides)
    return tuple(0 if size == 1 else stride for size, stride in zip(padded_shape, padded_stride))


def multiply(gradient, data, step, minimum, *, options, shape, product_dtype, target_shape,
             target_dtype=None, scratch_bytes=8 * 1024 * 1024, launch_plan=None):
    plan = launch_plan or plan_packed_multiply(shape, target_shape, scratch_bytes)
    if plan.shape != tuple(shape) or plan.target_shape != tuple(target_shape):
        raise ValueError("packed multiply launch plan has different geometry")
    coefficient_shape = tuple(options["logical_shape"])
    gs = broadcast_strides(gradient.shape, gradient.stride(), shape)
    cs = broadcast_strides(coefficient_shape, contiguous_strides(coefficient_shape), shape)
    group, bits = options["group_size"], options["bits"]
    output = torch.empty(target_shape, dtype=target_dtype or product_dtype, device=gradient.device)
    args = (gradient, data, step, minimum)
    common = (tuple(shape), contiguous_strides(shape), gs, cs)
    codec = (group, data.shape[1], bits, options["logical_dtype"] == torch.bfloat16,
             product_dtype == torch.bfloat16)
    with torch.cuda.device(gradient.device):
        if not prod(shape):
            return output.zero_()
        if plan.direct:
            _packed_multiply[(triton.cdiv(prod(shape), 256),)](
                *args, output, prod(shape), *common, *codec, 256, enable_fp_fusion=False)
        else:
            partial = torch.empty((plan.parts, plan.output_tile), dtype=torch.float32, device=gradient.device)
            block_outputs = min(16, plan.output_tile)
            grid = (triton.cdiv(plan.output_tile, block_outputs), plan.parts)
            for offset in range(0, plan.outputs, plan.output_tile):
                _packed_multiply_partial[grid](
                    *args, partial, *common, plan.padded_target, contiguous_strides(plan.padded_target),
                    plan.reduction_shape, contiguous_strides(plan.reduction_shape), plan.outputs,
                    plan.reduced_elements, offset, plan.output_tile, plan.parts, plan.reduction_loops,
                    *codec, plan.reduction_block, block_outputs, enable_fp_fusion=False)
                _packed_multiply_finish[(grid[0],)](
                    partial, output, plan.outputs, plan.parts, offset, plan.output_tile,
                    product_dtype == torch.bfloat16, triton.next_power_of_2(plan.parts), block_outputs,
                    enable_fp_fusion=False)
    return output
