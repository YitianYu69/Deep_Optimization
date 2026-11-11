"""Implicit FP8 convolution gradients without atomics or unfolded tensors."""

import torch
import triton
import triton.language as tl

from .common import (_load_operand, _load_scale, _scale_chunk_length, _coordinates,
                     _tile_coordinates, validate, arguments, grid)
from .launch import weight_splits
from .weight_gradient_blockwise import _blockwise_weight_gradient_kernel
from ..gradient import _gradient_tile


@triton.jit
def _input_gradient_kernel(
    a, b, a_scale, b_scale, output,
    BATCH: tl.constexpr, ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    VALID_I0: tl.constexpr, VALID_I1: tl.constexpr, VALID_I2: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    DD: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    CI: tl.constexpr, CO: tl.constexpr, GROUPS: tl.constexpr,
    A_RS: tl.constexpr, A_CS: tl.constexpr, A_BR: tl.constexpr, A_BC: tl.constexpr,
    A_SRS: tl.constexpr, A_SCS: tl.constexpr,
    B_RS: tl.constexpr, B_CS: tl.constexpr, B_BR: tl.constexpr, B_BC: tl.constexpr,
    B_SRS: tl.constexpr, B_SCS: tl.constexpr,
    TENSORWISE: tl.constexpr, INDEX64: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    gradient_scale=None, gradient_inverse=None, gradient_observations=None,
    GRADIENT_FP8: tl.constexpr = False, GRAD_CURRENT: tl.constexpr = False,
    GRAD_BR: tl.constexpr = 1, GRAD_BC: tl.constexpr = 1,
    GRAD_MAXIMUM: tl.constexpr = 448.0, GRAD_OBSERVE: tl.constexpr = False,
    bias=None, HAS_BIAS: tl.constexpr = False, BIAS_STRIDE: tl.constexpr = 1,
):
    volume: tl.constexpr = ID * IH * IW
    kernel_volume: tl.constexpr = KD * KH * KW
    group, pm, pn = _tile_coordinates(BATCH * volume, CI, BLOCK_M, BLOCK_N)
    if INDEX64:
        group, pm, pn = group.to(tl.int64), pm.to(tl.int64), pn.to(tl.int64)
    rows = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    id_, ih, iw = _coordinates(rows % volume, IH, IW)
    batch = rows // volume
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # Each CTA owns its destination, including all overlapping windows.
    if TENSORWISE:
        for tile in range(tl.cdiv(CO * kernel_volume, BLOCK_K)):
            if INDEX64:
                reduction = tile.to(tl.int64) * BLOCK_K + tl.arange(0, BLOCK_K)
            else:
                reduction = tile * BLOCK_K + tl.arange(0, BLOCK_K)
            co = reduction // kernel_volume
            kp = reduction % kernel_volume
            kd, kh, kw = _coordinates(kp, KH, KW)
            od_numerator = id_[:, None] + PD - kd[None, :] * DD
            oh_numerator = ih[:, None] + PH - kh[None, :] * DH
            ow_numerator = iw[:, None] + PW - kw[None, :] * DW
            od, oh, ow = od_numerator // SD, oh_numerator // SH, ow_numerator // SW
            valid = ((rows[:, None] < BATCH * volume) & (reduction[None, :] < CO * kernel_volume)
                     & (od_numerator % SD == 0) & (oh_numerator % SH == 0) & (ow_numerator % SW == 0)
                     & (od >= 0) & (od < OD) & (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW))
            source = batch[:, None] * (OD * OH * OW) + (od * OH + oh) * OW + ow
            av = _load_operand(a, source, group * CO + co[None, :], valid, A_RS, A_CS)
            bv = _load_operand(b, group * CO + co[:, None], cols[None, :] * kernel_volume + kp[:, None],
                               (reduction[:, None] < CO * kernel_volume) & (cols[None, :] < CI), B_RS, B_CS)
            acc = tl.dot(av, bv, acc, out_dtype=tl.float32)
        acc = (acc * tl.load(a_scale)) * tl.load(b_scale)
    else:
        for kernel_index in range(kernel_volume):
            if INDEX64:
                kernel_pos = kernel_index.to(tl.int64)
            else:
                kernel_pos = kernel_index
            kd, kh, kw = _coordinates(kernel_pos, KH, KW)
            od_numerator = id_ + PD - kd * DD
            oh_numerator = ih + PH - kh * DH
            ow_numerator = iw + PW - kw * DW
            od, oh, ow = od_numerator // SD, oh_numerator // SH, ow_numerator // SW
            valid_rows = ((rows < BATCH * volume) & (od_numerator % SD == 0)
                          & (oh_numerator % SH == 0) & (ow_numerator % SW == 0)
                          & (od >= 0) & (od < OD) & (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW))
            source = batch * (OD * OH * OW) + (od * OH + oh) * OW + ow
            channel = group * 0
            while channel < CO:
                absolute_channel = group * CO + channel
                # qG varies across its channel blocks; qW varies across its
                # output-channel (row) blocks in this transposed reduction.
                length = _scale_chunk_length(absolute_channel, 1, A_BC, absolute_channel, 1, B_BR,
                                             tl.minimum(BLOCK_K, CO - channel))
                offsets = tl.arange(0, BLOCK_K)
                av = _load_operand(a, source[:, None], absolute_channel + offsets[None, :],
                                   valid_rows[:, None] & (offsets[None, :] < length), A_RS, A_CS)
                bv = _load_operand(b, absolute_channel + offsets[:, None],
                                   cols[None, :] * kernel_volume + kernel_pos,
                                   (offsets[:, None] < length) & (cols[None, :] < CI), B_RS, B_CS)
                partial = tl.dot(av, bv, out_dtype=tl.float32)
                sa = _load_scale(a_scale, source, absolute_channel, valid_rows, A_BR, A_BC, A_SRS, A_SCS)
                sb = _load_scale(b_scale, absolute_channel, cols * kernel_volume + kernel_pos, cols < CI,
                                 B_BR, B_BC, B_SRS, B_SCS)
                acc += (partial * sa[:, None]) * sb[None, :]
                channel += length
    valid = (rows[:, None] < BATCH * volume) & (cols[None, :] < CI)
    if HAS_BIAS:
        acc += tl.load(bias + (group * CI + cols) * BIAS_STRIDE, cols < CI, other=0).to(tl.float32)[None, :]
    if GRADIENT_FP8:
        acc = _gradient_tile(acc, rows, group * CI + cols, valid, gradient_scale, gradient_inverse,
                             gradient_observations, BATCH * volume, CI * GROUPS, BLOCK_M, BLOCK_N,
                             GRAD_BR, GRAD_BC, GRAD_MAXIMUM, GRAD_CURRENT, GRAD_OBSERVE)
        tl.store(output + rows[:, None] * (CI * GROUPS) + group * CI + cols[None, :],
                 acc.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
    else:
        tl.store(output + rows[:, None] * (CI * GROUPS) + group * CI + cols[None, :], acc, valid)


@triton.jit
def _weight_gradient_kernel(
    a, b, a_scale, b_scale, output,
    BATCH: tl.constexpr, ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    VALID_I0: tl.constexpr, VALID_I1: tl.constexpr, VALID_I2: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    DD: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr,
    CI: tl.constexpr, CO: tl.constexpr, GROUPS: tl.constexpr,
    A_RS: tl.constexpr, A_CS: tl.constexpr, A_BR: tl.constexpr, A_BC: tl.constexpr,
    A_SRS: tl.constexpr, A_SCS: tl.constexpr,
    B_RS: tl.constexpr, B_CS: tl.constexpr, B_BR: tl.constexpr, B_BC: tl.constexpr,
    B_SRS: tl.constexpr, B_SCS: tl.constexpr,
    TENSORWISE: tl.constexpr, INDEX64: tl.constexpr, SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    volume: tl.constexpr = OD * OH * OW
    kernel_volume: tl.constexpr = KD * KH * KW
    group, pm, pn = _tile_coordinates(CO, CI * kernel_volume, BLOCK_M, BLOCK_N)
    if INDEX64:
        group, pm, pn = group.to(tl.int64), pm.to(tl.int64), pn.to(tl.int64)
    rows = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    ci = cols // kernel_volume
    kd, kh, kw = _coordinates(cols % kernel_volume, KH, KW)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for tile in range(tl.program_id(1), tl.cdiv(BATCH * volume, BLOCK_K), SPLIT_K):
        if INDEX64:
            reduction = tile.to(tl.int64) * BLOCK_K + tl.arange(0, BLOCK_K)
        else:
            reduction = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        batch = reduction // volume
        od, oh, ow = _coordinates(reduction % volume, OH, OW)
        id_ = od[:, None] * SD - PD + kd[None, :] * DD
        ih = oh[:, None] * SH - PH + kh[None, :] * DH
        iw = ow[:, None] * SW - PW + kw[None, :] * DW
        source = batch[:, None] * (ID * IH * IW) + (id_ * IH + ih) * IW + iw
        # A is the original output gradient matrix. Loading its transpose
        # here avoids storing a separate transposed FP8 gradient operand.
        av = _load_operand(a, reduction[None, :], group * CO + rows[:, None],
                           (rows[:, None] < CO) & (reduction[None, :] < BATCH * volume),
                           A_RS, A_CS)
        # Reconstructed structural right-padding must remain literal zero.
        # In particular, affine decoding need not map its zero code to zero.
        valid = ((reduction[:, None] < BATCH * volume) & (cols[None, :] < CI * kernel_volume)
                 & (id_ >= 0) & (id_ < VALID_I0) & (ih >= 0) & (ih < VALID_I1)
                 & (iw >= 0) & (iw < VALID_I2))
        bv = _load_operand(b, source, group * CI + ci[None, :], valid, B_RS, B_CS)
        acc = tl.dot(av, bv, acc, out_dtype=tl.float32)

    if TENSORWISE:
        acc = (acc * tl.load(a_scale)) * tl.load(b_scale)
    tl.store(output + tl.program_id(1) * (GROUPS * CO * CI * kernel_volume)
             + (group * CO + rows[:, None]) * (CI * kernel_volume) + cols[None, :], acc,
             mask=(rows[:, None] < CO) & (cols[None, :] < CI * kernel_volume))


@triton.jit
def _reduce_weight_gradient(partials, output, ELEMENTS: tl.constexpr,
                            SPLITS: tl.constexpr, SPLIT_BLOCK: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLIT_BLOCK)
    values = tl.load(partials + splits[:, None] * ELEMENTS + offsets[None, :],
                     mask=(splits[:, None] < SPLITS) & (offsets[None, :] < ELEMENTS), other=0.0)
    tl.store(output + offsets, tl.sum(values, axis=0), mask=offsets < ELEMENTS)


@torch.no_grad()
def input_gradient(qg, qw, geometry, *, out_dtype=torch.float32, num_stages="auto", gradient_output=None, bias=None):
    """Return ``[batch * input_volume, in_channels]`` input gradients."""
    validate(qg, qw, out_dtype)
    if tuple(qg.data.shape) != (geometry.batch_size * geometry.output_volume, geometry.out_channels):
        raise ValueError("Output gradient matrix does not match convolution geometry")
    if tuple(qw.data.shape) != (geometry.out_channels, geometry.in_channels_per_group * geometry.kernel_volume):
        raise ValueError("Weight matrix does not match convolution geometry")
    if bias is not None and (bias.device != qg.data.device or tuple(bias.shape) != (geometry.in_channels,)):
        raise ValueError("Bias must match the adjoint output channels and device")
    rows = geometry.batch_size * geometry.input_volume
    with torch.cuda.device(qg.data.device):
        fused = (gradient_output is not None and gradient_output.can_fuse(qg.data, bm=64, bn=32)
                 and (gradient_output.scale is not None or geometry.in_channels_per_group % gradient_output.block_shape[1] == 0))
        extra = {}
        if fused:
            out = gradient_output
            programs = triton.cdiv(rows, 64) * triton.cdiv(geometry.in_channels_per_group, 32) * geometry.groups
            out.allocate(rows, geometry.in_channels, qg.data.device, programs)
            br, bc = out.block_shape or (1, 1)
            extra = dict(gradient_scale=out.scale if out.scale is not None else out.inverse,
                         gradient_inverse=out.inverse,
                         gradient_observations=out.observations if out.observations is not None else out.inverse,
                         GRADIENT_FP8=True, GRAD_CURRENT=out.scale is None, GRAD_BR=br, GRAD_BC=bc,
                         GRAD_MAXIMUM=torch.finfo(out.dtype).max, GRAD_OBSERVE=out.observations is not None)
        output = torch.empty((rows, geometry.in_channels), device=qg.data.device,
                              dtype=gradient_output.dtype if fused else out_dtype)
        _input_gradient_kernel[grid(rows, geometry.in_channels_per_group, geometry.groups)](
            qg.data, qw.data, qg.scale_inv, qw.scale_inv, output,
            **arguments(qg, qw, geometry, num_stages, extra_tensors=() if bias is None else (bias,)),
            bias=output if bias is None else bias, HAS_BIAS=bias is not None,
            BIAS_STRIDE=1 if bias is None else bias.stride(0),
            **extra,
        )
    return (gradient_output.wrap(output) if fused else gradient_output.convert(output, matrix=True)
            if gradient_output is not None else output)


@torch.no_grad()
def weight_gradient(qx, qg, geometry, *, out_dtype=torch.float32, num_stages="auto"):
    """Return ``[out_channels, in_channels_per_group * kernel_volume]`` dW."""
    validate(qx, qg, out_dtype)
    if tuple(qx.data.shape) != (geometry.batch_size * geometry.input_volume, geometry.in_channels):
        raise ValueError("Activation matrix does not match convolution geometry")
    if tuple(qg.data.shape) != (geometry.batch_size * geometry.output_volume, geometry.out_channels):
        raise ValueError("Output gradient matrix does not match convolution geometry")
    cols = geometry.in_channels_per_group * geometry.kernel_volume
    with torch.cuda.device(qg.data.device):
        output = torch.empty((geometry.out_channels, cols), device=qg.data.device, dtype=out_dtype)
        kernel_arguments = arguments(qg, qx, geometry, num_stages)
        tensorwise = kernel_arguments["TENSORWISE"]
        tile_cols = cols if tensorwise else geometry.in_channels_per_group
        split_groups = geometry.groups if tensorwise else geometry.groups * geometry.kernel_volume
        splits = weight_splits(geometry.out_channels_per_group, tile_cols,
                               geometry.batch_size * geometry.output_volume, split_groups,
                               block_m=kernel_arguments["BLOCK_M"], block_n=kernel_arguments["BLOCK_N"])
        partials = (output if splits == 1 else torch.empty(
            (splits, geometry.out_channels, cols), device=qg.data.device, dtype=torch.float32))
        base_grid = grid(geometry.out_channels_per_group, tile_cols, split_groups)
        if tensorwise:
            _weight_gradient_kernel[lambda meta: (base_grid(meta)[0], splits)](
                qg.data, qx.data, qg.scale_inv, qx.scale_inv, partials, **kernel_arguments, SPLIT_K=splits)
        else:
            _blockwise_weight_gradient_kernel[lambda meta: (base_grid(meta)[0], splits)](
                qg.data, qx.data, qg.scale_inv, qx.scale_inv, partials,
                **kernel_arguments, SPLIT_K=splits, STRIPE_W=128)
        if splits > 1:
            _reduce_weight_gradient[(triton.cdiv(output.numel(), 256),)](
                partials, output, ELEMENTS=output.numel(), SPLITS=splits, BLOCK=256,
                SPLIT_BLOCK=triton.next_power_of_2(splits),
                num_warps=4, num_stages=1,
            )
    return output
