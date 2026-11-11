"""FP8 forward implementation, separate from the package's lazy callable."""

import torch
import triton
import triton.language as tl

from .common import (_load_operand, _load_scale, _scale_chunk_length, _coordinates,
                     _tile_coordinates, validate, arguments, grid)
from ..gradient import _gradient_tile


@triton.jit
def _forward_kernel(
    a, b, a_scale, b_scale, output, bias,
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
    TENSORWISE: tl.constexpr, INDEX64: tl.constexpr, HAS_BIAS: tl.constexpr, BIAS_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    gradient_scale=None, gradient_inverse=None, gradient_observations=None,
    GRADIENT_FP8: tl.constexpr = False, GRAD_CURRENT: tl.constexpr = False,
    GRAD_BR: tl.constexpr = 1, GRAD_BC: tl.constexpr = 1,
    GRAD_MAXIMUM: tl.constexpr = 448.0, GRAD_OBSERVE: tl.constexpr = False,
):
    volume: tl.constexpr = OD * OH * OW
    kernel_volume: tl.constexpr = KD * KH * KW
    group, pm, pn = _tile_coordinates(BATCH * volume, CO, BLOCK_M, BLOCK_N)
    if INDEX64:
        group, pm, pn = group.to(tl.int64), pm.to(tl.int64), pn.to(tl.int64)
    rows = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    od, oh, ow = _coordinates(rows % volume, OH, OW)
    batch = rows // volume
    channels_out = group * CO + cols
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    if TENSORWISE:
        for tile in range(tl.cdiv(CI * kernel_volume, BLOCK_K)):
            if INDEX64:
                reduction = tile.to(tl.int64) * BLOCK_K + tl.arange(0, BLOCK_K)
            else:
                reduction = tile * BLOCK_K + tl.arange(0, BLOCK_K)
            ci = reduction // kernel_volume
            kd, kh, kw = _coordinates(reduction % kernel_volume, KH, KW)
            id_ = od[:, None] * SD - PD + kd[None, :] * DD
            ih = oh[:, None] * SH - PH + kh[None, :] * DH
            iw = ow[:, None] * SW - PW + kw[None, :] * DW
            source = batch[:, None] * (ID * IH * IW) + (id_ * IH + ih) * IW + iw
            valid = ((rows[:, None] < BATCH * volume) & (reduction[None, :] < CI * kernel_volume)
                     & (id_ >= 0) & (id_ < VALID_I0) & (ih >= 0) & (ih < VALID_I1)
                     & (iw >= 0) & (iw < VALID_I2))
            av = _load_operand(a, source, group * CI + ci[None, :], valid, A_RS, A_CS)
            bv = _load_operand(b, channels_out[None, :], reduction[:, None],
                               (reduction[:, None] < CI * kernel_volume) & (cols[None, :] < CO), B_RS, B_CS)
            acc = tl.dot(av, bv, acc, out_dtype=tl.float32)
        acc = (acc * tl.load(a_scale)) * tl.load(b_scale)
    else:
        # Fix the spatial tap, then reduce channels only until either source
        # scale changes. Each FP8 dot has a row scale and a column scale;
        # applying them to its FP32 partial preserves the original FP8 values.
        for kernel_index in range(kernel_volume):
            if INDEX64:
                kernel_pos = kernel_index.to(tl.int64)
            else:
                kernel_pos = kernel_index
            kd, kh, kw = _coordinates(kernel_pos, KH, KW)
            id_ = od * SD - PD + kd * DD
            ih = oh * SH - PH + kh * DH
            iw = ow * SW - PW + kw * DW
            source = batch * (ID * IH * IW) + (id_ * IH + ih) * IW + iw
            valid_rows = ((rows < BATCH * volume) & (id_ >= 0) & (id_ < VALID_I0)
                          & (ih >= 0) & (ih < VALID_I1) & (iw >= 0) & (iw < VALID_I2))
            channel = group * 0
            while channel < CI:
                weight_col = channel * kernel_volume + kernel_pos
                length = _scale_chunk_length(group * CI + channel, 1, A_BC,
                                             weight_col, kernel_volume, B_BC,
                                             tl.minimum(BLOCK_K, CI - channel))
                offsets = tl.arange(0, BLOCK_K)
                av = _load_operand(a, source[:, None], group * CI + channel + offsets[None, :],
                                   valid_rows[:, None] & (offsets[None, :] < length), A_RS, A_CS)
                bv = _load_operand(b, channels_out[None, :],
                                   (channel + offsets[:, None]) * kernel_volume + kernel_pos,
                                   (offsets[:, None] < length) & (cols[None, :] < CO), B_RS, B_CS)
                partial = tl.dot(av, bv, out_dtype=tl.float32)
                sa = _load_scale(a_scale, source, group * CI + channel, valid_rows,
                                 A_BR, A_BC, A_SRS, A_SCS)
                sb = _load_scale(b_scale, channels_out, weight_col, cols < CO,
                                 B_BR, B_BC, B_SRS, B_SCS)
                acc += (partial * sa[:, None]) * sb[None, :]
                channel += length
    if HAS_BIAS:
        acc += tl.load(bias + channels_out * BIAS_STRIDE, mask=cols < CO, other=0.0).to(tl.float32)[None, :]
    valid = (rows[:, None] < BATCH * volume) & (cols[None, :] < CO)
    if GRADIENT_FP8:
        acc = _gradient_tile(acc, rows, channels_out, valid, gradient_scale, gradient_inverse,
                             gradient_observations, BATCH * volume, CO * GROUPS, BLOCK_M, BLOCK_N,
                             GRAD_BR, GRAD_BC, GRAD_MAXIMUM, GRAD_CURRENT, GRAD_OBSERVE)
        tl.store(output + rows[:, None] * (CO * GROUPS) + channels_out[None, :],
                 acc.to(output.dtype.element_ty, fp_downcast_rounding="rtne"), valid)
    else:
        tl.store(output + rows[:, None] * (CO * GROUPS) + channels_out[None, :], acc, valid)


@torch.no_grad()
def forward(qx, qw, geometry, *, bias=None, out_dtype=torch.float32, num_stages="auto", gradient_output=None):
    """Return canonical ``[batch * output_volume, out_channels]`` output.

    Kernel reductions use torch's flattened weight order: input channel first,
    then spatial kernel coordinates. No im2col tensor is allocated.
    """
    validate(qx, qw, out_dtype)
    if tuple(qx.data.shape) != (geometry.batch_size * geometry.input_volume, geometry.in_channels):
        raise ValueError("Activation matrix does not match convolution geometry")
    if tuple(qw.data.shape) != (geometry.out_channels, geometry.in_channels_per_group * geometry.kernel_volume):
        raise ValueError("Weight matrix does not match convolution geometry")
    if bias is not None:
        if bias.device != qx.data.device or tuple(bias.shape) != (geometry.out_channels,):
            raise ValueError("Bias must have shape [out_channels] on the convolution device")
        if bias.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise TypeError("Bias must have an FP32, BF16, or FP16 dtype")
    rows = geometry.batch_size * geometry.output_volume
    with torch.cuda.device(qx.data.device):
        fused = (gradient_output is not None and gradient_output.can_fuse(qx.data, bm=64, bn=32)
                 and (gradient_output.scale is not None or geometry.out_channels_per_group % gradient_output.block_shape[1] == 0))
        extra = {}
        if fused:
            out = gradient_output
            programs = triton.cdiv(rows, 64) * triton.cdiv(geometry.out_channels_per_group, 32) * geometry.groups
            out.allocate(rows, geometry.out_channels, qx.data.device, programs)
            br, bc = out.block_shape or (1, 1)
            extra = dict(gradient_scale=out.scale if out.scale is not None else out.inverse,
                         gradient_inverse=out.inverse,
                         gradient_observations=out.observations if out.observations is not None else out.inverse,
                         GRADIENT_FP8=True, GRAD_CURRENT=out.scale is None, GRAD_BR=br, GRAD_BC=bc,
                         GRAD_MAXIMUM=torch.finfo(out.dtype).max, GRAD_OBSERVE=out.observations is not None)
        output = torch.empty((rows, geometry.out_channels), device=qx.data.device,
                             dtype=gradient_output.dtype if fused else out_dtype)
        _forward_kernel[grid(rows, geometry.out_channels_per_group, geometry.groups)](
            qx.data, qw.data, qx.scale_inv, qw.scale_inv, output,
            output if bias is None else bias,
            **arguments(qx, qw, geometry, num_stages, extra_tensors=() if bias is None else (bias,)),
            HAS_BIAS=bias is not None, BIAS_STRIDE=1 if bias is None else bias.stride(0), **extra,
        )
    return (gradient_output.wrap(output) if fused else gradient_output.convert(output, matrix=True)
            if gradient_output is not None else output)
