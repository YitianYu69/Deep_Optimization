"""Opt-in depthwise stencil: FP8 storage and explicit FP32 accumulation.

No padded tensor-core GEMM is used for a one-channel reduction domain. This
candidate is deliberately separate from the strict FP8-dot executor.
"""

import torch
import triton
import triton.language as tl

from .common import _coordinates, _load_operand, _load_scale, validate
from .launch import geometry_arguments, operand_arguments, needs_int64, depthwise_eligible, depthwise_splits
from ..._shared.convolution import select_convolution_launch, convolution_launch_key


@triton.jit
def _depthwise_stencil(a, b, a_scale, b_scale, output, bias,
    BATCH: tl.constexpr, ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    VALID_I0: tl.constexpr, VALID_I1: tl.constexpr, VALID_I2: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    DD: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr, CHANNELS: tl.constexpr,
    A_RS: tl.constexpr, A_CS: tl.constexpr, A_BR: tl.constexpr, A_BC: tl.constexpr,
    A_SRS: tl.constexpr, A_SCS: tl.constexpr,
    B_RS: tl.constexpr, B_CS: tl.constexpr, B_BR: tl.constexpr, B_BC: tl.constexpr,
    B_SRS: tl.constexpr, B_SCS: tl.constexpr,
    INDEX64: tl.constexpr, TENSORWISE: tl.constexpr, TRANSPOSE: tl.constexpr,
    HAS_BIAS: tl.constexpr, BIAS_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if INDEX64:
        offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    volume: tl.constexpr = ID * IH * IW if TRANSPOSE else OD * OH * OW
    row, channel = offset // CHANNELS, offset % CHANNELS
    valid = row < BATCH * volume
    if TRANSPOSE:
        d, h, w = _coordinates(row % volume, IH, IW)
    else:
        d, h, w = _coordinates(row % volume, OH, OW)
    batch = row // volume
    acc = tl.zeros((BLOCK,), tl.float32)
    for tap in range(KD * KH * KW):
        kd, kh, kw = _coordinates(tap, KH, KW)
        if TRANSPOSE:
            nd, nh, nw = d + PD - kd * DD, h + PH - kh * DH, w + PW - kw * DW
            sd, sh, sw = nd // SD, nh // SH, nw // SW
            mask = (valid & (nd % SD == 0) & (nh % SH == 0) & (nw % SW == 0)
                    & (sd >= 0) & (sd < OD) & (sh >= 0) & (sh < OH) & (sw >= 0) & (sw < OW))
            source = batch * (OD * OH * OW) + (sd * OH + sh) * OW + sw
        else:
            sd, sh, sw = d * SD - PD + kd * DD, h * SH - PH + kh * DH, w * SW - PW + kw * DW
            mask = (valid & (sd >= 0) & (sd < VALID_I0) & (sh >= 0) & (sh < VALID_I1)
                    & (sw >= 0) & (sw < VALID_I2))
            source = batch * (ID * IH * IW) + (sd * IH + sh) * IW + sw
        av = _load_operand(a, source, channel, mask, A_RS, A_CS).to(tl.float32)
        bv = _load_operand(b, channel, tap, valid, B_RS, B_CS).to(tl.float32)
        if TENSORWISE:
            acc += av * bv
        else:
            sa = _load_scale(a_scale, source, channel, mask, A_BR, A_BC, A_SRS, A_SCS)
            sb = _load_scale(b_scale, channel, tap, valid, B_BR, B_BC, B_SRS, B_SCS)
            acc += (av * sa) * (bv * sb)
    if TENSORWISE:
        acc = (acc * tl.load(a_scale)) * tl.load(b_scale)
    if HAS_BIAS:
        acc += tl.load(bias + channel * BIAS_STRIDE, valid, other=0.0).to(tl.float32)
    tl.store(output + offset, acc, valid)


@triton.jit
def _depthwise_weight_partial(a, b, a_scale, b_scale, partial,
    BATCH: tl.constexpr, ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    VALID_I0: tl.constexpr, VALID_I1: tl.constexpr, VALID_I2: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    DD: tl.constexpr, DH: tl.constexpr, DW: tl.constexpr, CHANNELS: tl.constexpr,
    A_RS: tl.constexpr, A_CS: tl.constexpr, A_BR: tl.constexpr, A_BC: tl.constexpr,
    A_SRS: tl.constexpr, A_SCS: tl.constexpr,
    B_RS: tl.constexpr, B_CS: tl.constexpr, B_BR: tl.constexpr, B_BC: tl.constexpr,
    B_SRS: tl.constexpr, B_SCS: tl.constexpr,
    INDEX64: tl.constexpr, TENSORWISE: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    split = tl.program_id(2)
    tap = tl.program_id(1)
    channel_program = tl.program_id(0)
    if INDEX64:
        split, tap, channel_program = split.to(tl.int64), tap.to(tl.int64), channel_program.to(tl.int64)
    channel = channel_program * BLOCK_C + tl.arange(0, BLOCK_C)
    reduction: tl.constexpr = BATCH * OD * OH * OW
    chunk: tl.constexpr = tl.cdiv(reduction, SPLITS)
    kd, kh, kw = _coordinates(tap, KH, KW)
    acc = tl.zeros((BLOCK_C,), tl.float32)
    for tile in range(tl.cdiv(chunk, BLOCK_R)):
        if INDEX64:
            tile = tile.to(tl.int64)
        row = split * chunk + tile * BLOCK_R + tl.arange(0, BLOCK_R)
        valid = (row < reduction) & (row < (split + 1) * chunk)
        od, oh, ow = _coordinates(row % (OD * OH * OW), OH, OW)
        sd, sh, sw = od * SD - PD + kd * DD, oh * SH - PH + kh * DH, ow * SW - PW + kw * DW
        source = row // (OD * OH * OW) * (ID * IH * IW) + (sd * IH + sh) * IW + sw
        source_valid = (valid & (sd >= 0) & (sd < VALID_I0) & (sh >= 0) & (sh < VALID_I1)
                        & (sw >= 0) & (sw < VALID_I2))
        mask_a = source_valid[:, None] & (channel[None, :] < CHANNELS)
        mask_b = valid[:, None] & (channel[None, :] < CHANNELS)
        av = _load_operand(a, source[:, None], channel[None, :], mask_a, A_RS, A_CS).to(tl.float32)
        bv = _load_operand(b, row[:, None], channel[None, :], mask_b, B_RS, B_CS).to(tl.float32)
        if not TENSORWISE:
            av *= _load_scale(a_scale, source[:, None], channel[None, :], mask_a, A_BR, A_BC, A_SRS, A_SCS)
            bv *= _load_scale(b_scale, row[:, None], channel[None, :], mask_b, B_BR, B_BC, B_SRS, B_SCS)
        acc += tl.sum(av * bv, 0)
    if TENSORWISE:
        acc = (acc * tl.load(a_scale)) * tl.load(b_scale)
    tl.store(partial + (split * CHANNELS + channel) * (KD * KH * KW) + tap, acc, channel < CHANNELS)


@triton.jit
def _depthwise_weight_finish(partial, output, SIZE: tl.constexpr, SPLITS: tl.constexpr,
                              PAD_SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    split = tl.arange(0, PAD_SPLITS)
    values = tl.load(partial + split[:, None] * SIZE + index[None, :],
                     (split[:, None] < SPLITS) & (index[None, :] < SIZE), other=0.0)
    tl.store(output + index, tl.sum(values, 0), index < SIZE)


def _arguments(a, b, geometry, *, extra_tensors=()):
    if not depthwise_eligible(geometry):
        raise ValueError("Depthwise candidate requires one input/output channel per group")
    args = geometry_arguments(geometry)
    args.pop('CI'); args.pop('CO')
    args['CHANNELS'] = args.pop('GROUPS')
    return {**args, **operand_arguments('A', a), **operand_arguments('B', b),
            'INDEX64': needs_int64(a, b, geometry, extra_tensors=extra_tensors),
            'TENSORWISE': a.scale_inv.numel() == b.scale_inv.numel() == 1}


@torch.no_grad()
def forward(qx, qw, geometry, *, bias=None, out_dtype=torch.float32, num_stages='auto', autotune=True):
    return _stencil(qx,qw,geometry,bias=bias,out_dtype=out_dtype,transpose=False,autotune=autotune)


@torch.no_grad()
def input_gradient(qg, qw, geometry, *, out_dtype=torch.float32, num_stages='auto', autotune=True):
    return _stencil(qg,qw,geometry,out_dtype=out_dtype,transpose=True,autotune=autotune)


def _stencil(a,b,geometry,*,out_dtype,transpose,autotune,bias=None):
    validate(a,b,out_dtype)
    with torch.cuda.device(a.data.device):
        shape=geometry.input_matrix_shape if transpose else geometry.grad_matrix_shape
        output=torch.empty(shape,dtype=out_dtype,device=a.data.device)
        args=_arguments(a,b,geometry,extra_tensors=(bias,) if bias is not None else ())
        candidates=[dict(BLOCK=256,num_warps=4,num_stages=1)]
        if output.numel()>=1024:
            candidates += [dict(BLOCK=512,num_warps=4,num_stages=1),dict(BLOCK=1024,num_warps=4,num_stages=1)]
        def launch(config):
            return _depthwise_stencil[(triton.cdiv(output.numel(),config['BLOCK']),)](
                a.data,b.data,a.scale_inv,b.scale_inv,output,bias if bias is not None else output,
                **args,TRANSPOSE=transpose,HAS_BIAS=bias is not None,BIAS_STRIDE=bias.stride(0) if bias is not None else 1,
                **config,enable_fp_fusion=False)
        role='depthwise_dx' if transpose else 'depthwise_forward'
        selected=select_convolution_launch(convolution_launch_key(role,a,b,geometry,args,out_dtype,bias=bias),
                                            candidates,launch,autotune=autotune)
        launch(selected)
    return output


@torch.no_grad()
def weight_gradient(qx, qg, geometry, *, out_dtype=torch.float32, num_stages='auto', autotune=True):
    validate(qx, qg, out_dtype)
    splits = depthwise_splits(geometry.batch_size * geometry.output_volume, geometry.in_channels, geometry.kernel_volume)
    with torch.cuda.device(qx.data.device):
        partial = torch.empty((splits, geometry.in_channels, geometry.kernel_volume), dtype=torch.float32, device=qx.data.device)
        output = torch.empty(geometry.weight_matrix_shape, dtype=out_dtype, device=qx.data.device)
        args=_arguments(qx,qg,geometry)
        candidates=[dict(SPLITS=splits,BLOCK_R=128,BLOCK_C=32,num_warps=4,num_stages=1)]
        if geometry.batch_size*geometry.output_volume>=1024:
            candidates += [dict(SPLITS=splits,BLOCK_R=r,BLOCK_C=c,num_warps=4,num_stages=1)
                           for r,c in ((256,32),(128,16),(256,16))]
        def launch(config):
            _depthwise_weight_partial[(triton.cdiv(geometry.in_channels,config['BLOCK_C']),geometry.kernel_volume,splits)](
                qx.data,qg.data,qx.scale_inv,qg.scale_inv,partial,**args,**config,enable_fp_fusion=False)
            _depthwise_weight_finish[(triton.cdiv(output.numel(),64),)](partial,output,output.numel(),splits,
                triton.next_power_of_2(splits),64,num_warps=4,enable_fp_fusion=False)
        selected=select_convolution_launch(convolution_launch_key('depthwise_dw',qx,qg,geometry,args,out_dtype),
                                            candidates,launch,autotune=autotune)
        launch(selected)
    return output
