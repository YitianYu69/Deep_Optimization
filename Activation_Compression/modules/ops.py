import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.library import Library, register_fake
from torch.amp import custom_fwd, custom_bwd, autocast
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.grad import (
    conv1d_input, conv1d_weight,
    conv2d_input, conv2d_weight,
    conv3d_input, conv3d_weight,
)
import torch._dynamo as dynamo
from torch._C import Tag
from torch.library import custom_op, triton_op, wrap_triton

import triton
import triton.language as tl

from functools import reduce
import operator

from modules.module_utils import unified_quantize, unified_dequantize
import ACT6.cpp_extension as cpp_extension
import act_triton_kernel

from typing import Tuple, List


# act_lib = Library("act_cpp", "DEF")

# act_lib.define("quant_pack_bits(Tensor input, Tensor(a!) out, Tensor(b!) scaler, Tensor(e!) ema_min, float beta, int bits) -> ()", alias_analysis="FROM_SCHEMA")
# act_lib.define("dequant_unpack_bits(Tensor input, Tensor(a!) unpack, Tensor scaler, Tensor ema_min, int N, int G, int batchSize, int bits) -> ()", alias_analysis="FROM_SCHEMA")

# def quant_pack_bits_impl(input, out, scaler, ema_min, beta, bits):
#     if (bits == 1):
#         cpp_extension.pack1(input, out, scaler, ema_min, beta)
#     elif (bits == 2):
#         cpp_extension.pack2(input, out, scaler, ema_min, beta)
#     elif (bits == 3):
#         cpp_extension.pack3(input, out, scaler, ema_min, beta)
#     elif (bits == 4):
#         cpp_extension.pack4(input, out, scaler, ema_min, beta)
#     elif (bits == 8):
#         cpp_extension.pack8(input, out, scaler, ema_min, beta)
#     else:
#         raise ValueError(f'Bit {bits} is not supported!')

# def dequant_unpack_bits_impl(input, unpack, scaler, ema_min, N, G, batch_size, bits):
#     if (bits == 1):
#         cpp_extension.unpack1(input, unpack, scaler, ema_min, N, G, batch_size)
#     elif (bits == 2):
#         cpp_extension.unpack2(input, unpack, scaler, ema_min, N, G, batch_size)
#     elif (bits == 3):
#         cpp_extension.unpack3(input, unpack, scaler, ema_min, N, G, batch_size)
#     elif (bits == 4):
#         cpp_extension.unpack4(input, unpack, scaler, ema_min, N, G, batch_size)
#     elif (bits == 8):
#         cpp_extension.unpack8(input, unpack, scaler, ema_min, N, G, batch_size)
#     else:
#         raise ValueError(f'Bit {bits} is not supported!')
    

# act_lib.impl("quant_pack_bits", quant_pack_bits_impl, "CUDA")
# act_lib.impl("dequant_unpack_bits", dequant_unpack_bits_impl, "CUDA")

# torch.compiler.allow_in_graph(quant_pack_bits_impl)
# torch.compiler.allow_in_graph(dequant_unpack_bits_impl)

# @register_fake("act_cpp::quant_pack_bits")
# def quant_pack_bits_fake(input, out, scaler, ema_max, ema_min, beta, bits):
#     return None

# @register_fake("act_cpp::dequant_unpack_bits")
# def dequant_unpack_bits_fake(input, unpack, scaler, ema_min, N, G, batch_size, bits):
#     return None


# def quant_pack_bits(input, beta, bits):
#     N, G = input.shape[:2]
#     total_elt = N * G * 8 * bits

#     out = torch.ones((total_elt, ), dtype=torch.int32, device='cuda')
#     scaler = torch.zeros((N, G), dtype=input.dtype, device='cuda')
#     imin = torch.zeros((N, G), dtype=input.dtype, device='cuda')

#     if (bits == 1):
#         cpp_extension.pack1(input, out, scaler, imin, beta)
#     elif (bits == 2):
#         cpp_extension.pack2(input, out, scaler, imin, beta)
#     elif (bits == 3):
#         cpp_extension.pack3(input, out, scaler, imin, beta)
#     elif (bits == 4):
#         cpp_extension.pack4(input, out, scaler, imin, beta)
#     elif (bits == 8):
#         cpp_extension.pack8(input, out, scaler, imin, beta)
#     else:
#         raise ValueError(f'Bit {bits} is not supported!')
    
#     return out, scaler, imin

# def dequant_unpack_bits(input, scaler, imin, N, G, batch_size, bits):
#     unpack = torch.zeros((N, G, 256), dtype=scaler.dtype, device='cuda')

#     if (bits == 1):
#         cpp_extension.unpack1(input, unpack, scaler, imin, N, G, batch_size)
#     elif (bits == 2):
#         cpp_extension.unpack2(input, unpack, scaler, imin, N, G, batch_size)
#     elif (bits == 3):
#         cpp_extension.unpack3(input, unpack, scaler, imin, N, G, batch_size)
#     elif (bits == 4):
#         cpp_extension.unpack4(input, unpack, scaler, imin, N, G, batch_size)
#     elif (bits == 8):
#         cpp_extension.unpack8(input, unpack, scaler, imin, N, G, batch_size)
#     else:
#         raise ValueError(f'Bit {bits} is not supported!')

#     return unpack


def h4(x):
    a,b,c,d = x[...,0],x[...,1],x[...,2],x[...,3]
    return torch.stack([
        a+b+c+d,
        a+b-c-d,
        a-b+c-d,
        a-b-c+d
    ], dim=-1) * 0.5


def h4_inv(y):
    y0,y1,y2,y3 = y[...,0],y[...,1],y[...,2],y[...,3]
    return torch.stack([
        y0+y1+y2+y3,
        y0+y1-y2-y3,
        y0-y1+y2-y3,
        y0-y1-y2+y3
    ], dim=-1) * 0.5


def hadamard_auto(x):
    shp = x.shape
    B = x.shape[0]

    flat = x.reshape(B, -1).contiguous()
    F = flat.shape[1]

    pad = (-F) % 4

    if pad:
        flat = torch.cat(
            [flat, torch.zeros(B, pad, device=x.device, dtype=x.dtype)],
            dim=1
        )

    Fp = flat.shape[1]

    g = flat.reshape(B, Fp // 4, 4)
    y = h4(g)

    # drop padding before reshaping back
    y = y.reshape(B, Fp)
    if pad:
        y = y[:, :-pad]

    return y.reshape(shp), pad




def hadamard_auto_inv(y, pad):
    shp = y.shape
    B = y.shape[0]

    flat = y.reshape(B, -1).contiguous()
    F = flat.shape[1]

    if pad:
        flat = torch.cat(
            [flat, torch.zeros(B, pad, device=y.device, dtype=y.dtype)],
            dim=1
        )

    Fp = flat.shape[1]

    g = flat.reshape(B, Fp // 4, 4)
    x = h4_inv(g)

    x = x.reshape(B, Fp)
    if pad:
        x = x[:, :-pad]

    return x.reshape(shp)



# opt_stream = torch.cuda.Stream()
# lr = torch.tensor(1e-3, dtype=torch.float32, device='cuda')

class _DOConvnd(Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def run_forward(ctx, forward_op, input, weight, learnable_scale, stride=1, padding=0, dilation=1, groups=1,
                    clamp_alpha=3.0, target_name=None, meta=None, ema_grad_meta=None):
        weight_master = weight

        target_dtype = torch.float32
        if torch.is_autocast_enabled("cuda"):
            target_dtype = torch.get_autocast_dtype("cuda")
            input = input.to(target_dtype)
            weight_compute = weight.to(target_dtype)
        else:
            weight_compute = weight_master

        # weight_shape = weight_compute.shape
        # input_shape = input.shape
        # weight_flat = weight_compute.reshape(weight_shape[1], -1) # C, N
        # input_flat = input.permute(1, 0, 2, 3).reshape(input_shape[1], -1) # C x M
        # p = meta[f"{target_name}"]['low_rank_activations'] # C x L

        # low_rank_weight = p.T @ weight_flat # L x N
        # low_rank_input = p.T @ input_flat # L x M
        # _input = p @ low_rank_input # C x M
        # L = low_rank_input.shape[0]


        # beta = 0.99
        # R = input_flat - _input
        # res = (R * R).sum(dim=0)
        # idx = torch.topk(res, k=L).indices
        # Z = R[:, idx]

        # p = (1 - beta) * p + beta * Z
        # p, _ = torch.linalg.qr(p, mode='reduced')
        # meta[f"{target_name}"]['low_rank_activations'] = p


        # low_rank_input = low_rank_input.view(L, input_shape[0], input_shape[2], input_shape[3]).permute(1, 0, 2, 3) # N x L x H x W
        # low_rank_weight = low_rank_weight.view(L, weight_shape[0], weight_shape[2], weight_shape[3]).permute(1, 0, 2, 3) # N x L x kH x kW
        
        # print(f'Layer: {target_name}, fwd ori input shape: {input.shape}')
        # print(f'Layer: {target_name}, fwd act_padding: {meta[f"{target_name}"]["act_padding"]}, group_size: {meta[f"{target_name}"]["group_size"]}')

        input_l = None
        if meta[f'{target_name}']['DIVISION'] is not None and not meta[f'{target_name}']['pack_only']:
            HEIGHT, WIDTH = input.shape[-2], input.shape[-1]

            k = min(HEIGHT, WIDTH, meta[f'{target_name}']['DIVISION']['pool_kernel_size'])
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                input_l = F.avg_pool2d(input, k, stride=k, padding=0)
                # input_l = F.avg_pool2d(low_rank_input, k, stride=k, padding=0)
                # input_max = F.max_pool2d(input, k, stride=k, padding=0)
                # input_h = input - F.interpolate(input_l, size=(HEIGHT, WIDTH), scale_factor=None, mode="bilinear", align_corners=False)
                # input_stored = input_h
        
        # input_stored, had_pad = hadamard_auto(input)
        input_stored = input
        had_pad = None

        # print(f"Layer:  {target_name}")
        if meta[f'{target_name}']['DIVISION'] is not None:
            q_inputs_tensor, q_inputs_meta, input_l = unified_quantize((input_stored).to(torch.bfloat16), (input_l).to(torch.float8_e4m3fn), target_name, meta, clamp=True, clamp_alpha=clamp_alpha)
        else:
            q_inputs_tensor, q_inputs_meta, input_l = unified_quantize((input_stored).to(torch.bfloat16), input_l, target_name, meta, clamp=True, clamp_alpha=clamp_alpha)
            
        ctx.meta = stride, padding, dilation, groups, ema_grad_meta, clamp_alpha
        ctx.save_for_backward((weight_master), q_inputs_tensor[0], q_inputs_tensor[1], q_inputs_tensor[2], input_l, learnable_scale)
        ctx.q_inputs_meta = (q_inputs_meta)

        return forward_op(input, weight_compute, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
        # return forward_op(low_rank_input, low_rank_weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)

    @staticmethod
    @custom_bwd(device_type='cuda')
    def run_backward(ctx, dy, dim, pad_fn):
        q_inputs_meta = ctx.q_inputs_meta
        stride, padding, dilation, groups, ema_grad_meta, clamp_alpha = ctx.meta
        (weight_master, q_inputs_tensor_out, q_inputs_tensor_scaler, q_inputs_tensor_ema_min, input_l, learnable_scale) = ctx.saved_tensors
        q_inputs_tensor = (q_inputs_tensor_out, q_inputs_tensor_scaler, q_inputs_tensor_ema_min)

        if torch.is_autocast_enabled("cuda"):
            target_dtype = torch.get_autocast_dtype("cuda")
            dy = dy.to(target_dtype)
            weight_compute = weight_master.to(torch.bfloat16)
        else:
            weight_compute = weight_master

        stride, padding, dilation = pad_fn(stride), pad_fn(padding), pad_fn(dilation)


        input = unified_dequantize(q_inputs_tensor, q_inputs_meta, input_l) # B, C, H, W
        # low_rank_input = unified_dequantize(q_inputs_tensor, q_inputs_meta, input_l) # B, L, H, W
        # L = low_rank_input.shape[1]
        # low_rank_input = low_rank_input.permute(1, 0, 2, 3).reshape(L, -1) # L x M
        # input = p @ low_rank_input # C x M
        # input = input.view(input_shape[1], input_shape[0], input_shape[2], input_shape[3]).permute(1, 0, 2, 3) # N x C x H x W
        
        # input_shape = input.shape
        # input_flat = input.permute(1, 0, 2, 3).reshape(input_shape[1], -1) # C x M
        # p = meta[f"{target_name}"]['low_rank_activations'] # C x L
        # low_rank_input = p.T @ input_flat # L x M
        # L = low_rank_input.shape[0]
        # input = p @ low_rank_input # C x M

        # input = input.view(input_shape[1], input_shape[0], input_shape[2], input_shape[3]).permute(1, 0, 2, 3) # N x C x H x W


        del q_inputs_tensor, q_inputs_meta, input_l

        dx = dx2 = dw = None
        if dim == 1:
            if ctx.needs_input_grad[0]:
                dx = conv1d_input(
                    input.shape, weight_compute, dy,
                    stride=stride, padding=padding,
                    dilation=dilation, groups=groups
                )
            if ctx.needs_input_grad[1]:
                dw = conv1d_weight(
                    input, weight_compute.shape, dy,
                    stride=stride, padding=padding,
                    dilation=dilation, groups=groups
                )
        elif dim == 2:
            if ctx.needs_input_grad[0]:
                dx = conv2d_input(
                    input.shape, weight_compute, dy,
                    stride=stride, padding=padding,
                    dilation=dilation, groups=groups
                )

            if ctx.needs_input_grad[1]:
                dw = conv2d_weight(
                    input, weight_compute.shape, dy,
                    stride=stride, padding=padding,
                    dilation=dilation, groups=groups
                )
        else:
            if ctx.needs_input_grad[0]:
                dx = conv3d_input(
                    input.shape, weight_compute, dy,
                    stride=stride, padding=padding,
                    dilation=dilation, groups=groups
                )

            if ctx.needs_input_grad[1]:
                dw = conv3d_weight(
                    input, weight_compute.shape, dy,
                    stride=stride, padding=padding,
                    dilation=dilation, groups=groups
                )
        if dx is not None:
            dx = dx.to(torch.bfloat16)

        if dw is not None:
            dw = dw.to(torch.bfloat16)

        dw_scale = None
            
            # LRAS Scale
            # ema_grad_meta['ema_smooth'].mul_(ema_grad_meta['momentum']).add_(dw.float())
            # w_norm = torch.linalg.vector_norm(weight_master)
            # dw_norm = torch.linalg.vector_norm(dw.float())
            # trust_ratio = 0.1 * (w_norm / (dw_norm + 1e-5))
            # cond = (w_norm > 0) & (dw_norm > 0) & torch.isfinite(w_norm) & torch.isfinite(dw_norm)
            # trust_ratio = torch.where(cond, trust_ratio, torch.ones_like(trust_ratio))
            # scaled_lr = ema_grad_meta['lr'] * trust_ratio
            # update = dw.float() + ema_grad_meta['ema_smooth'] * ema_grad_meta['momentum']

            # print("ema_norm", ema_grad_meta['ema_smooth'].norm().item())
            # if not torch.isfinite(dw).all():
            #     print("dw has NaN/Inf")
            # if not torch.isfinite(w_norm):
            #     print("w_norm NaN/Inf:", w_norm.item())
            # if not torch.isfinite(dw_norm):
            #     print("dw_norm NaN/Inf:", dw_norm.item())
            # if not torch.isfinite(trust_ratio):
            #     print("trust_ratio NaN/Inf:", trust_ratio.item())
            # print("trust_ratio", trust_ratio.item(), "dw_norm", dw_norm.item(), "w_norm", w_norm.item())

            # with torch.no_grad():
            #     weight_master.add_(update * -scaled_lr)

        # with torch.cuda.stream(opt_stream):
        #     if dw is not None:
        #         weight.add(-1e-3 * dw)           
        return dx, dw, dw_scale, None, None, None, None, None, None, None, None, None, None, None, None
    

class _DOConv1d(Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, input, weight, learnable_scale,
                stride=1, padding=0, dilation=1, groups=1,
                clamp_alpha=3.0, target_name=None, meta=None, ema_grad_meta=None):
        return _DOConvnd.run_forward(ctx, F.conv1d, input, weight, learnable_scale,
                                     stride=stride, padding=padding, dilation=dilation, groups=groups,
                                     clamp_alpha=clamp_alpha, target_name=target_name, meta=meta, ema_grad_meta=ema_grad_meta
                                     )
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dy):
        return _DOConvnd.run_backward(ctx, dy, 1, _single)
    
class _DOConv2d(Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, input, weight, learnable_scale,
                stride=1, padding=0, dilation=1, groups=1,
                clamp_alpha=3.0, target_name=None, meta=None, ema_grad_meta=None):
        return _DOConvnd.run_forward(ctx, F.conv2d,input, weight, learnable_scale,
                                     stride=stride, padding=padding, dilation=dilation, groups=groups,
                                     clamp_alpha=clamp_alpha, target_name=target_name, meta=meta, ema_grad_meta=ema_grad_meta
                                     )
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dy):
        return _DOConvnd.run_backward(ctx, dy, 2, _pair)
    
class _DOConv3d(Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, input, weight, learnable_scale,
                stride=1, padding=0, dilation=1, groups=1,
                clamp_alpha=3.0, target_name=None, meta=None, ema_grad_meta=None):
        return _DOConvnd.run_forward(ctx, F.conv3d, input, weight, learnable_scale,
                                     stride=stride, padding=padding, dilation=dilation, groups=groups,
                                     clamp_alpha=clamp_alpha, target_name=target_name, meta=meta, ema_grad_meta=ema_grad_meta
                                     )
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dy):
        return _DOConvnd.run_backward(ctx, dy, 3, _triple)


class _DOLinear(Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, input, weight, clamp_alpha, target_name, meta, ema_grad_meta):
        weight_master = weight

        target_dtype = torch.float32
        if torch.is_autocast_enabled("cuda"):
            target_dtype = torch.get_autocast_dtype("cuda")
            input = input.to(target_dtype) 
            weight_compute = weight.to(target_dtype)
        else:
            weight_compute = weight_master

        # print(f'Layer: {target_name}, fwd ori input shape: {input.shape}')
        # print(f'Layer: {target_name}, fwd act_padding: {meta[f"{target_name}"]["act_padding"]}, group_size: {meta[f"{target_name}"]["group_size"]}')


        input_l = None
        if (meta[f'{target_name}']['DIVISION'] is not None and input.ndim != 1 and input.ndim != 2) and not meta[f'{target_name}']['pack_only']:
            channel = input.shape[2]

            k = min(channel, meta[f'{target_name}']['DIVISION']['pool_kernel_size'])
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                input_l = F.avg_pool1d(input, k, stride=k, padding=0)
                # input_max = F.max_pool1d(input, k, stride=k, padding=0)
                # input_h = input - F.interpolate(input_l, size=channel, scale_factor=None, mode="linear", align_corners=False)
                # input_stored = input_h


        # print(f"Layer:  {target_name}")
        input_stored = input
        # input_stored = input + 0.1 * (F.avg_pool1d(input, kernel_size=3, stride=1, padding=1) - input)
        if input_l is not None and input.ndim != 1 and input.ndim != 2:
            q_inputs_tensor, q_inputs_meta, input_l = unified_quantize(input_stored, (input_l), target_name, meta, clamp=True, clamp_alpha=clamp_alpha)
        else:
            q_inputs_tensor, q_inputs_meta, input_l = unified_quantize(input_stored, input_l, target_name, meta, clamp=True, clamp_alpha=clamp_alpha)

        ctx.target_name = target_name
        ctx.q_inputs_meta = q_inputs_meta
        ctx.meta = input.shape, ema_grad_meta, clamp_alpha
        ctx.save_for_backward(weight_master, q_inputs_tensor[0], q_inputs_tensor[1], q_inputs_tensor[2], input_l)
        # return F.linear(input, weight_compute, None)
        res = F.linear(input, weight_compute, None)
        return res


    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dy):
        # acc_var = ctx.acc_var
        q_inputs_meta = ctx.q_inputs_meta
        input_shape, ema_grad_meta, clamp_alpha = ctx.meta
        (weight_master, q_inputs_tensor_out, q_inputs_tensor_scaler, q_inputs_tensor_ema_min, input_l) = ctx.saved_tensors
        q_inputs_tensor = (q_inputs_tensor_out, q_inputs_tensor_scaler, q_inputs_tensor_ema_min)

        if torch.is_autocast_enabled("cuda"):
            target_dtype = torch.get_autocast_dtype("cuda")
            weight_compute = weight_master.to(target_dtype)
            dy = dy.to(target_dtype)
        else:
            weight_compute = weight_master

        input = unified_dequantize(q_inputs_tensor, q_inputs_meta, input_l)
        del q_inputs_tensor, q_inputs_meta, input_l
            
        C_in = input.shape[-1]
        C_out = dy.shape[-1]

        dy_flatten = dy.reshape(-1, C_out)
        input_flatten = input.view(-1, C_in)

        dw = dy_flatten.T.mm(input_flatten)
        dx = dy_flatten.mm(weight_compute)

        # # # LARS Scale
        # ema_grad_meta['ema_smooth'].mul_(ema_grad_meta['momentum']).add_(dw.float())
        # w_norm = torch.linalg.vector_norm(weight_master)
        # dw_norm = torch.linalg.vector_norm(dw.float())
        # trust_ratio = 0.1 * (w_norm / (dw_norm + 1e-5))
        # cond = (w_norm > 0) & (dw_norm > 0) & torch.isfinite(w_norm) & torch.isfinite(dw_norm)
        # trust_ratio = torch.where(cond, trust_ratio, torch.ones_like(trust_ratio))
        # scaled_lr = ema_grad_meta['lr'] * trust_ratio
        # update = dw.float() + ema_grad_meta['ema_smooth'] * ema_grad_meta['momentum']

        # print("ema_norm", ema_grad_meta['ema_smooth'].norm().item())
        # if not torch.isfinite(dw).all():
        #     print("dw has NaN/Inf")
        # if not torch.isfinite(dx).all():
        #     print("dx has NaN/Inf")
        # if not torch.isfinite(w_norm):
        #     print("w_norm NaN/Inf:", w_norm.item())
        # if not torch.isfinite(dw_norm):
        #     print("dw_norm NaN/Inf:", dw_norm.item())
        # if not torch.isfinite(trust_ratio):
        #     print("trust_ratio NaN/Inf:", trust_ratio.item())
        # print("trust_ratio", trust_ratio.item(), "dw_norm", dw_norm.item(), "w_norm", w_norm.item())

        # with torch.no_grad():
        #     weight_master.add_(update * -scaled_lr)

        return dx.view(*input_shape), dw, None, None, None, None, None, None, None
    






@triton.jit
def clamp_fused_fwd_triton(
    x_ptr, alpha, y_ptr, mask_ptr,
    num_elt,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < num_elt

    x = tl.load(x_ptr + offs, mask=mask, other=0.).to(tl.float32)

    inside = (tl.abs(x) <= alpha)
    y = tl.clamp(x, -alpha, alpha)

    tl.store(y_ptr + offs, y.to(tl.bfloat16), mask=mask)
    tl.store(mask_ptr + offs, inside.to(tl.int8), mask=mask)

@triton.jit
def clamp_fused_bwd_triton(
    mask_ptr, alpha, dy_ptr,
    dx_ptr,
    num_elt,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < num_elt

    inside = tl.load(mask_ptr + offs, mask=mask, other=0.).to(tl.int8)
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.).to(tl.float32)

    inside = inside != 0
    dx = tl.where(inside, dy, 0.0)
    tl.store(dx_ptr + offs, dx.to(tl.bfloat16), mask=mask)


@triton_op('act_lib::group_pact_clamp_fwd', mutates_args={})
def group_pact_clamp_fwd_impl(x: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    
    y = torch.empty_like(x, dtype=torch.bfloat16, device='cuda')
    mask = torch.empty(x.shape, dtype=torch.int8, device='cuda')

    num_elt = x.numel()
    grid = (triton.cdiv(x.numel(), 1024),)

    wrap_triton(clamp_fused_fwd_triton)[grid](
        x, alpha, y, mask,
        num_elt=num_elt,
        BLOCK_SIZE=1024,
        num_warps=2,
    )

    return y, mask


@triton_op('act_lib::group_pact_clamp_bwd', mutates_args={})
def group_pact_clamp_bwd_impl(mask: torch.Tensor, alpha: float, dy: torch.Tensor) -> torch.Tensor:

    mask = mask.view(*dy.shape)
    dx = torch.empty_like(dy, dtype=torch.bfloat16, device='cuda')

    num_elt = dy.numel()
    grid = (triton.cdiv(dy.numel(), 1024),)

    wrap_triton(clamp_fused_bwd_triton)[grid](
        mask, alpha, dy,
        dx,
        num_elt=num_elt,
        BLOCK_SIZE=1024,
        num_warps=2,
    )

    return dx


class GroupPACTClampFn(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, x, alpha, group_size, meta, target_name):
        y = torch.empty_like(x, dtype=torch.bfloat16, device='cuda')

        if torch.is_autocast_enabled("cuda"):
            target_dtype = torch.get_autocast_dtype("cuda")
            x = x.to(target_dtype)


        y, mask = torch.ops.act_lib.group_pact_clamp_fwd(x, alpha)

        clamp_meta = {}
        clamp_target_name = f"{target_name}_clamp"
        clamp_meta[clamp_target_name] = {
            'bits': 1,
            'pack_only' : True,
            'group_size': group_size,
            'act_padding': meta[f'{target_name}']['act_padding'],
            'N': meta[f'{target_name}']['N'],
            'AVG_ALAM': False,
            'ALAM_BITS': 0,
        }

        q_inputs_tensor, q_inputs_meta, input_l = unified_quantize(mask, None, clamp_target_name, clamp_meta, clamp=False, clamp_alpha=alpha)

        ctx.save_for_backward(q_inputs_tensor[0], q_inputs_tensor[1], q_inputs_tensor[2])
        ctx.meta = q_inputs_meta
        return y

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dy):
        (q_inputs_tensor_packed, q_inputs_tensor_scale, q_inputs_tensor_min) = ctx.saved_tensors
        q_inputs_meta = ctx.meta

        q_inputs_tensor = (q_inputs_tensor_packed, q_inputs_tensor_scale, q_inputs_tensor_min)
        mask = unified_dequantize(q_inputs_tensor, q_inputs_meta, None)

        # dx = torch.ops.act_lib.group_pact_clamp_bwd(mask, alpha, dy)
        return dy * mask, None, None, None, None


















def variance_analyze(original_act, quantized_act, quantized_meta, input_l, eps=1e-6):
    deq = unified_dequantize(quantized_act, quantized_meta, input_l)

    ori_m2 = (original_act ** 2).mean(dim=(0, 2, 3))      # [C]
    deq_m2 = (deq ** 2).mean(dim=(0, 2, 3))               # [C]

    shrink_ratio = deq_m2 / (ori_m2 + eps)                # <1 means shrink
    fix_scale = torch.sqrt((ori_m2 + eps) / (deq_m2 + eps)).detach()   # [C]

    return shrink_ratio, fix_scale











def analyze_outliers(x_flatten, threshold=0.3, name=None):
    mean = x_flatten.mean(dim=2, keepdim=True)
    std = x_flatten.std(dim=2, keepdim=True)

    z = (x_flatten - mean) / (std + 1e-5)
    outliers = (z.abs() > threshold)

    if name is None:
        print(f'Number of outliers: {outliers.sum()}')
        print(f'Precent of the outliers: {outliers.sum() / x_flatten.numel() * 100}%')
    else:
        print(f'{name} Number of outliers: {outliers.sum()}')
        print(f'{name} Precent of the outliers: {outliers.sum() / x_flatten.numel() * 100}%')


# @torch.compile(fullgraph=True)
def smooth(x, weight, layer):
    """
    Returns:
        s_input:  shape that divides the activation
        s_weight: shape that multiplies the weight
    """
    # if torch.isnan(weight).any():
    #     print("weight has nan")

    alpha = 0.5
    eps = 1e-6

    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
    else:
        dtype = x.dtype

    if layer == "conv":
        N, C_in, H, W = x.shape
        C_out = weight.shape[0]
        is_dw = (weight.shape[1] == 1 and C_out == C_in)

        # per input-channel max
        x_max = x.detach().abs().amax(dim=(0, 2, 3), keepdim=True)  # [1,C_in,1,1]
        w_max = weight.detach().abs().amax(dim=(0, 2, 3), keepdim=True)  # [1,C_in,1,1]

        # SmoothQuant per input-channel scale
        s = ((x_max + eps).pow(alpha) / (w_max + eps).pow(1-alpha)) # [1,C_in,1,1]
        s = s.clamp(min=1/16, max=16)

        # input always divides by [1,C_in,1,1]
        s_input = s  # shape [1,C_in,1,1]

        if is_dw:
            # weight: [C_in,1,kH,kW]
            s_weight = s.view(C_in, 1, 1, 1)  # reshape for weight
        else:
            # weight: [C_out, C_in, kH, kW]
            s_weight = s  # broadcast OK

        # if torch.isnan(x).any(): print("x is NaN")
        # if torch.isnan(weight).any(): print("weightis NaN")
        # if torch.isinf(x).any(): print("x is INF")
        # if torch.isinf(weight).any(): print("weight is INF")

        
        # if torch.isnan(x_max).any(): print("x_max is NaN")
        # if torch.isnan(w_max).any(): print("w_max is NaN")
        # if torch.isinf(x_max).any(): print("x_max is INF")
        # if torch.isinf(w_max).any(): print("w_max is INF")
        if torch.isnan(s_input).any(): print("s_input is NaN")
        if torch.isinf(s_input).any(): print("s is INF")
        if torch.isnan(s_weight).any(): print("s_weight is NaN")
        if torch.isinf(s_weight).any(): print("s_weight is INF")

        return s_input, s_weight

    elif layer == "linear":
        # x: [N, C_in]
        # w: [C_out, C_in]

        x_max = x.detach().abs().amax(dim=0, keepdim=True)   # [1,C_in]
        w_max = weight.detach().abs().amax(dim=0, keepdim=True)  # [1,C_in]

        s = ((x_max + eps).pow(alpha) / (w_max + eps).pow(1-alpha))  # [1,C_in]
        s = s.clamp(min=1/16, max=16)

        s_input  = s                         # [1,C_in]
        s_weight = s                        # same shape

        return s_input, s_weight



def fwht_transform_256(x):
    """
    x: [N, G, 256]
    Fast Walsh–Hadamard Transform applied to the last dimension.
    Works on GPU, pure PyTorch, no dependency.
    """

    N, G, C = x.shape
    assert C == 256, "Last dimension must be 256"

    # Flatten first two dims
    x = x.view(N*G, 256)  # [N*G, 256]

    h = 1
    while h < 256:
        # Each iteration: (a, b) -> (a+b, a-b)
        x = x.view(x.size(0), -1, h * 2)
        a, b = x[..., :h], x[..., h:]
        x = torch.cat([a + b, a - b], dim=-1)
        h *= 2

    # Restore original shape
    return x.view(N, G, C)