import torch
import torch.nn.functional as F

from .tensor_act_reshape_utils import (
    spatical_aware_act_tensor_reshape,
    spatical_aware_act_tensor_reshape_back
    )

from functools import reduce
import operator



def no_scheme_compute_quantization(input, bits, N, group_size):
    flat = input.view(N, -1)
    num_features = flat.size(1)
    pad_num = (-num_features) % group_size
    if pad_num > 0:
        flat = torch.cat(
            [flat, torch.zeros((N, pad_num), dtype=flat.dtype, device=flat.device)], dim=1
        )
        
    num_groups = flat.size(1) // group_size
    input_groups = flat.view(N, num_groups, group_size)
    return input_groups, bits


def quantize(input, bits, pack_only, group_size, act_padding, N, avg_alam, alam_bits, clamp, clamp_alpha):
    if act_padding:    
        # input_groups, q_bits = no_scheme_compute_quantization(input, bits, N, group_size)
        # N, G = input_groups.shape[:2]
        input_groups, N, G, new_H, new_W, ori_shape_new, spatical_padding_eligibility, spatical_reshape_eligibility, channel_mean = spatical_aware_act_tensor_reshape(input, Group_Size=group_size, act_padding=True, pack_only=pack_only)

        if not pack_only:
            out, scaler, ema_min = torch.ops.act_lib.quant_pack_triton(input_groups, bits, group_size, avg_alam, alam_bits, clamp, clamp_alpha)
            # return (out, scaler.view(N, G), ema_min.view(N, G)), (bits, pack_only, group_size, act_padding, N, G, avg_alam, alam_bits,input.shape)
            return (out, scaler, ema_min), (bits, pack_only, group_size, act_padding, N, G, new_H, new_W, spatical_padding_eligibility, spatical_reshape_eligibility, channel_mean,avg_alam, alam_bits, ori_shape_new)
        else:
            out = torch.ops.act_lib.pack_triton(input_groups, bits, group_size)
            return (out, None, None), (bits, pack_only, group_size, act_padding, N, G, 0, 0, False, False, channel_mean, None, None, ori_shape_new)

        
    else:
        # input_groups = input.view(N, -1, group_size)
        # N, G = input_groups.shape[:2]
        input_groups, N, G, new_H, new_W, ori_shape_new, spatical_padding_eligibility, spatical_reshape_eligibility, channel_mean = spatical_aware_act_tensor_reshape(input, Group_Size=group_size, act_padding=False, pack_only=pack_only)

        if not pack_only:
            out, scaler, ema_min = torch.ops.act_lib.quant_pack_triton(input_groups, bits, group_size, avg_alam, alam_bits, clamp, clamp_alpha)    
            # return (out, scaler.view(N, G), ema_min.view(N, G)), (bits, pack_only, group_size, act_padding, N, G, avg_alam, alam_bits, input.shape)
            return (out, scaler, ema_min), (bits, pack_only, group_size, act_padding, N, G, new_H, new_W, spatical_padding_eligibility, spatical_reshape_eligibility, channel_mean, avg_alam, alam_bits, ori_shape_new)
        else:
            out = torch.ops.act_lib.pack_triton(input_groups, bits, group_size)
            # return (out, None, None), (bits, pack_only, group_size, act_padding, N, G, None, None, input.shape)
            return (out, None, None), (bits, pack_only, group_size, act_padding, N, G, 0, 0, False, False, channel_mean, None, None, ori_shape_new)
    

def unified_quantize(input, input_l, target_name, meta, clamp, clamp_alpha):
        curr_meta = meta[f"{target_name}"]
        q_inputs_tensor, q_inputs_meta = quantize(input, curr_meta['bits'], curr_meta["pack_only"], curr_meta['group_size'], curr_meta['act_padding'], curr_meta['N'], curr_meta['AVG_ALAM'], curr_meta['ALAM_BITS'], clamp, clamp_alpha)

        return (q_inputs_tensor, q_inputs_meta, input_l)


def dequantize(quantized_tensor, quantized_meta):
    q_input, q_scale, ema_mn, = quantized_tensor
    # q_bits, pack_only, group_size, act_padding, N, G, avg_alam, alam_bits, input_shape = quantized_meta
    q_bits, pack_only, group_size, act_padding, N, G, new_H, new_W, spatical_padding_eligibility, spatical_reshape_eligibility, channel_mean, avg_alam, alam_bits, input_shape = quantized_meta

    if not pack_only:
        unpack = torch.ops.act_lib.dequant_unpack_triton(q_input, q_scale, ema_mn, q_bits, group_size, avg_alam, alam_bits)
    else:
        unpack = torch.ops.act_lib.unpack_triton(q_input, q_bits, N, G, group_size)

    # if act_padding:
    #     num_features = reduce(operator.mul, input_shape[1:], 1)
    #     flat = unpack.view(N, -1)[:, :num_features]
    #     return flat.view(*input_shape)
    # else:
    #     return unpack.view(*input_shape)

    #X: Tensor, ori_shape: Tuple[int, int, int, int], Group_Size: int, new_H: int, new_W: int, act_padding: bool, spatical_padding_eligibility: bool, spatical_reshape_eligibility: bool
    return spatical_aware_act_tensor_reshape_back(unpack, input_shape, group_size, new_H, new_W, act_padding, spatical_padding_eligibility, spatical_reshape_eligibility, channel_mean)


def unified_dequantize(ori_quantized_tensor, ori_quantized_meta, input_l):
    input_shape = ori_quantized_meta[-1]

    input = dequantize(ori_quantized_tensor, ori_quantized_meta)

    if input_l is None:
        return input
    else:
        if len(input_shape) == 4:
            HEIGHT, WIDTH = input_shape[-2], input_shape[-1]
            return input + F.interpolate(input_l, size=(HEIGHT, WIDTH), scale_factor=None, mode="nearest").to(torch.bfloat16)
        elif len(input_shape) == 3:
            channel = input_shape[2]
            return input + F.interpolate(input_l, size=channel, scale_factor=None, mode="nearest").to(torch.bfloat16)
        else:
            channel = input_shape[0]
            return input + F.interpolate(input_l, size=channel, scale_factor=None, mode="nearest").to(torch.bfloat16)
