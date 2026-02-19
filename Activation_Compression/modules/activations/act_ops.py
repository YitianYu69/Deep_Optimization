import torch
from torch import nn
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd
from torch.library import triton_op, wrap_triton

from modules.module_utils import unified_quantize, unified_dequantize
import modules.activations.kernel_registration

import triton
import triton.language as tl

from typing import Tuple


class _DOReLU(Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, input, relu, relu6, target_name, meta):
        act_padding = meta[f'{target_name}']['act_padding']
        ctx.act_padding = act_padding
        
        if act_padding:
            y, relu_mask = torch.ops.act_lib.relu_triton(input, relu, relu6)
            q_inputs_tensor, q_inputs_meta, input_l = unified_quantize(relu_mask, None, target_name, meta, clamp=False, clamp_alpha=0.0)
            
            ctx.q_meta = q_inputs_meta
            ctx.save_for_backward(q_inputs_tensor[0])
            return y
        else:
            bits = meta[f'{target_name}']['bits']
            group_size = meta[f'{target_name}']['group_size']

            y, relu_mask, N, G = torch.ops.act_lib.relu_fwd_fused_pack(input, relu, relu6,bits, group_size)

            ctx.meta = N, G, bits, group_size
            ctx.save_for_backward(relu_mask)
            return y


    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dy):
        act_padding = ctx.act_padding

        if act_padding:
            (q_inputs_tensor,) = ctx.saved_tensors
            q_inputs_meta = ctx.q_meta

            relu_mask = unified_dequantize((q_inputs_tensor, None, None), q_inputs_meta, None)

            return dy * relu_mask, None, None, None, None, None, None
        else:
            N, G, bits, group_size = ctx.meta
            (relu_mask,) = ctx.saved_tensors

            dx = torch.ops.act_lib.relu_bwd_fused_unpack(relu_mask, dy, bits, N, G, group_size)

            return dx, None, None, None, None, None, None
            

class _DOSiLU(Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, input, target_name, meta):
        pack_only = meta[f'{target_name}']["pack_only"]
        act_padding = meta[f'{target_name}']['act_padding']
        ctx.act_padding = act_padding
        ctx.pack_only = pack_only

        # print(f'Layer: {target_name}, fwd ori input shape: {input.shape}')
        # print(f'Layer: {target_name}, fwd act_padding: {act_padding}, group_size: {meta[f"{target_name}"]["group_size"]}')
        
        if act_padding:
            y, act = torch.ops.act_lib.silu_triton(input)
            q_inputs_tensor, q_inputs_meta, input_l = unified_quantize(act, None, target_name, meta, clamp=False, clamp_alpha=0.0)
            
            ctx.q_meta = q_inputs_meta
            ctx.save_for_backward(q_inputs_tensor[0], q_inputs_tensor[1], q_inputs_tensor[2])
        else:
            bits = meta[f'{target_name}']['bits']
            group_size = meta[f'{target_name}']['group_size']
            avg_alam = meta[f'{target_name}']['AVG_ALAM']
            alam_bits = meta[f'{target_name}']['ALAM_BITS']

            y, act_packed, scale, min, N, G, new_H, new_W, spatical_padding_eligibility, spatical_reshape_eligibility = torch.ops.act_lib.silu_fwd_quan_pack(input, bits, group_size, pack_only, avg_alam, alam_bits)

            ctx.meta = N, G, bits, group_size, avg_alam, alam_bits, new_H, new_W, spatical_padding_eligibility, spatical_reshape_eligibility
            ctx.save_for_backward(act_packed, scale, min)

        # print(f"Layer: {target_name}, fwd output shape: {y.shape}")
        return y


    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dy):
        act_padding = ctx.act_padding
        pack_only = ctx.pack_only

        if act_padding:
            (q_inputs_tensor_packed, q_inputs_tensor_scale, q_inputs_tensor_min) = ctx.saved_tensors
            q_inputs_tensors = (q_inputs_tensor_packed, q_inputs_tensor_scale, q_inputs_tensor_min)
            q_inputs_meta = ctx.q_meta

            act = unified_dequantize(q_inputs_tensors, q_inputs_meta, None)

            return dy * act, None, None, None
        else:
            N, G, bits, group_size, avg_alam, alam_bits, new_H, new_W, spatical_padding_eligibility, spatical_reshape_eligibility = ctx.meta
            (act_packed, scale, min) = ctx.saved_tensors

            dx = torch.ops.act_lib.silu_bwd_fused_dequan_unpack(act_packed, dy, scale, min, bits, N, G, group_size, pack_only,new_H, new_W, spatical_padding_eligibility, spatical_reshape_eligibility, avg_alam, alam_bits)

            return dx, None, None, None




class _DOGELU(Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, input, target_name, meta):
        act_padding = meta[f'{target_name}']['act_padding']
        ctx.act_padding = act_padding

        if act_padding:
            y, act = torch.ops.act_lib.gelu_triton(input)
            q_inputs_tensor, q_inputs_meta, input_l = unified_quantize(act, None, target_name, meta, clamp=False, clamp_alpha=0.0)

            ctx.q_meta = q_inputs_meta
            ctx.save_for_backward(q_inputs_tensor[0], q_inputs_tensor[1], q_inputs_tensor[2])
            return y
        else:
            bits = meta[f'{target_name}']['bits']
            group_size = meta[f'{target_name}']['group_size']

            y, act_packed, scale, min, N, G = torch.ops.act_lib.gelu_fwd_fused_quan_pack(input, bits, group_size)

            ctx.meta = N, G, bits, group_size
            ctx.save_for_backward(act_packed, scale, min)
            return y


    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dy):
        act_padding = ctx.act_padding

        if act_padding:
            (q_inputs_tensor_packed, q_inputs_tensor_scale, q_inputs_tensor_min) = ctx.saved_tensors
            q_inputs_tensors = (q_inputs_tensor_packed, q_inputs_tensor_scale, q_inputs_tensor_min)
            q_inputs_meta = ctx.q_meta

            act = unified_dequantize(q_inputs_tensors, q_inputs_meta, None)

            return dy * act, None, None, None
        else:
            N, G, bits, group_size = ctx.meta
            (act_packed, scale, min) = ctx.saved_tensors

            dx = torch.ops.act_lib.gelu_bwd_fused_dequan_unpack(act_packed, dy, scale, min, bits, N, G, group_size)

            return dx, None, None, None
