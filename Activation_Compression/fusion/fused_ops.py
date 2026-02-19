import torch 
from torch import nn
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd

import triton_kernel.norm_act_fusion.kernel_registration


class _DOBatchNormReLU2d(Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, x, weight, bias, relu, relu6,running_mean, running_var, exponential_average_factor,
                target_name, meta, ema_grad_meta):
        ctx.set_materialize_grads(False)

        BLOCK_SIZE = meta[f'{target_name}']['group_size']
        BITS = meta[f'{target_name}']['bits']
        num_warps = 8
        num_stages = 5
        momentum  = exponential_average_factor

        N, C, H, W = x.shape
        HW = H * W
        M = N * HW
        x_compute = x.view(N, C, HW)

        stride_n, stride_c, _ = x_compute.stride()

        sum_, sum_square_ = torch.ops.act_lib.bn_fwd_reduce(
            x_compute,
            M=M, HW=HW, stride_n=stride_n, stride_c=stride_c,
            BLOCK_M=BLOCK_SIZE, num_warps=num_warps, num_stages=num_stages,
        )

        y, x_hat_packed, relu_mask_packed, scale, min, mean, var = torch.ops.act_lib.bn_relu_fwd_norm_fused_quant_pack(
            x_compute, sum_, sum_square_, weight, bias, None, None,
            M=M, HW=HW,
            stride_n=stride_n, stride_c=stride_c,
            relu=relu, relu6=relu6,
            BLOCK_SIZE=BLOCK_SIZE,
            BITS=BITS,
            sync=False
        )

        running_mean.mul_(1.0 - momentum).add_(mean, alpha=momentum)
        running_var.mul_(1.0 - momentum).add_(var, alpha=momentum)

        ctx.meta = meta
        ctx.other_meta = BLOCK_SIZE, BITS, target_name
        ctx.save_for_backward(weight, x_hat_packed, relu_mask_packed, var, scale, min)

        return y.view(N, C, H, W)
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dy):
        meta = ctx.meta
        BLOCK_SIZE, BITS, target_name = ctx.other_meta
        (weight, x_hat_packed, relu_mask_packed, var, scale, min) = ctx.saved_tensors

        N, C, H, W = dy.shape
        HW = H * W
        M = N * HW

        dy_compute = dy.view(N, C, HW)

        stride_n, stride_c, _ = dy_compute.stride()

        DW, DB = torch.ops.act_lib.bn_relu_bwd_reduce_fused_dequant_unpack(
            x_hat_packed, relu_mask_packed, dy_compute,
            scale, min,
            M=M, HW=HW,
            stride_n=stride_n, stride_c=stride_c,
            BLOCK_SIZE=BLOCK_SIZE, BITS=BITS
        )

        DX = torch.ops.act_lib.bn_relu_bwd_dx_fused_dequant_unpack(
            x_hat_packed, relu_mask_packed, dy_compute,
            DW, DB, weight, var, scale, min,
            M=M, HW=HW,
            stride_n=stride_n, stride_c=stride_c,
            BLOCK_SIZE=BLOCK_SIZE, BITS=BITS
        )

        return DX.view(N, C, H, W), DW, DB, None, None, None, None, None, None, None, None, None

