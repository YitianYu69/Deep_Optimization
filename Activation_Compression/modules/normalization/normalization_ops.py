import torch 
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd
import torch.distributed as dist

from modules.module_utils import unified_quantize, unified_dequantize
import modules.normalization.kernel_registration


class _DOBatchNorm(Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        exponential_average_factor: float,  
        eps: float,
        target_name,
        meta: dict,
        ema_grad_meta
    ):
        ctx.set_materialize_grads(False)
        # ---- fixed launch params (keep constant to avoid recompiles) ----
        BLOCK_M   = meta[f'{target_name}']['group_size']
        num_warps = 8
        num_stages = 5
        momentum  = exponential_average_factor

        N, C, H, W = input.shape

        HW = H * W
        x3 = input.view(N, C, HW)

        stride_n, stride_c, _ = x3.stride()
        M = N * HW

        C = x3.shape[1]
        sum_, sumsq_ = torch.ops.act_lib.bn_fwd_reduce(
            x3,
            M=M, HW=HW, stride_n=stride_n, stride_c=stride_c,
            BLOCK_M=BLOCK_M, num_warps=num_warps, num_stages=num_stages,
        )

        if meta[f'{target_name}']['act_padding']:
            y3, input_hat, mean, var = torch.ops.act_lib.bn_fwd_norm(
                x3, sum_, sumsq_, weight, bias, None, None,
                M=M, HW=HW, stride_n=stride_n, stride_c=stride_c,
                BLOCK_M=BLOCK_M, sync=False,
                num_warps=num_warps, num_stages=num_stages
            )

            running_mean.mul_(1.0 - momentum).add_(mean, alpha=momentum)
            running_var.mul_(1.0 - momentum).add_(var, alpha=momentum)

            y = y3.view(N, C, H, W)

            q_inputs_tensor, q_inputs_meta, input_hat_l = unified_quantize(input_hat, None, target_name, meta, clamp=False, clamp_alpha=0.0)

            # Save only what backward needs (small + stats)
            ctx.target_name = target_name
            ctx.q_meta = q_inputs_meta
            ctx.meta = (BLOCK_M, num_warps, num_stages, meta, ema_grad_meta, None, None)
            ctx.save_for_backward(
                weight,
                q_inputs_tensor[0], q_inputs_tensor[1], q_inputs_tensor[2],
                var
            )
        else:
            avg_alam = meta[f'{target_name}']['AVG_ALAM']
            alam_bits = meta[f'{target_name}']['ALAM_BITS']

            y, x_hat_packed, scale, min, mean, var = torch.ops.act_lib.bn_fwd_norm_quant_pack_fused(
                x3, sum_, sumsq_, weight, bias, None, None,
                M=M, HW=HW, stride_n=stride_n, stride_c=stride_c,
                BLOCK_M=BLOCK_M, bits=meta[f'{target_name}']['bits'],
                sync=False,
                avg_alam=avg_alam, alam_bits=alam_bits,
                num_warps=num_warps, num_stages=num_stages 
            )

            running_mean.mul_(1.0 - momentum).add_(mean, alpha=momentum)
            running_var.mul_(1.0 - momentum).add_(var, alpha=momentum)

            y = y.view(N, C, H, W)

            ctx.target_name = target_name
            ctx.meta = (BLOCK_M, num_warps, num_stages, meta, ema_grad_meta, avg_alam, alam_bits)
            ctx.save_for_backward(
                weight,
                x_hat_packed, scale, min,
                var
            )

        return y

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dy: torch.Tensor):
        target_name = ctx.target_name
        BLOCK_M, num_warps, num_stages, meta, ema_grad_meta, avg_alam, alam_bits = ctx.meta

        if meta[f'{target_name}']['act_padding']:
            q_inputs_meta = ctx.q_meta
            (weight, q_out, q_scale, q_ema_min, var) = ctx.saved_tensors
            q_inputs_tensor = (q_out, q_scale, q_ema_min)

            x_hat = unified_dequantize(q_inputs_tensor, q_inputs_meta, None)

            N2, C, H, W = dy.shape
            HW = H * W
            x3  = x_hat.view(N2, C, HW)
            dy3 = dy.view(N2, C, HW)

            torch._assert(x3.stride(2) == 1 and dy3.stride(2) == 1, "QuanBN Triton expects stride_s==1.")
            stride_n, stride_c, _ = x3.stride()
            M = N2 * HW

            dgamma, dbeta = torch.ops.act_lib.bn_bwd_reduce(
                x3, dy3,
                M=M, HW=HW, stride_n=stride_n, stride_c=stride_c,
                BLOCK_M=BLOCK_M, num_warps=num_warps, num_stages=num_stages,
            )

            dx = torch.ops.act_lib.bn_bwd_dx(
                x3, dy3, var, weight, dbeta, dgamma,
                M=M, HW=HW, stride_n=stride_n, stride_c=stride_c,
                BLOCK_M=BLOCK_M, sync=False,
                num_warps=num_warps, num_stages=num_stages,
            )

            dx = dx.view(N2, C, H, W)
        else:
            weight, x_hat_packed, scale, min, var = ctx.saved_tensors

            N2, C, H, W = dy.shape
            HW = H * W
            M = N2 * HW

            dy3 = dy.view(N2, C, HW)
            stride_n, stride_c, _ = dy3.stride()

            dgamma, dbeta = torch.ops.act_lib.bn_bwd_reduce_dequant_unpack_fused(
                x_hat_packed, dy3, scale, min,
                M=M, HW=HW, stride_n=stride_n, stride_c=stride_c, BLOCK_M=BLOCK_M, bits=meta[f'{target_name}']['bits'],
                avg_alam=avg_alam, alam_bits=alam_bits
            )

            dx = torch.ops.act_lib.bn_bwd_dx_dequant_unpack_fused(
                x_hat_packed, dy3, scale, min,
                dgamma, dbeta, weight, var,
                M=M, HW=HW, stride_n=stride_n, stride_c=stride_c, BLOCK_M=BLOCK_M, bits=meta[f'{target_name}']['bits'],
                sync=False,
                avg_alam=avg_alam, alam_bits=alam_bits
            )
            dx = dx.view(N2, C, H, W)


            # ema_grad_meta['ema_smooth'].mul_(ema_grad_meta['momentum']).add_(dgamma.float())
            # weight.add_(-ema_grad_meta['lr'] * (dgamma.float() + ema_grad_meta['momentum'] * ema_grad_meta['ema_smooth']))
            # bias.add_(-ema_grad_meta['lr'] * dbeta.float())

        return (
            dx,                    # grad input
            dgamma,  # grad weight
            dbeta,   # grad bias
            None, None, None, None, None, None, None, None, None, None
        )
    


class _DOSyncBatchNorm(Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size,
                target_name, meta, ema_grad_meta):
        
        count = torch.empty(1,
                            dtype=running_mean.dtype,
                            device=input.device).fill_(input.numel() // input.size(1))
        
        mean, invstd = torch.batch_norm_stats(input, eps)

        num_channels = input.shape[1]
        combined = torch.cat([mean, invstd, count], dim=0)
        combined_lists = [
            torch.empty_like(combined) for k in range(world_size)
        ]

        dist.all_gather(combined_lists, combined, process_group, async_op=False)
        combined = torch.stack(combined_lists, dim=0)
        mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)

        size = count_all.view(-1).long().sum()
        if size == 1:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
        
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1)
        )

        BLOCK_M = meta[f'{target_name}']['group_size']
        num_warps = 8
        num_stages = 5

        N, C, H, W = input.shape
        HW = H * W
        x3 = input.view(N, C, HW)

        stride_n, stride_c, _ = x3.stride()
        M = N * HW

        if meta[f'{target_name}']['act_padding']:
            y3, input_hat, _, _ = torch.ops.act_lib.bn_fwd_norm(
                x3, None, None, weight, bias, mean, invstd,
                M=M, HW=HW, stride_n=stride_n, stride_c=stride_c,
                BLOCK_M=BLOCK_M, sync=True,
                num_warps=num_warps, num_stages=num_stages
            )

            y = y3.view(N, C, H, W)

            q_inputs_tensor, q_inputs_meta, input_hat_l = unified_quantize(input_hat, None, target_name, meta, clamp=False, clamp_alpha=0.0)

            # Save only what backward needs (small + stats)
            ctx.target_name = target_name
            ctx.q_meta = q_inputs_meta
            ctx.meta = (BLOCK_M, num_warps, num_stages, meta, ema_grad_meta)
            ctx.save_for_backward(
                weight,
                q_inputs_tensor[0], q_inputs_tensor[1], q_inputs_tensor[2],
                invstd
            )
        else:
            avg_alam = meta[f'{target_name}']['AVG_ALAM']
            alam_bits = meta[f'{target_name}']['ALAM_BITS']

            y, x_hat_packed, scale, min, _, _ = torch.ops.act_lib.bn_fwd_norm_quant_pack_fused(
                x3, None, None, weight, bias, mean, invstd,
                M=M, HW=HW, stride_n=stride_n, stride_c=stride_c,
                BLOCK_M=BLOCK_M, bits=meta[f'{target_name}']['bits'],
                sync=True,
                avg_alam=avg_alam, alam_bits=alam_bits,
                num_warps=num_warps, num_stages=num_stages 
            )

            y = y.view(N, C, H, W)

            ctx.target_name = target_name
            ctx.meta = (BLOCK_M, num_warps, num_stages, meta, ema_grad_meta, avg_alam, alam_bits)
            ctx.save_for_backward(
                weight,
                x_hat_packed, scale, min,
                invstd
            )
        return y
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, dy: torch.Tensor):
        target_name = ctx.target_name
        BLOCK_M, num_warps, num_stages, meta, ema_grad_meta, avg_alam, alam_bits = ctx.meta

        if meta[f'{target_name}']['act_padding']:
            q_inputs_meta = ctx.q_meta
            (weight, q_out, q_scale, q_ema_min, invstd) = ctx.saved_tensors
            q_inputs_tensor = (q_out, q_scale, q_ema_min)

            x_hat = unified_dequantize(q_inputs_tensor, q_inputs_meta, None)

            N2, C, H, W = dy.shape
            HW = H * W
            x3  = x_hat.view(N2, C, HW)
            dy3 = dy.view(N2, C, HW)

            torch._assert(x3.stride(2) == 1 and dy3.stride(2) == 1, "QuanBN Triton expects stride_s==1.")
            stride_n, stride_c, _ = x3.stride()
            M = N2 * HW

            dgamma, dbeta = torch.ops.act_lib.bn_bwd_reduce(
                x3, dy3,
                M=M, HW=HW, stride_n=stride_n, stride_c=stride_c,
                BLOCK_M=BLOCK_M, num_warps=num_warps, num_stages=num_stages,
            )

            dx = torch.ops.act_lib.bn_bwd_dx(
                x3, dy3, invstd, weight, dbeta, dgamma,
                M=M, HW=HW, stride_n=stride_n, stride_c=stride_c,
                BLOCK_M=BLOCK_M, sync=True,
                num_warps=num_warps, num_stages=num_stages,
            )

            dx = dx.view(N2, C, H, W)
        else:
            weight, x_hat_packed, scale, min, invstd = ctx.saved_tensors
            BLOCK_M, num_warps, num_stages, meta, ema_grad_meta, avg_alam, alam_bits = ctx.meta

            N2, C, H, W = dy.shape
            HW = H * W
            M = N2 * HW

            dy3 = dy.view(N2, C, HW)
            stride_n, stride_c, _ = dy3.stride()

            dgamma, dbeta = torch.ops.act_lib.bn_bwd_reduce_dequant_unpack_fused(
                x_hat_packed, dy3, scale, min,
                M=M, HW=HW, stride_n=stride_n, stride_c=stride_c, BLOCK_M=BLOCK_M, bits=meta[f'{target_name}']['bits'],
                avg_alam=avg_alam, alam_bits=alam_bits
            )

            dx = torch.ops.act_lib.bn_bwd_dx_dequant_unpack_fused(
                x_hat_packed, dy3, scale, min,
                dgamma, dbeta, weight, invstd,
                M=M, HW=HW, stride_n=stride_n, stride_c=stride_c, BLOCK_M=BLOCK_M, bits=meta[f'{target_name}']['bits'],
                sync=True,
                avg_alam=avg_alam, alam_bits=alam_bits
            )
            dx = dx.view(N2, C, H, W)


            # ema_grad_meta['ema_smooth'].mul_(ema_grad_meta['momentum']).add_(dgamma.float())
            # weight.add_(-ema_grad_meta['lr'] * (dgamma.float() + ema_grad_meta['momentum'] * ema_grad_meta['ema_smooth']))
            # bias.add_(-ema_grad_meta['lr'] * dbeta.float())

        return (
            dx,                    # grad input
            dgamma,  # grad weight
            dbeta,   # grad bias
            None, None, None, None, None, None, None, None, None, None
        )
