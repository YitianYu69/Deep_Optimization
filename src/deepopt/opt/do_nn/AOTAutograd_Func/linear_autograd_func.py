import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ...backends import gemm_nt
from ...quantization.tensor import QuantizedTensor
from ...quantization.quantizer import quantize_fp8


class _DOLinearFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        plan: dict,
        out_dtype: torch.dtype
    ):
        fp8_trainig_plan = plan["fp8_training"]
        fp8_dtypes = fp8_trainig_plan["fp8_dtypes"]
        block_shapes = plan["block_shapes"]
        backend = fp8_trainig_plan.get("backend", "reference")
        num_stages = fp8_trainig_plan.get("num_stages", 3)

        x_matrix = x.reshape(-1, weight.shape[1])

        qx = quantize_fp8(
            x_matrix, fp8_dtype=fp8_dtypes["activation"],
            block_shape=block_shapes["activation"]
        )
        qw = quantize_fp8(
            weight, fp8_dtype=fp8_dtypes["weight"],
            block_shape=block_shapes["weight"]
        )

        output = gemm_nt(qx, qw, backend=backend, bias=bias, out_dtype=out_dtype, num_stages=num_stages)
        need_dx, need_dw = ctx.needs_input_grad[:2]

        # Saved-activation compression belongs at this boundary: a future
        # codec must restore each operand with its scale and block metadata.
        # Keep tensor payloads in save_for_backward so PyTorch saved-tensor
        # hooks, offloading, and version checks remain available.
        ctx.save_for_backward(
            qx.data if need_dw else None,
            qx.scale_inv if need_dw else None,
            qw.data if need_dx else None,
            qw.scale_inv if need_dx else None
        )

        ctx.input_shape = x.shape
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.out_features = weight.shape[0]
        ctx.fp8_dtypes = dict(fp8_dtypes)
        ctx.block_shapes = dict(block_shapes)
        ctx.backend = backend
        ctx.num_stages = num_stages

        return output.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        need_dx, need_dw, need_db = ctx.needs_input_grad[:3]
        x_data, x_scale_inv, w_data, w_scale_inv = ctx.saved_tensors

        grad_matrix = grad_output.reshape(-1, ctx.out_features)

        dx = None
        dw = None
        db = None
        if need_dx or need_dw:
            qg = quantize_fp8(
                grad_matrix, fp8_dtype=ctx.fp8_dtypes["grad"], block_shape=ctx.block_shapes["grad"]
            )

            if need_dx:
                qw = QuantizedTensor(
                    data=w_data, scale_inv=w_scale_inv, block_shape=ctx.block_shapes["weight"]
                )

                # Dense/sparse kernel selection belongs to backend dispatch;
                # gradient orientation and dtype remain the autograd contract.
                dx = gemm_nt(
                    qg, qw.transpose(),
                    backend=ctx.backend,
                    out_dtype=ctx.input_dtype,
                    num_stages=ctx.num_stages
                ).reshape(ctx.input_shape)
            
            if need_dw:
                qx = QuantizedTensor(
                    data=x_data, scale_inv=x_scale_inv, block_shape=ctx.block_shapes["activation"]
                )

                dw = gemm_nt(
                    qg.transpose(),
                    qx.transpose(),
                    backend=ctx.backend,
                    out_dtype=ctx.weight_dtype,
                    num_stages=ctx.num_stages
                )
        
        if need_db:
            db = grad_matrix.sum(dim=0, dtype=torch.float32).to(dtype=ctx.bias_dtype)
        
        return dx, dw, db, None, None
