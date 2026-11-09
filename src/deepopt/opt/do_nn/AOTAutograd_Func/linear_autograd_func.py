import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ...backends import gemm_nt
from ...quantization.tensor import QuantizedTensor
from ...quantization.quantizer import quantize_fp8, _quantize_tensorwise_pair, _quantize_tensorwise_transpose
from ..._execution import backward_finite_check_policy


class _DOLinearFunction(Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        plan: dict,
        out_dtype: torch.dtype,
        backward_enabled: bool,
        weight_cache,
        scaling_state=None,
        state_training=True,
    ):
        fp8_trainig_plan = plan["fp8_training"]
        fp8_dtypes = fp8_trainig_plan["fp8_dtypes"]
        block_shapes = plan["block_shapes"]
        backend = fp8_trainig_plan.get("backend", "reference")
        num_stages = fp8_trainig_plan.get("num_stages", "auto")

        x_matrix = x.reshape(-1, weight.shape[1])
        need_dx, need_dw = ctx.needs_input_grad[:2]
        delayed = scaling_state is not None
        paired_layouts = backward_enabled and (delayed or (backend == "triton" and x.device.type == "cuda"))
        paired_x = paired_layouts and need_dw and block_shapes["activation"] is None and (delayed or x_matrix.numel() > 4096)
        paired_w = paired_layouts and need_dx and block_shapes["weight"] is None and (delayed or weight.numel() > 4096)
        if delayed:
            qx, qx_transposed, _ = scaling_state.quantize(x_matrix, "activation", transposed=paired_x, training=state_training)
        elif paired_x:
            qx, qx_transposed = _quantize_tensorwise_pair(x_matrix, fp8_dtype=fp8_dtypes["activation"])
        else:
            qx = quantize_fp8(x_matrix, fp8_dtype=fp8_dtypes["activation"], block_shape=block_shapes["activation"])
        weight_quantizer = _quantize_tensorwise_pair if paired_w else quantize_fp8
        if delayed:
            def weight_quantizer(value, **kwargs):
                normal, transposed, _ = scaling_state.quantize(value, "weight", transposed=paired_w, training=state_training)
                return (normal, transposed) if paired_w else normal

            from ...scaling_strategy.delayed import checkpoint_recomputing, current_scaling_batch

            ctx.scaling_batch = current_scaling_batch()
            if checkpoint_recomputing():
                weight_cache = None
        if weight_cache is None:
            quantized_weight = weight_quantizer(weight, fp8_dtype=fp8_dtypes["weight"],
                                                block_shape=block_shapes["weight"])
        else:
            quantized_weight = weight_cache.quantize(weight, fp8_dtype=fp8_dtypes["weight"],
                                                     block_shape=block_shapes["weight"], quantizer=weight_quantizer,
                                                     paired=paired_w,
                                                     recipe=(scaling_state.cache_token, state_training) if delayed else None)
        qw, qw_transposed = quantized_weight if paired_w else (quantized_weight, None)

        output = gemm_nt(qx, qw, backend=backend, bias=bias, out_dtype=out_dtype, num_stages=num_stages)
        if not backward_enabled:
            return output.reshape(*x.shape[:-1], weight.shape[0])

        # Saved-activation compression belongs at this boundary: a future
        # codec must restore each operand with its scale and block metadata.
        # Keep tensor payloads in save_for_backward so PyTorch saved-tensor
        # hooks, offloading, and version checks remain available.
        save_qx_transpose = (
            need_dw
            and block_shapes["activation"] is not None
            and block_shapes["activation"][0] == 1
            and block_shapes["activation"][1] > 1
        )
        if save_qx_transpose:
            # Rowwise scaling is orientation-dependent. Quantize the original
            # transpose independently to retain reduction-axis scale blocks.
            qx = quantize_fp8(
                x_matrix.T, fp8_dtype=fp8_dtypes["activation"],
                block_shape=block_shapes["activation"]
            )
        elif paired_x:
            qx = qx_transposed

        ctx.save_for_backward(
            qx.data if need_dw else None,
            qx.scale_inv if need_dw else None,
            (qw_transposed.data if paired_w else qw.data) if need_dx else None,
            (qw_transposed.scale_inv if paired_w else qw.scale_inv) if need_dx else None
        )

        ctx.input_shape = x.shape
        ctx.input_dtype = x.dtype
        # Match the projection's compute precision. Autograd promotes dW when
        # accumulating into an FP32 master parameter's gradient.
        ctx.weight_grad_dtype = out_dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.out_features = weight.shape[0]
        ctx.fp8_dtypes = dict(fp8_dtypes)
        ctx.block_shapes = dict(block_shapes)
        ctx.backend = backend
        ctx.backward_backend = backend if fp8_trainig_plan["backward_backend"] == "inherit" else fp8_trainig_plan["backward_backend"]
        ctx.num_stages = num_stages
        ctx.save_qx_transpose = save_qx_transpose or paired_x
        ctx.weight_transposed = paired_w
        ctx.check_finite = backward_finite_check_policy()
        ctx.scaling_state, ctx.state_training = scaling_state, state_training

        return output.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if ctx.scaling_state is None:
            return _DOLinearFunction._backward(ctx, grad_output)
        from ...scaling_strategy.delayed import autograd_scaling_scope

        # CUDA autograd workers do not inherit Python ContextVars. Restore the
        # forward's owner before reading saved tensors (which may recompute).
        with autograd_scaling_scope(ctx.scaling_batch):
            return _DOLinearFunction._backward(ctx, grad_output)

    @staticmethod
    def _backward(ctx, grad_output):
        need_dx, need_dw, need_db = ctx.needs_input_grad[:3]
        x_data, x_scale_inv, w_data, w_scale_inv = ctx.saved_tensors

        grad_matrix = grad_output.reshape(-1, ctx.out_features)

        dx = None
        dw = None
        db = None
        if need_dx or need_dw:
            delayed = ctx.scaling_state is not None
            paired_g = delayed and need_dw or (need_dw and ctx.backend == "triton" and grad_matrix.device.type == "cuda"
                        and ctx.block_shapes["grad"] is None and grad_matrix.numel() > 4096)
            if delayed:
                qg, qg_transposed, db = ctx.scaling_state.quantize(
                    grad_matrix, "grad", normal=need_dx, transposed=need_dw, bias=need_db,
                    training=ctx.state_training, check_finite=ctx.check_finite)
            elif paired_g and need_dx:
                qg, qg_transposed = _quantize_tensorwise_pair(
                    grad_matrix, fp8_dtype=ctx.fp8_dtypes["grad"], check_finite=ctx.check_finite)
            elif paired_g:
                qg_transposed = _quantize_tensorwise_transpose(
                    grad_matrix, fp8_dtype=ctx.fp8_dtypes["grad"], check_finite=ctx.check_finite)
            else:
                qg = quantize_fp8(grad_matrix, fp8_dtype=ctx.fp8_dtypes["grad"],
                                  block_shape=ctx.block_shapes["grad"], check_finite=ctx.check_finite)

            if need_dx:
                qw = QuantizedTensor(
                    data=w_data, scale_inv=w_scale_inv, block_shape=ctx.block_shapes["weight"]
                )

                # Dense/sparse kernel selection belongs to backend dispatch;
                # gradient orientation and dtype remain the autograd contract.
                dx = gemm_nt(
                    qg, qw if ctx.weight_transposed else qw.transpose(contiguous=False),
                    backend=ctx.backward_backend,
                    out_dtype=ctx.input_dtype,
                    num_stages=ctx.num_stages,
                ).reshape(ctx.input_shape)
            
            if need_dw:
                qx = QuantizedTensor(
                    data=x_data, scale_inv=x_scale_inv, block_shape=ctx.block_shapes["activation"]
                )

                needs_qg_special_transpose = (
                    ctx.block_shapes["grad"] is not None
                    and ctx.block_shapes["grad"][0] == 1
                    and ctx.block_shapes["grad"][1] > 1
                )
                if needs_qg_special_transpose:
                    qg = quantize_fp8(
                        grad_matrix.T,
                        fp8_dtype=ctx.fp8_dtypes["grad"],
                        block_shape=ctx.block_shapes["grad"],
                        check_finite=ctx.check_finite,
                    )
                elif paired_g:
                    qg = qg_transposed

                dw = gemm_nt(
                    qg if needs_qg_special_transpose or paired_g else qg.transpose(contiguous=False),
                    qx if ctx.save_qx_transpose else qx.transpose(contiguous=False),
                    backend=ctx.backward_backend,
                    out_dtype=ctx.weight_grad_dtype,
                    num_stages=ctx.num_stages,
                )
        
        if need_db and db is None:
            db = grad_matrix.sum(dim=0, dtype=torch.float32).to(dtype=ctx.bias_dtype)
        elif need_db:
            db = db.to(dtype=ctx.bias_dtype)
        
        return dx, dw, db, None, None, None, None, None, None
