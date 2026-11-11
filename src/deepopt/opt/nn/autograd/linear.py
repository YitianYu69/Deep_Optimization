import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ...backends import gemm_nt
from ...quantization.tensor import QuantizedTensor
from ...quantization.quantizer import quantize_fp8, _quantize_tensorwise_pair, _quantize_tensorwise_transpose
from ..._execution import backward_finite_check_policy
from ...quantization.activation_tensor import FP8Activation
from ...precision.gradient import capture_gradient, record_gradient_input
from ...quantization.gradient import enabled, gradient_operand, gradient_matrix, gradient_transpose, gradient_bias


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
        ctx.gradient_slot = capture_gradient(x)
        fp8_dtypes = dict(fp8_trainig_plan["fp8_dtypes"])
        block_shapes = dict(plan["block_shapes"])
        backend = fp8_trainig_plan.get("backend", "reference")
        num_stages = fp8_trainig_plan.get("num_stages", "auto")

        supplied = x.quantized("last") if isinstance(x, FP8Activation) else None
        x_matrix = None if supplied is not None else x.reshape(-1, weight.shape[1])
        if supplied is not None:
            block_shapes["activation"] = supplied.block_shape
            fp8_dtypes["activation"] = supplied.data.dtype
        need_dx, need_dw = ctx.needs_input_grad[:2]
        compression = fp8_trainig_plan.get("activation_compression", {})
        compress_activation = backward_enabled and need_dw and compression.get("enabled", False)
        save_qx_transpose = (
            need_dw
            and supplied is None and block_shapes["activation"] is not None
            and block_shapes["activation"][0] == 1
            and block_shapes["activation"][1] > 1
        )
        delayed = scaling_state is not None
        paired_layouts = backward_enabled and (delayed or (backend == "triton" and x.device.type == "cuda"))
        paired_x = (supplied is None and not compress_activation and paired_layouts and need_dw
                    and block_shapes["activation"] is None and (delayed or x_matrix.numel() > 4096))
        paired_w = paired_layouts and need_dx and block_shapes["weight"] is None and (delayed or weight.numel() > 4096)
        if supplied is not None:
            qx = supplied
            if compress_activation:
                from ...quantization.activation import quantize_pack_activation

                packed_x = quantize_pack_activation(qx.data, compression["group_size"], qx.block_shape,
                                           bits=compression.get("bits", 2))
        elif compress_activation and delayed:
            qx, packed_x = scaling_state.quantize_pack_activation(
                x_matrix, group_size=compression["group_size"], training=state_training,
                bits=compression.get("bits", 2))
        elif compress_activation and not save_qx_transpose:
            from ...quantization.activation_fp8 import quantize_pack_fp8_activation

            qx, packed_x = quantize_pack_fp8_activation(
                x_matrix, fp8_dtype=fp8_dtypes["activation"], block_shape=block_shapes["activation"],
                group_size=compression["group_size"], bits=compression.get("bits", 2))
        elif delayed:
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

            from deepopt.opt.scaling.delayed import checkpoint_recomputing, current_scaling_batch

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

        if save_qx_transpose:
            # Rowwise scaling is orientation-dependent. Quantize the original
            # transpose independently to retain reduction-axis scale blocks.
            if compress_activation:
                from ...quantization.activation_fp8 import quantize_pack_fp8_activation

                qx, packed_x = quantize_pack_fp8_activation(
                    x_matrix.T, fp8_dtype=fp8_dtypes["activation"],
                    block_shape=block_shapes["activation"], group_size=compression["group_size"],
                    bits=compression.get("bits", 2))
            else:
                qx = quantize_fp8(
                    x_matrix.T, fp8_dtype=fp8_dtypes["activation"],
                    block_shape=block_shapes["activation"]
                )
        elif paired_x:
            qx = qx_transposed

        # Payloads, including the codec metadata, remain visible to saved-tensor
        # hooks/offloading/version checks. Context retains only shape settings.
        codec_tensors = (packed_x.step, packed_x.minimum) if compress_activation else ()
        ctx.save_for_backward(
            (packed_x.data if compress_activation else qx.data) if need_dw else None,
            qx.scale_inv if need_dw else None,
            (qw_transposed.data if paired_w else qw.data) if need_dx else None,
            (qw_transposed.scale_inv if paired_w else qw.scale_inv) if need_dx else None,
            *codec_tensors,
        )
        ctx.compress_activation = compress_activation
        if compress_activation:
            ctx.compression_group_size = packed_x.group_size
            ctx.compression_shape = packed_x.shape
            ctx.compression_block_shape = packed_x.block_shape
            ctx.compression_saved_transpose = save_qx_transpose
            ctx.compression_bits = packed_x.bits

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
        from deepopt.opt.scaling.delayed import autograd_scaling_scope

        # CUDA autograd workers do not inherit Python ContextVars. Restore the
        # forward's owner before reading saved tensors (which may recompute).
        with autograd_scaling_scope(ctx.scaling_batch):
            return _DOLinearFunction._backward(ctx, grad_output)

    @staticmethod
    def _backward(ctx, grad_output):
        need_dx, need_dw, need_db = ctx.needs_input_grad[:3]
        x_data, x_scale_inv, w_data, w_scale_inv, *codec_tensors = ctx.saved_tensors

        gradient_layout = getattr(ctx, "gradient_layout", "last")
        supplied = gradient_operand(grad_output, gradient_layout)
        grad_blocks = supplied.block_shape if supplied is not None else ctx.block_shapes["grad"]
        record_gradient_input(ctx.gradient_slot, reused=supplied is not None,
                              transpose=bool(need_dw and supplied is not None and grad_blocks is not None
                                             and grad_blocks[0] == 1 and grad_blocks[1] > 1))
        if supplied is not None:
            grad_matrix = None
        elif gradient_layout == "channels" or isinstance(grad_output, FP8Activation):
            grad_matrix = gradient_matrix(grad_output, gradient_layout)
        else:
            grad_matrix = grad_output.reshape(-1, ctx.out_features)

        dx = None
        dw = None
        db = None
        if need_dx or need_dw:
            delayed = ctx.scaling_state is not None
            paired_g = supplied is None and (delayed and need_dw or (need_dw and ctx.backend == "triton" and grad_matrix.device.type == "cuda"
                        and ctx.block_shapes["grad"] is None and grad_matrix.numel() > 4096))
            if supplied is not None:
                qg = supplied
            elif delayed:
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
                if enabled(ctx.gradient_slot):
                    with ctx.gradient_slot.output(ctx.input_shape, ctx.input_dtype, check_finite=ctx.check_finite) as output:
                        dx = gemm_nt(qg, qw if ctx.weight_transposed else qw.transpose(contiguous=False),
                                     backend=ctx.backward_backend, out_dtype=torch.float32,
                                     num_stages=ctx.num_stages, gradient_output=output)
                else:
                    dx = gemm_nt(qg, qw if ctx.weight_transposed else qw.transpose(contiguous=False),
                                 backend=ctx.backward_backend, out_dtype=ctx.input_dtype,
                                 num_stages=ctx.num_stages)
                    if gradient_layout == "channels":
                        from ...quantization.activation_tensor import restore_activation

                        dx = restore_activation(dx, ctx.input_shape, "channels")
                    else:
                        dx = dx.reshape(ctx.input_shape)

            if need_dw:
                if ctx.compress_activation:
                    from ...quantization.activation import PackedActivation, dequantize_unpack_activation

                    packed = PackedActivation(x_data, *codec_tensors, ctx.compression_shape,
                                              ctx.compression_group_size, ctx.compression_block_shape,
                                              ctx.compression_bits)
                    transpose = not ctx.compression_saved_transpose
                    data = dequantize_unpack_activation(packed, dtype=ctx.fp8_dtypes["activation"], transpose=transpose)
                    blocks = ctx.compression_block_shape
                    if transpose and blocks is not None:
                        blocks = (blocks[1], blocks[0])
                    qx = QuantizedTensor(data, x_scale_inv.T if transpose else x_scale_inv, blocks)
                else:
                    qx = QuantizedTensor(
                        data=x_data, scale_inv=x_scale_inv, block_shape=ctx.block_shapes["activation"]
                    )

                needs_qg_special_transpose = (
                    grad_blocks is not None
                    and grad_blocks[0] == 1
                    and grad_blocks[1] > 1
                )
                if needs_qg_special_transpose:
                    qg = (gradient_transpose(grad_output, ctx.fp8_dtypes["grad"], grad_blocks, ctx.check_finite,
                                             layout=gradient_layout)
                          if supplied is not None else quantize_fp8(grad_matrix.T, fp8_dtype=ctx.fp8_dtypes["grad"],
                                                                 block_shape=grad_blocks, check_finite=ctx.check_finite))
                elif paired_g:
                    qg = qg_transposed

                dw = gemm_nt(
                    qg if needs_qg_special_transpose or paired_g else qg.transpose(contiguous=False),
                    qx if ctx.compress_activation or ctx.save_qx_transpose else qx.transpose(contiguous=False),
                    backend=ctx.backward_backend,
                    out_dtype=ctx.weight_grad_dtype,
                    num_stages=ctx.num_stages,
                )

        if need_db and db is None:
            db = (gradient_bias(supplied) if supplied is not None else
                  grad_matrix.sum(dim=0, dtype=torch.float32)).to(dtype=ctx.bias_dtype)
        elif need_db:
            db = db.to(dtype=ctx.bias_dtype)

        return dx, dw, db, None, None, None, None, None, None
