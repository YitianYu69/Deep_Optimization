"""FP8 transposed convolution with unique-source activation compression."""

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ....backends import convolution_transpose as executor
from ....quantization.activation import PackedActivation, dequantize_unpack_activation
from ....quantization.activation_fp8 import quantize_pack_fp8_activation
from ....quantization.quantizer import quantize_fp8
from ....quantization.tensor import QuantizedTensor
from ....quantization.activation_tensor import FP8Activation
from ...._execution import backward_finite_check_policy
from ....precision.gradient import capture_gradient, record_gradient_input
from ....quantization.gradient import enabled, gradient_operand, gradient_matrix, gradient_bias
from ...convolution import restore_matrix, source_matrix
from ...convolution.geometry import _contiguous
from .native import _memory_format


class _FP8ConvTransposeFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, geometry, plan, out_dtype, backward_enabled,
                weight_cache, scaling_state, state_training, output_slot):
        settings, blocks = dict(plan["fp8_training"]), dict(plan["block_shapes"])
        ctx.gradient_slot = capture_gradient(x)
        types = dict(settings["fp8_dtypes"])
        settings["fp8_dtypes"] = types
        compression = settings["activation_compression"]
        need_dx, need_dw = ctx.needs_input_grad[:2]
        compress = backward_enabled and need_dw and compression["enabled"]
        supplied = x.quantized("channels") if isinstance(x, FP8Activation) else None
        matrix = None if supplied is not None else source_matrix(x)
        if supplied is not None:
            blocks["activation"] = supplied.block_shape
            types["activation"] = supplied.data.dtype
        packed = None
        if supplied is not None:
            qx = supplied
            if compress:
                from ....quantization.activation import quantize_pack_activation

                packed = quantize_pack_activation(qx.data, compression["group_size"], qx.block_shape,
                                         bits=compression.get("bits", 2))
        elif scaling_state is not None:
            if compress:
                qx, packed = scaling_state.quantize_pack_activation(
                    matrix, group_size=compression["group_size"], bits=compression.get("bits", 2),
                    training=state_training)
            else:
                qx, _, _ = scaling_state.quantize(matrix, "activation", training=state_training)
        elif compress:
            qx, packed = quantize_pack_fp8_activation(
                matrix, fp8_dtype=types["activation"], block_shape=blocks["activation"],
                group_size=compression["group_size"], bits=compression.get("bits", 2))
        else:
            qx = quantize_fp8(matrix, fp8_dtype=types["activation"], block_shape=blocks["activation"])

        recipe = None

        def quantizer(value, *, fp8_dtype, block_shape):
            matrix = value.reshape(geometry.weight_matrix_shape)
            if scaling_state is not None:
                return scaling_state.quantize(matrix, "weight", training=state_training)[0]
            return quantize_fp8(matrix, fp8_dtype=fp8_dtype, block_shape=block_shape)

        if scaling_state is not None:
            from ....scaling.delayed import checkpoint_recomputing, current_scaling_batch

            ctx.scaling_batch = current_scaling_batch()
            if checkpoint_recomputing():
                weight_cache = None
            recipe = (scaling_state.cache_token, state_training)

        if weight_cache is None:
            qw = quantizer(weight, fp8_dtype=types["weight"], block_shape=blocks["weight"])
        else:
            qw = weight_cache.quantize(weight, fp8_dtype=types["weight"],
                                       block_shape=blocks["weight"], quantizer=quantizer, recipe=recipe)
        from contextlib import nullcontext

        output_scope = (output_slot.output(geometry.output_shape, out_dtype, training=state_training)
                        if output_slot is not None else nullcontext(None))
        with output_scope as planned_output:
            output = executor.forward(qx, qw, geometry, backend=settings["backend"], bias=bias,
                                      out_dtype=torch.float32 if planned_output is not None else out_dtype,
                                      num_stages=settings["num_stages"], output=planned_output)
        if not backward_enabled:
            return output if isinstance(output, FP8Activation) else restore_matrix(output, geometry.output_shape, _memory_format(x))
        ctx.save_for_backward(
            (packed.data if compress else qx.data) if need_dw else None,
            qx.scale_inv if need_dw else None,
            qw.data if need_dx else None,
            qw.scale_inv if need_dx else None,
            packed.step if compress else None,
            packed.minimum if compress else None,
        )
        ctx.geometry, ctx.settings, ctx.blocks = geometry, dict(settings), dict(blocks)
        ctx.compress, ctx.compression = compress, dict(compression)
        ctx.input_dtype, ctx.weight_dtype, ctx.compute_dtype = x.dtype, weight.dtype, out_dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.input_format, ctx.weight_format = _memory_format(x), _memory_format(weight)
        ctx.scaling_state, ctx.state_training = scaling_state, state_training
        ctx.check_finite = backward_finite_check_policy()
        return output if isinstance(output, FP8Activation) else restore_matrix(output, geometry.output_shape, ctx.input_format)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if ctx.scaling_state is None:
            return _FP8ConvTransposeFunction._backward(ctx, grad_output)
        from ....scaling.delayed import autograd_scaling_scope

        with autograd_scaling_scope(ctx.scaling_batch):
            return _FP8ConvTransposeFunction._backward(ctx, grad_output)

    @staticmethod
    def _backward(ctx, grad_output):
        x_data, x_scale, w_data, w_scale, step, minimum = ctx.saved_tensors
        need_dx, need_dw, need_db = ctx.needs_input_grad[:3]
        settings, blocks, geometry = ctx.settings, ctx.blocks, ctx.geometry
        dx = dw = db = None
        supplied = gradient_operand(grad_output, "channels")
        record_gradient_input(ctx.gradient_slot, reused=supplied is not None)
        matrix = (None if supplied is not None else gradient_matrix(grad_output, "channels")
                  if isinstance(grad_output, FP8Activation) else source_matrix(grad_output))
        if need_dx or need_dw:
            if supplied is not None:
                qg = supplied
            elif ctx.scaling_state is not None:
                qg, _, db = ctx.scaling_state.quantize(
                    matrix, "grad", bias=False, training=ctx.state_training, check_finite=ctx.check_finite)
            else:
                qg = quantize_fp8(matrix, fp8_dtype=settings["fp8_dtypes"]["grad"],
                                  block_shape=blocks["grad"], check_finite=ctx.check_finite)
            backend = settings["backend"] if settings["backward_backend"] == "inherit" else settings["backward_backend"]
            if need_dx:
                qw = QuantizedTensor(w_data, w_scale, blocks["weight"])
                if enabled(ctx.gradient_slot):
                    with ctx.gradient_slot.output(geometry.input_shape, ctx.input_dtype, check_finite=ctx.check_finite) as output:
                        dx = executor.input_gradient(qg, qw, geometry, backend=backend,
                                                     out_dtype=torch.float32, num_stages=settings["num_stages"],
                                                     gradient_output=output)
                else:
                    values = executor.input_gradient(qg, qw, geometry, backend=backend,
                                                     out_dtype=ctx.input_dtype, num_stages=settings["num_stages"])
                    dx = restore_matrix(values, geometry.input_shape, ctx.input_format)
            if need_dw:
                if ctx.compress:
                    packed = PackedActivation(x_data, step, minimum, geometry.input_matrix_shape,
                                              ctx.compression["group_size"], blocks["activation"],
                                              ctx.compression.get("bits", 2))
                    x_data = dequantize_unpack_activation(packed, dtype=settings["fp8_dtypes"]["activation"])
                qx = QuantizedTensor(x_data, x_scale, blocks["activation"])
                values = executor.weight_gradient(qx, qg, geometry, backend=backend,
                                                  out_dtype=ctx.compute_dtype, num_stages=settings["num_stages"])
                # Like linear FP8, accumulate compute-precision dW into the
                # caller's master-parameter dtype through autograd promotion.
                dw = values.reshape(geometry.weight_shape).to(ctx.weight_dtype)
                dw = _contiguous(dw, ctx.weight_format)
        if need_db:
            db = (gradient_bias(supplied) if supplied is not None else
                  grad_output.sum((0, *range(2, grad_output.ndim)), dtype=torch.float32)).to(ctx.bias_dtype)
        return dx, dw, db, None, None, None, None, None, None, None, None
