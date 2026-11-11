"""First-order pooling autograd; no raw-input activation shadows."""

from contextlib import nullcontext

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ....backends.pooling import executor
from ....precision.gradient import capture_gradient
from ....quantization.gradient import enabled
from ....quantization.activation_tensor import FP8Activation
from ....quantization.activation import PackedActivation
from ....quantization.indices import PackedIndices
from ...._execution import backward_finite_check_policy


def storage_bytes(tensors):
    storages = {t.untyped_storage()._cdata: t.untyped_storage() for t in tensors if t is not None}
    return sum(s.nbytes() for s in storages.values())


class PoolFunction(Function):
    @staticmethod
    def forward(ctx, x, g, settings, scaling_state, state_training, execution_info):
        ctx.gradient_slot, ctx.gradient_check = capture_gradient(x), backward_finite_check_policy()
        ctx.geometry, ctx.backend, ctx.input_dtype = g, settings["backend"], x.dtype
        ctx.execution_info = execution_info
        arguments = {key: value for key, value in settings.items() if key not in ("backend", "logical_dtype")}
        compute = executor(x, ctx.backend)
        if scaling_state is not None and settings["dtype"] in (torch.float8_e4m3fn, torch.float8_e5m2):
            result = scaling_state.activation_forward(x, g, compute, arguments, training=state_training)
        else:
            result = compute.forward(x, g, **arguments)
        output, inverse, state, _, _, info = result
        routing, coefficient = state.routing, state.coefficient
        ctx.route_meta = (routing.shape, routing.value_count) if isinstance(routing, PackedIndices) else None
        ctx.coefficient_meta = (coefficient.shape, coefficient.group_size, coefficient.block_shape, coefficient.bits) if isinstance(coefficient, PackedActivation) else None
        ctx.coefficient_layout = state.coefficient_layout
        guard = settings["options"].get("indices") if settings["need_state"] else None
        public = state.public_indices
        saved = (routing.data if isinstance(routing, PackedIndices) else routing,
                 coefficient.data if isinstance(coefficient, PackedActivation) else coefficient,
                 coefficient.step if isinstance(coefficient, PackedActivation) else None,
                 coefficient.minimum if isinstance(coefficient, PackedActivation) else None,
                 state.factor, public if settings["need_state"] else None, guard, state.samples)
        ctx.save_for_backward(*saved)
        if execution_info is not None:
            compression = settings["compression"]["enabled"]
            target = ("eliminate_saved_activations" if "average" in g.family else
                      "pack_routing" if g.routes else "quantize_pack_derivative")
            execution_info.update(info, family=g.family, dimensions=g.rank, input="fp8" if isinstance(x, FP8Activation) else str(x.dtype),
                                  output="fp8" if inverse is not None else str(settings["dtype"]),
                                  compression=compression, compression_strategy=target if compression else "ordinary_state",
                                  compression_active=compression and settings["need_state"],
                                  routing_bits=routing.bits if isinstance(routing, PackedIndices) else 64 if routing is not None else 0,
                                  routing_bytes=saved[0].numel() * saved[0].element_size() if saved[0] is not None else 0,
                                  coefficient_bytes=sum(t.numel() * t.element_size() for t in saved[1:4] if t is not None),
                                  index_guard_bytes=storage_bytes((saved[5], guard)),
                                  sample_bytes=state.samples.numel() * state.samples.element_size() if state.samples is not None else 0,
                                  sample_source=("explicit" if settings["options"].get("_random_samples") is not None else "generated") if g.family == "fractional_max" else None,
                                  factor_bytes=state.factor.numel() * state.factor.element_size() if state.factor is not None else 0,
                                  saved_state_bytes=sum(t.numel() * t.element_size() for t in saved if t is not None),
                                  retained_storage_bytes=storage_bytes(saved), saved_raw_input_bytes=0)
        value = (FP8Activation(output, inverse, g.output_shape, settings["block_shape"], g.layout, settings["logical_dtype"])
                 if inverse is not None else g.transport.restore(output))
        if public is not None:
            ctx.mark_non_differentiable(public)
            return value, public
        return value

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient, index_gradient=None):
        data, coefficient, step, minimum, factor, _, _, samples = ctx.saved_tensors
        routing = PackedIndices(data, *ctx.route_meta) if ctx.route_meta is not None else data
        if ctx.coefficient_meta is not None:
            coefficient = PackedActivation(coefficient, step, minimum, *ctx.coefficient_meta)
        args = dict(routing=routing, coefficient=coefficient, factor=factor, samples=samples,
                    coefficient_layout=ctx.coefficient_layout, input_dtype=ctx.input_dtype, execution_info=ctx.execution_info)
        with torch.autocast(gradient.device.type, enabled=False) if gradient.device.type != "meta" else nullcontext():
            compute = executor(gradient, ctx.backend)
            if enabled(ctx.gradient_slot):
                with ctx.gradient_slot.output(ctx.geometry.shape, ctx.input_dtype, check_finite=ctx.gradient_check) as out:
                    dx = compute.backward(gradient, ctx.geometry, gradient_output=out, **args)
            else:
                dx = compute.backward(gradient, ctx.geometry, **args)
        if ctx.execution_info is not None:
            ctx.execution_info.update(dy="fp8" if isinstance(gradient, FP8Activation) else str(gradient.dtype),
                                      dx="fp8" if isinstance(dx, FP8Activation) else str(dx.dtype))
        return dx, None, None, None, None, None
