"""Native global average reduction with shape-only backward state."""

from deepopt.kernel_policy import kernel_forward, kernel_backward

from torch.autograd import Function
from torch.autograd.function import once_differentiable

from .._output import restore_fresh_output


class GlobalAverageFunction(Function):
    @staticmethod
    @kernel_forward
    def forward(ctx, x, geometry, options, compression, backward, execution_info):
        ctx.shape, ctx.dtype = tuple(x.shape), x.dtype
        ctx.elements = geometry.input_elements
        # Torch uses the native mean reduction for this geometry. The custom
        # owner retains no input tensor, including when followed by dropout.
        output = geometry.spec.native(x, **options)
        ctx.save_for_backward()
        if execution_info is not None:
            execution_info.update(backend="native", family=geometry.family, dimensions=geometry.rank,
                input=str(x.dtype), output=str(output.dtype), compression=compression,
                compression_strategy="eliminate_saved_activations", compression_active=compression and backward,
                saved_state_bytes=0, retained_storage_bytes=0, saved_raw_input_bytes=0,
                routing_bits=0, routing_bytes=0, coefficient_bytes=0, index_guard_bytes=0,
                sample_bytes=0, factor_bytes=0, forward_variant="native_global_average")
        return restore_fresh_output(output, lambda value: value)

    @staticmethod
    @once_differentiable
    @kernel_backward
    def backward(ctx, gradient):
        return (gradient.to(ctx.dtype) / ctx.elements).expand(ctx.shape), None, None, None, None, None
