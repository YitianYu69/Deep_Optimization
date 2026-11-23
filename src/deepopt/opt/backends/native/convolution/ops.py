"""Native PyTorch convolution kernels selected by spatial rank."""

from math import prod
from contextlib import nullcontext

import torch
from torch.nn import functional as F


def native_convolution(x, weight, bias, geometry):
    """Preserve PyTorch AMP and native convolution algorithm selection."""
    convolution = (F.conv1d, F.conv2d, F.conv3d)[geometry.rank - 1]
    return convolution(x, weight, bias, geometry.stride, geometry.padding,
                       geometry.dilation, geometry.groups)


def native_backward(grad_output, x, weight, geometry, output_mask, *, bias_sizes=None):
    """Select native convolution gradients using the original layout metadata.

    Values of ``x`` are unused unless the weight gradient is requested; values
    of ``weight`` are unused unless the input gradient is requested. Native
    kernels still inspect both tensors' strides when selecting an algorithm.
    """
    return torch.ops.aten.convolution_backward(
        grad_output, x, weight, bias_sizes, geometry.stride, geometry.padding,
        geometry.dilation, False, (0,) * geometry.rank, geometry.groups, output_mask,
    )


def native_input_gradient(weight, grad_output, geometry, *, input_memory_format=torch.contiguous_format):
    # On CPU, convolution's native input adjoint is a transposed convolution.
    # CUDA/explicit deterministic policies retain native_backward algorithm
    # selection until the alternate route has matching device qualification.
    if (_shape_only_cpu_eligible(grad_output)
            and grad_output.is_contiguous(memory_format=input_memory_format)):
        output_padding = tuple(size - ((output - 1) * stride - 2 * padding
            + dilation * (kernel - 1) + 1) for size, output, stride, padding, dilation, kernel in
            zip(geometry.input_spatial, geometry.output_spatial, geometry.stride,
                geometry.padding, geometry.dilation, geometry.kernel_size))
        if all(0 <= extra < stride for extra, stride in zip(output_padding, geometry.stride)):
            call = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[geometry.rank - 1]
            return call(grad_output, weight, None, geometry.stride, geometry.padding,
                        output_padding, geometry.groups, geometry.dilation)
    x = torch.empty(geometry.input_shape, dtype=grad_output.dtype,
                    device=grad_output.device, memory_format=input_memory_format)
    return native_backward(grad_output, x, weight, geometry, (True, False, False))[0]


def _shape_only_cpu_eligible(gradient):
    return (gradient.device.type == "cpu" and gradient.dtype in
            (torch.float32, torch.float64, torch.bfloat16)
            and not torch.are_deterministic_algorithms_enabled()
            and not torch.backends.cudnn.deterministic)


def _cuda_geometry_reason(geometry):
    """The narrow geometry whose CUDA descriptor change is opt-in."""
    from deepopt.opt.nn.convolution.geometry import ConvGeometry

    if type(geometry) is not ConvGeometry:
        return "ordinary_convolution_geometry_required"
    if (len(geometry.input_shape) != 4 or len(geometry.weight_shape) != 4
            or any(type(size) is not int or size <= 0
                   for size in (*geometry.input_shape, *geometry.weight_shape))):
        return "positive_batched_rank2_geometry_required"
    if (geometry.kernel_size != (1, 1) or geometry.stride != (1, 1)
            or geometry.dilation != (1, 1) or geometry.padding != (0, 0)
            or type(geometry.groups) is not int or geometry.groups != 1
            or any(geometry.synthetic_padding_right)):
        return "unsupported_convolution_geometry"
    if geometry.input_shape[1] != geometry.weight_shape[1]:
        return "convolution_channel_geometry_mismatch"
    return None


def _contiguous_stride(shape):
    stride, step = [], 1
    for size in reversed(shape):
        stride.insert(0, step)
        step *= size
    return tuple(stride)


def _cuda_geometry_backend_reason(descriptor=None, weight=None, geometry=None):
    """Require the cuDNN path that keeps dx-only input values untouched."""
    if not torch.backends.cudnn.enabled or not torch.backends.cudnn.is_available():
        return "enabled_available_cudnn_backend_required"
    selector = getattr(torch._C, "_select_conv_backend", None)
    backend_type = getattr(torch._C, "_ConvBackend", None)
    if not callable(selector) or backend_type is None or not hasattr(backend_type, "Cudnn"):
        return "native_convolution_backend_selector_unavailable"
    if descriptor is not None:
        backend = selector(descriptor, weight, None, list(geometry.stride), list(geometry.padding),
                           list(geometry.dilation), False, [0, 0], geometry.groups)
        if backend != backend_type.Cudnn:
            return "native_backend_not_cudnn:" + str(backend)
    return None


def _cuda_geometry_input_gradient(grad_output, weight, geometry, output_mask, *,
                                  input_memory_format, weight_memory_format,
                                  transposed, execution_info):
    """Use the one-element descriptor from Torch's ``nn.grad.conv2d_input``.

    This changes the input descriptor's strides, so it is deliberately a
    separate CUDA-only route. Never repair an unsupported input or fall back
    to a full-shape placeholder after this route has been selected.
    """
    reason = _cuda_geometry_reason(geometry)
    if reason is None and (not isinstance(output_mask, (tuple, list))
                           or tuple(output_mask) != (True, False, False)
                           or any(type(flag) is not bool for flag in output_mask)):
        reason = "dx_only_output_mask_required"
    if reason is None and (transposed or input_memory_format is not torch.contiguous_format
                           or weight_memory_format is not torch.contiguous_format):
        reason = "ordinary_nchw_convolution_required"
    if reason is None and (type(grad_output) is not torch.Tensor or type(weight) is not torch.Tensor):
        reason = "ordinary_dense_tensor_operands_required"
    if reason is None and (grad_output.device.type != "cuda" or weight.device != grad_output.device
                           or grad_output.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16):
        reason = "same_device_cuda_bf16_operands_required"
    if reason is None and (grad_output.layout is not torch.strided or weight.layout is not torch.strided
                           or tuple(grad_output.shape) != (geometry.batch_size, geometry.out_channels,
                                                          *geometry.output_spatial)
                           or tuple(weight.shape) != geometry.weight_shape
                           or tuple(grad_output.stride()) != _contiguous_stride(grad_output.shape)
                           or tuple(weight.stride()) != _contiguous_stride(weight.shape)):
        reason = "contiguous_nchw_operand_metadata_required"
    if reason is None:
        reason = _cuda_geometry_backend_reason()
    if reason is not None:
        if execution_info is not None:
            execution_info.update(input_bias_gradient_route="cuda_geometry_rejected",
                                  input_bias_gradient_fallback_reason=reason)
        raise ValueError("CUDA convolution dX geometry route: " + reason)

    # PyTorch 2.15's torch.nn.grad.conv2d_input uses exactly this descriptor.
    # Input values are unused for the strict dx-only mask; do not call
    # contiguous() on this expanded view or allocate the original input size.
    descriptor = grad_output.new_empty(1).expand(geometry.input_shape)
    reason = _cuda_geometry_backend_reason(descriptor, weight, geometry)
    if reason is not None:
        if execution_info is not None:
            execution_info.update(input_bias_gradient_route="cuda_geometry_rejected",
                                  input_bias_gradient_fallback_reason=reason)
        raise ValueError("CUDA convolution dX geometry route: " + reason)
    if execution_info is not None:
        execution_info.update(input_bias_gradient_route="cuda_geometry_selected",
                              input_bias_gradient_fallback_reason=None)
    values = native_backward(grad_output, descriptor, weight, geometry, (True, False, False))
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise ValueError("CUDA convolution dX geometry result has an invalid output mask ABI")
    dx = values[0]
    if (type(dx) is not torch.Tensor or tuple(dx.shape) != geometry.input_shape
            or dx.dtype != torch.bfloat16 or dx.device != grad_output.device
            or dx.layout is not torch.strided or tuple(dx.stride()) != _contiguous_stride(geometry.input_shape)
            or values[1] is not None or values[2] is not None):
        if execution_info is not None:
            execution_info.update(input_bias_gradient_route="cuda_geometry_rejected",
                                  input_bias_gradient_fallback_reason="result_metadata")
        raise ValueError("CUDA convolution dX geometry result violates the contiguous BF16 ABI")
    if execution_info is not None:
        execution_info.update(input_bias_gradient_route="cuda_geometry_shape_only",
            input_bias_gradient_fallback_reason=None,
            input_bias_gradient_metadata_storage_bytes=grad_output.element_size(),
            input_bias_gradient_full_placeholder_elided_bytes=prod(geometry.input_shape) * grad_output.element_size())
    return values


def native_input_bias_gradients(grad_output, weight, geometry, output_mask, *,
                                input_memory_format=torch.contiguous_format,
                                weight_memory_format=torch.contiguous_format,
                                transposed=False, execution_info=None, cuda_geometry=False):
    """Use qualified shape-only adjoints; preserve the native fallback recipe.

    Slow CPU depthwise/dilated bias reductions, mismatched upstream layouts
    and unqualified CUDA/deterministic policies keep the original joint
    native call and its layout-based algorithm choice. Report that fallback
    separately from buffer removal. The explicit CUDA geometry route instead
    rejects unsupported inputs before allocating its shape-only descriptor.
    """
    if cuda_geometry is not False:
        if cuda_geometry is not True:
            raise ValueError("cuda_geometry must be a boolean")
        return _cuda_geometry_input_gradient(grad_output, weight, geometry, output_mask,
            input_memory_format=input_memory_format, weight_memory_format=weight_memory_format,
            transposed=transposed, execution_info=execution_info)
    if len(output_mask) != 3 or output_mask[1]:
        raise ValueError("Input/bias gradient helper does not compute weight gradients")
    need_dx, _, need_db = output_mask
    depthwise_bias = (need_db and not transposed and geometry.rank < 3
        and geometry.groups == geometry.in_channels == geometry.out_channels
        and grad_output.dtype != torch.bfloat16)
    dilated_bias = need_db and any(dilation != 1 for dilation in geometry.dilation)
    # oneDNN transposed convolution can reduce the bias in a different order
    # from Tensor.sum. Preserve ATen's reduction when it can select oneDNN.
    onednn_transposed_bias = (need_db and transposed and grad_output.device.type == "cpu"
        and torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available())
    native_bias_rounding = depthwise_bias or dilated_bias or onednn_transposed_bias
    transposed_supported = (not transposed or geometry.rank == 1
                            or grad_output.dtype == torch.bfloat16)
    layout_supported = grad_output.is_contiguous(memory_format=input_memory_format)
    optimized = (_shape_only_cpu_eligible(grad_output) and not native_bias_rounding
                 and transposed_supported and layout_supported)
    if execution_info is not None:
        execution_info["input_bias_gradient_route"] = ("shape_only_cpu" if optimized
            else "native_backward_layout_fallback")
        execution_info["input_bias_gradient_fallback_reason"] = (None if optimized else
            "device_dtype_or_deterministic_policy" if not _shape_only_cpu_eligible(grad_output) else
            "upstream_layout" if not layout_supported else
            "native_bias_rounding" if native_bias_rounding else "transposed_dtype_rank")
    if optimized:
        dx = None
        if need_dx:
            if transposed:
                call = (F.conv1d, F.conv2d, F.conv3d)[geometry.rank - 1]
                dx = call(grad_output, weight, None, geometry.stride, geometry.padding,
                          geometry.dilation, geometry.groups)
                # output_padding >= stride is legal when dilation permits it.
                # Its virtual extra positions are outside the original source.
                dx = dx[(slice(None), slice(None), *(slice(0, size) for size in geometry.input_spatial))]
            else:
                dx = native_input_gradient(weight, grad_output, geometry,
                    input_memory_format=input_memory_format)
        db = grad_output.sum((0, *range(2, grad_output.ndim))) if need_db else None
        return dx, None, db
    x = torch.empty(geometry.input_shape, dtype=grad_output.dtype,
                    device=grad_output.device, memory_format=input_memory_format)
    operand = weight if need_dx else torch.empty(geometry.weight_shape,
        dtype=grad_output.dtype, device=grad_output.device, memory_format=weight_memory_format)
    if transposed:
        from ..convolution_transpose import native_backward as backward
    else:
        backward = native_backward
    return backward(grad_output, x, operand, geometry, output_mask,
                    bias_sizes=(geometry.out_channels,) if need_db else None)


def native_weight_gradient(x, grad_output, geometry, *, weight_memory_format=torch.contiguous_format):
    weight = torch.empty(geometry.weight_shape, dtype=grad_output.dtype,
                         device=grad_output.device, memory_format=weight_memory_format)
    return native_backward(grad_output, x, weight, geometry, (False, True, False))[1]


def _workspace_geometry_reason(geometry):
    """Two original VAE families with no crop or spatial descriptor change."""
    from deepopt.opt.nn.convolution.geometry import ConvGeometry

    if type(geometry) is not ConvGeometry or geometry.rank != 2:
        return "batched_rank2_convolution_required"
    if (any(type(size) is not int or size <= 0 for size in (*geometry.input_shape, *geometry.weight_shape))
            or geometry.stride != (1, 1) or geometry.dilation != (1, 1)
            or any(geometry.synthetic_padding_right)):
        return "positive_uncropped_stride1_geometry_required"
    pointwise = geometry.groups == 1 and geometry.kernel_size == (1, 1) and geometry.padding == (0, 0)
    depthwise = (geometry.groups == geometry.in_channels == geometry.out_channels
                 and geometry.kernel_size == (3, 3) and geometry.padding == (1, 1))
    return None if pointwise or depthwise else "ordinary_1x1_or_depthwise_3x3_required"


def _native_full_dw_tiled_dx(source, grad_output, weight, geometry, *, batch_size,
                            need_bias=False, execution_info=None, profile_ranges=False):
    """Overwrite only a freshly decoded, invocation-private source workspace.

    The compiler's trusted decode producer owns ``source``; this is not a
    public donation API. Full-batch dW and bias reduction finish their reads
    first. Native dX uses the corresponding source view only for metadata,
    then its bounded result is copied over that dead view on the same stream.
    This deliberately does not use convolution_backward.out or split dW.
    """
    from deepopt.opt.nn.convolution.geometry import ConvGeometry

    reason = _workspace_geometry_reason(geometry)
    if reason is None and (type(batch_size) is not int or not 0 < batch_size < geometry.batch_size):
        reason = "strict_partial_batch_bound_required"
    if reason is None and type(need_bias) is not bool:
        reason = "bias_mask_must_be_boolean"
    if reason is None and any(type(value) is not torch.Tensor for value in (source, grad_output, weight)):
        reason = "ordinary_dense_tensor_operands_required"
    if reason is None and (source.device.type not in ("cpu", "cuda")
            or any(value.device != source.device or value.dtype != torch.bfloat16
                   or value.layout is not torch.strided for value in (source, grad_output, weight))):
        reason = "same_device_bf16_operands_required"
    if reason is None and (tuple(source.shape) != geometry.input_shape
            or tuple(grad_output.shape) != geometry.output_shape or tuple(weight.shape) != geometry.weight_shape
            or any(tuple(value.stride()) != _contiguous_stride(value.shape) for value in (source, weight))):
        reason = "contiguous_nchw_operand_metadata_required"
    if reason is None and (source._base is not None or source.storage_offset() != 0
            or source.untyped_storage().nbytes() != source.numel() * source.element_size()
            or source.untyped_storage().data_ptr() in
                (grad_output.untyped_storage().data_ptr(), weight.untyped_storage().data_ptr())):
        reason = "private_full_decoded_storage_required"
    if reason is None and (torch.is_grad_enabled() or any(value.requires_grad for value in (source, grad_output, weight))):
        reason = "first_order_private_workspace_required"
    if reason is not None:
        raise ValueError("Private convolution workspace: " + reason)

    # Keep the existing complete reduction and BF16 result, including the
    # native weight-layout descriptor, before any source byte is overwritten.
    profiling = profile_ranges or torch.autograd._profiler_enabled()
    def phase(name):
        return torch.profiler.record_function("deepopt.convolution.private_workspace." + name) if profiling else nullcontext()
    with phase("full_dw"):
        dw = native_weight_gradient(source, grad_output, geometry)
    with phase("full_db") if need_bias else nullcontext():
        db = (native_backward(grad_output, source, weight, geometry, (False, False, True),
                              bias_sizes=(geometry.out_channels,))[2] if need_bias else None)
    if (type(dw) is not torch.Tensor or tuple(dw.shape) != geometry.weight_shape
            or dw.dtype != torch.bfloat16 or dw.device != source.device
            or need_bias and (type(db) is not torch.Tensor or tuple(db.shape) != (geometry.out_channels,)
                             or db.dtype != torch.bfloat16 or db.device != source.device)):
        raise ValueError("Private convolution workspace full-batch reduction ABI changed")

    largest = 0
    for first in range(0, geometry.batch_size, batch_size):
        count = min(batch_size, geometry.batch_size - first)
        shape = (count, *geometry.input_shape[1:])
        part_geometry = ConvGeometry(shape, geometry.weight_shape, geometry.stride, geometry.padding,
                                    geometry.dilation, geometry.groups)
        destination = source.narrow(0, first, count)
        # dX does not read source values. Supplying the private view avoids a
        # separate full/partial shape-only metadata allocation.
        with phase("partial_dx"):
            partials = native_backward(grad_output.narrow(0, first, count), destination, weight,
                                      part_geometry, (True, False, False))
        if not isinstance(partials, (tuple, list)) or len(partials) != 3 or any(value is not None for value in partials[1:]):
            raise ValueError("Private convolution workspace partial mask ABI changed")
        partial = partials[0]
        del partials
        if (type(partial) is not torch.Tensor or tuple(partial.shape) != shape
                or partial.dtype != source.dtype or partial.device != source.device
                or tuple(partial.stride()) != _contiguous_stride(shape)
                or partial.untyped_storage().data_ptr() in
                    (source.untyped_storage().data_ptr(), grad_output.untyped_storage().data_ptr(),
                     weight.untyped_storage().data_ptr())):
            raise ValueError("Private convolution workspace partial dX ABI changed")
        largest = max(largest, partial.numel() * partial.element_size())
        with phase("partial_copy"):
            destination.copy_(partial)
        del partial, destination
    if execution_info is not None:
        execution_info.update(convolution_workspace_route="full_dw_tiled_dx",
            convolution_workspace_bytes=source.numel() * source.element_size(),
            convolution_dx_batch_size=batch_size,
            convolution_dx_slices=(geometry.batch_size + batch_size - 1) // batch_size,
            convolution_partial_dx_max_bytes=largest, convolution_metadata_placeholder_bytes=0,
            convolution_partial_upstream_layout_upper_bytes=batch_size * prod(geometry.output_shape[1:]) * grad_output.element_size(),
            convolution_dw_full_batch=True, convolution_db_full_batch=need_bias,
            convolution_vendor_workspace_bytes=None, convolution_destination_is_decoded_source=True)
    return source, dw, db
