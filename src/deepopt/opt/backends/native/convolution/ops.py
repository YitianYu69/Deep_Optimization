"""Native PyTorch convolution kernels selected by spatial rank."""

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
    x = torch.empty(geometry.input_shape, dtype=grad_output.dtype,
                    device=grad_output.device, memory_format=input_memory_format)
    return native_backward(grad_output, x, weight, geometry, (True, False, False))[0]


def native_weight_gradient(x, grad_output, geometry, *, weight_memory_format=torch.contiguous_format):
    weight = torch.empty(geometry.weight_shape, dtype=grad_output.dtype,
                         device=grad_output.device, memory_format=weight_memory_format)
    return native_backward(grad_output, x, weight, geometry, (False, True, False))[1]
