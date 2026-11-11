"""Native transposed compute and independently requested gradients."""

import torch
from torch.nn import functional as F


def native_convolution(x, weight, bias, geometry):
    call = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[geometry.rank - 1]
    return call(x, weight, bias, geometry.stride, geometry.padding, geometry.output_padding,
                geometry.groups, geometry.dilation)


def native_backward(grad_output, x, weight, geometry, output_mask, *, bias_sizes=None):
    return torch.ops.aten.convolution_backward(
        grad_output, x, weight, bias_sizes, geometry.stride, geometry.padding,
        geometry.dilation, True, geometry.output_padding, geometry.groups, output_mask)


def native_weight_gradient(x, grad_output, geometry, *, weight_memory_format=torch.contiguous_format):
    weight = torch.empty(geometry.weight_shape, device=x.device, dtype=x.dtype,
                         memory_format=weight_memory_format)
    return native_backward(grad_output, x, weight, geometry, (False, True, False))[1]
