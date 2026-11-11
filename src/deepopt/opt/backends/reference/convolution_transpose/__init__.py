"""Explicit dequantization oracles independent of implicit FP8 addressing."""

import torch

from ....nn.convolution import restore_matrix, source_matrix
from ...native.convolution_transpose import native_convolution, native_backward, native_weight_gradient


@torch.no_grad()
def forward(qx, qw, geometry, *, bias=None, out_dtype=torch.float32, num_stages="auto", output=None):
    with torch.autocast(qx.data.device.type, enabled=False):
        x = restore_matrix(qx.dequantize(), geometry.input_shape)
        w = qw.dequantize().reshape(geometry.weight_shape)
        y = source_matrix(native_convolution(x, w, None if bias is None else bias.float(), geometry))
        return output.convert(y, matrix=True) if output is not None else y.to(out_dtype)


@torch.no_grad()
def input_gradient(qg, qw, geometry, *, out_dtype=torch.float32, num_stages="auto", gradient_output=None):
    with torch.autocast(qg.data.device.type, enabled=False):
        grad = restore_matrix(qg.dequantize(), geometry.output_shape)
        w = qw.dequantize().reshape(geometry.weight_shape)
        dummy = torch.empty(geometry.input_shape, dtype=grad.dtype, device=grad.device)
        value = source_matrix(native_backward(grad, dummy, w, geometry, (True, False, False))[0])
        return gradient_output.convert(value, matrix=True) if gradient_output is not None else value.to(out_dtype)


@torch.no_grad()
def weight_gradient(qx, qg, geometry, *, out_dtype=torch.float32, num_stages="auto"):
    with torch.autocast(qx.data.device.type, enabled=False):
        x = restore_matrix(qx.dequantize(), geometry.input_shape)
        grad = restore_matrix(qg.dequantize(), geometry.output_shape)
        return native_weight_gradient(x, grad, geometry).reshape(geometry.weight_matrix_shape).to(out_dtype)
