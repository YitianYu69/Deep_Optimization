"""Convolution oracles over independently quantized source operands."""

import torch

from deepopt.opt.backends.native.convolution import native_convolution, native_input_gradient, native_weight_gradient
from deepopt.opt.nn.convolution import restore_matrix, source_matrix
from deepopt.opt.nn.convolution.geometry import zero_synthetic_padding


@torch.no_grad()
def forward(qx, qw, geometry, *, bias=None, out_dtype=torch.float32, num_stages="auto"):
    with torch.autocast(qx.data.device.type, enabled=False):
        x = restore_matrix(qx.dequantize(), geometry.input_shape)
        zero_synthetic_padding(x, geometry)
        weight = qw.dequantize().reshape(geometry.weight_shape)
        result = native_convolution(x, weight, None if bias is None else bias.float(), geometry)
        return source_matrix(result).to(out_dtype)


@torch.no_grad()
def input_gradient(qg, qw, geometry, *, out_dtype=torch.float32, num_stages="auto"):
    with torch.autocast(qg.data.device.type, enabled=False):
        grad = restore_matrix(qg.dequantize(), geometry.output_shape)
        weight = qw.dequantize().reshape(geometry.weight_shape)
        return source_matrix(native_input_gradient(weight, grad, geometry)).to(out_dtype)


@torch.no_grad()
def weight_gradient(qx, qg, geometry, *, out_dtype=torch.float32, num_stages="auto"):
    with torch.autocast(qx.data.device.type, enabled=False):
        x = restore_matrix(qx.dequantize(), geometry.input_shape)
        zero_synthetic_padding(x, geometry)
        grad = restore_matrix(qg.dequantize(), geometry.output_shape)
        return native_weight_gradient(x, grad, geometry).reshape(geometry.weight_matrix_shape).to(out_dtype)
