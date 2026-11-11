"""Pooling oracle with no saved raw inputs and exact/processed saved state."""

from dataclasses import replace
from math import isinf

import torch
from torch._subclasses.fake_tensor import FakeTensor

from deepopt.opt.nn.activation.geometry import ReLUGeometry
from deepopt.opt.nn.pooling.state import PoolState
from deepopt.opt.quantization.indices import pack_indices, unpack_indices, empty_indices, PackedIndices
from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation, PackedActivation
from deepopt.opt.quantization.activation_tensor import FP8Activation
from .maximum import forward as maximum
from .power import forward as power
from .output import convert
from .windows import average_backward, decode_locations


def dense(value):
    return value.dequantize(torch.float32) if isinstance(value, FP8Activation) else value if value.dtype == torch.float64 else value.float()


def coefficient_geometry(x, options):
    layout = x._activation_layout if isinstance(x, FP8Activation) else "last"
    transport = ReLUGeometry(tuple(x.shape), layout)
    block = x._block_shape if isinstance(x, FP8Activation) else options.get("input_block_shape")
    return transport, block


def forward(x, g, *, options, dtype, block_shape, compression, check_finite, need_state,
            input_block_shape=None, samples=None, scale=None, observe=False, capture_scale=False):
    fake = isinstance(getattr(x, "_data", x), FakeTensor) or x.device.type == "meta"
    work_dtype = torch.float64 if getattr(x, "_data", x).dtype == torch.float64 else torch.float32
    public = codes = coefficient = factor = routing = None
    coeff_transport, coeff_block = coefficient_geometry(x, dict(compression=compression, input_block_shape=input_block_shape))
    if fake:
        values = torch.empty(g.output_shape, device=x.device, dtype=work_dtype)
        if options.get("return_indices", False):
            public = torch.empty(g.output_shape, device=x.device, dtype=torch.int64)
        if need_state and g.routes:
            shape = ReLUGeometry(g.shape, g.layout).matrix_shape if g.family == "unpool" else g.transport.matrix_shape
            routing = empty_indices(shape, g.route_values, x.device) if compression["enabled"] else torch.empty(shape, device=x.device, dtype=torch.int64)
        if need_state and g.family == "power" and not isinf(g.power):
            a = torch.empty(coeff_transport.matrix_shape, device=x.device, dtype=work_dtype)
            coefficient = quantize_pack_activation(a, compression["group_size"], coeff_block, bits=compression["bits"]) if compression["enabled"] else a
            factor = torch.empty(g.output_shape, device=x.device, dtype=work_dtype)
    else:
        source = dense(x)
        if g.family in ("average", "adaptive_average"):
            values = g.spec.native(source, **options)
        elif g.family == "power" and not isinf(g.power):
            if need_state:
                values, coefficient, factor = power(source, g, options)
            else:
                values = g.spec.native(source, **options)
        elif g.family == "unpool":
            indices = options["indices"]
            if ((indices < 0) | (indices >= g.output_elements)).any():
                raise RuntimeError("unpool indices outside output bounds")
            values = g.spec.native(source, **options)
            codes = indices
        else:
            values, public, codes = maximum(source, g, options, samples)
            if not options.get("return_indices", False):
                public = None
        if need_state and codes is not None:
            transport = ReLUGeometry(g.shape, g.layout) if g.family == "unpool" else g.transport
            codes = transport.matrix(codes).clamp(0, g.route_values - 1)
            routing = pack_indices(codes, g.route_values) if compression["enabled"] else codes.clone()
        if need_state and coefficient is not None:
            coefficient = coeff_transport.matrix(coefficient)
            coefficient = (quantize_pack_activation(coefficient, compression["group_size"], coeff_block, bits=compression["bits"])
                           if compression["enabled"] else coefficient)
        elif not need_state:
            coefficient = factor = None
    factor = g.transport.matrix(factor) if factor is not None else None
    state = PoolState(routing, coefficient, factor, public, samples if need_state else None, coeff_transport.layout)
    matrix = g.transport.matrix(values)
    output, inverse, snapshot, observations = convert(matrix, dtype, block_shape, scale, observe, capture_scale, check_finite, fake)
    info = {"backend": "reference", "forward_launches": 0, "packing_launches": 0, "scale_launches": 0,
            "workspace_dtype": None, "scale_mode": "provided" if scale is not None else "current" if inverse is not None else "none"}
    return output, inverse, state, snapshot, observations, info


def backward(gradient, g, *, routing, coefficient, factor, samples, coefficient_layout,
             input_dtype, gradient_output=None, execution_info=None):
    fake = isinstance(getattr(gradient, "_data", gradient), FakeTensor) or gradient.device.type == "meta"
    if fake or not gradient.numel():
        dx = torch.empty(g.shape, device=gradient.device, dtype=input_dtype)
    else:
        dy = dense(gradient)
        if g.family in ("average", "adaptive_average"):
            dx = average_backward(dy, g)
        elif g.family == "power" and not isinf(g.power):
            summed = average_backward(dy * g.transport.restore(factor), replace(g, divisor=1))
            a = dequantize_unpack_activation(coefficient, dtype=dy.dtype) if isinstance(coefficient, PackedActivation) else coefficient
            a = ReLUGeometry(g.shape, coefficient_layout).restore(a)
            dx = a * summed
        else:
            codes = unpack_indices(routing) if isinstance(routing, PackedIndices) else routing
            if g.family == "unpool":
                indices = ReLUGeometry(g.shape, g.layout).restore(codes).reshape(g.batch * g.channels, -1)
                dx = dy.reshape(g.batch * g.channels, -1).gather(1, indices).reshape(g.shape)
            else:
                codes = g.transport.restore(codes)
                sign = None
                if g.family == "power":
                    sign = ((codes % 3 == 1).to(dy.dtype) - (codes % 3 == 2).to(dy.dtype)).reshape_as(dy)
                    codes = codes // 3
                indices, valid = decode_locations(codes, g, samples)
                values = (dy if sign is None else dy * sign).reshape_as(indices)
                dx = torch.zeros((g.batch * g.channels, g.input_elements), device=dy.device, dtype=dy.dtype)
                dx.scatter_add_(1, indices.clamp(0, g.input_elements - 1), torch.where(valid, values, 0))
                dx = dx.reshape(g.shape)
    if execution_info is not None:
        execution_info.update(backward_launches=0, backward_workspace_dtype=None)
    return gradient_output.convert(dx) if gradient_output is not None else dx.to(input_dtype)
