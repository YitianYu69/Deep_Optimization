"""Lazy spatial convolution executors; operands retain source scaling geometry."""

from importlib import import_module

import torch
from torch._subclasses.fake_tensor import FakeTensor


def _backend(name):
    if name == "reference":
        return import_module("deepopt.opt.backends.reference.convolution")
    if name in ("triton", "native_fast", "native_precise"):
        # Native scaled GEMM is reused by the pointwise fast path. General
        # convolution needs an implicit gather/reduction executor.
        return import_module("deepopt.opt.backends.triton.convolution")
    raise ValueError("Unsupported convolution backend")


def forward(qx, qw, geometry, *, backend, **kwargs):
    if isinstance(qx.data, FakeTensor) and backend != "reference":
        return torch.empty(geometry.grad_matrix_shape, device=qx.data.device,
                           dtype=kwargs.get("out_dtype", torch.float32))
    return _backend(backend).forward(qx, qw, geometry, **kwargs)


def input_gradient(qg, qw, geometry, *, backend, **kwargs):
    gradient_output = kwargs.pop("gradient_output", None)
    if isinstance(qg.data, FakeTensor) and backend != "reference":
        value = torch.empty(geometry.input_matrix_shape, device=qg.data.device,
                            dtype=kwargs.get("out_dtype", torch.float32))
        return gradient_output.convert(value, matrix=True) if gradient_output is not None else value
    if gradient_output is not None and backend in ("triton", "native_fast", "native_precise"):
        return _backend(backend).input_gradient(qg, qw, geometry, gradient_output=gradient_output, **kwargs)
    value = _backend(backend).input_gradient(qg, qw, geometry, **kwargs)
    return gradient_output.convert(value, matrix=True) if gradient_output is not None else value


def weight_gradient(qx, qg, geometry, *, backend, **kwargs):
    if isinstance(qx.data, FakeTensor) and backend != "reference":
        return torch.empty(geometry.weight_matrix_shape, device=qx.data.device,
                           dtype=kwargs.get("out_dtype", torch.float32))
    return _backend(backend).weight_gradient(qx, qg, geometry, **kwargs)
