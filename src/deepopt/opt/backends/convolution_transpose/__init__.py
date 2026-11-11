"""Lazy transposed backend selection preserving the real operand recipes."""

from importlib import import_module

import torch
from torch._subclasses.fake_tensor import FakeTensor


def _backend(name):
    if name == "reference":
        return import_module("deepopt.opt.backends.reference.convolution_transpose")
    if name in ("triton", "native_fast", "native_precise"):
        return import_module("deepopt.opt.backends.triton.convolution_transpose")
    raise ValueError("Unsupported transposed convolution backend")


def forward(qx, qw, geometry, *, backend, **kwargs):
    if isinstance(qx.data, FakeTensor):
        value = torch.empty(geometry.grad_matrix_shape, dtype=kwargs.get("out_dtype", torch.float32), device=qx.data.device)
        output = kwargs.get("output")
        return output.convert(value, matrix=True) if output is not None else value
    if backend != "reference":
        kwargs["gemm_backend"] = backend
    return _backend(backend).forward(qx, qw, geometry, **kwargs)


def input_gradient(qg, qw, geometry, *, backend, **kwargs):
    if isinstance(qg.data, FakeTensor):
        value = torch.empty(geometry.input_matrix_shape, dtype=kwargs.get("out_dtype", torch.float32), device=qg.data.device)
        output = kwargs.get("gradient_output")
        return output.convert(value, matrix=True) if output is not None else value
    if backend != "reference":
        kwargs["gemm_backend"] = backend
    return _backend(backend).input_gradient(qg, qw, geometry, **kwargs)


def weight_gradient(qx, qg, geometry, *, backend, **kwargs):
    if isinstance(qx.data, FakeTensor):
        return torch.empty(geometry.weight_matrix_shape, dtype=kwargs.get("out_dtype", torch.float32), device=qx.data.device)
    if backend != "reference":
        kwargs["gemm_backend"] = backend
    return _backend(backend).weight_gradient(qx, qg, geometry, **kwargs)
