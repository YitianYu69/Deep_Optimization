"""Lazy transposed backend selection preserving the real operand recipes."""

from importlib import import_module

import torch
from torch._subclasses.fake_tensor import FakeTensor
from deepopt.kernel_policy import kernel_backend, kernel_role


def _backend(name):
    if name == "reference":
        return import_module("deepopt.opt.backends.reference.convolution_transpose")
    if name in ("triton", "tilelang", "mixed", "inherit", "native_fast", "native_precise"):
        family = kernel_backend("convolution", backend=name)
        from ..dispatch import backend_module
        return backend_module("convolution", "convolution_transpose", backend=family)
    raise ValueError("Unsupported transposed convolution backend")


@kernel_role("convolution.forward")
def forward(qx, qw, geometry, *, backend, **kwargs):
    if backend not in ("reference", "native_fast", "native_precise"):
        backend = kernel_backend("convolution", backend=backend, role="convolution.forward")
    if isinstance(qx.data, FakeTensor):
        value = torch.empty(geometry.grad_matrix_shape, dtype=kwargs.get("out_dtype", torch.float32), device=qx.data.device)
        output = kwargs.get("output")
        return output.convert(value, matrix=True) if output is not None else value
    if backend != "reference":
        kwargs["gemm_backend"] = backend
    return _backend(backend).forward(qx, qw, geometry, **kwargs)


@kernel_role("convolution.dx")
def input_gradient(qg, qw, geometry, *, backend, **kwargs):
    if backend not in ("reference", "native_fast", "native_precise"):
        backend = kernel_backend("convolution", backend=backend, role="convolution.dx")
    if isinstance(qg.data, FakeTensor):
        value = torch.empty(geometry.input_matrix_shape, dtype=kwargs.get("out_dtype", torch.float32), device=qg.data.device)
        output = kwargs.get("gradient_output")
        return output.convert(value, matrix=True) if output is not None else value
    if backend != "reference":
        kwargs["gemm_backend"] = backend
    return _backend(backend).input_gradient(qg, qw, geometry, **kwargs)


@kernel_role("convolution.dw")
def weight_gradient(qx, qg, geometry, *, backend, **kwargs):
    if backend not in ("reference", "native_fast", "native_precise"):
        backend = kernel_backend("convolution", backend=backend, role="convolution.dw")
    if isinstance(qx.data, FakeTensor):
        return torch.empty(geometry.weight_matrix_shape, dtype=kwargs.get("out_dtype", torch.float32), device=qx.data.device)
    if backend != "reference":
        kwargs["gemm_backend"] = backend
    return _backend(backend).weight_gradient(qx, qg, geometry, **kwargs)
