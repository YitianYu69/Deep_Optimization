"""Lazy spatial convolution executors; operands retain source scaling geometry."""

from importlib import import_module

import torch
from torch._subclasses.fake_tensor import FakeTensor
from deepopt.kernel_policy import kernel_backend, kernel_role


def _backend(name):
    if name == "reference":
        return import_module("deepopt.opt.backends.reference.convolution")
    if name in ("triton", "tilelang", "mixed", "inherit", "native_fast", "native_precise"):
        # Native scaled GEMM is reused by the pointwise fast path. General
        # convolution needs an implicit gather/reduction executor.
        family = kernel_backend("convolution", backend=name)
        from ..dispatch import backend_module
        return backend_module("convolution", "convolution", backend=family)
    raise ValueError("Unsupported convolution backend")


def _executor(backend, geometry, kernel_variant):
    if kernel_variant == "depthwise" and backend != "reference":
        from deepopt.opt.backends._shared.convolution import depthwise_eligible
        if depthwise_eligible(geometry):
            family = kernel_backend("convolution", backend=backend)
            from ..dispatch import backend_module
            return backend_module("convolution", "convolution.depthwise", backend=family)
    return _backend(backend)


@kernel_role("convolution.forward")
def forward(qx, qw, geometry, *, backend, kernel_variant="strict", **kwargs):
    if backend != "reference":
        backend = kernel_backend("convolution", backend=backend, role="convolution.forward")
    if isinstance(qx.data, FakeTensor) and backend != "reference":
        return torch.empty(geometry.grad_matrix_shape, device=qx.data.device,
                           dtype=kwargs.get("out_dtype", torch.float32))
    return _executor(backend, geometry, kernel_variant).forward(qx, qw, geometry, **kwargs)


@kernel_role("convolution.dx")
def input_gradient(qg, qw, geometry, *, backend, kernel_variant="strict", **kwargs):
    if backend != "reference":
        backend = kernel_backend("convolution", backend=backend, role="convolution.dx")
    gradient_output = kwargs.pop("gradient_output", None)
    if isinstance(qg.data, FakeTensor) and backend != "reference":
        value = torch.empty(geometry.input_matrix_shape, device=qg.data.device,
                            dtype=kwargs.get("out_dtype", torch.float32))
        return gradient_output.convert(value, matrix=True) if gradient_output is not None else value
    implementation = _executor(backend, geometry, kernel_variant)
    if (gradient_output is not None and kernel_variant == "depthwise"
            and getattr(implementation, "__name__", "").endswith(".depthwise")):
        value = implementation.input_gradient(qg, qw, geometry, **kwargs)
        return gradient_output.convert(value, matrix=True)
    if gradient_output is not None and backend in ("triton", "tilelang", "native_fast", "native_precise"):
        return _backend(backend).input_gradient(qg, qw, geometry, gradient_output=gradient_output, **kwargs)
    value = implementation.input_gradient(qg, qw, geometry, **kwargs)
    return gradient_output.convert(value, matrix=True) if gradient_output is not None else value


@kernel_role("convolution.dw")
def weight_gradient(qx, qg, geometry, *, backend, kernel_variant="strict", **kwargs):
    if backend != "reference":
        backend = kernel_backend("convolution", backend=backend, role="convolution.dw")
    if isinstance(qx.data, FakeTensor) and backend != "reference":
        return torch.empty(geometry.weight_matrix_shape, device=qx.data.device,
                           dtype=kwargs.get("out_dtype", torch.float32))
    return _executor(backend, geometry, kernel_variant).weight_gradient(qx, qg, geometry, **kwargs)
