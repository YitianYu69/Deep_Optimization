"""Lazy normalization executor selection."""

from importlib import import_module

from torch._subclasses.fake_tensor import FakeTensor
from deepopt.opt.quantization.activation_tensor import FP8Activation
from deepopt.kernel_policy import kernel_backend


def executor(x, backend):
    if backend not in ("reference", "triton", "tilelang", "mixed", "inherit"):
        raise ValueError("normalization backend must be reference, triton, tilelang, or mixed")
    storage = x._data if isinstance(x, FP8Activation) else x
    if x.device.type == "cuda" and not isinstance(storage, FakeTensor) and backend != "reference":
        family = kernel_backend("normalization", backend=backend)
        from ..dispatch import backend_module
        return backend_module("normalization", "normalization", backend=family)
    return import_module("deepopt.opt.backends.reference.normalization")
