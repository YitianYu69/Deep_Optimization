"""Lazy normalization executor selection."""

from importlib import import_module

from torch._subclasses.fake_tensor import FakeTensor
from deepopt.opt.quantization.activation_tensor import FP8Activation


def executor(x, backend):
    if backend not in ("reference", "triton"):
        raise ValueError("normalization backend must be reference or triton")
    storage = x._data if isinstance(x, FP8Activation) else x
    if x.device.type == "cuda" and not isinstance(storage, FakeTensor) and backend == "triton":
        return import_module("deepopt.opt.backends.triton.normalization")
    return import_module("deepopt.opt.backends.reference.normalization")
