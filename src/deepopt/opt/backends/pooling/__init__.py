"""Lazy pooling executor selection; reference execution covers Fake/meta."""

from importlib import import_module

import torch
from torch._subclasses.fake_tensor import FakeTensor
from deepopt.kernel_policy import kernel_backend


def executor(x, backend):
    data = getattr(x, "_data", x)
    family = (kernel_backend("pooling", backend=backend)
              if backend != "reference" and x.device.type == "cuda" and data.dtype != torch.float64
              and not isinstance(data, FakeTensor) else "reference")
    if family != "reference":
        from ..dispatch import backend_module
        return backend_module("pooling", "pooling", backend=family)
    return import_module(f"deepopt.opt.backends.{family}.pooling")
