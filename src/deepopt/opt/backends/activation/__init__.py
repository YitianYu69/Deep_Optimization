"""Lazy selection of reference or fused Triton activation implementations."""

from importlib import import_module

import torch
from torch._subclasses.fake_tensor import FakeTensor
from deepopt.kernel_policy import kernel_backend


def executor(x, kind="relu"):
    storage = getattr(x, "_data", x)
    family = (kernel_backend("activation") if x.device.type == "cuda" and not isinstance(storage, FakeTensor)
              and storage.dtype != torch.float64 else "reference")
    operation = "relu" if kind == "relu" else "elementwise"
    if family != "reference":
        from ..dispatch import backend_module
        return backend_module("activation", f"activation.{operation}", backend=family)
    return import_module(f"deepopt.opt.backends.{family}.activation.{operation}")
