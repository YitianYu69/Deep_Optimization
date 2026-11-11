"""Lazy selection of reference or fused Triton activation implementations."""

from importlib import import_module

import torch
from torch._subclasses.fake_tensor import FakeTensor


def executor(x, kind="relu"):
    storage = getattr(x, "_data", x)
    family = ("triton" if x.device.type == "cuda" and not isinstance(storage, FakeTensor)
              and storage.dtype != torch.float64 else "reference")
    operation = "relu" if kind == "relu" else "elementwise"
    return import_module(f"deepopt.opt.backends.{family}.activation.{operation}")
