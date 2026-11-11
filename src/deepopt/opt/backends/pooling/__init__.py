"""Lazy pooling executor selection; reference execution covers Fake/meta."""

from importlib import import_module

import torch
from torch._subclasses.fake_tensor import FakeTensor


def executor(x, backend):
    data = getattr(x, "_data", x)
    family = "triton" if backend == "triton" and x.device.type == "cuda" and data.dtype != torch.float64 and not isinstance(data, FakeTensor) else "reference"
    return import_module(f"deepopt.opt.backends.{family}.pooling")
