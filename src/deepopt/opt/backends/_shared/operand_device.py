"""Operand-owned CUDA launch scopes; no allocation or stream changes."""

from contextlib import contextmanager
from functools import wraps

import torch


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    else:
        # Internal packed descriptors, rather than arbitrary object traversal.
        for name in ("data", "step", "minimum"):
            item = getattr(value, name, None)
            if isinstance(item, torch.Tensor):
                yield item


@contextmanager
def cuda_operand_scope(*operands):
    tensors = [tensor for value in operands for tensor in _tensors(value)]
    if not tensors or tensors[0].device.type != "cuda":
        raise ValueError("CUDA launch requires CUDA tensor operands")
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("CUDA launch operands must share one device")
    with torch.cuda.device(device):
        yield device


def operand_device_guard(function):
    @wraps(function)
    def call(*args, **kwargs):
        with cuda_operand_scope(args, kwargs):
            return function(*args, **kwargs)
    return call
