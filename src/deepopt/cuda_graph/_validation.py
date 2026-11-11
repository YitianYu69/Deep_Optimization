"""Metadata-only checks: replay must not synchronize tensors with the host."""

from contextlib import contextmanager
from threading import Lock

import torch


_capture_lock = Lock()


def cuda_device(device):
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise ValueError("CUDA Graphs require a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Graphs require a CUDA-enabled PyTorch build and an available GPU")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


@contextmanager
def capture_guard():
    if not _capture_lock.acquire(blocking=False):
        raise RuntimeError("Only one DeepOpt CUDA capture may run in a process at a time")
    try:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Nested CUDA capture is unsupported")
        yield
    finally:
        _capture_lock.release()


def warmup_count(value):
    if type(value) is not int or value < 1:
        raise ValueError("warmup must be a positive integer")


def tensor_spec(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ValueError("Capture arguments must be strided CUDA tensors")
    return (tensor.device, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), tensor.requires_grad)


def arguments(args, device):
    if not isinstance(args, tuple):
        raise TypeError("sample_args must be a tuple of positional tensors")
    specs = tuple(tensor_spec(arg) for arg in args)
    if any(spec[0] != device for spec in specs):
        raise ValueError("All capture inputs must be on the graph's CUDA device")
    return specs


def check_arguments(args, kwargs, specs):
    if kwargs:
        raise TypeError("Captured calls accept positional tensors only")
    if len(args) != len(specs) or tuple(tensor_spec(arg) for arg in args) != specs:
        raise ValueError("Captured input count, shape, stride, dtype, device and requires_grad must match; recapture")


def autocast_spec():
    enabled = torch.is_autocast_enabled("cuda")
    return (enabled, torch.get_autocast_dtype("cuda") if enabled else None)


def operation_policy_spec():
    """Snapshot scoped projection choices without reading device tensors.

    Missing compression policy means adapters retain their prepared setting;
    an explicit disabled policy can select different backward work. Preserve
    that distinction. FP8 plans are owned immutable execution dictionaries.
    """
    from deepopt.opt._autocast import (_active_interception, _active_convolution_interception,
                                      _active_normalization_interception, _active_pooling_interception, _get_activation_compression,
                                      _get_fp8_plan)

    compression = _get_activation_compression("cuda")
    compression_spec = (None if compression is None else
                        (compression["enabled"], compression["group_size"],
                         compression.get("bits", 2), compression.get("attention_bits"),
                         compression.get("norm_xhat_recenter", False)))
    return (_get_fp8_plan("cuda"), compression_spec, _active_interception.get().get("cuda", True),
            _active_convolution_interception.get().get("cuda", True),
            _active_normalization_interception.get().get("cuda", True),
            _active_pooling_interception.get().get("cuda", True))


def static_arguments(args):
    # Own the storage: callers can release or overwrite their example tensors.
    # Preserve arbitrary non-overlapping strides, including sliced inputs.
    result = []
    for arg in args:
        value = torch.empty_strided(arg.shape, arg.stride(), dtype=arg.dtype, device=arg.device)
        value.copy_(arg.detach())
        value.requires_grad_(arg.requires_grad)
        result.append(value)
    return tuple(result)
