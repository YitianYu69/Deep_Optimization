"""Version-aware pooling signatures and native/DO call-site matching."""

from dataclasses import dataclass
from math import prod

import torch

from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PoolSpec:
    name: str
    family: str
    rank: int
    fields: tuple
    required: int

    def arguments(self, args, kwargs):
        names = tuple(name for name, _ in self.fields)
        if len(args) > len(names) or set(kwargs) - set(names):
            raise TypeError(f"{self.name}: invalid pooling arguments")
        if any(name in kwargs for name in names[:len(args)]):
            raise TypeError(f"{self.name}: duplicate pooling argument")
        values = {name: default for name, default in self.fields}
        values.update(zip(names, args))
        values.update(kwargs)
        if any(name not in kwargs and position >= len(args) for position, name in enumerate(names[:self.required])):
            raise TypeError(f"{self.name}: missing required pooling argument")
        return values

    @property
    def native(self):
        native = getattr(F, self.name, None)
        if native is None and self.name == "lp_pool3d":
            return _lp_pool3d
        if native is None:
            raise NotImplementedError(f"{self.name} is unavailable in this Torch version")
        return native


def _lp_pool3d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    """Power-pooling expression for Torch releases without a native 3D wrapper."""
    if norm_type == 0:
        raise ValueError("norm_type must be nonzero")
    if norm_type == float("inf"):
        return F.max_pool3d(input.abs(), kernel_size, stride, ceil_mode=ceil_mode)
    if norm_type == -float("inf"):
        return -F.max_pool3d(-input.abs(), kernel_size, stride, ceil_mode=ceil_mode)
    kernel = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
    if len(kernel) == 1:
        kernel *= 3
    value = F.avg_pool3d(input.pow(norm_type), kernel_size, stride, ceil_mode=ceil_mode)
    return (torch.sign(value) * F.relu(value.abs()) * prod(kernel)).pow(1.0 / norm_type)


SPECS = {}
MODULE_SPECS = {}
FUNCTIONS = {}
FORCE_INDICES = set()
for _rank in (1, 2, 3):
    for _prefix, _class, _family, _fields, _required in (
        ("max_pool", "MaxPool", "max", (("kernel_size", None), ("stride", None), ("padding", 0),
            ("dilation", 1), ("ceil_mode", False), ("return_indices", False)), 1),
        ("avg_pool", "AvgPool", "average", (("kernel_size", None), ("stride", None), ("padding", 0),
            ("ceil_mode", False), ("count_include_pad", True)) + ((("divisor_override", None),) if _rank > 1 else ()), 1),
        ("adaptive_max_pool", "AdaptiveMaxPool", "adaptive_max", (("output_size", None), ("return_indices", False)), 1),
        ("adaptive_avg_pool", "AdaptiveAvgPool", "adaptive_average", (("output_size", None),), 1),
        ("lp_pool", "LPPool", "power", (("norm_type", None), ("kernel_size", None), ("stride", None), ("ceil_mode", False)), 2),
        ("max_unpool", "MaxUnpool", "unpool", (("indices", None), ("kernel_size", None), ("stride", None),
            ("padding", 0), ("output_size", None)), 2),
    ):
        _name = f"{_prefix}{_rank}d"
        _spec = PoolSpec(_name, _family, _rank, _fields, _required)
        SPECS[_name] = _spec
        if hasattr(nn, f"{_class}{_rank}d"):
            MODULE_SPECS[getattr(nn, f"{_class}{_rank}d")] = _spec
        if hasattr(F, _name):
            FUNCTIONS[getattr(F, _name)] = _name
        if "max" in _family and hasattr(F, _name + "_with_indices"):
            _function = getattr(F, _name + "_with_indices")
            FUNCTIONS[_function] = _name
            FORCE_INDICES.add(_function)
    if _rank > 1:
        _name = f"fractional_max_pool{_rank}d"
        _spec = PoolSpec(_name, "fractional_max", _rank, (("kernel_size", None), ("output_size", None),
                        ("output_ratio", None), ("return_indices", False), ("_random_samples", None)), 1)
        SPECS[_name] = _spec
        MODULE_SPECS[getattr(nn, f"FractionalMaxPool{_rank}d")] = _spec
        FUNCTIONS[getattr(F, _name)] = _name
        _function = getattr(F, _name + "_with_indices")
        FUNCTIONS[_function] = _name
        FORCE_INDICES.add(_function)


def call_kind(target):
    return FUNCTIONS.get(target) or getattr(target, "_deepopt_pool_kind", None)


def native_call(spec, x, options, *, indices=False):
    options = dict(options)
    if indices:
        options["return_indices"] = True
    return spec.native(x, **options)
