"""Canonical spatial pooling domains, independent of activation transport."""

from dataclasses import dataclass
from functools import lru_cache
from math import gcd, prod, isinf

import torch

from ..activation.geometry import ReLUGeometry
from .spec import SPECS


def spatial_tuple(value, rank, default=None):
    if value is None or isinstance(value, (tuple, list)) and not value:
        value = default
    values = (value,) * rank if isinstance(value, int) else tuple(value)
    if len(values) == 1:
        values *= rank
    return (1,) * (3 - rank) + values


@dataclass(frozen=True)
class PoolGeometry:
    kind: str
    shape: tuple
    output_shape: tuple
    kernel: tuple
    stride: tuple
    padding: tuple
    dilation: tuple
    include_pad: bool
    divisor: int | None
    power: float
    layout: str

    @property
    def spec(self):
        return SPECS[self.kind]

    @property
    def family(self):
        return self.spec.family

    @property
    def rank(self):
        return self.spec.rank

    @property
    def batched(self):
        return len(self.shape) == self.rank + 2

    @property
    def batch(self):
        return self.shape[0] if self.batched else 1

    @property
    def channels(self):
        return self.shape[1] if self.batched else self.shape[0]

    @property
    def input_spatial(self):
        return (1,) * (3 - self.rank) + self.shape[-self.rank:]

    @property
    def output_spatial(self):
        return (1,) * (3 - self.rank) + self.output_shape[-self.rank:]

    @property
    def input_elements(self):
        return prod(self.input_spatial)

    @property
    def output_elements(self):
        return prod(self.output_spatial)

    @property
    def transport(self):
        return ReLUGeometry(self.output_shape, self.layout)

    @property
    def max_window(self):
        if self.family.startswith("adaptive"):
            return tuple(i // o + (0 if i % o == 0 else 1 if i % o == gcd(i, o) else 2) if o else 1
                         for i, o in zip(self.input_spatial, self.output_spatial))
        return self.kernel

    @property
    def window_volume(self):
        return prod(self.max_window)

    @property
    def routes(self):
        return "max" in self.family or self.family == "unpool" or self.family == "power" and isinf(self.power)

    @property
    def route_values(self):
        return self.output_elements if self.family == "unpool" else self.window_volume * (3 if self.family == "power" else 1)

    def canonical(self, value, output=False):
        spatial = self.output_spatial if output else self.input_spatial
        return value.reshape(self.batch, self.channels, *spatial)

    def bounds(self, output, *, sample=None):
        starts, extents = [], []
        for axis, (o, i, out) in enumerate(zip(output, self.input_spatial, self.output_spatial)):
            if self.family.startswith("adaptive"):
                start, end = o * i // out, ((o + 1) * i + out - 1) // out
                extent = end - start
            elif self.family == "fractional_max":
                if axis < 3 - self.rank:
                    start = 0
                elif out == 1 or o == out - 1:
                    start = i - self.kernel[axis]
                else:
                    alpha = (i - self.kernel[axis]) / (out - 1)
                    r = float(sample[2 - axis if self.rank == 2 else axis])
                    start = int((o + r) * alpha) - int(r * alpha)
                extent = self.kernel[axis]
            else:
                start, extent = o * self.stride[axis] - self.padding[axis], self.kernel[axis]
            starts.append(start)
            extents.append(extent)
        return tuple(starts), tuple(extents)


@lru_cache(maxsize=512)
def _output_shape(kind, shape, options, dtype, indices_shape):
    spec = SPECS[kind]
    args = dict(options)
    x = torch.empty(shape, dtype=dtype, device="meta")
    if spec.family == "unpool":
        args["indices"] = torch.empty(indices_shape, device="meta", dtype=torch.int64)
    if spec.family == "fractional_max":
        args.pop("_random_samples", None)
    try:
        result = spec.native(x, **args)
    except NotImplementedError:
        # Older Torch releases lack some Meta registrations. A zero-batch CPU
        # probe validates spatial arguments without allocating input values.
        batched = len(shape) == spec.rank + 2
        probe_shape = (0,) + (shape[1:] if batched else shape)
        probe = torch.empty(probe_shape, device="cpu", dtype=torch.float32)
        if spec.family == "unpool":
            args["indices"] = torch.empty(probe_shape, device="cpu", dtype=torch.int64)
        if spec.family == "fractional_max":
            args["_random_samples"] = torch.empty(0, probe_shape[1], spec.rank, dtype=torch.float32)
        result = spec.native(probe, **args)
        result_shape = tuple((result[0] if isinstance(result, tuple) else result).shape)
        return ((shape[0],) + result_shape[1:]) if batched else result_shape[1:]
    return tuple((result[0] if isinstance(result, tuple) else result).shape)


def geometry(kind, x, options, layout):
    spec = SPECS[kind]
    if x.ndim not in (spec.rank + 1, spec.rank + 2):
        raise RuntimeError(f"{kind} expects {spec.rank + 1}D or {spec.rank + 2}D input")
    canonical = {key: tuple(value) if isinstance(value, list) else value for key, value in options.items()
                 if key not in ("indices", "_random_samples")}
    indices = options.get("indices")
    if spec.family == "unpool":
        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64 or indices.shape != x.shape:
            raise RuntimeError("unpool indices must be int64 and match input shape")
        if indices.device != x.device:
            raise RuntimeError("unpool indices and input must share a device")
    shape = _output_shape(kind, tuple(x.shape), tuple(canonical.items()), x.dtype,
                          tuple(indices.shape) if indices is not None else ())
    k = spatial_tuple(options.get("kernel_size"), spec.rank, 1)
    stride = spatial_tuple(options.get("stride"), spec.rank, options.get("kernel_size", 1))
    pad = (0,) * (3 - spec.rank) + spatial_tuple(options.get("padding", 0), spec.rank)[3 - spec.rank:]
    dilation = spatial_tuple(options.get("dilation", 1), spec.rank)
    power = float(options.get("norm_type", 1))
    return PoolGeometry(kind, tuple(x.shape), shape, k, stride, pad, dilation,
                        options.get("count_include_pad", True), options.get("divisor_override"), power, layout)
