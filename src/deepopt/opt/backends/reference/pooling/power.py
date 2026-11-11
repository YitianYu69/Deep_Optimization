"""LP value and preprocessed derivative-factor oracle."""

from math import prod

import torch
from torch.nn import functional as F


def forward(x, g, options):
    average = getattr(F, f"avg_pool{g.rank}d")
    z = average(x.pow(g.power), options["kernel_size"], options.get("stride"), 0, options.get("ceil_mode", False))
    base = z.sign() * F.relu(z.abs()) * prod(g.kernel)
    values = base.pow(1.0 / g.power)
    upstream = base.pow(1.0 / g.power - 1.0) * (prod(g.kernel) / g.power) * z.sign()
    derivative = torch.where(z.abs() <= 0, 0.0, upstream) * z.sign()
    # Divide out the average divisor once using the same spatial windows.
    # Native LP has padding zero: the valid divisor is the clipped window volume.
    from .windows import window_mapping

    starts, extents = window_mapping(g, None, x.device)
    count = torch.ones_like(starts[0], dtype=x.dtype)
    for start, extent, length in zip(starts, extents, g.input_spatial):
        count *= (start + extent).clamp_max(length) - start.clamp_min(0)
    factor = derivative / count.reshape(g.output_shape)
    coefficient = g.power * x.pow(g.power - 1)
    return values.detach(), coefficient, factor
