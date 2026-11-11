"""Tensor reference for exact regular/adaptive/fractional window mappings."""

from math import prod

import torch


def window_mapping(g, samples, device):
    coordinates = torch.meshgrid(*(torch.arange(size, device=device) for size in g.output_spatial), indexing="ij")
    coords = [value.reshape(1, -1).expand(g.batch * g.channels, -1) for value in coordinates]
    starts, extents = [], []
    for axis, (coordinate, length, output) in enumerate(zip(coords, g.input_spatial, g.output_spatial)):
        if g.family.startswith("adaptive"):
            start = coordinate * length // output
            end = ((coordinate + 1) * length + output - 1) // output
            extent = end - start
        elif g.family == "fractional_max":
            if axis < 3 - g.rank:
                start = torch.zeros_like(coordinate)
            elif output == 1:
                start = torch.full_like(coordinate, length - g.kernel[axis])
            else:
                alpha = (length - g.kernel[axis]) / (output - 1)
                r = samples.reshape(-1, g.rank)[:, 2 - axis if g.rank == 2 else axis].to(
                    torch.float64 if samples.dtype == torch.float64 else torch.float32)[:, None]
                start = ((coordinate + r) * alpha).long() - (r * alpha).long()
                start = torch.where(coordinate == output - 1, length - g.kernel[axis], start)
            extent = torch.full_like(coordinate, g.kernel[axis])
        else:
            start = coordinate * g.stride[axis] - g.padding[axis]
            extent = torch.full_like(coordinate, g.kernel[axis])
        starts.append(start)
        extents.append(extent)
    return starts, extents


def encode_locations(indices, g, samples):
    flat = indices.reshape(g.batch * g.channels, -1)
    starts, extents = window_mapping(g, samples, indices.device)
    h, w = g.input_spatial[1:]
    coordinates = (flat // (h * w), flat // w % h, flat % w)
    step = (1, 1, 1) if g.family.startswith("adaptive") or g.family == "fractional_max" else g.dilation
    local = [(coordinate - start) // d for coordinate, start, d in zip(coordinates, starts, step)]
    return ((local[0] * extents[1] + local[1]) * extents[2] + local[2]).reshape(g.output_shape)


def decode_locations(codes, g, samples):
    flat = codes.reshape(g.batch * g.channels, -1)
    starts, extents = window_mapping(g, samples, codes.device)
    local = (flat // (extents[1] * extents[2]), flat // extents[2] % extents[1], flat % extents[2])
    step = (1, 1, 1) if g.family.startswith("adaptive") or g.family == "fractional_max" else g.dilation
    coordinates = [start + offset * d for start, offset, d in zip(starts, local, step)]
    valid = torch.ones_like(flat, dtype=torch.bool)
    for coordinate, length in zip(coordinates, g.input_spatial):
        valid &= (coordinate >= 0) & (coordinate < length)
    index = (coordinates[0] * g.input_spatial[1] + coordinates[1]) * g.input_spatial[2] + coordinates[2]
    return index, valid


def average_backward(dy, g):
    values = dy.reshape(g.batch * g.channels, -1)
    dx = torch.zeros((g.batch * g.channels, g.input_elements), device=dy.device, dtype=dy.dtype)
    for linear in range(g.output_elements):
        output = (linear // (g.output_spatial[1] * g.output_spatial[2]),
                  linear // g.output_spatial[2] % g.output_spatial[1], linear % g.output_spatial[2])
        starts, extents = g.bounds(output)
        step = (1, 1, 1) if g.family.startswith("adaptive") else g.dilation
        ranges = [torch.arange(start, start + count * stride, stride, device=dy.device)
                  for start, count, stride in zip(starts, extents, step)]
        padded = prod(max(0, min(start + extent, length + pad) - start)
                      for start, extent, length, pad in zip(starts, extents, g.input_spatial, g.padding))
        ranges = [value[(value >= 0) & (value < length)] for value, length in zip(ranges, g.input_spatial)]
        count = prod(v.numel() for v in ranges)
        divisor = g.divisor if g.divisor is not None else padded if g.include_pad else count
        if g.family.startswith("adaptive"):
            divisor = count
        coordinates = torch.meshgrid(*ranges, indexing="ij")
        index = ((coordinates[0] * g.input_spatial[1] + coordinates[1]) * g.input_spatial[2] + coordinates[2]).flatten()
        if index.numel():
            dx[:, index] += values[:, linear:linear + 1] / divisor
    return dx.reshape(g.shape)
