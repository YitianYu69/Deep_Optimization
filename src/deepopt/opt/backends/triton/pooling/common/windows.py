"""Exact spatial window mapping and bounded inverse candidate ranges."""

import triton
import triton.language as tl


@triton.jit
def _fractional_parameters(nc, valid, samples, GEO: tl.constexpr, AXIS: tl.constexpr):
    # Each static axis gets its own specialization, rather than redefining
    # annotated constexpr names on successive unrolled loop iterations.
    sample_axis: tl.constexpr = 2 - AXIS if GEO[8] == 2 else AXIS
    index = (nc + valid.to(tl.int64) * 0) * GEO[8] + sample_axis
    sample = tl.load(samples + index, valid, other=0.0).to(tl.float32)
    alpha: tl.constexpr = (GEO[1][AXIS] - GEO[3][AXIS]) / (GEO[2][AXIS] - 1)
    return sample, alpha


@triton.jit
def _window(nc, coordinates, samples, valid, GEO: tl.constexpr):
    starts = ()
    extents = ()
    for axis in tl.static_range(3):
        o = coordinates[axis]
        if GEO[0] == 2 or GEO[0] == 3:
            start = o * GEO[1][axis] // GEO[2][axis]
            end = ((o + 1) * GEO[1][axis] + GEO[2][axis] - 1) // GEO[2][axis]
            extent = end - start
        elif GEO[0] == 4:
            if axis < 3 - GEO[8]:
                start = o * 0
            elif GEO[2][axis] == 1:
                start = o * 0 + GEO[1][axis] - GEO[3][axis]
            else:
                r, alpha = _fractional_parameters(nc, valid, samples, GEO, axis)
                start = ((o.to(tl.float32) + r) * alpha).to(tl.int64) - (r * alpha).to(tl.int64)
                start = tl.where(o == GEO[2][axis] - 1, GEO[1][axis] - GEO[3][axis], start)
            extent = o * 0 + GEO[3][axis]
        else:
            start, extent = o * GEO[4][axis] - GEO[5][axis], o * 0 + GEO[3][axis]
        starts += (start,)
        extents += (extent,)
    return starts, extents


@triton.jit
def _divisor(starts, extents, GEO: tl.constexpr):
    count = extents[0] * 0 + 1
    for axis in tl.static_range(3):
        if GEO[0] == 2 or GEO[0] == 3:
            size = extents[axis]
        else:
            start = starts[axis]
            if GEO[10]:
                size = tl.maximum(0, tl.minimum(start + GEO[3][axis], GEO[1][axis] + GEO[5][axis]) - start)
            else:
                size = tl.maximum(0, tl.minimum(start + GEO[3][axis], GEO[1][axis]) - tl.maximum(start, 0))
        count *= size
    if GEO[11] is not None:
        count = count * 0 + GEO[11]
    return count


@triton.jit
def _candidates(nc, positions, samples, valid, GEO: tl.constexpr):
    lows = ()
    highs = ()
    for axis in tl.static_range(3):
        i = positions[axis]
        if GEO[0] == 2 or GEO[0] == 3:
            low = i * GEO[2][axis] // GEO[1][axis]
            high = ((i + 1) * GEO[2][axis] + GEO[1][axis] - 1) // GEO[1][axis]
        elif GEO[0] == 4:
            if axis < 3 - GEO[8] or GEO[2][axis] == 1:
                low, high = i * 0, i * 0 + 1
            else:
                r, alpha = _fractional_parameters(nc, valid, samples, GEO, axis)
                base = (r * alpha).to(tl.int64)
                low = tl.floor((i - GEO[3][axis] + base + 1).to(tl.float32) / alpha - r).to(tl.int64) - 1
                high = tl.ceil((i + base + 1).to(tl.float32) / alpha - r).to(tl.int64) + 1
        else:
            low = (i + GEO[5][axis] - (GEO[3][axis] - 1) * GEO[6][axis] + GEO[4][axis] - 1) // GEO[4][axis]
            high = (i + GEO[5][axis]) // GEO[4][axis] + 1
        lows += (tl.maximum(low, 0),)
        highs += (tl.minimum(high, GEO[2][axis]),)
    return lows, highs
