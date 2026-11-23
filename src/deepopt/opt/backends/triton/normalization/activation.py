"""Activation epilogues shared by normalization kernels.

The affine result is rounded before activation, and the derivative product is
rounded before normalization reductions. These are the materialized recipe's
two observable dtype boundaries, even when no intermediate is allocated.
"""

import triton
import triton.language as tl

from ..activation.math import _evaluate


@triton.jit
def _activate(value, output, preactivation, index, valid,
              ACTIVATION: tl.constexpr = "", APPROXIMATE: tl.constexpr = "none",
              SAVE_PREACTIVATION: tl.constexpr = False):
    if ACTIVATION != "":
        rounded = value.to(output.dtype.element_ty)
        if SAVE_PREACTIVATION:
            tl.store(preactivation + index, rounded, valid)
        x = rounded.to(tl.float32)
        if ACTIVATION == "silu":
            # Match the compiler's standalone pointwise forward expression.
            return x / (1.0 + tl.exp(-x))
        result, _ = _evaluate(x, ACTIVATION, APPROXIMATE)
        return result
    return value


@triton.jit
def _adjust_gradient(value, preactivation, index, valid,
                     ACTIVATION: tl.constexpr = "", APPROXIMATE: tl.constexpr = "none"):
    if ACTIVATION != "":
        x = tl.load(preactivation + index, valid, other=0).to(tl.float32)
        _, coefficient = _evaluate(x, ACTIVATION, APPROXIMATE)
        return (value * coefficient).to(preactivation.dtype.element_ty).to(tl.float32)
    return value


@triton.jit
def _adjust_spatial_gradient(value, preactivation, n, c, s, valid,
                             C: tl.constexpr, S: tl.constexpr, META: tl.constexpr,
                             ACTIVATION: tl.constexpr = "", APPROXIMATE: tl.constexpr = "none",
                             derivative_saved=None, derivative_step=None, derivative_minimum=None,
                             PACKED_DERIVATIVE: tl.constexpr = False):
    if PACKED_DERIVATIVE:
        # The compiler admits only BF16 contiguous flat1024 two-bit states.
        # n/c/s are widened by the launch kernels before address arithmetic.
        index = (n * C + c) * S + s
        group, offset = index // 1024, index % 1024
        word = tl.load(derivative_saved + group * 64 + offset // 16, valid, other=0)
        code = ((word >> ((offset % 16) * 2)) & 3).to(tl.float32)
        coefficient = code * tl.load(derivative_step + group, valid, other=0) + tl.load(
            derivative_minimum + group, valid, other=0)
        return (value * coefficient).to(tl.bfloat16).to(tl.float32)
    if ACTIVATION != "":
        address = n * META[1][0] + c * META[1][1]
        for axis in tl.static_range(len(META[2])):
            address += s // META[3][axis] % META[2][axis] * META[1][axis + 2]
        return _adjust_gradient(value, preactivation, address, valid, ACTIVATION, APPROXIMATE)
    return value
