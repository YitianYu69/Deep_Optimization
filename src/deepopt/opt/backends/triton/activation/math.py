"""Pointwise activation values and pre-output-cast local derivatives."""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _evaluate(x, KIND: tl.constexpr, APPROXIMATE: tl.constexpr):
    if KIND == "sigmoid" or KIND == "silu":
        s = 1.0 / (1.0 + tl.exp(-x))
        if KIND == "sigmoid":
            return s, s * (1.0 - s)
        return x * s, s * (1.0 + x * (1.0 - s))
    elif KIND == "tanh":
        y = libdevice.tanh(x)
        return y, 1.0 - y * y
    elif KIND == "gelu":
        if APPROXIMATE == "tanh":
            u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
            t = libdevice.tanh(u)
            derivative = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 0.134145 * x * x)
            return 0.5 * x * (1.0 + t), derivative
        else:
            cdf = 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))
            return x * cdf, cdf + x * 0.3989422804014327 * tl.exp(-0.5 * x * x)
    else:
        # Hardtanh/ReLU6 backward passes NaN inputs, like native threshold tests.
        predicate = ~((x <= 0.0) | (x >= 6.0))
        return tl.where(x < 0.0, 0.0, tl.where(x > 6.0, 6.0, x)), predicate.to(tl.float32)
