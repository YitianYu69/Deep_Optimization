"""Implicit-GEMM FP8 spatial convolutions, without unfolded activation storage.

Imports stay lazy so importing DeepOpt does not require Triton or CUDA.
"""


def forward(*args, **kwargs):
    # Importing a submodule binds its name on this package. A module named
    # `forward` would replace this callable after the first warmup invocation.
    from .forward_impl import forward as implementation

    return implementation(*args, **kwargs)


def input_gradient(*args, **kwargs):
    from .backward import input_gradient as implementation

    return implementation(*args, **kwargs)


def weight_gradient(*args, **kwargs):
    from .backward import weight_gradient as implementation

    return implementation(*args, **kwargs)


def execution_mode(*operands):
    """Describe arithmetic without reading device values.

    Both tensorwise and blockwise kernels retain FP8 dot operands. Blockwise
    reductions scale each FP32 partial before accumulation, without changing
    the saved values or requantizing them into a different scaling geometry.
    """
    return "native_fp8"


__all__ = ["forward", "input_gradient", "weight_gradient", "execution_mode"]
