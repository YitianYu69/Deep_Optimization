"""Lazy FP8 gradient output/transpose helpers shared across DO operations."""


def bias(q):
    from .reduction import bias as implementation
    return implementation(q)


def transpose_quantize(*args, **kwargs):
    from .transpose import transpose_quantize as implementation
    return implementation(*args, **kwargs)


def __getattr__(name):
    if name == "_gradient_tile":
        from .epilogue import _gradient_tile
        return _gradient_tile
    raise AttributeError(name)
