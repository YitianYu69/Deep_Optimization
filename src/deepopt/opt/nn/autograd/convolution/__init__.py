"""Native and FP8 spatial convolution autograd implementations."""

from .native import _NativeConvolutionFunction

__all__ = ["_NativeConvolutionFunction"]
