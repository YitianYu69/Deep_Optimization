"""Geometry-specific pooling kernels with an explicit general-path control."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps


_variant=ContextVar("deepopt_pooling_kernel_variant",default="auto")


def pooling_kernel_variant():
    return _variant.get()


def resnet_maxpool(geometry):
    return (geometry.family=="max" and geometry.rank==2 and geometry.kernel==(1,3,3)
            and geometry.stride==(1,2,2) and geometry.padding==(0,1,1) and geometry.dilation==(1,1,1)
            and pooling_kernel_variant()!="generic")


@contextmanager
def pooling_kernel_scope(variant="auto"):
    if variant not in ("auto","generic"):
        raise ValueError("pooling variant must be auto or generic")
    token=_variant.set(variant)
    try:yield
    finally:_variant.reset(token)


def saved_pooling_kernel(function):
    @wraps(function)
    def call(ctx,*args):
        with pooling_kernel_scope(ctx.pooling_variant):
            return function(ctx,*args)
    return call
