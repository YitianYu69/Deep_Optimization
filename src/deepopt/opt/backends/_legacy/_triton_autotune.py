"""Compatibility alias for deepopt.opt.backends.triton.gemm.autotune."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("deepopt.opt.backends.triton.gemm.autotune")
