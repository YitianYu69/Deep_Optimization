"""Compatibility alias for deepopt.opt.backends.native.fp8."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("deepopt.opt.backends.native.fp8")
