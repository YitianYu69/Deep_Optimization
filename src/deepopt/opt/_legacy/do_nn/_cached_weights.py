"""Compatibility alias for deepopt.opt.nn._cached_weights."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("deepopt.opt.nn._cached_weights")
