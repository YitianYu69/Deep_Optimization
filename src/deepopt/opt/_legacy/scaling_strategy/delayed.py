"""Compatibility alias for deepopt.opt.scaling.delayed."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("deepopt.opt.scaling.delayed")
