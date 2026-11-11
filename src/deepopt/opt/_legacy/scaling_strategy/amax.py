"""Compatibility alias for deepopt.opt.scaling.amax."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("deepopt.opt.scaling.amax")
