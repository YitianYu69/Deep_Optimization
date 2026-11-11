"""Compatibility alias for deepopt.opt.backends._shared.cuda."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("deepopt.opt.backends._shared.cuda")
