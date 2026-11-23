"""Compatibility alias for shared, GPU-language-independent launch metadata."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("deepopt.opt.backends._shared.delayed_launch")
