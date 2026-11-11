"""Compatibility alias for deepopt.opt.nn.hf_conv1d."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("deepopt.opt.nn.hf_conv1d")
