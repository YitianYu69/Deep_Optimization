"""Numerical recipe identity for checkpoints and installed-library reports."""

from importlib.metadata import PackageNotFoundError, version
import sys


FP8_RECIPE_VERSION = 2


def fp8_recipe(backend):
    triton_version = None
    if backend == "triton":
        try:
            triton_version = version("triton")
        except PackageNotFoundError:
            triton_version = getattr(sys.modules.get("triton"), "__version__", None)
    return {"version": FP8_RECIPE_VERSION,
            "accumulation": "triton-native" if backend == "triton" else "reference-fp32",
            "triton_version": triton_version}
