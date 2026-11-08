"""Numerical recipe identity for checkpoints and installed-library reports."""

from importlib.metadata import PackageNotFoundError, version
import sys


FP8_RECIPE_VERSION = 2


def fp8_recipe(backend, settings=None):
    triton_version = None
    if backend == "triton":
        try:
            triton_version = version("triton")
        except PackageNotFoundError:
            triton_version = getattr(sys.modules.get("triton"), "__version__", None)
    result = {"version": FP8_RECIPE_VERSION,
            "accumulation": "triton-native" if backend == "triton" else "reference-fp32",
            "triton_version": triton_version}
    if settings is not None and settings.scaling_recipe == "delayed":
        from dataclasses import asdict

        result.update(version=3, scaling="delayed", delayed_scaling=asdict(settings.delayed_scaling))
    return result
