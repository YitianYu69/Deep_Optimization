"""Numerical recipe identity for checkpoints and installed-library reports."""

from importlib.metadata import PackageNotFoundError, version
import sys


FP8_RECIPE_VERSION = 5


def fp8_recipe(backend, settings=None):
    triton_version = None
    if backend == "triton":
        try:
            triton_version = version("triton")
        except PackageNotFoundError:
            triton_version = getattr(sys.modules.get("triton"), "__version__", None)
    result = {"version": FP8_RECIPE_VERSION,
            "accumulation": "triton-native" if backend == "triton" else "reference-fp32",
            "triton_version": triton_version,
            "weight_gradient_output": "forward-dtype"}
    if settings is not None and settings.scaling_recipe == "delayed":
        from dataclasses import asdict

        result.update(scaling="delayed", delayed_scaling=asdict(settings.delayed_scaling))
    if settings is not None and (settings.backward_backend != "inherit" or settings.exclude_modules):
        result.update(backward_backend=settings.backward_backend,
                      exclude_modules=list(settings.exclude_modules))
        if settings.backward_backend != "inherit":
            import torch

            result.update(backward_accumulation="native-fast" if settings.backward_backend == "native_fast" else "native-precise",
                          backward_fallback="triton-unsupported-layout", torch_version=str(torch.__version__),
                          cuda_version=torch.version.cuda)
    return result
