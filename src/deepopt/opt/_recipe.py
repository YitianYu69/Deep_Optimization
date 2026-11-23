"""Numerical recipe identity for checkpoints and installed-library reports."""

from importlib.metadata import PackageNotFoundError, version
import sys


FP8_RECIPE_VERSION = 5


def bf16_recipe(value=None):
    """Independent rounding/storage identity; missing legacy values mean AMP."""
    from deepopt._bf16_policy import resolve_bf16_policy

    policy = resolve_bf16_policy(value)
    result = {**policy.as_dict(), "parameter_storage": "fp32-master",
            "parameter_gradient": "fp32", "reduction_accumulation": "fp32",
            "external_leaf_gradient": "source-dtype",
            "activation_storage": "bf16" if policy.storage_enabled else "torch-amp"}
    if policy.storage_enabled:
        from deepopt._bf16_policy import STORAGE_PRIMITIVE_RECIPE_VERSION

        result.update(primitive_recipe_version=STORAGE_PRIMITIVE_RECIPE_VERSION,
                      addition="promoted-arithmetic-bf16-store-reduce-in-promoted-domain",
                      concatenation="branch-store-bf16-cast-before-master-reduction")
    return result


def activation_compression_recipe():
    """Version mandatory gradient semantics independently of compute precision."""
    return {"version": 2, "layer_norm_xhat": "centered-forward-second-moment",
            "multiplication_coefficient": "consumer-specific-bf16-affine-snapshot",
            "multiplication_gradient": "promoted-product-round-then-sum-to-size"}


def fp8_recipe(backend, settings=None, *, kernel_policy=None):
    if backend == "inherit":
        from deepopt.kernel_policy import resolve_kernel_policy
        backend = resolve_kernel_policy(kernel_policy).backend
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
    if backend in ("tilelang", "mixed"):
        try:
            tilelang_version = version("tilelang")
        except PackageNotFoundError:
            tilelang_version = None
        result.update(accumulation=f"{backend}-explicit-kernel-policy", tilelang_version=tilelang_version)
    if settings is not None and settings.scaling_recipe == "delayed":
        from dataclasses import asdict

        result.update(scaling="delayed", delayed_scaling=asdict(settings.delayed_scaling))
    if settings is not None and (settings.backward_backend != "inherit" or settings.exclude_modules):
        result.update(backward_backend=settings.backward_backend,
                      exclude_modules=list(settings.exclude_modules))
        if settings.backward_backend != "inherit":
            import torch

            result.update(backward_accumulation="native-fast" if settings.backward_backend == "native_fast" else "native-precise",
                          backward_fallback=f"{backend}-unsupported-layout", torch_version=str(torch.__version__),
                          cuda_version=torch.version.cuda)
    return result


def kernel_recipe(model, options=None):
    """Record actual prepared policies, including compression-only adapters.

    Historical Triton defaults add no checkpoint keys. Version discovery uses
    package metadata and never imports either optional compiler.
    """
    from deepopt.kernel_policy import plan_kernel_policy, resolve_kernel_policy, KernelPolicy
    from deepopt.operations import operation_recipes

    base = resolve_kernel_policy(getattr(model, "_deepopt_kernel_policy", options))
    operations = operation_recipes(model)
    routes = {}
    for name, module in model.named_modules():
        plan = getattr(module, "plan", None)
        if isinstance(plan, dict):
            selected = plan_kernel_policy(plan)
            if selected != KernelPolicy():
                routes[name or "<root>"] = selected.as_dict()
    if base == KernelPolicy() and not routes and not operations:
        return None
    versions = {}
    for backend in ("triton", "tilelang"):
        policies = [base, *(resolve_kernel_policy(value) for value in routes.values())]
        if not any(policy.backend == backend or policy.backend == "mixed" for policy in policies):
            continue
        try:
            versions[backend] = version(backend)
        except PackageNotFoundError:
            versions[backend] = None
    result = {"version": 1, "policy": base.as_dict(), "prepared_operations": routes, "compiler_versions": versions}
    if operations:
        result["registered_operations"] = operations
    return result
