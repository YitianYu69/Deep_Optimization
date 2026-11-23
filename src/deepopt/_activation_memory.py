"""Owned activation-memory recipes; validation and scoping never import Torch."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


MEMORY_RECIPE_VERSION = 1
DEFAULT_REDUCTION_SCRATCH_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ActivationMemoryConfig:
    multiply_backward: str = "inherit"
    pack_reuse: str = "inherit"
    reduction_scratch_bytes: int = DEFAULT_REDUCTION_SCRATCH_BYTES
    recompute_modules: tuple[str, ...] = ()

    def __post_init__(self):
        if self.multiply_backward not in ("inherit", "materialized", "packed_streaming"):
            raise ValueError("multiply_backward must be inherit, materialized, or packed_streaming")
        if self.pack_reuse not in ("inherit", "off", "forward"):
            raise ValueError("pack_reuse must be inherit, off, or forward")
        if type(self.reduction_scratch_bytes) is not int or self.reduction_scratch_bytes < 4:
            raise ValueError("reduction_scratch_bytes must be an integer >= 4")
        paths = self.recompute_modules
        if not isinstance(paths, (tuple, list)) or any(
            not isinstance(path, str) or not path or any(
                not part or not (part.isidentifier() or part.isdecimal())
                for part in path.split(".")) for path in paths
        ):
            raise ValueError("recompute_modules must contain valid nonempty module paths")
        paths = tuple(sorted(paths))
        if len(set(paths)) != len(paths) or any(
            child.startswith(parent + ".") for index, parent in enumerate(paths)
            for child in paths[index + 1:]
        ):
            raise ValueError("recompute_modules must be unique and must not overlap")
        object.__setattr__(self, "recompute_modules", paths)


@dataclass(frozen=True)
class ActivationMemoryPlan(ActivationMemoryConfig):
    """Portable forward-time selection retained by operation backward owners."""

    version: int = MEMORY_RECIPE_VERSION

    def __post_init__(self):
        super().__post_init__()
        if type(self.version) is not int or self.version != MEMORY_RECIPE_VERSION:
            raise ValueError(f"activation memory recipe version must be the integer {MEMORY_RECIPE_VERSION}")

    def as_dict(self):
        return {"version": self.version, **self.as_config()}

    def as_config(self):
        return {"multiply_backward": self.multiply_backward, "pack_reuse": self.pack_reuse,
                "reduction_scratch_bytes": self.reduction_scratch_bytes,
                "recompute_modules": list(self.recompute_modules)}

    @property
    def is_default(self):
        return self == _DEFAULT


_DEFAULT = ActivationMemoryPlan()
_plans = ContextVar("deepopt_activation_memory", default=None)


def resolve_activation_memory(value=None, *, compiler_options=None):
    """Normalize configuration/descriptor and reject explicit native conflicts.

    Inherit leaves historical compiler booleans intact. Explicit selections
    constrain those booleans through the same resolver used by eager scopes.
    """
    if value is None:
        plan = _DEFAULT
    elif isinstance(value, ActivationMemoryPlan):
        plan = value
    elif isinstance(value, ActivationMemoryConfig):
        plan = ActivationMemoryPlan(value.multiply_backward, value.pack_reuse,
                                    value.reduction_scratch_bytes, value.recompute_modules)
    elif isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("activation memory keys must be strings")
        unknown = set(value) - {"multiply_backward", "pack_reuse", "reduction_scratch_bytes",
                                "recompute_modules", "version"}
        if unknown:
            raise ValueError(f"activation memory: unknown fields: {sorted(unknown)}")
        plan = ActivationMemoryPlan(**value)
    else:
        raise ValueError("activation memory must be a configuration or versioned descriptor")
    options = compiler_options or {}
    for key, choice, enabled in (("fuse_packed_multiply", plan.multiply_backward, "packed_streaming"),
                                 ("share_compatible_packs", plan.pack_reuse, "forward")):
        if key in options and type(options[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
        if choice != "inherit" and key in options and options[key] != (choice == enabled):
            raise ValueError(f"activation memory {choice} conflicts with explicit compiler option {key}")
    if plan.multiply_backward == "packed_streaming":
        for key in ("enable_aot_autodiff", "use_native_autodiff"):
            if options.get(key) is False:
                raise ValueError(f"activation memory packed_streaming conflicts with explicit compiler option {key}")
        phases = options.get("fusion_phases", ("forward", "backward"))
        if phases in ("none", "forward") or isinstance(phases, (tuple, list)) and "backward" not in phases:
            raise ValueError("activation memory packed_streaming requires the backward fusion phase")
    return plan


def activation_memory_compiler_options(value, options=None):
    """Return native options matching the explicitly selected eager recipe."""
    result = dict(options or {})
    plan = resolve_activation_memory(value, compiler_options=result)
    if plan.multiply_backward != "inherit":
        result["fuse_packed_multiply"] = plan.multiply_backward == "packed_streaming"
    if plan.pack_reuse != "inherit":
        result["share_compatible_packs"] = plan.pack_reuse == "forward"
    return result


def validate_activation_memory_execution(value, *, kernel_policy=None, model=None):
    """Reject features lacking an execution owner before model/RNG effects."""
    plan = resolve_activation_memory(value)
    if model is not None:
        names = dict(model.named_modules())
        missing = set(plan.recompute_modules) - names.keys()
        if missing:
            raise ValueError(f"activation memory recompute_modules do not exist: {sorted(missing)}")
    if plan.pack_reuse == "forward":
        raise NotImplementedError("activation memory pack_reuse=forward is not implemented for the shared eager/native recipe")
    if plan.recompute_modules:
        raise NotImplementedError("activation memory module recomputation is not implemented")
    if plan.multiply_backward == "packed_streaming" and kernel_policy is not None:
        from .kernel_policy import resolve_kernel_policy

        policy = resolve_kernel_policy(kernel_policy)
        if (policy.route("activation_codec") != "triton"
                or policy.route("activation_codec", "compiler.backward") != "triton"
                or policy.route("compiler_pointwise", "compiler.backward") != "triton"):
            raise ValueError("activation memory packed_streaming requires Triton activation_codec and compiler.backward kernel routes")
    return plan


def _device_key(device_type):
    if device_type not in ("cpu", "cuda"):
        raise ValueError("activation memory device_type must be cpu or cuda")
    return device_type


def active_activation_memory(device_type):
    return (_plans.get() or {}).get(_device_key(device_type))


def current_activation_memory(device_type):
    return active_activation_memory(device_type) or _DEFAULT


def invocation_activation_memory(plan, device_type):
    """An owned nondefault plan survives an empty qualification scope."""
    owned = validate_activation_memory_execution(plan.get("activation_memory") if isinstance(plan, Mapping) else plan)
    active = active_activation_memory(device_type)
    if not owned.is_default and active is not None and active != owned:
        raise ValueError("Activation memory differs from the owned operation/model plan")
    return owned if not owned.is_default else active or owned


@contextmanager
def activation_memory_scope(value, device_type):
    plan = validate_activation_memory_execution(value)
    key = _device_key(device_type)
    token = _plans.set({**(_plans.get() or {}), key: plan})
    try:
        yield plan
    finally:
        _plans.reset(token)
