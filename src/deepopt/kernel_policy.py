"""Explicit, Torch-free custom-kernel selection and execution ownership.

Package installation never selects a backend. Policies contain only immutable
metadata and can safely be retained by autograd, checkpoints and compiler caches.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps, cached_property
from hashlib import sha256
import json


BACKENDS = ("triton", "tilelang")
KERNEL_FAMILIES = frozenset((
    "gemm", "quantization", "scaling", "activation_codec", "gradient",
    "activation", "normalization", "pooling", "convolution",
    "compiler_pointwise", "compiler_preparation",
))
_ROLES = frozenset(
    f"{operation}.{role}"
    for operation, roles in (
        ("linear", ("forward", "dx", "dw", "db", "prepare_input", "prepare_weight", "prepare_gradient")),
        ("convolution", ("forward", "dx", "dw", "db")),
        ("activation", ("forward", "backward")),
        ("normalization", ("forward", "backward")),
        ("pooling", ("forward", "backward")),
        ("compiler", ("forward", "backward", "preparation", "add", "sub", "mul", "div", "neg",
                      "relu", "sigmoid", "tanh", "gelu", "silu", "leaky_relu", "hardtanh",
                      "hardswish", "hardsigmoid", "softplus", "mish")),
    ) for role in roles
)


@dataclass(frozen=True)
class KernelConfig:
    backend: str = "inherit"
    preferred: str = "triton"
    fallback: str = "error"
    families: dict[str, str] = field(default_factory=dict)
    roles: dict[str, str] = field(default_factory=dict)


def kernel_config_from_dict(raw):
    if isinstance(raw, KernelConfig):
        raw = {name: getattr(raw, name) for name in ("backend", "preferred", "fallback", "families", "roles")}
    if not isinstance(raw, Mapping):
        raise ValueError("runtime.kernels must be a table")
    if any(not isinstance(key, str) for key in raw):
        raise ValueError("runtime.kernels keys must be strings")
    unknown = set(raw) - {"backend", "preferred", "fallback", "families", "roles"}
    if unknown:
        raise ValueError(f"runtime.kernels: unknown fields: {sorted(unknown)}")
    backend, preferred, fallback = (raw.get("backend", "inherit"), raw.get("preferred", "triton"),
                                     raw.get("fallback", "error"))
    if backend not in ("inherit", "triton", "tilelang", "mixed"):
        raise ValueError("runtime.kernels.backend must be inherit, triton, tilelang, or mixed")
    if preferred not in BACKENDS:
        raise ValueError("runtime.kernels.preferred must be triton or tilelang")
    if fallback not in ("error", "other"):
        raise ValueError("runtime.kernels.fallback must be error or other")
    maps = {}
    for name, allowed in (("families", KERNEL_FAMILIES), ("roles", _ROLES)):
        values = raw.get(name, {})
        if not isinstance(values, Mapping):
            raise ValueError(f"runtime.kernels.{name} must be a table")
        if any(not isinstance(key, str) for key in values):
            raise ValueError(f"runtime.kernels.{name} route keys must be strings")
        if set(values) - allowed:
            raise ValueError(f"runtime.kernels.{name}: unknown routes: {sorted(set(values) - allowed)}")
        if any(value not in BACKENDS for value in values.values()):
            raise ValueError(f"runtime.kernels.{name} routes must be triton or tilelang")
        maps[name] = dict(values)
    if backend != "mixed" and (preferred != "triton" or fallback != "error" or any(maps.values())):
        raise ValueError("preferred, fallback and route overrides require runtime.kernels.backend=mixed")
    return KernelConfig(backend, preferred, fallback, **maps)


@dataclass(frozen=True)
class KernelPolicy:
    backend: str = "triton"
    preferred: str = "triton"
    fallback: str = "error"
    families: tuple[tuple[str, str], ...] = ()
    roles: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.backend not in ("triton", "tilelang", "mixed"):
            raise ValueError("Resolved kernel policy must use triton, tilelang, or mixed")
        object.__setattr__(self, "families", tuple(sorted(dict(self.families).items())))
        object.__setattr__(self, "roles", tuple(sorted(dict(self.roles).items())))
        kernel_config_from_dict(self.as_dict())

    def route(self, family, role=None):
        if family not in KERNEL_FAMILIES:
            raise ValueError(f"Unknown custom kernel family: {family!r}")
        if self.backend != "mixed":
            return self.backend
        return dict(self.roles).get(role, dict(self.families).get(family, self.preferred))

    def pinned(self, family, role=None):
        return self.backend != "mixed" or role in dict(self.roles) or family in dict(self.families)

    def as_dict(self):
        return {"backend": self.backend, "preferred": self.preferred, "fallback": self.fallback,
                "families": dict(self.families), "roles": dict(self.roles)}

    @cached_property
    def fingerprint(self):
        return sha256(json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_DEFAULT = KernelPolicy()
_TILELANG = KernelPolicy("tilelang")
_policy = ContextVar("deepopt_kernel_policy", default=None)
_role = ContextVar("deepopt_kernel_role", default=None)


def current_kernel_policy():
    return _policy.get() or _DEFAULT


def current_kernel_role():
    return _role.get()


def resolve_kernel_policy(options=None, *, fp8_backend=None):
    if isinstance(options, KernelPolicy):
        policy = options
    elif options is None:
        policy = ((_TILELANG if fp8_backend == "tilelang" else _DEFAULT) if fp8_backend in BACKENDS
                  else _policy.get() or _DEFAULT)
    elif isinstance(options, str) and options in BACKENDS:
        policy = _TILELANG if options == "tilelang" else _DEFAULT
    else:
        if isinstance(options, str):
            options = {"backend": options}
        if options is None and _policy.get() is not None:
            policy = _policy.get()
        else:
            config = kernel_config_from_dict(options or {})
            mode = config.backend
            if mode == "inherit":
                mode = fp8_backend if fp8_backend in BACKENDS else "triton"
            policy = KernelPolicy(mode, config.preferred if mode == "mixed" else "triton",
                                  config.fallback, tuple(sorted(config.families.items())),
                                  tuple(sorted(config.roles.items())))
    if fp8_backend in BACKENDS and (policy.backend == "mixed" or policy.backend != fp8_backend):
        raise ValueError("Conflicting FP8 and kernel backends; use fp8.backend=inherit with the kernel policy")
    return policy


def kernel_backend(family, backend=None, role=None):
    if backend in ("reference", "native_fast", "native_precise", "inherit"):
        backend = None
    if backend == "mixed" and _policy.get() is not None and _policy.get().backend == "mixed":
        backend = _policy.get()
    policy = resolve_kernel_policy(backend)
    ambient = _policy.get()
    if backend is not None and ambient is not None and ambient.backend != "mixed" and policy.backend != ambient.backend:
        raise ValueError("A custom backend override conflicts with the bound pure kernel policy")
    return policy.route(family, role or _role.get())


def kernel_role(role):
    """Mixed products own auxiliary routing; pure modes keep their fast path."""
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            policy = current_kernel_policy()
            if policy.backend != "mixed":
                return function(*args, **kwargs)
            selected_role = role(*args, **kwargs) if callable(role) else role
            with kernel_policy_scope(policy, role=selected_role):
                return function(*args, **kwargs)
        return call
    return decorate


@contextmanager
def kernel_policy_scope(policy=None, *, role=None):
    selected = resolve_kernel_policy(policy)
    token = _policy.set(selected) if _policy.get() != selected else None
    role_token = _role.set(role) if role is not None and role != _role.get() else None
    try:
        yield selected
    finally:
        if role_token is not None:
            _role.reset(role_token)
        if token is not None:
            _policy.reset(token)


def select_kernel_backend(family, eligibility, *, policy=None, role=None):
    """Select only from metadata eligibility; execution errors never enter here."""
    selected = resolve_kernel_policy(policy)
    role = role or _role.get()
    primary = selected.route(family, role)
    reason = eligibility.get(primary, "no capability decision supplied")
    if reason is None:
        return primary
    if selected.backend == "mixed" and selected.fallback == "other" and not selected.pinned(family, role):
        other = "tilelang" if primary == "triton" else "triton"
        if other in eligibility and eligibility[other] is None:
            return other
    raise ValueError(f"{family} ({role or 'unspecified'}) is unsupported by {primary}: {reason}")


def plan_kernel_policy(plan):
    cached = getattr(plan, "kernel_policy", None)
    if cached is not None:
        return cached
    if not isinstance(plan, Mapping):
        return current_kernel_policy()
    training = plan.get("fp8_training", {})
    return resolve_kernel_policy(plan.get("kernel_policy", training.get("kernel_policy")),
                                 fp8_backend=training.get("backend"))


def kernel_forward(function):
    """Bind autograd's complete route program before allocating any saved state."""
    family = next((name for name in ("linear", "convolution", "activation", "normalization", "pooling")
                   if name in function.__module__.split(".")), None)
    if "convolution_transpose" in function.__module__.split("."):
        family = "convolution"
    @wraps(function)
    def forward(ctx, *args, **kwargs):
        plans = [value for value in (*args, *kwargs.values())
                 if isinstance(value, Mapping) and ("fp8_training" in value or "kernel_policy" in value)]
        ctx.kernel_policy = plan_kernel_policy(plans[0]) if plans else current_kernel_policy()
        role = f"{family}.forward" if family and ctx.kernel_policy.backend == "mixed" else None
        with kernel_policy_scope(ctx.kernel_policy, role=role):
            return function(ctx, *args, **kwargs)
    return forward


def kernel_backward(function):
    family = next((name for name in ("linear", "convolution", "activation", "normalization", "pooling")
                   if name in function.__module__.split(".")), None)
    if "convolution_transpose" in function.__module__.split("."):
        family = "convolution"
    @wraps(function)
    def backward(ctx, *args, **kwargs):
        policy = getattr(ctx, "kernel_policy", current_kernel_policy())
        role = f"{family}.backward" if family and policy.backend == "mixed" else None
        with kernel_policy_scope(policy, role=role):
            return function(ctx, *args, **kwargs)
    return backward


def kernel_entry(function):
    """Enter an explicit operation plan without changing an enclosing model."""
    @wraps(function)
    def call(*args, **kwargs):
        plan = kwargs.get("plan")
        if plan is None:
            return function(*args, **kwargs)
        selected = plan_kernel_policy(plan)
        if isinstance(plan, Mapping) and "activation_memory" in plan:
            from ._activation_memory import invocation_activation_memory, validate_activation_memory_execution

            device = next((getattr(getattr(value, "device", None), "type", None)
                           for value in (*args, *kwargs.values())
                           if getattr(getattr(value, "device", None), "type", None) in ("cpu", "cuda")), None)
            if device is not None:
                memory = invocation_activation_memory(plan, device)
                validate_activation_memory_execution(memory, kernel_policy=selected)
        if selected == current_kernel_policy():
            return function(*args, **kwargs)
        with kernel_policy_scope(selected):
            return function(*args, **kwargs)
    return call


def decoder_kernel_policy(owner=None):
    """An explicit consumer scope may override the producer's default decoder."""
    return _policy.get() or owner or _DEFAULT


def owned_kernel_policy(function):
    """Use saved state ownership when called after its execution scope exits."""
    @wraps(function)
    def call(owner, *args, **kwargs):
        selected = resolve_kernel_policy(getattr(owner, "kernel_policy", None))
        ambient = _policy.get()
        if ambient is not None and ambient.backend != "mixed" and ambient.backend != selected.backend:
            raise ValueError("Delayed state kernel policy conflicts with the bound pure kernel policy")
        with kernel_policy_scope(selected):
            return function(owner, *args, **kwargs)
    return call
