"""Versioned BF16 storage policy, independent of Torch and ambient AMP state.

Operator forwards must retain the returned descriptor for their backward.  The
scope only chooses new forward invocations; it never changes a tensor's dtype.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class BF16Policy:
    name: str = "torch_amp"
    version: int = 1

    def __post_init__(self):
        if self.name not in ("torch_amp", "bf16_storage"):
            raise ValueError("BF16 policy must be torch_amp or bf16_storage")
        if type(self.version) is not int or self.version != 1:
            raise ValueError("BF16 policy version must be the integer 1")

    @property
    def storage_enabled(self):
        return self.name == "bf16_storage"

    def as_dict(self):
        return {"name": self.name, "version": self.version}


_DEFAULT = BF16Policy()
_policies = ContextVar("deepopt_bf16_policy", default=None)
_disabled = ContextVar("deepopt_bf16_rewrites_disabled", default=frozenset())
_model_regions = ContextVar("deepopt_bf16_model_regions", default=frozenset())
STORAGE_PRIMITIVE_RECIPE_VERSION = 3


def managed_bf16_primitives(device_type):
    return device_type in _model_regions.get() and current_bf16_policy(device_type).storage_enabled


@contextmanager
def bf16_model_region(device_type):
    token = _model_regions.set(_model_regions.get() | {device_type})
    try:
        yield
    finally:
        _model_regions.reset(token)


def resolve_bf16_policy(value=None):
    """Validate a name or portable descriptor, returning an immutable policy."""
    if value is None:
        return _DEFAULT
    if isinstance(value, BF16Policy):
        return value
    if isinstance(value, str):
        return BF16Policy(value)
    if isinstance(value, Mapping):
        if set(value) != {"name", "version"}:
            raise ValueError("BF16 policy descriptor requires exactly name and version")
        return BF16Policy(value["name"], value["version"])
    raise ValueError("BF16 policy must be a name or a versioned descriptor")


def _device_key(device_type):
    if not isinstance(device_type, str) or device_type not in ("cpu", "cuda"):
        raise ValueError("BF16 policy device_type must be cpu or cuda")
    return device_type


def current_bf16_policy(device_type):
    key = _device_key(device_type)
    if key in _disabled.get():
        return _DEFAULT
    values = _policies.get()
    return _DEFAULT if values is None else values.get(key, _DEFAULT)


def active_bf16_policy(device_type):
    """Distinguish an explicit scope from the implicit AMP default."""
    return (_policies.get() or {}).get(_device_key(device_type))


def bf16_rewrites_disabled(device_type):
    return _device_key(device_type) in _disabled.get()


@contextmanager
def bf16_rewrite_scope(enabled, device_type):
    key = _device_key(device_type)
    if type(enabled) is not bool:
        raise ValueError("BF16 rewrite enabled must be a boolean")
    token = _disabled.set(_disabled.get() - {key} if enabled else _disabled.get() | {key})
    try:
        yield
    finally:
        _disabled.reset(token)


def suspend_bf16_rewrites(device_type):
    return bf16_rewrite_scope(False, device_type)


@contextmanager
def bf16_policy_scope(value, device_type):
    """Select forwards on one device, restoring nested scopes on every exit."""
    policy = resolve_bf16_policy(value)
    key = _device_key(device_type)
    values = dict(_policies.get() or {})
    values[key] = policy
    token = _policies.set(values)
    try:
        yield policy
    finally:
        _policies.reset(token)
