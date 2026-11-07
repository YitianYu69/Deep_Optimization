"""Scoped native AMP and FP8 linear execution, without modifying model classes."""

from contextlib import ContextDecorator, ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import asdict

import torch
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode, _get_current_function_mode, _pop_mode_temporarily
from deepopt._config_resolution import _UNSET, resolve_config
from deepopt.config import FP8Config
from ._execution import finite_check_scope


# Values are normalized operation plans or False (explicit dense override).
# Missing device keys mean that an existing DO operation keeps its own plan.
_active_fp8 = ContextVar("deepopt_active_fp8", default={})
_active_interception = ContextVar("deepopt_fp8_linear_interception", default={})
_autocast_entries = ContextVar("deepopt_autocast_entries", default=())
_FLOAT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _get_fp8_plan(device_type):
    return _active_fp8.get().get(device_type)


def _linear_arguments(args, kwargs):
    """Leave malformed calls to PyTorch, including duplicate/unknown keywords."""
    names = ("input", "weight", "bias")
    if len(args) > 3 or set(kwargs) - set(names) or any(name in kwargs for name in names[:len(args)]):
        return None
    values = tuple(args[index] if index < len(args) else kwargs.get(name)
                   for index, name in enumerate(names))
    x, weight, bias = values
    tensors = (x, weight) if bias is None else values
    if any(not isinstance(value, torch.Tensor) or value.is_nested
           or value.layout != torch.strided or value.dtype not in _FLOAT_DTYPES for value in tensors):
        return None
    if x.device.type not in ("cpu", "cuda") or any(value.device != x.device for value in tensors):
        return None
    if x.ndim < 1 or weight.ndim != 2 or not x.numel() or not weight.numel():
        return None
    return values


class _FP8LinearMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is F.linear:
            arguments = _linear_arguments(args, kwargs)
            if arguments is not None:
                plan = _get_fp8_plan(arguments[0].device.type)
                if (plan is not None and plan is not False
                        and _active_interception.get().get(arguments[0].device.type, True)):
                    from .do_nn.functional import do_linear

                    return do_linear(*arguments, plan=plan)
        return func(*args, **kwargs)


@contextmanager
def _suspend_fp8_interception():
    """Skip our broad dispatcher inside an already-selected FP8 operation.

    Other user modes and tensor subclass handlers remain active. If a user mode
    is above ours, leave its stack intact instead of disabling all Torch modes.
    """
    if isinstance(_get_current_function_mode(), _FP8LinearMode):
        with _pop_mode_temporarily():
            yield
    else:
        yield


class _FP8Scope:
    """Resolve once; each entry owns only its context token and mode lifetime."""

    def __init__(self, device_type, fp8=_UNSET, *, config=None):
        self.device_type = device_type
        self.inherit_outer = fp8 is _UNSET
        self.plan = None
        self.intercept_linear = True
        if fp8 is None:
            return
        from .policy import fp8_plan, normalize_fp8_config

        settings = normalize_fp8_config(
            {"enabled": True, **fp8} if isinstance(fp8, Mapping) else fp8, config=config)
        if fp8 is _UNSET and not settings.enabled:
            return
        if settings.enabled and device_type not in ("cpu", "cuda"):
            raise ValueError("FP8 autocast supports CPU and CUDA devices")
        from .do_nn.functional import _prepare_fp8_plan

        self.plan = _prepare_fp8_plan(fp8_plan(settings)) if settings.enabled else False
        self.intercept_linear = settings.intercept_linear

    @contextmanager
    def enter(self):
        previous = _active_fp8.get()
        previous_interception = _active_interception.get()
        if (self.plan is None or (self.inherit_outer and previous.get(self.device_type) is not None)
                or (previous.get(self.device_type) == self.plan
                    and previous_interception.get(self.device_type) == self.intercept_linear)):
            yield
            return
        token = _active_fp8.set({**previous, self.device_type: self.plan})
        interception_token = _active_interception.set({**previous_interception,
                                                      self.device_type: self.intercept_linear})
        try:
            with _FP8LinearMode() if self.intercept_linear else nullcontext():
                yield
        finally:
            _active_interception.reset(interception_token)
            _active_fp8.reset(token)


def fp8_context(device_type, fp8=_UNSET, *, config=None):
    """Apply FP8 policy, preserving ambient AMP and explicit disable/inherit."""
    return _FP8Scope(device_type, fp8, config=config).enter()


class autocast(ContextDecorator):
    """Compose torch.autocast with optional FP8 linear execution.

    ``enabled`` controls native AMP only. ``fp8=True`` enables ordinary
    nn.Linear/F.linear calls; a dictionary or FP8Config selects its policy.
    ``fp8=False`` disables FP8 in nested scopes; ``None`` inherits it.
    Other operations keep their normal PyTorch/AMP behavior. Backward uses the
    FP8 plan saved during forward, even when executed after leaving the scope.

    Omitted AMP/FP8 settings use an owned configuration snapshot. An active
    native AMP dtype and outer FP8 policy are preserved when their controls are
    omitted. With no defaults loaded, native AMP keeps torch.autocast defaults.

    Like native AMP, use scopes within one thread's synchronous execution; do
    not suspend an active scope across interleaved async tasks or generators.
    """

    def __init__(self, device_type, dtype=_UNSET, enabled=_UNSET, cache_enabled=None, fp8=_UNSET,
                 *, config=None, check_finite=None):
        if check_finite is not None and type(check_finite) is not bool:
            raise TypeError("check_finite must be a boolean or None")
        self.check_finite = check_finite
        self.device_type = device_type
        self.dtype = dtype
        self.enabled = enabled
        self.cache_enabled = cache_enabled
        self.fp8 = _UNSET if fp8 is _UNSET else deepcopy(fp8)
        explicit_fp8 = _UNSET
        if isinstance(fp8, FP8Config):
            explicit_fp8 = asdict(fp8)
        elif isinstance(fp8, Mapping):
            explicit_fp8 = {"enabled": True, **fp8}
        elif type(fp8) is bool:
            explicit_fp8 = {"enabled": fp8}
        overrides = None if explicit_fp8 is _UNSET else {"runtime": {"fp8": explicit_fp8}}
        self.config = resolve_config(config, overrides=overrides)
        self._fp8_scope = _FP8Scope(device_type, self.fp8, config=self.config)

    def _recreate_cm(self):
        # Entries store their state in context variables, so decorators can
        # reuse the resolved immutable policy without resolving configuration.
        return copy(self)

    def __enter__(self):
        stack = ExitStack()
        try:
            precision = self.config.get("runtime", {}).get("precision")
            ambient_amp = torch.is_autocast_enabled(self.device_type)
            enabled = self.enabled
            if enabled is _UNSET:
                enabled = ambient_amp or (precision == "bf16" if precision is not None else True)
            dtype = self.dtype
            if dtype is _UNSET:
                dtype = (torch.get_autocast_dtype(self.device_type) if ambient_amp else
                         torch.bfloat16 if precision == "bf16" else None)
            stack.enter_context(torch.autocast(self.device_type, dtype=dtype,
                                               enabled=enabled, cache_enabled=self.cache_enabled))
            if self.check_finite is not None:
                stack.enter_context(finite_check_scope(self.check_finite))
            stack.enter_context(self._fp8_scope.enter())
        except BaseException:
            stack.close()
            raise
        _autocast_entries.set((*_autocast_entries.get(), stack))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        entries = _autocast_entries.get()
        stack = entries[-1]
        _autocast_entries.set(entries[:-1])
        return stack.__exit__(exc_type, exc_value, traceback)
