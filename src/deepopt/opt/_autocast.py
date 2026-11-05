"""Scoped native AMP and FP8 linear execution, without modifying model classes."""

from contextlib import ContextDecorator, ExitStack, contextmanager
from contextvars import ContextVar
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict

import torch
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode
from deepopt._config_resolution import _UNSET, resolve_config
from deepopt.config import FP8Config


# Values are normalized operation plans or False (explicit dense override).
# Missing device keys mean that an existing DO operation keeps its own plan.
_active_fp8 = ContextVar("deepopt_active_fp8", default={})
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
                if plan is not None and plan is not False:
                    from .do_nn.functional import do_linear

                    return do_linear(*arguments, plan=plan)
        return func(*args, **kwargs)


@contextmanager
def fp8_context(device_type, fp8=_UNSET, *, config=None):
    """Apply only FP8 policy; preserve ambient native AMP for core composition.

    Omitted policy inherits an outer scope or the selected configuration.
    None inherits only the surrounding policy. False explicitly disables FP8, also
    for existing DO modules. A dictionary enables FP8 unless enabled=False.
    """
    if fp8 is None:
        yield
        return

    from .policy import fp8_plan, normalize_fp8_config

    if fp8 is _UNSET and _get_fp8_plan(device_type) is not None:
        yield
        return
    if isinstance(fp8, FP8Config):
        settings = normalize_fp8_config(fp8)
    else:
        settings = normalize_fp8_config({"enabled": True, **fp8} if isinstance(fp8, Mapping) else fp8,
                                        config=config)
    if fp8 is _UNSET and not settings.enabled:
        yield
        return
    if settings.enabled and device_type not in ("cpu", "cuda"):
        raise ValueError("FP8 autocast supports CPU and CUDA devices")
    plan = fp8_plan(settings) if settings.enabled else False
    previous = _active_fp8.get()
    if previous.get(device_type) == plan:
        # Runtime and core may independently scope the same policy.
        yield
        return
    token = _active_fp8.set({**previous, device_type: plan})
    try:
        with _FP8LinearMode():
            yield
    finally:
        _active_fp8.reset(token)


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

    def __init__(self, device_type, dtype=_UNSET, enabled=_UNSET, cache_enabled=None, fp8=_UNSET, *, config=None):
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

    def _recreate_cm(self):
        return type(self)(self.device_type, dtype=self.dtype, enabled=self.enabled,
                          cache_enabled=self.cache_enabled, fp8=self.fp8, config=self.config)

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
            stack.enter_context(fp8_context(self.device_type, self.fp8, config=self.config))
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
