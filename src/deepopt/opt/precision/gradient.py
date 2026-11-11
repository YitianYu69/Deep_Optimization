"""Immutable backward edge policies and forward-only call-site binding."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps


_active = ContextVar("deepopt_gradient_edge", default=None)
_native_module = ContextVar("deepopt_native_gradient_boundary", default=False)


def native_boundary_active():
    return _native_module.get()


@dataclass(frozen=True)
class GradientEdgePlan:
    precision: str = "native"
    dtype: str = "e5m2"
    layout: str = "last"
    block_shape: tuple | None = None
    scaling_recipe: str = "current"
    reason: str = "unplanned backward boundary"
    orientation: str = "normal"


def current_gradient():
    """Read only during forward; backward retains the resolved slot on ctx."""
    return _active.get()


def capture_gradient(x):
    from .._autocast import _get_fp8_plan

    if _get_fp8_plan(x.device.type) is False:
        return False
    binding = _active.get()
    if isinstance(binding, GradientBinding):
        if binding.slot.scaling is not None and binding.slot.scaling.scales.device != x.device:
            raise ValueError("Move the prepared model and gradient scales to the input device")
        if binding.edge.precision == "fp8" and x.requires_grad and x.is_leaf:
            return False
        from ..scaling.delayed import current_scaling_batch

        return GradientBinding(binding.slot, binding.training, current_scaling_batch())
    return binding


def prepared_functional_plan(kind):
    from .planner import _frame

    if _native_module.get():
        return None
    frame = _frame.get()
    if frame is not None and getattr(frame.root, "_deepopt_backward_functional", {}).get(frame.training, {}).get(kind):
        settings = frame.root._deepopt_fp8_config
        active = settings.intercept_linear
        if kind.startswith("conv") and settings.intercept_convolution is not None:
            active = settings.intercept_convolution
        if kind in ("batch_norm", "group_norm", "layer_norm") and settings.intercept_normalization is not None:
            active = settings.intercept_normalization
        if "pool" in kind and settings.intercept_pooling is not None:
            active = settings.intercept_pooling
        if not active:
            return None
        return frame.root._deepopt_backward_plan
    return None


def record_gradient_input(binding, *, reused, transpose=False):
    if isinstance(binding, GradientBinding):
        binding.slot.last_input_execution = {"reused_fp8": reused,
                                             "independent_transpose_stages": int(transpose)}


@dataclass(frozen=True)
class GradientBinding:
    slot: object
    training: bool
    scaling_batch: object = None

    @property
    def edge(self):
        return self.slot.edge

    @contextmanager
    def output(self, *args, **kwargs):
        from ..scaling.delayed import autograd_scaling_scope

        with autograd_scaling_scope(self.scaling_batch):
            with self.slot.output(*args, training=self.training, **kwargs) as output:
                yield output


@contextmanager
def bind_gradient(slot, training=True):
    token = _active.set(False if slot is False else GradientBinding(slot, training) if slot is not None else None)
    try:
        yield
    finally:
        _active.reset(token)


def gradient_operation(kind):
    """Bind a prepared module's independent per-call-site gradient decision."""
    def decorate(function):
        @wraps(function)
        def call(module, *args, **kwargs):
            if not hasattr(module, "_gradient_slots"):
                return function(module, *args, **kwargs)
            if _native_module.get():
                return function(module, *args, **kwargs)
            from .planner import _frame

            frame = _frame.get()
            if frame is None or not hasattr(module, "_gradient_slots"):
                return function(module, *args, **kwargs)
            key = "gradient:" + module._gradient_key
            index = frame.positions.get(key, 0)
            frame.positions[key] = index + 1
            slots = module._gradient_slots.get(frame.training, ())
            slot = frame.root.get_submodule(slots[index]) if index < len(slots) else False
            with bind_gradient(slot, module.training):
                return function(module, *args, **kwargs)
        call._deepopt_gradient_operation = True
        return call
    return decorate


class NativeGradientBoundary:
    """Isolate internal calls of opaque/native leaf modules from outer counters."""

    def __init__(self, owner):
        self.owner = owner
        self.override = owner.__dict__.get("forward")

    def __call__(self, *args, **kwargs):
        function = self.override or type(self.owner).forward.__get__(self.owner)
        from .planner import _frame

        frame = _frame.get()
        if frame is None:
            return function(*args, **kwargs)
        key = "gradient:" + self.owner._gradient_key
        index = frame.positions.get(key, 0)
        frame.positions[key] = index + 1
        paths = self.owner._gradient_slots.get(frame.training, ())
        slot = frame.root.get_submodule(paths[index]) if index < len(paths) else None
        token = _native_module.set(True)
        try:
            with bind_gradient(slot, self.owner.training):
                return function(*args, **kwargs)
        finally:
            _native_module.reset(token)


def gradient_entry(kind):
    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            with functional_gradient(kind):
                return function(*args, **kwargs)
        return call
    return decorate


@contextmanager
def functional_gradient(kind):
    from .planner import _frame

    frame = _frame.get()
    if _active.get() is not None or frame is None or _native_module.get():
        yield
        return
    paths = getattr(frame.root, "_deepopt_backward_functional", {}).get(frame.training, {}).get(kind, ())
    key = "gradient_functional:" + kind
    index = frame.positions.get(key, 0)
    frame.positions[key] = index + 1
    slot = frame.root.get_submodule(paths[index]) if index < len(paths) else None
    with bind_gradient(slot, frame.training):
        yield
