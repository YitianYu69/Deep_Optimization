"""Optional, passive tensor diagnostics shared by automatic and manual loops."""

from dataclasses import dataclass
from collections.abc import Callable, Iterable

import torch
from torch import Tensor

from ._diagnostics import passive_checks_enabled


@dataclass(frozen=True)
class TensorEvent:
    """Borrowed, detached tensors at a named point in training.

    Tensors stay on their original CPU/CUDA device. Observers must treat them
    as read-only. Detachment shares storage; it does not copy data. Store
    scalar summaries rather than these references.
    Raising aborts the operation; returning a value does not modify training.
    """

    kind: str
    name: str
    tensors: tuple[Tensor, ...]


def all_finite(tensors: Iterable[Tensor]) -> bool:
    """Check finite values on each tensor's device, including CPU and CUDA.

    Large real CPU inputs avoid a full-size mask. Accelerator flags are combined
    before a host read (at most one per device, with early exit on failure).
    Grouping by device permits events containing tensors from multiple devices.
    """
    flags = {}
    for tensor in tensors:
        if tensor.layout == torch.sparse_coo:
            # Duplicate coordinates can overflow only after summation.
            tensor = tensor.coalesce().values()
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue  # Integer and boolean tensors cannot contain NaN/Inf.
        if tensor.device.type == "cpu":
            if tensor.numel() >= 32768 and tensor.dtype in (
                    torch.float16, torch.bfloat16, torch.float32, torch.float64):
                checked = tensor.detach() if tensor.requires_grad else tensor
                minimum, maximum = torch.aminmax(checked)
                finite = torch.isfinite(minimum) & torch.isfinite(maximum)
            else:
                finite = torch.isfinite(tensor).all()
            if not finite:
                return False
        else:
            flags.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
    return all(bool(torch.stack(device_flags).all()) for device_flags in flags.values())


class FiniteCheck:
    """Reject NaN/Inf in selected event kinds; other events cost no scan.

    The default covers losses, gradients, reports, and final metrics. Include
    "weights", "activation", or a caller-defined kind to inspect those events.
    Set enabled=False to temporarily disable this particular observer.
    """

    def __init__(self, *, kinds=None, enabled: bool = True):
        kinds = ("loss", "gradients", "report", "metrics") if kinds is None else kinds
        if isinstance(kinds, str):
            raise TypeError("kinds must be a collection of event names, not one string")
        self.kinds = frozenset(kinds)
        if any(not isinstance(kind, str) or not kind for kind in self.kinds):
            raise ValueError("Observer event kinds must be nonempty strings")
        if type(enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        self.enabled = enabled

    def __call__(self, event: TensorEvent) -> None:
        if self.enabled and event.kind in self.kinds and not all_finite(event.tensors):
            if event.kind == "metrics":
                raise ValueError("Metric.compute() must map string names to finite scalar tensors")
            raise FloatingPointError(f"Non-finite {event.name or event.kind}; operation aborted")


class CheckObservers:
    """Internal dispatcher; no observed tensors or training state are stored."""

    def __init__(self, checks=True, *, gate=None):
        if checks is True:
            observers = (FiniteCheck(),)
        elif checks is False:
            observers = ()
        else:
            try:
                observers = tuple(checks)
            except TypeError as exc:
                raise TypeError("checks must be True, False, or an iterable of callable observers") from exc
        if any(not callable(observer) for observer in observers):
            raise TypeError("Every check observer must be callable")
        self._observers: tuple[Callable, ...] = observers
        self._gate = gate

    def allows(self, kind, *, collective=False):
        return ((self._gate is None or self._gate(kind))
                and passive_checks_enabled(kind, collective=collective))

    def interests(self):
        """Static subscriptions for one rank-wide registration, not per-batch RPC."""
        kinds = set()
        for observer in self._observers:
            if not bool(getattr(observer, "enabled", True)):
                continue
            selected = getattr(observer, "kinds", None)
            if selected is None:
                return None
            kinds.update(selected)
        return sorted(kinds)

    @staticmethod
    def _accepts(observer, kind):
        kinds = getattr(observer, "kinds", None)
        return bool(getattr(observer, "enabled", True)) and (kinds is None or kind in kinds)

    def interested(self, kind: str) -> bool:
        return self.allows(kind) and any(self._accepts(observer, kind) for observer in self._observers)

    def observe(self, kind: str, tensors, *, name: str = "") -> None:
        if not isinstance(kind, str) or not kind or not isinstance(name, str):
            raise ValueError("Observations require a nonempty kind and a string name")
        if not self.allows(kind):
            return
        selected = tuple(observer for observer in self._observers if self._accepts(observer, kind))
        if not selected:
            return
        values = (tensors,) if isinstance(tensors, Tensor) else tuple(tensors)
        if any(not isinstance(value, Tensor) for value in values):
            raise TypeError("Observed values must be tensors")
        with torch.no_grad():
            event = TensorEvent(kind, name, tuple(value.detach() for value in values))
            for observer in selected:
                observer(event)


class TensorStatistics:
    """Analyze-mode summaries; never retain activations or their graphs."""

    kinds = frozenset({"weights", "activation", "gradients"})

    def __call__(self, event):
        from deepopt.logger import init_logger

        for index, value in enumerate(event.tensors):
            if value.layout == torch.sparse_coo:
                value = value.coalesce().values()
            if value.numel() == 0:
                continue
            value = value.abs() if value.is_complex() else value
            value = value.to(dtype=torch.float64 if value.dtype == torch.float64 else torch.float32)
            summary = torch.stack((value.mean(), value.std(unbiased=False), value.min(), value.max())).tolist()
            init_logger("deepopt.diagnostics").info("%s[%s] mean=%g std=%g min=%g max=%g",
                                                  event.name or event.kind, index, *summary)
