"""Create isolated metric collections and validate their reporting results."""

from collections.abc import Mapping
from weakref import WeakValueDictionary

import torch
from torchmetrics import Metric

from deepopt.observers import CheckObservers
from deepopt.engine.reporting.metrics import MetricAccumulator


def _flatten_results(results: Mapping) -> dict[str, torch.Tensor]:
    """Preserve scalar leaves under explicit names; never reduce metric values."""
    flattened = {}
    active = set()

    def visit(mapping, prefix=""):
        if id(mapping) in active:
            raise ValueError("Metric result dictionaries must not contain cycles")
        active.add(id(mapping))
        try:
            for name, value in mapping.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("Metric.compute() must use nonempty string names")
                path = f"{prefix}/{name}" if prefix else name
                if isinstance(value, Mapping):
                    if not prefix and name in {"loss", "samples", "batches"}:
                        raise ValueError(f"Reserved root metric {name!r} must be a scalar tensor")
                    visit(value, path)
                    continue
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"Metric result {path!r} must be a scalar tensor or a dictionary of named scalars")
                if value.numel() != 1:
                    raise ValueError(f"Metric result {path!r} is not scalar; reduce it explicitly "
                                     "or return a dictionary of named scalar tensors")
                if path in flattened:
                    raise ValueError(f"Metric names collide after flattening: {path!r}")
                flattened[path] = value
        finally:
            active.remove(id(mapping))

    visit(results)
    return flattened


class MetricLifecycle:
    def __init__(self, device, *, observe=None):
        self.device = device
        if observe is not None and not callable(observe):
            raise TypeError("observe must be callable")
        self._observe = CheckObservers().observe if observe is None else observe
        self._check_loss = observe is None
        # TorchMetrics hashes include mutable states; track identity by id.
        self._instances = WeakValueDictionary()

    def create(self, source, factory=None) -> Metric:
        metric = (MetricAccumulator(source.create_metrics(), update_metrics=source.update_metrics,
                                    observe=self._observe, check_loss=self._check_loss)
                  if factory is None else factory())
        if not isinstance(metric, Metric):
            raise ValueError("metric_factory must return a torchmetrics.Metric subclass instance")
        instances = [child for child in metric.modules() if isinstance(child, Metric)]
        if any(self._instances.get(id(child)) is child for child in instances):
            raise ValueError("Metric factories must return fresh instances for each phase")
        self._instances.update({id(child): child for child in instances})
        metric = metric.to(self.device)
        metric.reset()
        return metric

    @staticmethod
    def compute(metric: Metric, *, observe=None) -> dict:
        if observe is not None and not callable(observe):
            raise TypeError("observe must be callable")
        results = metric.compute()
        if not isinstance(results, Mapping):
            raise ValueError("Metric.compute() must return a dictionary")
        if {"seconds", "samples_per_second"}.intersection(results):
            raise ValueError("Metric.compute() returned a reserved timing metric")
        results = _flatten_results(results)
        observe = CheckObservers().observe if observe is None else observe
        observe("metrics", tuple(results.values()), name="metric results")
        scalars = {name: value.item() for name, value in results.items()}
        for name in ("samples", "batches"):
            if name in scalars and (type(scalars[name]) is not int or scalars[name] <= 0):
                raise ValueError(f"Metric.compute()[{name!r}] must be a positive integer tensor")
        return scalars
