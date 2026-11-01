"""Use ordinary functions, existing tasks, or module hooks in the same loop."""

from collections.abc import Mapping
from copy import copy, deepcopy

import torch
from torch import Tensor
from torchmetrics import Metric

from deepopt.module import TrainingModule
from deepopt.tasks.base import BatchOutput
from deepopt.tasks.loss import LossResult


def _detach(value):
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return {key: _detach(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_detach(item) for item in value)
    return value


def _callable_name(value):
    if value is None:
        return None
    return (f"{value.__module__}.{value.__qualname__}" if hasattr(value, "__qualname__") else
            f"{type(value).__module__}.{type(value).__qualname__}")


class StepAdapter:
    def __init__(self, model, task=None, *, loss_fn=None, step_fn=None,
                 validation_step_fn=None, metrics=None, metric_inputs=None):
        self.model = model
        for name, function in (("loss_fn", loss_fn), ("step_fn", step_fn),
                               ("validation_step_fn", validation_step_fn)):
            if function is not None and not callable(function):
                raise TypeError(f"{name} must be callable")
        if loss_fn is not None and step_fn is not None:
            raise ValueError("Choose loss_fn or step_fn, not both")
        self.loss_fn, self.step_fn = loss_fn, step_fn
        self.validation_step_fn, self.metric_inputs = validation_step_fn, metric_inputs
        self.functional = loss_fn is not None or step_fn is not None
        if validation_step_fn is not None and not self.functional:
            raise ValueError("validation_step_fn requires loss_fn or step_fn")
        if metrics is not None:
            if not isinstance(metrics, Mapping):
                raise TypeError("metrics must map names to torchmetrics.Metric instances")
            if any(not isinstance(name, str) or not name or name in
                   {"loss", "samples", "batches", "seconds", "samples_per_second"} for name in metrics):
                raise ValueError("Metric names must be nonempty and must not use reserved reporting names")
            if any(not isinstance(metric, Metric) for metric in metrics.values()):
                raise TypeError("Custom metrics must inherit torchmetrics.Metric")
            if len({id(metric) for metric in metrics.values()}) != len(metrics):
                raise ValueError("Each metric name must have its own Metric instance")
        self.metrics = dict(metrics) if metrics is not None else None
        if isinstance(metric_inputs, Mapping):
            if self.metrics is None or not set(metric_inputs).issubset(self.metrics):
                raise ValueError("Per-metric input functions must name metrics supplied in metrics")
            if any(not callable(function) for function in metric_inputs.values()):
                raise TypeError("metric_inputs must contain callable input functions")
        elif metric_inputs is not None and not callable(metric_inputs):
            raise TypeError("metric_inputs must be callable or map metric names to input functions")
        self.task = task

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, task):
        if self.functional:
            if task is not None:
                raise ValueError("Choose task or loss_fn/step_fn, not both")
        elif isinstance(self.model, TrainingModule):
            if task is not None:
                raise ValueError("Pass the task to TrainingModule, or override its step methods")
        elif task is None:
            raise ValueError("Provide loss_fn or step_fn; alternatively provide a Task or use TrainingModule")
        self._task = task

    def create_metrics(self):
        if self.metrics is not None:
            # Templates are never updated; each phase owns independent states.
            return {name: deepcopy(metric) for name, metric in self.metrics.items()}
        if self.functional:
            return {}
        source = self.model if self.task is None else self.task
        return source.create_metrics()

    def update_metrics(self, metrics, output):
        if not self.functional and self.metrics is None and self.metric_inputs is None:
            source = self.model if self.task is None else self.task
            return source.update_metrics(metrics, output)
        if not metrics:
            return
        per_metric_inputs = isinstance(self.metric_inputs, Mapping)
        shared_data = None if per_metric_inputs else _detach(output.metric_data)
        for name, metric in metrics.items():
            data = _detach(output.metric_data[name]) if per_metric_inputs else shared_data
            if not isinstance(data, (tuple, list, Mapping)):
                raise ValueError("Metrics need inputs: return metric_data as a tuple or keyword dictionary, "
                                 "or supply metric_inputs(output, batch)")
            if isinstance(data, Mapping):
                metric.update(**data)
            else:
                metric.update(*data)

    def checkpoint_config(self):
        if not self.functional and self.metrics is None and self.metric_inputs is None:
            return None
        return {"loss_fn": _callable_name(self.loss_fn), "step_fn": _callable_name(self.step_fn),
                "validation_step_fn": _callable_name(self.validation_step_fn),
                "metric_inputs": ({name: _callable_name(function) for name, function in self.metric_inputs.items()}
                                  if isinstance(self.metric_inputs, Mapping) else _callable_name(self.metric_inputs)),
                "metrics": ({name: _callable_name(metric) for name, metric in self.metrics.items()}
                            if self.metrics is not None else None)}

    def training_step(self, batch) -> BatchOutput:
        return self._compute(batch, training=True)

    def validation_step(self, batch) -> BatchOutput:
        return self._compute(batch, training=False)

    def _compute(self, batch, *, training):
        if self.functional:
            function = self.step_fn if training or self.validation_step_fn is None else self.validation_step_fn
            if function is not None:
                raw = function(self.model, batch)
            else:
                if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                    raise ValueError("loss_fn uses (inputs, targets) batches; use step_fn for other batch interfaces")
                inputs, targets = batch
                predictions = self.model(inputs)
                raw = {"loss": self.loss_fn(predictions, targets),
                       "sample_count": targets.shape[0] if isinstance(targets, Tensor) and targets.ndim else None,
                       "metric_data": (predictions, targets), "predictions": predictions, "targets": targets}
        elif self.task is not None:
            raw = self.task.compute(self.model, batch)
        else:
            step = self.model.training_step if training else self.model.validation_step
            raw = step(batch)
        if isinstance(raw, BatchOutput):
            output = raw
        elif isinstance(raw, (Tensor, LossResult)):
            output = BatchOutput(raw)
        elif isinstance(raw, Mapping) and "loss" in raw:
            metric_data = raw.get("metric_data")
            if "metric_data" not in raw and "predictions" in raw and "targets" in raw:
                metric_data = (raw["predictions"], raw["targets"])
            output = BatchOutput(raw["loss"], raw.get("sample_count"), metric_data=metric_data)
        else:
            raise ValueError("Step computation must return a scalar loss, a mapping containing loss, "
                             "LossResult, or BatchOutput")
        output.validate(check_finite=False)
        if self.metric_inputs is not None:
            # Metric-only transformations do not need a backward graph. Detach
            # existing tensors too: no_grad does not detach a returned input.
            with torch.no_grad():
                data = ({name: self.metric_inputs[name](raw, batch) if name in self.metric_inputs else output.metric_data
                         for name in self.metrics} if isinstance(self.metric_inputs, Mapping) else
                        self.metric_inputs(raw, batch))
                data = _detach(data)
            # Preserve the original loss/reduction without constructing its
            # objective or report, and leave the caller's output untouched.
            output = copy(output)
            object.__setattr__(output, "metric_data", data)
        return output
