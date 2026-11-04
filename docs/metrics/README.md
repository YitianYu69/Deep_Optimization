# Metrics, loss reports, and metric inputs

[Documentation index](../README.md) · [Training APIs](../training/README.md) · [Examples](../../examples/README.md)

Pass TorchMetrics instances directly. Every custom metric must inherit
`torchmetrics.Metric`; a task or wrapper is optional.

## Runnable recipes

After installing the project, run these commands from the repository root.
All use generated CPU data and write no files.

| Goal | Command | Result to inspect |
| --- | --- | --- |
| Understand unequal-batch loss reports | `python examples/metrics/loss_reporting.py` | Plain loss is 50; explicitly weighted loss and MSE are about 33.33 |
| Define registered metric state | `python examples/metrics/custom_metric.py` | MAE is 2 over three samples, then reset clears the count |
| Route different update signatures | `python examples/metrics/metric_routing.py` | Nested `quality/absolute/mean` plus a separate prediction mean |
| Weight variable-length sequences | `python examples/metrics/weighted_tokens.py` | Six optimizer updates across three epochs, with explicit valid-token weighting |

Source: [loss reporting](../../examples/metrics/loss_reporting.py),
[custom metric](../../examples/metrics/custom_metric.py),
[routing](../../examples/metrics/metric_routing.py), and
[weighted tokens](../../examples/metrics/weighted_tokens.py).

## Add metrics to training

```python
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from deepopt import Trainer

trainer = Trainer(model, optimizer, loss_fn=loss_function,
                  metrics={"mae": MeanAbsoluteError(sync_on_compute=True),
                           "mse": MeanSquaredError(sync_on_compute=True)})
trainer.fit(train_loader, validation_loader, epochs=5,
            monitor="mae", monitor_mode="min")
```

Metric names must be nonempty strings, and each name must have its own metric
instance. `loss`, `samples`, `batches`, `seconds`, and `samples_per_second` are
reserved engine names. Supplied instances are templates: training and
validation use separate copied instances, moved to the training device and
reset for the phase. The templates are not updated.

## How inputs reach a metric

The supervised `loss_fn` route supplies `(predictions, targets)` automatically.
For arbitrary steps, choose an explicit route:

| Step output / trainer option | Metric call |
| --- | --- |
| `{"loss": loss, "predictions": predictions, "targets": targets}` | `metric.update(predictions, targets)` |
| `{"loss": loss, "metric_data": (a, b, c)}` | `metric.update(a, b, c)` |
| `{"loss": loss, "metric_data": {"preds": p, "target": t}}` | `metric.update(preds=p, target=t)` |
| `metric_inputs=function` | `function(raw_output, prepared_batch)` supplies positional or keyword arguments for all metrics |
| `metric_inputs={"metric_name": function}` | The named metric gets its own arguments; unlisted metrics use the step's `metric_data` |

An explicit `metric_data` field takes precedence over `predictions`/`targets`.
Unrecognized field names or targets in an arbitrary batch are not inferred.

```python
from torchmetrics import MeanMetric

trainer = Trainer(
    model, optimizer, step_fn=step,
    metrics={"mse": MeanSquaredError(), "prediction_mean": MeanMetric()},
    metric_inputs={"prediction_mean": lambda output, batch: (output["predictions"],)},
)
```

Here `step` returns `loss`, `predictions`, and
`metric_data=(predictions, targets)`. MSE uses that shared data; MeanMetric
receives predictions alone. The complete
[metric-routing recipe](../../examples/metrics/metric_routing.py) combines this
route with a custom metric returning nested scalar results.

Input functions receive the original output form and the device-prepared batch.
If a step returns a scalar tensor, its input function receives that tensor;
if it returns a dictionary, the function receives that dictionary. Input
functions run under `no_grad()`, and their returned tensors are detached before
metric updates. The same routing options work with optional module/task hooks.

## Metric lifecycle

For each training or validation phase the trainer:

1. Creates fresh metric instances and moves them to its device.
2. Resets them, then calls `update()` for each batch under `no_grad()`.
3. Calls `compute()` for the whole phase and converts scalar results to Python
   values for rows, progress, monitors, and logging.

Child metrics own their statistics. F1, AUROC, and similar metrics compute
across their stored phase state; the trainer does not average their per-batch
scores. Some metrics retain predictions and therefore need memory proportional
to the amount of data observed. That state can also enlarge step checkpoints.

The default `MetricAccumulator` itself subclasses `Metric`. It aggregates loss
report numerators/weights and sample or batch counts, and registers user metrics
as child modules. The training path uses `update()` / `compute()` rather than
calling `metric(output)` per batch, avoiding the extra snapshots needed by the
accumulator's standalone forward behavior.

Metrics and input functions are for reporting. Keep differentiable losses in
your step or loss function. Detached tensors share storage with their originals;
treat metric inputs as read-only. Copy a tensor explicitly inside a metric when
its calculation needs to mutate data.

## Define a custom Metric

Implement `update(...)` and `compute()`, and register running tensor/list states
with `add_state`. Use appropriate `dist_reduce_fx` values when synchronization
is needed. For an additive error statistic, store a total and count rather than
averaging batch means. The runnable
[ErrorSummary recipe](../../examples/metrics/metric_routing.py) does exactly
this; [metrics/custom_metric.py](../../examples/metrics/custom_metric.py) is a smaller reusable
metric definition.

`compute()` may return a tensor containing one value or a nested dictionary of
named one-value tensors. Nested keys are flattened with `/`:

| Registered name | Returned result | Reported name |
| --- | --- | --- |
| `mae` | One-value tensor | `mae` |
| `quality` | `{"mae": value, "mse": value}` | `quality/mae`, `quality/mse` |
| `quality` | `{"absolute": {"mean": value}}` | `quality/absolute/mean` |

Use the complete reported name in `fit(..., monitor="quality/absolute/mean")`.
Vectors must be explicitly reduced or separated into named one-value tensors
by the metric. Python numbers, unnamed vectors, recursive dictionaries, and
flattened name collisions are rejected. Default observers check that metric
values are finite; a selected best-score monitor must remain finite even when
optional diagnostics are disabled.

TorchMetrics owns synchronization through options such as `sync_on_compute`.
Changing a parent's setting does not change its children's settings. Ranks
using synchronized metrics must use consistent schemas and participate in
compute. With an initialized multi-rank process group, the trainer's DDP core
handles gradient synchronization while TorchMetrics handles metric-state
synchronization. Validation shards can have unequal batch counts, including
an empty local shard; every rank still enters the final metric computation.
See [distributed training](../distributed/README.md) for sampler and launch rules.

For manual integration through `trainer.new_metrics()`, keep the trainer or its
runtime alive while updating or computing that collection: its diagnostic
callback borrows the runtime. A standalone `torchmetrics.Metric` has its own
ordinary lifecycle.

## Loss reporting and sample counts

Plain scalar objectives produce a report that is the mean of batch objectives.
An optional `LossResult` can supply explicit report weights or omit the report.
Optimization weights and reporting weights are separate concepts; see
[explicit loss weights](../optimization/README.md#explicit-weights-and-reports).

The default accumulator reports `samples` only when **every** batch supplies a
positive integer `sample_count`. Otherwise it reports `batches`; sample totals
and sample throughput are omitted. Generic scalar steps do not invent sample
counts. The supervised `(inputs, targets)` path counts the first dimension of
tensor targets when it exists; another task or layout should declare its count.

`sample_count` does not determine a loss denominator, metric weighting, token
count, or optimizer normalization. Each metric controls its own counting. For
example, a custom metric may count every valid token while throughput counts
sequences. `samples` and `batches`, when supplied as metric results, must be
positive integer tensors.

Training rows additionally include `seconds` and, when counts are known,
`samples_per_second`. Timing fields are owned by the loop rather than user
metrics. `evaluate()` returns only the metric results.

## Factories for reusable tasks

Optional tasks can create metrics through `metrics_factory`:

```python
from torchmetrics import MeanAbsoluteError
from deepopt.tasks.regression import RegressionTask

task = RegressionTask(metrics_factory=lambda: {
    "mae": MeanAbsoluteError(sync_on_compute=True),
})
```

Each factory call must return new instances. Built-in tasks pass predictions
and targets to each metric. A custom task implements
`update_metrics(metrics, output)` when it needs another signature or selected
slices of `output.metric_data`. That method calls `update()` only; the lifecycle
owns compute and reset. Reused instances, including nested reused children,
are rejected. Supplying explicit `metrics` to `Trainer` overrides task-created
metric templates.

## Replace the whole accumulator

For advanced reporting, `Trainer(..., metric_factory=MyAccumulator)` accepts a
factory returning a fresh `Metric` subclass with:

- `update(output)`, receiving the normalized `BatchOutput`;
- `compute()`, returning a dictionary of named scalar tensors or nested scalar
  dictionaries.

This replaces the default accumulator and task metric-update route. Include
`loss` if your accumulator defines a loss report. `samples` and `batches` are
optional positive integer counts; `seconds` and `samples_per_second` remain
reserved. The trainer still checks the backward objective independently.
Do not combine `metric_factory` with `metrics` or `metric_inputs`.

This hook is useful when reporting genuinely requires a different accumulator;
ordinary recipes can pass a dictionary of Metric instances instead.

## Step checkpoint state

Step checkpoints save registered metric states, metric types, and update
counters, including nested children. Snapshots are detached copies, and restore
invalidates compute/synchronization caches. State that exists only in arbitrary
Python attributes is not captured automatically; register it with TorchMetrics.

The adapter reads the public `update_count` property and restores TorchMetrics'
underlying `_update_count`, which is absent from its state dict. For that reason,
step resume requires the same TorchMetrics version. The user does not need to
set these counters manually. See the full
[checkpoint contract](../checkpoints/README.md).

Related: [observers](../observers/README.md), [memory estimates](../memory/README.md),
[configured runs](../runs/README.md), and [architecture](../architecture/README.md).
