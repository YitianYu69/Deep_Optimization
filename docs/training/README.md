# Training APIs and custom computations

[Documentation index](../README.md) · [Runnable examples](../../examples/README.md)

Start with a PyTorch model, optimizer, and iterable of batches. No configuration
file, model schema, task subclass, or wrapper is required. Choose `Trainer.fit()`
when DeepOpt should control training, or `TrainingRuntime` when your recipe owns
the loop and update order.

## Choose an entry point

| Your computation | Entry point | Runnable example |
| --- | --- | --- |
| `model(inputs)` and one scalar loss | `Trainer(..., loss_fn=...)` | [regression.py](../../examples/training/regression.py) |
| Arbitrary batches, multiple forwards, or combined losses | `Trainer(..., step_fn=...)` | [train_functional.py](../../examples/training/train_functional.py), `--mode custom` |
| Reusable methods on a model wrapper | `TrainingModule.training_step()` | [train_module.py](../../examples/training/train_module.py), `--mode fit` |
| A recipe that controls backward and updates | `TrainingRuntime` | [train_module.py](../../examples/training/train_module.py), `--mode manual` |

From the repository root, after `python -m pip install -e .`:

```bash
python examples/training/regression.py
python examples/training/train_functional.py --mode custom --epochs 2
python examples/training/train_module.py --mode manual
python examples/training/multimodal.py
```

These examples run on CPU with generated data and write no files. The module
example also accepts `--mode wrapper` for ordinary inference with no trainer.

## Use shared configuration defaults

`load_config(source)` validates and installs process-local defaults. A source can
be a full dictionary, `ExperimentConfig`, TOML path, or `builtin:NAME`. Loading
does not construct models/loaders, seed random generators, start training, or
create files. `get_config()` returns a copy; `reset_config()` clears defaults.

This complete CPU recipe uses global defaults for its loader, optimizer, and
epoch target while keeping the model, data, and loss in Python:

```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from deepopt import Trainer, load_config
from deepopt.data import build_loaders

load_config({
    "name": "shared-defaults", "seed": 7,
    "training": {"epochs": 2, "batch_size": 4, "learning_rate": 0.03,
                 "optimizer": "sgd", "momentum": 0.0},
    "runtime": {"device": "cpu", "precision": "fp32"},
})
torch.manual_seed(7)  # The recipe still owns model/data randomness.
inputs = torch.randn(12, 3)
dataset = TensorDataset(inputs, inputs.sum(dim=1, keepdim=True))
loader, _, _ = build_loaders(dataset, None)
trainer = Trainer(nn.Linear(3, 1), loss_fn=F.mse_loss)
trainer.fit(loader, progress_bar=False)
```

For newly constructed components, explicit arguments win over `config=`, which
wins over loaded defaults. `config=None` uses those defaults. A partial dictionary
merges into them; a typed experiment, selected TOML file, or built-in preset is
a complete source. Passing `config=` does not replace the process-global state.
See the [configuration guide](../runs/README.md) for the schema and presets.

Trainer/runtime/core objects retain their own snapshots. Later `load_config()`
calls affect new objects, not existing ones; `.config` returns an owned copy.
With no loaded defaults, direct APIs keep their ordinary behavior. Required
information is still required: an automatic optimizer needs a learning rate,
and `fit()` needs an epoch target either explicitly or in the trainer snapshot.

`Trainer(model, optimizer=None, ...)` can create the configured optimizer after
device placement, FP8 preparation, and compilation. A supplied optimizer is
retained, including its parameter groups and learning rates. Omitted schedulers
are built when an LR policy is configured; `scheduler=None` explicitly disables
that construction. Similarly, `schedules=None` disables inherited coefficients
and `grad_clip_norm=None` disables inherited norm clipping.

`TrainingRuntime` and the bare cores do not create optimizers. When passing a
runtime to `Trainer`, that runtime keeps ownership of its device, precision,
FP8, diagnostics, and optimizer; configure those on the runtime. Trainer-level
epochs, accumulation, metrics, and scheduling remain trainer responsibilities.

## Train with a loss function

For batches containing exactly `(inputs, targets)`:

```python
from torch.nn import functional as F
from deepopt import Trainer

trainer = Trainer(model, optimizer, loss_fn=F.mse_loss)
state = trainer.fit(train_loader, validation_loader, epochs=5)
results = trainer.evaluate(validation_loader)
```

This path calls `model(inputs)`, then `loss_fn(predictions, targets)`. Use any
callable, including a PyTorch loss module. Return a differentiable scalar tensor;
the callable owns `.mean()`, `.sum()`, masks, and combined loss terms. The
framework does not infer the scalar's original denominator. See
[loss and accumulation rules](../optimization/README.md#losses-and-gradient-accumulation).

Move the model to the chosen device **before** constructing its optimizer, then
pass that device to `Trainer`. Without configured defaults, a direct trainer
uses CPU; configured device selection also applies to direct construction.
Direct construction does not seed your run. The optimizer must contain every
trainable model parameter exactly once and may not contain outside parameters.
Frozen model parameters may remain in its groups.

Complete programs: [regression recipe](../../examples/training/regression.py) and
[functional loss/step examples](../../examples/training/train_functional.py).

## Write a step for an arbitrary task

Use `step_fn(model, batch)` for multiple model inputs, multiple forward passes,
self-supervised objectives, or a different batch structure:

```python
def step(model, batch):
    predictions = model(batch["features"], batch["context"])
    targets = batch["targets"]
    loss = custom_loss(predictions, targets)
    return {"loss": loss, "predictions": predictions, "targets": targets}

trainer = Trainer(model, optimizer, step_fn=step,
                  validation_step_fn=validation_step)
```

The validation function is optional; it defaults to the training computation.
Steps run after batch transfer and inside the precision context. A step computes
the objective and reporting inputs. `fit()` owns backward, accumulation, and
optimizer updates. Use a manual loop for alternating or custom updates.

| Step return value | Use |
| --- | --- |
| Scalar tensor | Ordinary training; reported loss is a mean of batch objectives |
| Dictionary containing `loss` | Add metric inputs and an optional `sample_count` |
| `LossResult` | Explicit optimization weights, independent reports, or no loss report |
| `BatchOutput` | Optional structured form, mainly for reusable tasks/strategies |

A dictionary's `loss` can itself be a tensor or `LossResult`. `predictions` and
`targets` route to metrics together; explicit `metric_data` takes precedence.
Other fields are available to `metric_inputs(raw_output, batch)`. No arbitrary
field names are guessed. See [metrics and routing](../metrics/README.md).

The default transfer handles tensors nested in mappings, lists, tuples,
named tuples, and dataclass fields. Non-tensor values pass through. Mapping
types are preserved when their constructor can reproduce the transferred data;
otherwise a plain dictionary is used. For another transfer contract, override
`TrainingRuntime.prepare_batch()` and pass that runtime to `Trainer`.

The [multimodal recipe](../../examples/training/multimodal.py) demonstrates a
self-supervised step with two forward passes and no classification labels.

## Fit, evaluate, and control epochs

| Method or option | Behavior |
| --- | --- |
| `fit(train_loader, validation_loader=None, epochs=N)` | Train until the **total** completed epoch count reaches `N`; return `TrainingState` |
| `train_epoch(loader, step_callbacks=...)` | Run/finalize one training epoch, advance counters and the configured scheduler, return training metrics; no validation or best-score selection |
| `evaluate(loader)` | Return fresh whole-phase metrics; use eval mode and `no_grad()`, then restore previous module training flags |
| `monitor="loss"`, `monitor_mode="min"` | Default best-score selection when validation is present |
| `monitor=None` | Keep validation while disabling best-score selection |
| `progress_bar=False` | Disable the default tqdm epoch bar on stderr |
| `request_stop()` | Stop after the current epoch completes |
| `new_metrics()` | Create a fresh phase metric collection for advanced integration |

Omitted `epochs`, `monitor`, and `monitor_mode` use the trainer's snapshot, not
defaults loaded later. A call can override only these fit controls:

```python
trainer.fit(train_loader, config={"training": {"epochs": 4}},
            monitor=None, progress_bar=False)
```

Explicit fit arguments have priority over that per-call configuration. It cannot
change the existing runtime, optimizer, batch size, or accumulation policy.
The existing run's monitor/direction compatibility checks still apply.

`fit(train_loader, epochs=N)` supports training without validation; validation
results are empty and best-score selection is disabled. An explicit monitor
requires validation. The monitored name must exist, be finite, and remain
consistent with the run's previous monitor and direction. Ties preserve the
earlier best epoch. `N` must exceed the already completed count for direct
`fit()`; the configured runner separately supports artifact repair for a
completed run.

Training and validation loaders must provide at least one batch. The exception
is a restored step checkpoint at an exhausted epoch: previously saved progress
allows that epoch to finish without replaying its data.

`TrainingState` exposes `completed_epochs`, `optimizer_steps`, `best_score`,
`best_epoch`, `monitor`, and `monitor_mode`. A completed training epoch is counted
before validation and callbacks; a later failure still makes automatic
continuation unsafe. Restore a checkpoint or reconstruct the run before retrying.

## Exponential moving averages for validation

Enable EMA in the trainer's configuration when validation should use averaged
model parameters:

```toml
[training.ema]
enabled = true
decay = 0.999
```

Use `load_config()` before constructing the trainer, or pass the settings with
`Trainer(..., config={"training": {"ema": {"enabled": True, "decay": 0.999}}})`.
EMA is disabled by default. Decay must be finite and satisfy `0 <= decay < 1`.
The initial shadow equals the initial model. After every successful optimizer
update, `shadow = decay * shadow + (1 - decay) * live_parameter`; accumulation
microbatches do not advance it separately. Step callbacks see the completed
average, and a failed optimizer step does not advance its counter.

Training and optimizer updates always use live parameters. `Trainer.evaluate()`
and validation inside `fit()` temporarily install the EMA values in existing
parameter storage, preserving parameter identities and optimizer references.
Validation starts with current buffers, including BatchNorm statistics and
integer counters; buffers are not averaged. Live parameters and buffers are
restored even when validation raises. Best-checkpoint selection therefore uses
the EMA validation result when this option is enabled.

Shadows use FP32 for lower-precision floating-point parameters and otherwise
retain their dtype. Enabled EMA requires materialized dense parameters and adds
shadow storage; validation also allocates temporary parameter/buffer backups.
The trainer logs the additional shadow bytes and a persistent estimate including
EMA. Model wrappers are not deep-copied. Configure EMA before training and keep
the model's parameters, shapes, and dtypes fixed afterward.

Checkpoints store live model weights together with EMA shadows and their update
count. Resume requires the same enabled policy and decay. Configured checkpoint
evaluation restores EMA automatically; see the [checkpoint guide](../checkpoints/README.md)
for direct evaluation.

## Callbacks, progress, and saving

```python
from deepopt.logger import init_logger

logger = init_logger(__name__)

def on_epoch(trainer, row):
    logger.info("Epoch report: %s", row)
    if row["epoch"] == 3:
        trainer.request_stop()

trainer.fit(train_loader, validation_loader, epochs=10,
            callbacks=[on_epoch], progress_bar=False)
```

Epoch callbacks receive `(trainer, row)` after training, validation, and
best-score/state updates. The progress bar advances after every callback
succeeds. Each row contains:

- `epoch` and cumulative `optimizer_steps`;
- `batch_size`, the training loader's batch size per rank before gradient
  accumulation, captured when starting or resuming the epoch (`null` in JSON
  when the loader does not expose a fixed batch size);
- `learning_rate` and `learning_rates`, captured at the epoch's start;
- `train` and `validation` metric dictionaries.

Training metrics include elapsed `seconds`; `samples_per_second` is present
when all sample counts are known. Validation results contain metrics rather
than training timing. User metric names and loss reports are described in the
[metrics guide](../metrics/README.md).

`step_callbacks=[callback]` receives `(trainer, event)` after each successful
optimizer update, including any per-update scheduler step. The event contains
`epoch`, `optimizer_step`, `batches_processed`, and `samples_processed` (possibly
`None`). Pending gradients have been cleared at this boundary. A stop request
from any callback still takes effect at the epoch boundary.

Callable epoch callbacks run in their supplied order; exceptions propagate. `fit()` does not
create files, save a best model, or retain a history list automatically.
Attach `Checkpoint(directory, every_steps=N)` to `callbacks` for those behaviors
without maintaining history yourself.
[Checkpoint helpers](../checkpoints/README.md) and the
[configured runner](../runs/README.md) attach those behaviors. Automatic batch
size planning belongs to that runner or the separate
[memory-planning helpers](../memory/README.md), not to bare `fit()`.

Use `EveryNSteps(callback, N)` in `step_callbacks` to run a function at a chosen
optimizer-update cadence. Between due callbacks, the loop skips callback-related
synchronization. Plain callables keep their original every-update behavior.
Callback objects may expose `on_fit_start(trainer, train_loader, validation_loader)`
and `on_optimizer_step(trainer, event)`; those optimizer hooks run in callback-list
order before explicit `step_callbacks`. A `should_run_step(next_update)` predicate
on an object controls its update-hook cadence. Checkpoint uses these hooks; normal
recipes need not implement them.

The complete [callback recipe](../../examples/observability/callbacks.py) logs updates,
stops after epoch two, and continues the same trainer to a total of three epochs.

## Constructor options

| Options | Purpose |
| --- | --- |
| `model, optimizer` | Supply an optimizer, or let Trainer build it from configured learning rate/settings; an existing `TrainingRuntime` may be supplied as `model` |
| `config` | Partial overrides or a full configuration source; `None` inherits defaults when constructing the trainer |
| `loss_fn` / `step_fn` / `task` | Choose one computation route; a `TrainingModule` may supply its own hooks |
| `validation_step_fn` | Separate validation computation with the functional loss/step route |
| `device`, `precision` | Trainer-level AMP: FP32 without AMP on CPU/CUDA, or BF16 AMP on CUDA; see below |
| `fp8` | Opt-in automatic projection conversion and FP8 execution; `True` or `FP8Config(enabled=True, ...)` |
| `accumulation_steps` | Positive number of microbatches per optimizer update |
| `grad_clip_norm` / `gradient_strategy` | Choose a built-in clipping convenience or custom gradient processing |
| `loss_strategy` | Optional replacement for objective normalization |
| `scheduler`, `scheduler_interval` | Scheduler attached to this optimizer; interval is `"epoch"` or `"step"` |
| `metrics`, `metric_inputs` | TorchMetrics templates and optional input routing |
| `metric_factory` | Advanced replacement for the whole metric accumulator; mutually exclusive with `metrics`/`metric_inputs` |
| `checks` | Default finite checks, `False`, or selected observer callables |
| `execution_mode` / `diagnostics` | Select a diagnostic preset or immutable custom policy |
| `mode_schedule` | Apply mode transitions at optimizer-update boundaries |
| `non_blocking` | Request nonblocking transfers through nested batches |
| `experiment_identity` | Optional caller-declared recipe/data/code versions, checked for equality on resume |
| `schedules` | Optional shared `ScheduleController`; prepare coefficients at accumulation-window boundaries; see [scheduling](../scheduling/README.md) |

When passing an existing runtime, configure optimizer, device, precision, FP8,
clipping/gradient strategy, and checks on that runtime. Do not supply those
options again to `Trainer`. Configure metrics, loss computation, accumulation,
and scheduling on the trainer.

## Precision

```python
model = model.to("cuda:0")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
trainer = Trainer(model, optimizer, device="cuda:0", precision="bf16",
                  loss_fn=loss_function)
```

This section describes **trainer-level automatic mixed precision (AMP)**.
The example requires a CUDA device supporting BF16. CPU uses FP32 with AMP
disabled; CUDA supports FP32 with AMP disabled or BF16 with AMP enabled.
Trainer-level FP16 AMP and gradient scaling are not implemented.
[FP8 projection training](../fp8/README.md) is a separate operation-level feature,
configured through its FP8 plan rather than `precision="fp8"`.
Enable automatic preparation with `Trainer(..., fp8=True)` or
`TrainingRuntime(..., fp8=True)`; `runtime.autocast()` combines its AMP and FP8
policies. Custom loops can use `deepopt.autocast(..., fp8=True)` directly.
The [FP8 engine example](../fp8/README.md#train-with-the-regular-engine-core)
shows `Trainer`, a manual runtime loop and direct `TrainingCore` usage.
`TrainingRuntime.backward()` owns
backward through the selected execution core. The regular core uses one device;
the [DDP core](../distributed/README.md) runs one process per device with shared
trainer/runtime orchestration.

## Own the loop with TrainingRuntime

```python
from deepopt import TrainingRuntime

runtime = TrainingRuntime(model, optimizer)

def compute_loss(batch):
    inputs, targets = batch
    return loss_function(model(inputs), targets)

model.train()
for batch in train_loader:
    runtime.zero_grad()
    loss = runtime.compute(batch, compute_loss)
    runtime.backward(loss)
    runtime.step()
```

`compute()` applies transfer and autocast and enters the selected execution core.
Use this form for loops that may run under DDP: calling the raw model and then
`runtime.backward()` would bypass the DDP forward contract. Direct
`prepare_batch()`/`autocast()` remain available for single-device integrations.

| Runtime method | Responsibility |
| --- | --- |
| `prepare_batch(batch)` | Device transfer without prescribing the task |
| `autocast()` | Context manager for the configured precision |
| `compute(batch, step)` | Transfer, then call `step(prepared_batch)` inside autocast; no train/eval or grad-mode change |
| `backward(loss, **kwargs)` | Optional loss observation, then PyTorch backward; accepts options such as `retain_graph` |
| `step(gradient_divisor=1)` | Normalize owned gradients, process/check them, update the optimizer, clear gradients |
| `zero_grad()` | Clear the optimizer's gradients using `set_to_none=True` |
| `observe(kind, tensors, name=...)` | Emit an optional diagnostic event |
| `synchronize()` | CUDA synchronization when using CUDA; a no-op on CPU |

A runtime may have no optimizer when only transfer/autocast/backward helpers
are needed. `zero_grad()` and `step()` require one. A runtime's optimizer may
own a subset of model parameters; its processing and cleanup cover that subset.
Manual loops own counters, metrics, schedules, saving, mode changes, and update
timing. `step()` does not maintain an accumulation counter. The optional
[`OptimizationWindow`](../optimization/README.md#accumulation-in-a-manual-loop)
helper reuses accumulation without taking over the outer loop.

Use ordinary `optimizer.step(closure)` in your own loop for closure-based
optimizers. The [alternating-update recipe](../../examples/optimization/alternating_updates.py)
shows two runtimes, parameter freezing, and gradient isolation between updates.
Do not interleave independent manual updates with an active `fit()` lifecycle.

A failed backward, observer, or optimizer update clears owned gradients and
blocks further runtime updates. A failed accumulation window is also unusable.
An update may already have changed parameters; clearing gradients cannot undo
it. Restore/reconstruct state, then create a fresh window when needed.
`invalidate()` and `reset_after_restore()` support that recovery boundary;
calling the latter alone does not restore training state.

## Optional modules and tasks

`TrainingModule(model)` can be used solely as an ordinary `nn.Module` wrapper.
It forwards positional/keyword model inputs and registers the wrapped model's
parameters and buffers under a `model.` state-dict prefix. No trainer is needed.

For reusable computations, override `training_step(batch)` and optionally
`validation_step(batch)`, then give the module to `Trainer`. These hooks accept
the same return forms as functional steps. The default validation hook calls
the training hook. Alternatively, provide `TrainingModule(model, task=task)`.
See the [wrapper, manual, and fit examples](../../examples/training/train_module.py).

`Task` is a structural protocol in `deepopt.tasks.base`. A compatible object
implements `compute(model, batch)`, `create_metrics()`, and
`update_metrics(metrics, output)`. Explicit inheritance is optional; it does not
mean overriding an implementation from a concrete base. `compute` returns a
step result; `create_metrics` returns fresh TorchMetrics instances; the update
method calls metric `update()` without `compute()` or `reset()`.

The built-in presets are optional:

| Task | Built-in contract |
| --- | --- |
| `ClassificationTask` | `(inputs, targets)` with nonempty class-index targets `[batch]`; model logits `[batch, classes]`; caller-supplied `num_classes` enables micro accuracy and checks the logit count; omit it for loss-only training |
| `RegressionTask` | Matching nonempty prediction/target shapes; default MSE averages elements within each example, then applies the selected mean/sum across examples; default metric is MSE |

Both accept `loss_fn`, `metrics_factory`, and a built-in `reduction_type` of
`"mean"` or `"sum"`. For a custom loss, omit `reduction_type` and reduce inside
the callable. Built-in means use explicit example weights during accumulation;
generic scalar loss functions follow the batch-objective rules instead.
Custom masking or target formats belong in your loss or step.

Explicit `metrics` templates and `metric_inputs` also work with module/task
computations. When omitted, their own metric methods remain active.

## Next guides

- [Execution modes and compilation](../execution/README.md), [FP8 transformation](../fp8/README.md), and [CUDA Graphs](../cuda/README.md)
- [Distributed training](../distributed/README.md): the same trainer with the DDP core
- [Optimization, loss weights, and strategies](../optimization/README.md)
- [Metrics and reporting](../metrics/README.md)
- [Observers and diagnostics](../observers/README.md)
- [Configured runs](../runs/README.md), [checkpoints](../checkpoints/README.md), and [memory planning](../memory/README.md)
- [Architecture and contributor reference](../architecture/README.md)

## Console and file logging

`fit()` emits start, completed-epoch, interruption, and completion diagnostics.
It does not own log paths. Wrap a recipe in `logging_session(config, "logs")`
for file capture; configured runs do this automatically. See the
[logging guide](../logging/README.md) and
[complete example](../../examples/observability/logging_recipe.py).
