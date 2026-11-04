# Optional diagnostics with observers

[Documentation index](../README.md) · [Training APIs](../training/README.md) · [Optimization](../optimization/README.md) · [Memory planning](../memory/README.md)

Observers inspect tensors at named points in training without owning the
training computation. Use the default finite checks, turn them off, or pass
ordinary callables for selected events. No observer subclass is required.

For `max_speed`, periodic `balanced` checks, detailed `analyze` diagnostics,
or boundary-safe mode schedules, see [execution policies](../execution/README.md).
Omitting a mode preserves the legacy `checks=True` behavior below.

Run the small offline CPU examples after installing the package:

```bash
python examples/observability/train_observers.py --mode all
python examples/observability/callbacks.py
```

The first prints activation summaries only for the selective observer recipe
and a final loss/update summary for each configuration. The second reports
every fifth update, stops after two completed epochs, and continues to a total
of three. Neither creates files. The example's `--mode` selects its observer
demonstration; it is separate from the trainer's `execution_mode` policy.

## Choose the checks

```python
from deepopt import Trainer, FiniteCheck

trainer = Trainer(model, optimizer, loss_fn=loss_fn)  # Default finite checks.
trainer = Trainer(model, optimizer, loss_fn=loss_fn, checks=False)
trainer = Trainer(model, optimizer, loss_fn=loss_fn,
                  checks=[FiniteCheck(kinds={"loss", "gradients"})])
```

These lines show alternative constructions. An explicit iterable replaces
the defaults: include `FiniteCheck()` in that iterable when you also want its
normal coverage. `checks=True` enables one default `FiniteCheck`; `checks=False`
disables the optional diagnostic observers.

Configure checks on `TrainingRuntime` when supplying a runtime to a trainer:

```python
from deepopt import TrainingRuntime

runtime = TrainingRuntime(model, optimizer, checks=[FiniteCheck()])
trainer = Trainer(runtime, loss_fn=loss_fn)
```

Do not also pass `checks=` to `Trainer(runtime, ...)`. The supplied runtime owns
its precision, optimizer, gradient processing, and diagnostic configuration.
Configured runs accept a boolean TOML toggle:

```toml
[runtime]
checks = false
```

Custom observers are configured in Python. Diagnostic choices and
`runtime.checks` may change on checkpoint restore. Observer objects and their
internal counters are not checkpointed; recreate them yourself.

For standalone inspection, `deepopt.observers.all_finite(tensors)` accepts an
iterable of tensors and returns a boolean. It handles real/complex tensors,
skips integer/boolean values, and coalesces sparse COO values before checking.
Calling a checker directly does not invoke the runtime's failure-state handling.

## CPU and CUDA

Observers already work with CPU and CUDA tensors. The dispatcher detaches
without moving or copying storage, so custom observers receive tensors on their
original devices. The finite checker runs its reductions on those devices and
combines accelerator flags before reading at most one boolean per device. A
single event may include CPU tensors and tensors from multiple CUDA devices.
Returning a Python boolean or raising immediately still requires a host read;
these checks are not fully asynchronous.

Move the model before constructing its optimizer, and select the same device
on the runtime or trainer:

```python
import torch
from deepopt import Trainer, FiniteCheck

device = torch.device("cuda:0")
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
trainer = Trainer(model, optimizer, device=device, loss_fn=loss_fn,
                  checks=[FiniteCheck()])
trainer.fit(train_loader, epochs=3)
```

The trainer moves batches to its device. Custom observers should compute
reductions on the event tensors directly; calling `.cpu()` on a full activation
would introduce an unnecessary device transfer. CUDA use requires a CUDA-enabled
PyTorch installation and a compatible GPU.

## Events and their timing

`TensorEvent` is a frozen container with three fields:

| Field | Value |
| --- | --- |
| `kind` | A nonempty event name, such as `"gradients"` |
| `name` | A descriptive string identifying this observation |
| `tensors` | A tuple of detached, borrowed tensor views |

The standard runtime/loop emits these events:

| Kind | When it is emitted | Default finite check? |
| --- | --- | --- |
| `loss` | The automatic loop inspects a step's loss; runtime backward also inspects its actual objective | Yes |
| `gradients` | Before an update; again after normalization or gradient processing when applicable | Yes |
| `report` | When accumulating reporting totals and weights | Yes |
| `metrics` | After metric computation and flattening, before scalar results are returned | Yes |
| `weights` | Before and after an optimizer update, only when an observer subscribes | No |
| `activation` or a custom name | When user code explicitly calls `runtime.observe()` | No |

These are observation points, not once-per-batch callbacks. For example, a
loss can be observed by the loop and again at backward, while an accumulation
window can produce several gradient observations at its update boundary.
Use [Trainer callbacks](../training/README.md) for epoch or optimizer-update
logging tied to training progress.

Declare periodic callback cadence with `EveryNSteps` so skipped updates avoid
callback-related synchronization:

```python
from deepopt import EveryNSteps
from deepopt.logger import log_output

def report_update(trainer, event):
    log_output(event["optimizer_step"])

trainer.fit(train_loader, epochs=3,
            step_callbacks=[EveryNSteps(report_update, 100)])
```

Observers inspect tensors; callbacks handle update/epoch reports and actions.
An observer's own sampling counter is not restored by a training checkpoint,
whereas `EveryNSteps` uses the trainer's restored global update count.

`FiniteCheck(kinds=None, enabled=True)` checks for NaN/Inf in the default event
kinds. Supply a collection of strings to select kinds; a single string is not
a valid collection here. Add `"weights"`, `"activation"`, or your own event name
to inspect those observations. Toggle an individual check with
`check.enabled = False` when needed.

## Write a callable observer

```python
from deepopt.logger import init_logger

logger = init_logger(__name__)

def activation_summary(event):
    logger.info("%s mean=%s", event.name, event.tensors[0].mean().item())

activation_summary.kinds = {"activation"}

runtime = TrainingRuntime(
    model, optimizer, checks=[FiniteCheck(), activation_summary]
)
```

The optional `kinds` attribute selects events. A callable without that attribute
receives all emitted events. An optional `enabled=False` attribute disables
that callable. Returning a value from an observer does not alter training.

Emit events at meaningful points in a custom step or manual loop:

```python
hidden = encoder(inputs)
runtime.observe("activation", hidden, name="encoder.hidden")
predictions = head(hidden)  # Keep the original differentiable tensor.
```

`observe(kind, tensors, *, name="")` accepts one tensor or an iterable of
tensors. It prepares an event only when an observer is interested. The same
API leaves room for attention scores, regularizer inputs, or other tensor
categories defined by a recipe.

The [observer recipe](../../examples/observability/train_observers.py) captures
its live runtime in a custom step to emit activations. That is appropriate for
execution but not for fake-tensor memory estimation, whose copied step would
still reference the original runtime. Use manual batch sizing for that recipe,
or remove external callbacks from the computation you want to estimate.

A forward hook is useful when a module produces the activation you need:

```python
handle = model.encoder.register_forward_hook(
    lambda module, args, output:
        runtime.observe("activation", output, name="encoder.output")
)
try:
    trainer.fit(train_loader, epochs=3)
finally:
    handle.remove()
```

This snippet assumes `model.encoder` returns a tensor. Select the relevant
tensor explicitly for tuple/dictionary outputs. Register hooks only on the
layers you intend to inspect and remove them when finished. The trainer does
not automatically attach activation hooks to every layer.

Run the complete [observer example](../../examples/observability/train_observers.py):

```bash
python examples/observability/train_observers.py --mode all
python examples/observability/train_observers.py --mode all --device cuda
```

It demonstrates default checks, disabled checks, and selected finite checks
plus activation logging. It uses a small local dataset without downloads or
checkpoint files; training runs on CPU by default or the selected CUDA device.
Its custom step keeps gradients through the original activation while
diagnostics receive a detached view on that same device.

## Read-only behavior and performance

Callbacks run under `torch.no_grad()`. Detachment shares the original storage;
it does not copy tensors or make them immutable. Observers must not mutate
event tensors, consume random numbers, update optimizer state, or retain tensor
references. Keep differentiable objectives in the step and update policies in
the [optimization APIs](../optimization/README.md).

Store small scalar summaries instead of activations. `.item()` can synchronize
an accelerator, so printing every event can substantially affect throughput.
Use selected kinds and a caller-owned sampling cadence for expensive diagnostics.
The internal finite checker combines accelerator flags before a host read and
avoids full-size finite masks for suitable large real CPU tensors; this does
not make arbitrary observers free.

Immediate finite checks and Python scalar logging are not CUDA Graph replay
operations. Keep them outside a captured region; the runtime's execution APIs
reject use during capture. Capture the model computation through the
[CUDA Graph APIs](../cuda/README.md), and observe its outputs afterward. Remove
forward hooks that call `runtime.observe()` before capturing that model.

## What disabling checks does and does not change

`checks=False` removes the optional finite-value scans. Structural requirements
still apply: scalar objectives where required, valid shapes/types, positive
denominators, valid optimizer ownership, and built-in classification label-range
validation. PyTorch guards and checks implemented in user code still run.

A monitored validation score must remain finite to select a best checkpoint.
Use `fit(..., monitor=None)` when no best-score selection is needed. Standalone
loss/`BatchOutput` helpers retain their own finite checks by default; callers
using `validate()` or `loss_sum()` can pass `check_finite=False` to delegate those
diagnostics to the runtime. See [loss semantics](../optimization/README.md).

## Failures and recovery

`FiniteCheck` raises `FloatingPointError` for a nonfinite observed loss,
gradient, report, or weight, and `ValueError` for nonfinite final metric results.
Custom observer exceptions also propagate. A runtime observer failure clears
owned gradients and blocks subsequent updates until state is restored or the
runtime is reconstructed. A check after an optimizer update cannot undo that
update; failed training must not simply continue with the same state.

Use [checkpoint restoration](../checkpoints/README.md) to recover a consistent
model, optimizer, and progress state. Clearing a failure flag alone does not
repair partially changed optimizer/model state.

Implementation: [observer types and dispatcher](../../src/deepopt/observers.py),
[runtime](../../src/deepopt/engine/execution/runtime.py),
[optimizer update sequence](../../src/deepopt/engine/execution/optimizer_step.py).
