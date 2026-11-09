# Losses, optimizers, accumulation, and strategies

[Documentation index](../README.md) · [Training APIs](../training/README.md) · [Examples](../../examples/README.md)

Use normal PyTorch loss functions and optimizers for an ordinary recipe.
Explicit weighting, multiple optimizers, and replaceable strategies are optional
tools for computations that need them.

## Run the examples

From the repository root after installation:

```bash
python examples/optimization/train_strategies.py --api fit --weighting samples
python examples/optimization/train_strategies.py --api manual --weighting samples
python examples/optimization/train_strategies.py --api manual --weighting microbatches
python examples/optimization/multiple_optimizers.py
python examples/optimization/alternating_updates.py
```

These are small CPU programs with generated data and no output files. The
[strategy example](../../examples/optimization/train_strategies.py) uses five
microbatches per epoch, so accumulation by three leaves a final window with
batch sizes six and three. This exposes the difference between sample weighting
and equal microbatch weighting. Compare `fit` and `manual` with the same
weighting option before changing the weighting policy.

For explicit DDP `rank_mean`, `global_mean`, and `global_sum` objectives, see
[distributed reduction contracts](../execution/README.md#ddp-accumulation-and-reductions).
The legacy `LossResult.from_sum` behavior remains rank-averaged under DDP unless
an explicit global-sum strategy is selected.

## Losses and gradient accumulation

```python
from torch.nn import functional as F
from deepopt import Trainer

trainer = Trainer(model, optimizer, loss_fn=F.smooth_l1_loss,
                  accumulation_steps=4, grad_clip_norm=1.0)
```

A custom loss owns its reduction and can combine terms. Return a differentiable
scalar; return `.mean()`, `.sum()`, or your chosen scalar objective for unreduced
values. Plain tensors do not declare whether their values were originally
means, sums, or something else.

For the default loss strategy on one device:

| Result | One microbatch per update | Several microbatches per update | Reported phase `loss` |
| --- | --- | --- | --- |
| Plain scalar tensor | Backward the scalar unchanged | Mean of the supplied scalars over the actual window | Mean of batch objectives |
| `LossResult.from_mean(value, weight=w)` | Backward the scalar unchanged | Weighted mean using explicit weights | Weighted mean using the same weights |
| `LossResult.from_sum(value, report_weight=w)` | Backward the supplied sum | Add supplied sums; no division of gradients | Sum of values divided by total report weight |
| `LossResult(value)` | Backward the scalar unchanged | Add supplied objectives; no division | Omitted |

If a plain scalar is already a sum, automatic accumulation still averages
those batch objectives. Select `LossResult.from_sum` when addition across
microbatches is the intended objective. `sample_count` counts throughput only;
it never supplies an inferred loss denominator.

A final incomplete accumulation window is applied using its actual number of
microbatches or total explicit weight. Gradients are normalized before clipping
and the optimizer update. Accumulation does not reproduce a large physical
batch for batch-dependent operations such as BatchNorm.

The optional classification/regression task presets define their own explicit
example weights: built-in means use example-weighted accumulation, built-in
sums accumulate sums, and both report a mean per example. A custom `loss_fn`
on those tasks should omit `reduction_type` and return its own scalar or
`LossResult`. See [task contracts](../training/README.md#optional-modules-and-tasks).

## Explicit weights and reports

```python
from deepopt import LossResult

def step(model, batch):
    predictions = model(batch["features"])
    objective = masked_token_loss(predictions, batch)
    return {"loss": LossResult.from_mean(objective, weight=batch["valid_tokens"]),
            "sample_count": batch["features"].shape[0]}
```

Here the recipe knows that `objective` is a mean over `valid_tokens`. DeepOpt
backpropagates weighted numerators and divides accumulated gradients by the
window's total weight. Weights may be example counts, token counts, or
fractional importance weights; they must be positive finite Python real
numbers, excluding booleans. The recipe must define how to handle a batch with
no valid values rather than supplying zero as a denominator.

The complete [weighted-token recipe](../../examples/metrics/weighted_tokens.py)
uses variable sequence lengths and unequal token counts, including a partial
accumulation window. It supplies the valid-token counts as batch metadata.

To choose optimization and reporting independently:

```python
from deepopt import LossReport, LossResult

result = LossResult(
    value=objective,
    normalization_weight=optimization_weight,
    report=LossReport(report_total, reporting_weight),
)
```

`LossReport` contains a scalar tensor numerator and a positive finite weight.
The phase report is `sum(total) / sum(weight)`, accumulated from detached values.
Reports must be present for every batch in a phase or absent for all of them.
Weighted and unweighted objectives cannot be mixed within one accumulation
window. When the loss report is absent, select a reported validation metric or
use `fit(..., monitor=None)`.

Trainable loss modules belong on the model with parameters registered in the
optimizer and checkpoint state. Fixed loss weights must be on the correct
device. Passing a separate trainable loss object as `loss_fn` does not
automatically add its parameters to an optimizer. Checkpoints do not serialize
Python function bodies or captured arguments; reconstruct them from your
recipe. See [checkpoint compatibility](../checkpoints/README.md).

## Accumulation in a manual loop

For complete control, accumulate your numerator with `runtime.backward()` and
call `runtime.step(gradient_divisor=total_weight)` at your chosen boundary.
`samples=N` is a legacy integer-only alias for the divisor; use only one option.

The optional `OptimizationWindow` shares the automatic trainer's accumulation
behavior without controlling the outer loop:

```python
from deepopt.engine import OptimizationWindow
from deepopt.tasks.base import BatchOutput

runtime.zero_grad()
window = OptimizationWindow(runtime, accumulation_steps=4)
for batch in train_loader:
    loss = runtime.compute(batch, compute_loss)  # compute_loss(prepared_batch)
    window.backward(BatchOutput(loss))
window.flush()  # Apply a pending partial window; does nothing when empty.
```

`backward(output)` returns `True` when it performs an optimizer update.
`flush()` returns whether it applied a pending update. Manual code owns counters
and scheduler timing. Pass a `loss_strategy` to the window when needed.
After a failed window, restore/reconstruct the affected training state and make
a new window; its failed state cannot be reset for a blind retry.

Run [optimization/train_strategies.py](../../examples/optimization/train_strategies.py) with
`--api fit --weighting samples` or `--api manual --weighting microbatches` to
compare the same policies through both APIs.

## Create an optimizer

Direct recipes pass an ordinary optimizer:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
trainer = Trainer(model, optimizer, loss_fn=loss_function)
```

The optional configured factory `deepopt.optimizer_factory.build_optimizer(model, config)`
supports `adamw`, `sgd`, and a `custom` choice. AdamW uses learning rate and
weight decay; SGD additionally uses momentum. For a parameter-iterable factory:

```python
from deepopt.optimizer_factory import build_optimizer

def apply_optimizer(parameters, settings):
    return torch.optim.AdamW(parameters, lr=settings.learning_rate)

optimizer = build_optimizer(model, training_config, apply_func=apply_optimizer)
```

`apply_func(parameters, settings)` runs once during creation, after model device
placement. It returns a `torch.optim.Optimizer`; it does not customize each
update. For layer-aware routing, use an `optimizer_factory(model, settings)`
with `build_experiment`, because that factory receives the model itself.
`optimizer="custom"` requires a supplied creation function/factory.
See [component wiring](../runs/README.md).

## Different optimizers for different layers

```python
from deepopt import MultiOptimizer

optimizer = MultiOptimizer({
    "encoder": torch.optim.AdamW(model.encoder.parameters(), lr=0.0001),
    "head": torch.optim.SGD(model.head.parameters(), lr=0.01, momentum=0.9),
})
trainer = Trainer(model, optimizer, step_fn=step)
```

The caller decides which parameters belong to each optimizer. `MultiOptimizer`
does not inspect layer names or choose a routing policy. It exposes a combined
`param_groups`, coordinates `zero_grad()` and `step()`, and stores named child
states. Children update in insertion order, once per wrapper update.

Child parameters must be disjoint and cannot repeat. With `Trainer`, their
combined coverage must include every trainable parameter. Configure child
groups before constructing the wrapper; changing assignments/groups afterward
requires rebuilding it. `optimizers` exposes a read-only mapping to the child
optimizers. A scheduler on the wrapper updates the same group dictionaries
used by its children.

Children must support ordinary closure-free updates. The wrapper rejects
LBFGS and supplied closures. A failed child update is not transactional; an
earlier child may already have updated its parameters. Restore the run before
continuing. Saved child names, order, types, layouts, and the checkpoint's named
parameter routing must match on resume.

Complete examples: [multiple optimizers with fit](../../examples/optimization/multiple_optimizers.py)
and [configured layer routing](../../examples/configured/train_multi_optimizer.py).
For alternating schedules, separate losses/backward passes, or different update
frequencies, use the [manual recipe](../../examples/optimization/alternating_updates.py).

## Schedulers

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
trainer = Trainer(model, optimizer, loss_fn=loss_function,
                  scheduler=scheduler, scheduler_interval="step")
```

The scheduler must belong to this optimizer and support `step()` without
required arguments. The default `scheduler_interval="epoch"` steps after a
training epoch that contains an optimizer update, before validation. `"step"`
steps after each successful optimizer update, including the final partial
accumulation window, and before step callbacks. Step counters count updates,
not microbatches. No scheduler is created by direct `Trainer` use.

The configured factory uses `training.lr_schedule` when present, including
warmup, step decay, cosine, and warmup/cosine policies. Without that table it
builds `ExponentialLR` using the legacy `training.lr_gamma`. Use
`scheduler_factory` to choose another scheduler or return `None`. The
[scheduling guide](../scheduling/README.md#learning-rate-policies) documents exact
boundaries, and [learning_rate.py](../../examples/scheduling/learning_rate.py)
shows both update and epoch intervals.
Metric-based schedulers such as `ReduceLROnPlateau` and custom policies belong
in a caller-owned loop. An automatic physical-batch resize changes update
counts; OneCycleLR resizing is rejected and other fixed-horizon schedules need
explicit recipe planning. Learning rates and accumulation are never silently
rescaled by [batch-size planning](../memory/README.md).

## Replace a loss or gradient strategy

Strategies change optimization mechanics; forward/loss computations usually
remain ordinary functions.

| Strategy | Behavior |
| --- | --- |
| `ExplicitObjective()` | Default; follow the scalar and explicit-weight rules above |
| `MicrobatchMean()` | Average supplied scalar objectives equally across actual microbatches, ignoring explicit optimization weights; does not change reporting |
| `ClipGradNorm(max_norm)` | Clip normalized gradients by their norm |
| `ClipGradValue(clip_value)` | Clip normalized gradient values |

```python
from deepopt.engine.strategies import ClipGradValue, MicrobatchMean

trainer = Trainer(model, optimizer, loss_fn=loss_function,
                  accumulation_steps=4, loss_strategy=MicrobatchMean(),
                  gradient_strategy=ClipGradValue(0.5))
```

Both supplied loss strategies agree on plain scalar outputs. They differ when
an optional task or `LossResult` defines another optimization weight. Neither
infers a mean from a sum. `grad_clip_norm=...` is a convenience for
`ClipGradNorm`; choose it or `gradient_strategy`, not both. Built-in clipping
does not support sparse gradients. Default checks inspect dense gradients and
coalesced sparse COO values; custom optimizers/strategies must support the
actual gradient layout.

For a new policy, inherit the relevant class from `deepopt.engine.strategies`:

- `LossStrategy.prepare(output, *, accumulation_steps)` returns
  `BackwardLoss(scalar_loss, normalization_weight=None)` for no gradient
  division, or a positive weight contributing to the final divisor. The
  strategy supplies the corresponding numerator and leaves the report alone.
- `GradientStrategy.process(parameters)` transforms gradients in place after
  normalization and before the update. It must not change parameters or clear
  gradients. An optional tensor return is a diagnostic for observers.

Implement `get_config()` with comparable primitive settings. Stateless
strategies inherit empty `state_dict()` / `load_state_dict()` implementations;
stateful policies override both. Construct fresh instances for each run.
Checkpoints record strategy type/configuration/state and reject a different
type or configuration; they do not instantiate Python classes automatically.
Arbitrary runtime attributes are not persisted. Keep persistent behavior in
strategies, model parameters/buffers, or optimizer state.

## Compatibility forms

Existing `BatchOutput(tensor, count, "mean")`, `"sum"`, and `"none"` are explicit
per-example shorthands. Mean/sum describe already reduced scalars; `"none"`
requires exactly one value per example and selects its mean. Those forms use
`count` as the declared normalization/reporting denominator. A plain
`BatchOutput(tensor, count)` instead uses batch-objective averaging and treats
the count as throughput only.

`output.loss_sum()` is an explicit differentiable conversion for callers that
need it; ordinary non-accumulated backward does not require it. Validation
helpers check declared contracts; they do not discover a tensor's reduction
history. For new recipes, prefer scalar returns or explicit `LossResult` when
weights are needed.

## FP8 model computation

FP8 transformation changes supported model operations and their backward
computations. Ordinary optimizer state, accumulation, clipping, and scheduling
remain owned by the APIs in this guide. Start with
[FP8 training and model transformation](../fp8/README.md) and the
[FP8 example](../../examples/optimization/fp8_training.py). The CPU reference
backend checks FP8 quantization behavior. CUDA forward uses Triton on supported
NVIDIA hardware; tensorwise recipes can opt into native scaled GEMM for dX/dW.
See [native backward and projection selection](../fp8/eager_linear_benchmark.md#native-backward-and-selective-preparation)
for explicit accumulation policies, dense exclusions and benchmark controls.

Related: [metrics](../metrics/README.md), [observers](../observers/README.md),
[checkpoints](../checkpoints/README.md), and [architecture](../architecture/README.md).
