# Checkpoints and resume

[Documentation index](../README.md) · [Configured runs](../runs/README.md) · [Training APIs](../training/README.md)

Checkpoint helpers save the state of an existing trainer. Your recipe constructs
the model, optimizer, computation, metrics, strategies, and loaders before
restoring. The configured runner supplies this wiring through
`experiment_factory`; direct recipes call the helpers themselves.

## Start with the ready-made callback

```python
from deepopt import Checkpoint

checkpoint = Checkpoint("runs/my-recipe", every_steps=0)
trainer.fit(train_loader, validation_loader, epochs=5, callbacks=[checkpoint])
```

This application-level callback owns completed-epoch history, discovers common
public loader/sampler generators, including nested `sampler`/`batch_sampler`
links, and writes `last.pt`, monitored `best.pt`, and
`metrics.jsonl`. `every_steps=N` also writes `last_step.pt` every N completed
optimizer updates, including accumulation. `0` disables step saves.
A stateful loader is still required for any positive interval. Ordinary epoch
checkpoints work with normal DataLoaders; absent validation produces no best file.

When `every_steps` is omitted, the callback inherits
`training.checkpoint_every_steps` from the trainer's effective configuration at
first attachment. With no configured cadence it remains zero. Explicit
`every_steps=0` disables inherited step saves. The directory is always explicit;
loading global configuration alone does not create checkpoint files.

Reconstruct the same components and attach a fresh callback:

```python
checkpoint = Checkpoint("runs/my-recipe")
checkpoint.restore(trainer, train_loader, validation_loader)
trainer.fit(train_loader, validation_loader, epochs=8, callbacks=[checkpoint])
```

`restore()` selects the latest last checkpoint by saved progress; a completed
epoch wins a tie. Selection inspects memory-mapped metadata with meta tensors,
then loads only the selected model/optimizer payload on CPU. An explicit `path=`
must refer to a last checkpoint in this directory and cannot discard newer
saved progress; a basename such as `path="last.pt"` is relative to this directory.
The callback repairs its
metrics file and a best-file write interrupted after an authoritative last epoch.
Restoring does not recreate arbitrary models, datasets or Python functions.

Global defaults select settings when new trainers/loaders are constructed.
They do not replace saved policy during restoration: `restore_checkpoint(...,
config=None)` starts from the checkpoint's configuration, and a supplied partial
dictionary overlays that saved base. The callback follows the same saved-base
rule and derives an omitted cadence from it. Reconstruct compatible components
first; loading unrelated defaults later is not a checkpoint migration. The
configured `train(resume=...)` entry point owns this reconstruction for its recipe.

Optional `config=` overlays recipe metadata on the trainer snapshot at first
attachment; omitted configuration uses that snapshot. The callback does not
re-read changing globals during saves. `generators=` supplies an explicit
mapping when your recipe owns additional Torch generators. Call `restore` with
`batch_size_mode="auto"`, `sample_batch_fn` and optional `memory_budget_bytes` for
preflight resume sizing; its default is manual. Common public loader/sampler
generators are discovered by identity, but arbitrary custom/dataset RNG state
still belongs to the recipe/loader. `checkpoint.history` is available for plotting.
Discovery does not iterate loaders, draw random values, or inspect dataset
internals. Shared generators are saved once, and cyclic sampler links are safe.
An explicit `generators=` mapping replaces discovery; include all recipe RNGs
that need recovery when supplying one.

To resume with a different batch size, rebuild the loader with that size and
call `restore()` normally. When `config` is omitted, the callback reuses saved
recipe metadata and updates its batch-size field from the new loader. You do
not need to repeat the saved configuration. An explicitly supplied configuration
must agree with the loader; automatic planning records the final selected size.
For step saves, keep a stateful loader and set `every_steps=N` on the callback.
Changing the cadence alone does not convert an ordinary loader into a stateful one.

Attach from the first `fit()` or restore first. A callback cannot reconstruct
metrics from earlier unsaved epochs; attaching late raises an actionable error.
Use a new directory for an independent run. See the
[progressive example](../../examples/progressive_training.py) and
[worker/skip loader guide](../data/README.md). The lower-level helpers below remain
available when you deliberately own checkpoint bookkeeping.

## Choose the saved boundary

| Kind | Safe place to save | What resume does |
| --- | --- | --- |
| Epoch | Initial/completed epoch boundary; normally an epoch callback after validation | Starts the next epoch |
| Step | Step callback after a complete optimizer update, with no pending gradients | Continues the active epoch at the next unconsumed batch |

An optimizer step may contain several accumulated microbatches. A final partial
window is also one completed update. Saving halfway through accumulation is
unsupported because pending gradients are not serialized.

The configured runner maintains `last.pt`, `best.pt` when monitoring is enabled,
and `last_step.pt` when a positive step interval is configured.
Best selection happens after epoch validation. A step save never selects a new
best model. `training.checkpoint_every_steps` counts optimizer updates; use a
larger interval to reduce I/O or `0` to disable step checkpoints.

## Runnable direct example

The complete [checkpoint recipe](../../examples/checkpointing/checkpoint_resume.py)
uses `Checkpoint` to own history and generator bookkeeping, reconstructs its
components, restores with a newly requested batch size, and evaluates the best
saved model. It uses a temporary directory by default:

```bash
python examples/checkpointing/checkpoint_resume.py --kind epoch
python examples/checkpointing/checkpoint_resume.py --kind step
python examples/checkpointing/checkpoint_resume.py --kind step --resume-batch-size 4
```

The first two commands demonstrate changed-size continuation; the last keeps
the original size. Add `--output runs/checkpoint-demo` to retain the files in a
new directory. The step example deliberately interrupts immediately after a
durable save, then creates a new trainer before restoring.
Best selection monitors dataset-wide MSE, so changing physical batch sizes does
not change the metric's weighting. The ordinary scalar loss report retains its
documented mean-of-batch-objectives meaning.

## Evaluate best weights in a direct recipe

`trainer.evaluate(loader)` evaluates the current model; it does not load
`best.pt`. Recreate separate evaluation components using your recipe, then load
only the model weights. For the repository's progressive example:

```python
import torch
from examples.progressive_training import make_components

evaluator, _, validation_loader = make_components()
saved = torch.load("runs/progressive/checkpoints/best.pt", map_location="cpu", weights_only=True)
evaluator.model.load_state_dict(saved["model"])
del saved
results = evaluator.evaluate(validation_loader)
```

Run this from the repository root after the progressive checkpoint stage.
For your model, use your own architecture and evaluation loss/metrics. The
separate evaluator leaves the training model, optimizer and counters intact.
For inference directly, load the same `saved["model"]` state into your model,
call `model.eval()`, and run under `torch.no_grad()`.
There is no best file without monitored validation. Resume training through
`Checkpoint.restore()` and a last checkpoint, which restore optimizer/RNG state
as well as weights.

When `training.ema.enabled = true`, the checkpoint's `model` remains the live
training state and its separate `ema` entry contains validation shadows, decay,
and the completed optimizer-update count. Construct an evaluator with the same
EMA policy, load the live model as above, then restore the averages before
calling `evaluate()`:

```python
from deepopt.engine.checkpointing.checkpoint import restore_ema_for_evaluation

evaluator.model.load_state_dict(saved["model"])
restore_ema_for_evaluation(saved, evaluator)
results = evaluator.evaluate(validation_loader)
```

Keep `saved` until both loads finish. `deepopt.run.evaluate()` performs both
steps automatically. Validation uses current checkpoint buffers with averaged
parameters, and leaves the evaluator's live parameters intact. EMA checkpoint
state is validated before resume mutates training components. Changing EMA's
enabled policy or decay during resume is rejected; older checkpoints without
EMA remain compatible when it is disabled.

## Lower-level helper signatures

The [advanced checkpoint example](../../examples/checkpointing/checkpoint_resume_low_level.py)
demonstrates manual history, generator and callback ownership. Use it when
integrating a different artifact workflow; ordinary recipes need only `Checkpoint`.

Import helpers from `deepopt.engine.checkpointing.checkpoint`:

```text
save_checkpoint(path, trainer, config=None, *, generators, history,
                kind="epoch", train_loader=None)

read_checkpoint(path) -> dict

restore_checkpoint(
    payload, trainer, config=None, *, generators, train_loader=None,
    validation_loader=None, batch_size_mode="auto", sample_batch_fn=None,
    memory_budget_bytes=None,
) -> list[dict]
```

The `*` above denotes keyword-only arguments in the API signature. Callers pass
normal keyword arguments, as in the following callback:

```python
def record_epoch(trainer, row):
    history.append(row)
    save_checkpoint(path, trainer, config, generators=generators,
                    history=history, train_loader=train_loader)
```

`config` may be an `ExperimentConfig` or a dictionary of recipe metadata. On
save, omitted configuration uses `trainer.config`; on restore it uses the saved
configuration. Direct recipes may still supply explicit metadata. `history`
must contain exactly one row per completed epoch;
an in-progress step checkpoint retains completed rows only. Append the row before
saving in an epoch callback. `generators` maps stable names to Torch generators;
use `{}` if the recipe has none. Include a `train_loader` for step saves and
restores, and to record its effective batch size for epoch saves.

`read_checkpoint()` returns a format-validated payload loaded on CPU with
`torch.load(..., weights_only=True)`. The parent directory must already exist
when saving. Helpers do not create run directories, select filenames, print
epoch logs, or register callbacks. See [configured runs](../runs/README.md) for
that application layer.

## What is saved

Current files use checkpoint `format_version=5`; supported version-4 files remain readable:

Version 5 adds named coefficient schedule identity, state and last applied
values. Restoring validates schedule progress against the trainer's optimizer
updates. Version-4 checkpoints without schedules restore without a controller;
they cannot silently acquire one. See [scheduling resume rules](../scheduling/README.md#reports-and-restoration).

- Model parameters/buffers, optimizer state, scheduler state, and strategy state.
- Completed epoch/update counters, best score/epoch, and monitor selection.
- Completed-epoch history, configuration, component types, device, library
  versions, functional-computation metadata, and optimizer parameter routing.
- Python and Torch CPU RNG state, CUDA RNG state for CUDA training, and the
  explicitly supplied named Torch generators.
- For a step save: active-epoch metrics and their registered states/update
  counters, batch/sample counters, epoch-start learning rates, timing, and the
  loader's own snapshot.

Datasets, architecture definitions, Python callables, external logging services,
and arbitrary application state are not packaged into the file. Recreate those
from your recipe and its recorded settings. Non-Torch random generators or
stochastic data transforms need their own state contract; global NumPy RNG state
is not part of these checkpoint helpers.

Automatic FP8 settings from `Trainer(..., fp8=...)` or `TrainingRuntime(...,
fp8=...)` are recorded in trainer compatibility metadata. Configured runs also
save `[runtime.fp8]` in the experiment configuration. Reconstruct with the same
enabled settings before restoring; changed formats, scaling, backend, or tracing
policy are rejected. Master parameters and optimizer state remain ordinary
floating-point checkpoint tensors; temporary FP8 operands are not stored.
Older checkpoints without FP8 metadata remain compatible when automatic FP8 is
disabled.

An explicit `transform_model(..., plan=...)` call without automatic FP8 settings
still leaves plan reconstruction with the recipe. Record a recipe/plan version
in `experiment_identity` when it must be checked; matching tensor keys alone
cannot detect a changed manual plan. Custom conversion rules and Python code
are not reconstructed from checkpoint metadata. See
[automatic FP8 configuration](../runs/README.md#configure-fp8).

Compiled code and captured CUDA Graph objects are also not serialized. Recreate
their execution setup through the corresponding [execution](../execution/README.md)
and [CUDA Graph](../cuda/README.md) APIs.

## Restore compatibility

The helper validates the following before enabling training:

| Requirement | Reason |
| --- | --- |
| Same component types, model state keys/shapes/dtypes, and functional computation metadata | Restore state into compatible computation |
| Same optimizer parameter names/order/groups and child routing | Avoid loading state into a different same-shaped layer |
| Same precision, accumulation, clipping, scheduler timing, and compatible strategy configuration | Preserve the optimization policy |
| Same enabled automatic FP8 configuration | Preserve operand formats, scaling, backend, and transformation policy |
| Same resolved device and exact PyTorch version; same visible CUDA device count for single-process CUDA runs | Match runtime/RNG state; DDP records each rank's current-device RNG |
| Same named generator keys | Restore each generator to its intended role |
| For step resume: same loader type, metric classes/registered state structure, and exact TorchMetrics version | Restore the active phase and next data position |

Configuration comparisons permit changes to total `training.epochs`, physical
`training.batch_size`, checkpoint cadence, diagnostic policies/mode schedules,
and nonblocking transfer settings. Active diagnostics are recorded separately.
Other configuration changes, including application metadata, are rejected.
Callable names and type metadata do not establish that their implementations
are unchanged: use the same recipe code and data/transforms for exact replay.

Pass `experiment_identity={"recipe_version": "v2", "dataset_version": "split-a"}`
to `Trainer`, or use an `[experiment_identity]` TOML table. These caller-declared
fields are strictly checked on resume. Checkpoints also record best-effort Git
commit/dirty status (when Git is available) and a normalized configuration
fingerprint. Git observations are cached per process, are informational, and do
not prove external data or generated code identity. Supply required versions for
the compatibility guarantees your experiment needs.

The reader accepts formats 4 and 5. Some older format-4 files without named
optimizer topology cannot be resumed safely and are rejected. The built-in
loader's current snapshot format 3 records `drop_last` and can read its legacy
format 1 and 2 snapshots as `drop_last=False`;
this is separate from the outer checkpoint format. No arbitrary checkpoint
migration is performed.

Older checkpoints made with a nested sampler whose RNG was omitted cannot
recover that missing state. Discovery now rejects the generator mismatch rather
than silently replaying the wrong order. Existing checkpoints with complete RNG
state retain their generator names and remain compatible.

## Change batch size

Construct the loader at the new requested size, then restore with
`batch_size_mode="manual"` to retain that size. Do not change accumulation or
learning-rate settings as an implicit consequence of resizing.

For automatic selection, pass `sample_batch_fn(size)` with a complete
representative batch. On CUDA—or with an explicit positive
`memory_budget_bytes`—the helper plans after model/optimizer restoration and
before training resumes. It needs a `train_loader` so it can apply the result.
CPU automatic mode without a budget records a skipped plan and keeps the
requested size. See [memory planning](../memory/README.md) for the estimation
contract and fresh-run behavior.

The built-in resumable loader preserves the saved permutation and exact consumed
sample cursor. It groups only the remaining examples at the new size, including
when the old cursor is not divisible by that size. Repeated resizes are
supported. Existing metric totals and actual batch counts remain intact.

Changed batching changes the numerical trajectory: batch-dependent layers,
random draws, scalar-objective averaging, effective accumulated batch size, and
the number of optimizer updates can differ. Learning rates, accumulation, and
scheduler state remain unchanged. Automatic resizing rejects a changed size for
`OneCycleLR`; its fixed update horizon needs an explicit recipe decision before
using manual sizing. Other custom or per-update schedules also remain the
caller's responsibility.

`trainer.resume_adjustment` and `trainer.batch_size_adjustment` expose the
decision after successful restoration. Direct helpers emit logging messages but
do not write decision files. The configured runner persists the decision in
`batch_size.jsonl` and `resume.jsonl`.

## Step-resume loader contract

`require_step_loader(loader)` checks for callable `state_dict()` and
`load_state_dict(state)` methods. It imposes no universal worker-count limit.
Your loader implementation must also satisfy these behaviors:

1. Snapshot the position after batches **consumed by training**. Account for
   prefetched but unconsumed work; a dispatched sampler cursor is insufficient
   with prefetching workers.
2. Restore the next unconsumed batch, including relevant dataset, sampler,
   worker, and random-generator state. Taking a snapshot must not advance data.
3. Preserve epoch exhaustion. After restoring a save taken at the final update,
   the first iteration must yield no more batches; only the following iteration
   starts a new epoch. This lets the trainer finish metrics, validation, and
   epoch scheduling exactly once.
4. Return state compatible with `weights_only=True` loading, and validate the
   dataset/settings needed for consistent restoration.
5. For changed-size step resume, declare `supports_batch_size_change=True` and
   preserve the requested new size while restoring the consumed-data position.
   Automatic resizing also needs a cursor-preserving `set_batch_size(size)`.

The built-in `ResumableDataLoader(dataset, *, batch_size, generator, num_workers=0)`
restores shuffled indices for a finite map-style dataset. It requires a CPU
Torch generator and supports multiple workers, prefetch and persistent workers.
It checkpoints the delivered cursor separately from dispatched indices, so
queued work is fetched again after restore. Worker count and batch size may
change on restore. It provides `state_dict`, `load_state_dict`,
`set_batch_size`, and `reset_epoch`. Its `batch_size` attribute is `None` because
it uses a custom batch sampler; the effective size is
`loader.batch_sampler.batch_size`.

`ResumableDataLoader` does not capture worker-local dataset or transform RNG
state. For recovery of that state, the optional `deepopt.data.StatefulDataLoader`
adapter uses TorchData to restore consumed batches and worker snapshots. See the
[worker loading guide](../data/README.md#worker-aware-step-checkpoints) for
installation and dataset RNG requirements. Custom or streaming pipelines may
provide their own loader. Interface availability alone cannot verify replay,
worker-state capture, or exhaustion behavior.

## Recovery and durability

With the [DDP core](../distributed/README.md), all ranks participate in save and
restore. Rank zero writes the shared checkpoint, including each rank's RNG,
loader cursor, metric progress, buffers and strategy state. Resume requires the
same world size. Model keys stay unwrapped, so distributed weights can also be
evaluated in a single process. Do not rank-zero-guard the checkpoint call itself.

Each save writes a temporary file in the destination directory, flushes and
syncs it, then replaces the destination. A failed write leaves the preceding
checkpoint intact. Files within a run are not one multi-file transaction:
`last.pt` is authoritative for completed-epoch history, and the configured
runner can repair logs and the latest best-file write from it where possible.

Component `load_state_dict()` hooks may fail after mutating their objects.
Once mutable restoration starts, the trainer/runtime stays disabled until an
entire restore succeeds. A failed update, callback, or restore is a reason to
reload a valid checkpoint or reconstruct the components—not to continue with
partially changed weights or optimizer state. MultiOptimizer child updates can
likewise be partially applied if a later child fails.

The configured runner accepts `last.pt` and `last_step.pt` for continuation;
`best.pt` is for evaluation. If `last_step.pt` has fewer optimizer updates than
the current `last.pt`, it is stale and the runner rejects it. Direct helpers
leave filename selection and staleness checks to the application. Keep one
writer per run directory and keep its files together.

## Example artifacts and replay expectations

All checkpoint examples use generated CPU data and download nothing. Run them
from the repository root after installing the package. With no `--output`, files
live in a temporary directory and disappear on normal completion.

| Recipe | Persistent output with `--output NEW_DIRECTORY` |
| --- | --- |
| [Ready-made callback](../../examples/checkpointing/checkpoint_resume.py) | `last.pt`, `best.pt`, `metrics.jsonl`; step mode also leaves `last_step.pt` |
| [Low-level saves](../../examples/checkpointing/checkpoint_resume_low_level.py) | `last.pt`; step mode also leaves `last_step.pt`; history is stored inside the checkpoint |

Both examples print a final JSON result with restored update count, completed
epochs, batch size, and samples per epoch. The ready-made callback example also
evaluates the best weights with an independently reconstructed evaluator. These
direct examples do not create configured-run decision files or persistent text
logs; wrap them in [logging_session](../logging/README.md) when those are needed.

For an unchanged-batching replay example, use `--resume-batch-size 4`. The
default of six deliberately changes update grouping while preserving the
sample cursor. A matching seed and checkpoint do not promise bitwise results
across different devices, kernels, batching, or software versions.
