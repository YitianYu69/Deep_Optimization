# Execution modes and performance contracts

[Documentation index](../README.md) · [Training](../training/README.md) · [DDP](../distributed/README.md) · [Observers](../observers/README.md) · [Checkpoints](../checkpoints/README.md)

Execution modes control passive diagnostics, not the optimization algorithm.
Loss reduction, precision, accumulation, clipping, optimizer updates, structural
validation and distributed failure coordination remain separate responsibilities.
When used by `TrainingRuntime`, both `TrainingCore` and `DDPCore` execute under
the same runtime policy scopes. Bare cores do not own optimizer-update counters;
mode selection and scheduling belong to the runtime/trainer.

Model compilation is independently configurable through
[`[runtime.compile]`](../runs/README.md#configure-torchcompile). It is disabled
by default and does not compile the runtime or its diagnostic scheduling.

FP8 is a separate opt-in computation policy through
[`[runtime.fp8]`](../runs/README.md#configure-fp8) or `Trainer(..., fp8=...)`.
It can be combined with AMP precision; changing diagnostic modes does not change
FP8 formats or scaling.

Mode coverage includes prepared/native FP8 projections, direct quantization and
scaling calls, default loss/report validators, custom backward and gradient
processing invoked through the runtime, trainer metric callbacks, and default
observers owned by a custom metric factory.

## Run the examples

```bash
python examples/execution/execution_modes.py
python examples/execution/compile_training.py
python examples/execution/compile_training.py --backend inductor --fullgraph
python examples/optimization/fp8_training.py
```

[Execution modes](../../examples/execution/execution_modes.py) runs seven
microbatches with accumulation 2. Its four optimizer updates use `max_speed`,
`analyze`, `analyze`, then `balanced`, and it demonstrates an explicit safe mode
switch after the final partial window. It prints the mode at each update and a
final loss; diagnostics may also emit log messages to stderr.

[Compiled training](../../examples/execution/compile_training.py) compiles a
small MLP before optimizer/trainer construction and performs three updates on
generated data. Its default `aot_eager` backend exercises compiler/autograd
integration without native kernel generation. Select `inductor` to exercise its
compiler toolchain. Neither example downloads data or writes run artifacts.
Both accept `--device cuda` when CUDA is available; unavailable devices and
compiler failures propagate instead of silently changing execution.

The [FP8 example](../../examples/optimization/fp8_training.py) passes `FP8Config`
to the trainer, prints the automatic transformation report, and performs three
updates. Its default CPU reference backend emulates FP8 quantization; select
`--device cuda --backend triton` for the actual CUDA backend.

## Enable FP8 in a direct recipe

```python
import torch

from deepopt import Trainer
from deepopt.config import FP8Config

model = make_model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
trainer = Trainer(model, optimizer, device=device, loss_fn=loss_fn,
                  fp8=FP8Config(enabled=True, backend="reference"))
model = trainer.model
report = trainer.runtime.transformation_report
```

FP8 preparation preserves master parameter objects, so the supplied optimizer
continues to own them. Both regular and DDP cores apply the same preparation
policy before their execution setup. The default module strategy keeps a
container's root class and methods; exact registered root operations return
their DO adapter. Explicit `full`/`auto` tracing can replace the root with an FX
graph. Use the model exposed by the trainer/runtime afterward.

The core also enters the FP8 context around its step, so supported functional
linear calls in untraced Python code use the same policy. The transformation
report lists graph/module rewrites; it does not enumerate runtime-dispatched
calls or promise that every operation uses FP8. `precision="bf16"` continues to
control CUDA AMP independently. The [FP8 guide](../fp8/README.md) also covers a
standalone combined autocast context for caller-owned loops.

## Compile the model

The direct Python equivalent of `[runtime.compile]` is:

```python
import torch

from deepopt import Trainer
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model

model = make_model().to(device)
model = compile_model(model, CompileConfig(enabled=True, backend="inductor"))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
trainer = Trainer(model, optimizer, device=device, loss_fn=loss_fn)
```

This helper uses `nn.Module.compile()` in place, retaining the model class,
parameter objects, and checkpoint keys. The first forward performs compilation;
new shapes or guards can cause recompilation. `fullgraph=True` makes graph breaks
an error. Backend-specific `mode` and `options` are mutually exclusive in the
configuration. See the [complete compiler field reference](../runs/README.md#configure-torchcompile).

Configured recipes apply compilation automatically through `build_experiment`.
When automatic FP8 is enabled too, construction follows model placement, FP8
preparation, compilation, optimizer creation, then runtime/DDP setup. Direct
recipes combining the two can call `prepare_fp8_model(model, fp8_settings)` from
`deepopt.opt.policy` before `compile_model`, then pass the same `fp8_settings`
to the trainer. Matching preparation is reused; preparing an already compiled
or captured unprepared model is rejected.

Direct recipes using only compilation call the compiler helper themselves.
Checkpoint state contains ordinary
weights and optimizer state, not generated code; recreate compilation when
rebuilding a direct recipe. Diagnostic-mode switching operates outside the
compiled model. Compiler-managed CUDA graphs and explicit
[CUDA Graph capture](../cuda/README.md) are separate mechanisms; combining them
requires validation for that recipe. Current [FP8 operations](../fp8/README.md)
also have their own constraints and do not inherit compiler/capture support
merely because the surrounding runtime supports those features.

## Diagnostic presets

| Selection | Loss/gradient and FP8 finite checks | Other diagnostics |
| --- | --- | --- |
| No mode; legacy `checks=True` | Existing strict checks at every observation | Existing default event kinds; behavior preserved |
| `max_speed` | Disabled | No passive events, scans or observer-specific hot-path collectives |
| `balanced` | Loss/FP8 forward and gradients/FP8 backward follow their independent update cadences (100 by default) | Finite checks at metric computation; no weight/activation inspection |
| `analyze` | Every update | Weight and explicitly emitted activation checks/statistics, plus profiler scopes |

`max_speed` can continue through numerical anomalies until another operation
detects them. It does not disable required shape/ownership checks, loader/rank
consistency, checkpoint compatibility or monitored-score validity. Epoch-end
Python scalar results still require host reads. Analysis mode intentionally
adds scans, scalar reads and logging overhead.

The runtime establishes phase-local policy in `compute()`, manual `autocast()`,
`backward()`, optimizer/gradient processing, loss preparation, and trainer metric
updates/computation. `quantize_fp8`, `AmaxScaling.compute`, `BatchOutput.validate`,
and loss/report validation inherit it when their `check_finite` argument is
omitted. Standalone calls remain strict; an explicit `check_finite=True` requests
a check even in `max_speed`. Default validators still check shapes, dtypes,
normalization weights and reporting structure in every mode.

FP8 forward checks follow the loss cadence (or explicit weight/activation
inspection); FP8 backward checks follow the gradient cadence. The latter choice
is captured during forward, so leaving its autocast scope does not discard the
backward policy. An unscoped FP8 forward defers its backward check decision to
`runtime.backward()` when that helper is used; otherwise standalone backward
remains strict. Explicit user checks and third-party library validation are
not disabled automatically. All microbatches in an accumulation window use the same update
number. Reenter a saved `runtime.autocast()` context for each manual forward;
the next scheduled mode is selected on entry. A manual `runtime.step()` also
prepares the next mode when its forward/backward were performed externally.

Gradient normalization, clipping, and required sparse coalescing keep their
mathematical behavior in all modes. Diagnostic-only gradient value lists and
post-processing scans are omitted on unchecked updates. Classification label
range validation, DDP coordination, monitored-score validation, and phase-end
scalar conversion remain required. Memory estimation uses a separate
`max_speed` metadata probe and preserves the source runtime's diagnostic mode
and weight cache; reference FP8 cache layouts can be traced without real data
addresses or CUDA stream access.

## Choose and schedule diagnostics

```python
from deepopt import DiagnosticPolicy, Trainer, TrainingRuntime

trainer = Trainer(model, optimizer, loss_fn=loss_fn,
                  execution_mode="max_speed", accumulation_steps=4)

policy = DiagnosticPolicy(loss_finite_every=20, gradient_finite_every=20)
runtime = TrainingRuntime(model, optimizer, diagnostics=policy)
```

Choose `execution_mode` or `diagnostics`, not both. Use
`DiagnosticPolicy.for_mode("analyze")` to construct a complete preset. Custom
policies are frozen and validated at construction. Explicit policies replace
the legacy boolean `checks` setting; custom observer collections still replace
the built-in observer set and are filtered by the selected policy.

```python
trainer = Trainer(
    model, optimizer, loss_fn=loss_fn,
    execution_mode="max_speed",
    mode_schedule={1000: "analyze", 1011: "max_speed"},
)
```

Schedule keys refer to the **next optimizer update**, starting at 1, not to
microbatches. The example analyzes updates 1000–1010 inclusive. All microbatches
of an update use the same diagnostic policy. The schedule is evaluated in Python
outside the compiled model. Resume restores the optimizer-update counter and
evaluates the caller's requested schedule at that point.

`trainer.set_execution_mode("analyze")` and the corresponding runtime method
are permitted at completed update/epoch boundaries. They reject transitions
inside a forward/backward pair or while gradients are pending. An explicit mode
change replaces the current policy and clears the previous schedule. In DDP,
every rank must call the transition with the same mode. Observer subscriptions
are registered at construction/transitions; do not independently mutate observer
subscriptions on individual ranks during a distributed run.
Local legacy check opt-out does not remove a rank from observer collectives
requested by another rank; only that rank's local callback execution is skipped.

Analysis statistics include mean, standard deviation, minimum and maximum for
gradient/weight/activation events without retaining tensors. Activation events
still require an explicit `runtime.observe("activation", tensor)` call or hook.
Profiler scopes `deepopt.compute`, `deepopt.backward`, `deepopt.step`,
`deepopt.loss`, and `deepopt.metrics` are
visible when a caller enables `torch.profiler`; the engine does not silently
start a trace or write profiler files.

## TOML and CLI

```toml
[runtime]
device = "cuda"
precision = "bf16"
execution_mode = "balanced"
non_blocking = true

[runtime.mode_schedule]
"1000" = "analyze"
"1011" = "max_speed"

[experiment_identity]
recipe_version = "vae-v2"
dataset_version = "imagenet-split-2026-01"
```

Alternatively, omit `execution_mode` and use a `[runtime.diagnostics]` table
with `mode`, `loss_finite_every`, `gradient_finite_every`, `inspect_weights`, and
`inspect_activations`. Omitted fields use that mode's preset. The original
`checks=true/false` configuration remains supported when no mode is selected.

```bash
python -m deepopt train --resume runs/example/last_step.pt --execution-mode analyze
```

The CLI override clears the saved custom diagnostic policy/schedule. It does
not change precision, accumulation, learning rates, clipping or loss semantics.

## DDP accumulation and reductions

For finite loaders with a trustworthy count, `DDPCore` uses native DDP's `no_sync()` for both
forward and backward on intermediate microbatches. The final microbatch of each
full or partial window synchronizes. With 10 batches and accumulation 4, the
reduction points are batches 4, 8 and 10. No lookahead consumes extra data before
a step checkpoint.

`ResumableDataLoader.remaining_batches()` uses the delivered cursor, including
after resume/rebatching and when worker prefetch has dispatched further ahead.
Ordinary finite map-style DataLoaders with their standard batch sampler and
lists/tuples have known lengths. Unknown-length streams and loaders without a
reliable remaining-count contract retain synchronization on every backward.

`ExplicitObjective(distributed_reduction=...)` provides these explicit choices:

| Policy | Objective over ranks within an update window |
| --- | --- |
| `legacy` | Existing behavior: weighted global mean when weights are supplied; otherwise native DDP rank averaging |
| `rank_mean` | Average rank-local sums of supplied scalar microbatch objectives; normalization weights are not used |
| `global_mean` | Sum weighted scalar numerators and divide by total weight across ranks/microbatches; explicit weights required |
| `global_sum` | Sum supplied scalar objectives across ranks/microbatches without world-size averaging |

```python
from deepopt.engine.strategies import ExplicitObjective

trainer = Trainer(model, optimizer, step_fn=sum_loss_step,
                  loss_strategy=ExplicitObjective(distributed_reduction="global_sum"))
```

`LossResult.from_sum` retains its legacy meaning unless a different strategy is
selected. Reporting remains independent of the backward objective. Global-sum
scaling is applied before gradient clipping. Reduction policy is part of strict
checkpoint strategy metadata and cannot silently change on resume.

Static observer interests are registered once, and diagnostic gating happens
before payload construction. Normalization compatibility uses a tensor check
once per update rather than Python object gathering per microbatch. Unweighted
divisor 1 needs no denominator reduction. Necessary data-loading, failure and
rank-consistency collectives remain enabled in every mode.

## Transfers and reproducibility

`non_blocking=True` propagates through tensor batches, mappings, dataclasses,
named tuples and lists. It permits asynchronous copies where PyTorch supports
them, especially with pinned CPU input. Copies on the default stream alone do
not guarantee overlap with model computation. No additional prefetch stream is
introduced in this change.

Resume guarantees are distinct:

| Mechanism | Guarantee |
| --- | --- |
| `ResumableDataLoader` | Shuffled indices and delivered cursor; not arbitrary worker RNG recovery |
| `StatefulDataLoader` | Worker/dataset recovery when the dataset supplies the required state |
| `SeededTransformDataset` + resumable loader | Exact transform inputs/outputs when the base dataset is deterministic and all transform randomness uses the supplied generator |

Seeded transforms derive a stable seed from `(base_seed, epoch, sample_index,
view)` using BLAKE2, not Python's randomized `hash`. Epoch/index keys travel to
persistent workers with each request. See the [data guide](../data/README.md).

Exact augmented tensors do not guarantee bitwise-identical distributed optimizer
state. In the four-rank Gloo acceptance test, rebuilding native DDP buckets after
resume changed floating-point summation order and produced moment differences
around `1.7e-21`. The ordinary-bucket test therefore requires exact inputs and
tight numerical state agreement; a separate fixed-bucket case also checks
bitwise state equality. Deterministic kernels, unchanged software/hardware,
reducer configuration and recipe/data identity remain relevant.

## Verification and benchmarks

- White-box tests cover diagnostic cadence, no-event/no-scan fast paths, cursor
  accounting, recursive transfer flags and zero scalar reads during metric update.
- Black-box tests compare modes and weighted partial windows against plain
  PyTorch updates, and check safe switching, checkpoint compatibility and model
  compilation without mode-induced recompilation.
- Real two/four-rank Gloo tests cover asymmetric validation failures, empty/unequal
  validation shards, DDP reduction counts for accumulation 1/2/4/8, all reduction
  semantics, and persistent-worker crop/flip replay through step resume.
- CUDA/NCCL, pinned transfers, BF16 and compiled residual/VAE tests are hardware
  gated. A CPU-only run does not verify GPU correctness or performance.

```bash
python benchmarks/benchmark_execution.py --device cpu --batches 24 --repeats 7 --profile
torchrun --standalone --nproc-per-node=2 benchmarks/benchmark_execution.py \
  --device cuda --precision bf16 --accumulation 4 --pinned --non-blocking --profile
```

The benchmark checks parameters and reported loss against a plain PyTorch
reference before reporting median timings, relative overhead, throughput and
profiler scalar-read counts. CUDA runs also compare pinned/pageable and
blocking/nonblocking copies. Results are workload/environment-specific; no GPU
speedup is inferred from CPU results.

One local CPU run (PyTorch 2.14.1, one CPU thread, 24 preloaded batches of 32,
64 input features, hidden width 128, accumulation 4, two warmups and seven
measured repetitions, `DEEPOPT_LOGGING_LEVEL=ERROR`) produced these
reference-checked results:

| Execution | Median epoch time | Profiled scalar reads per epoch |
| --- | ---: | ---: |
| Plain PyTorch | 1.506 ms | 1 |
| Legacy strict checks | 3.459 ms | 152 |
| `max_speed` | 2.586 ms | 7 |
| `balanced` | 2.565 ms | 8 |
| `analyze` | 5.275 ms | 272 |

These timings describe this small CPU workload, not a general speed guarantee.
The fast-mode reads are phase-end reporting; the separate profiler test requires
zero scalar reads in per-batch metric bookkeeping. GPU measurements remain a
hardware-validation task.

CI separates fast CPU checks, worker/DDP tests, and opt-in scheduled/manual GPU
jobs. GPU jobs require repository variable `DEEPOPT_GPU_CI=true` and a compatible
self-hosted runner. FSDP/sharded-state support remains a separate future feature.
