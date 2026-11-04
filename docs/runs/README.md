# Configuration, CLI, and managed runs

[Documentation index](../README.md) · [Direct training](../training/README.md) · [Checkpoint contracts](../checkpoints/README.md)

Use `Trainer` with ready-made components when you own the surrounding program.
Use `deepopt.run.train()` when you want configuration snapshots, run directories,
checkpoint restoration and optional batch planning around the same trainer.
The CLI is an offline synthetic classification recipe; it does not discover or
import arbitrary user models from TOML.

## Run the bundled recipe

From the repository root, after installing the project:

```bash
python -m deepopt inspect-config --config configs/baseline.toml
python -m deepopt train --config configs/baseline.toml --output runs/baseline --epochs 2 --no-progress
python -m deepopt train --resume runs/baseline/last.pt --epochs 3 --no-progress
python -m deepopt evaluate --checkpoint runs/baseline/best.pt --device cpu
```

Use a new output path for a fresh run. Resume continues in the checkpoint's
directory; omit `--output` and normally omit `--config` so saved configuration is
used. Epochs are the **total target**, so resuming epoch two with `--epochs 3`
trains one further epoch. `best.pt` is for evaluation; resume from `last.pt` or
`last_step.pt`. The installed `deepopt` console command is equivalent to
`python -m deepopt`.

Epoch records and evaluation results are JSON on stdout. Progress and diagnostic
messages go to stderr; `--no-progress` disables progress bars only. See the
[logging guide](../logging/README.md) for log levels and file routing.

| Command | Options |
| --- | --- |
| `inspect-config` | Required `--config PATH`; validates TOML and prints resolved settings |
| `train` | Fresh `--config PATH --output NEW_DIR`, or `--resume last.pt/last_step.pt` |
| Training overrides | `--epochs N`, `--batch-size N`, `--batch-size-mode auto/manual`, `--checkpoint-every-steps N`, `--execution-mode max_speed/balanced/analyze`, `--no-progress` |
| `evaluate` | Required `--checkpoint PATH`; optional `--device DEVICE` (an override uses FP32) |

`python -m deepopt --config PATH` remains a legacy inspection shorthand. Put
`--config` after a subcommand when using `train` or `inspect-config`.

## Configuration fields

The complete baseline lives in [configs/baseline.toml](../../configs/baseline.toml).
Only `name`, `seed`, and the training `epochs`, `batch_size`, and `learning_rate`
fields are required:

```toml
name = "my-run"
seed = 7

[training]
epochs = 3
batch_size = 16
learning_rate = 0.001

[runtime]
device = "cpu"
precision = "fp32"
```

| Table | Supported settings and defaults |
| --- | --- |
| `training` | Required fields above; `optimizer="adamw"` (`sgd` or recipe-supplied `custom`), `weight_decay=0`, `momentum=0.9`, `accumulation_steps=1`, optional `grad_clip_norm` |
| Reporting/checkpoints in `training` | `monitor="loss"`, `monitor_mode="min"`, `checkpoint_every_steps=0`; TOML `monitor=false` disables best selection |
| LR scheduling in `training` | `lr_gamma=1`, `scheduler_interval="epoch"` (`step` counts optimizer updates), optional `lr_schedule` table |
| `runtime` | `device="cpu"`, `precision="fp32"`, `deterministic=true`, `checks=true`, optional execution/diagnostic policies, `non_blocking=false` |
| `runtime.compile` | `enabled=false`, `backend="inductor"`, `fullgraph=false`, optional `mode`, `dynamic`, and scalar-valued `options` |
| `runtime.fp8` | `enabled=false`, `backend="reference"`, `trace_mode="modules"`; operand formats and scaling described below |
| `schedules` | Named coefficient curves consumed explicitly by the recipe |
| `application` | Recipe-owned JSON-compatible dataset/model/settings metadata |
| `experiment_identity` | Optional JSON-compatible semantic identity used for reconstruction/resume checks |

Unknown common fields are rejected. `application` does not impose a universal
dataset or model schema; each recipe validates the keys it consumes. TOML has no
null literal: omit optional values to use their defaults. See
[config.py](../../src/deepopt/config.py) for exact field validation.

`runtime.precision` configures trainer-level AMP: `fp32` disables AMP on CPU/CUDA,
and `bf16` enables AMP on CUDA. Configure FP8 operand formats through
`runtime.fp8`; `precision="fp8"` is not an AMP setting.

The bundled CLI interprets only its own optional data table:

```toml
[application.synthetic]
train_samples = 256
validation_samples = 64
input_dim = 16
num_classes = 4
```

## Configure FP8

`build_experiment()` prepares supported operations automatically when FP8 is
enabled. No model-factory transformation code is needed:

```toml
[runtime]
device = "cpu"
precision = "fp32"

[runtime.fp8]
enabled = true
backend = "reference"
trace_mode = "modules"
activation_dtype = "e4m3fn"
weight_dtype = "e4m3fn"
grad_dtype = "e4m3fn"
```

The complete [configs/fp8.toml](../../configs/fp8.toml) uses small generated data:

```bash
python -m deepopt inspect-config --config configs/fp8.toml
python -m deepopt train --config configs/fp8.toml --output runs/fp8 --batch-size-mode manual --no-progress
python -m deepopt train --resume runs/fp8/last.pt --epochs 3 --batch-size-mode manual --no-progress
python -m deepopt evaluate --checkpoint runs/fp8/best.pt --device cpu
```

| FP8 field | Meaning |
| --- | --- |
| `enabled` | Opt in to automatic preparation and supported FP8 computation; default `false` |
| `backend` | `reference` emulates quantized operands; `triton` runs native FP8 GEMM on supported NVIDIA CUDA hardware |
| `trace_mode` | `modules` (default) preserves a container root; `full` requests one FX graph; `auto` tries full tracing then falls back |
| `activation_dtype`, `weight_dtype`, `grad_dtype` | Each accepts `e4m3fn` (default) or `e5m2`; Triton currently requires all three to match |
| `block_shapes` | Optional table with `activation`, `weight`, and `grad` two-integer arrays; omitted roles use tensorwise scaling |
| `num_stages` | Triton pipeline stages, integer 1–4; default 3 |
| `out_dtype` | Optional `fp32`, `bf16`, or `fp16`; omit to follow the active AMP/input dtype |

For example, `[runtime.fp8.block_shapes]` with `weight = [32, 64]` applies
blockwise weight scaling. Triton requires both dimensions to be multiples of
32. TOML has no null literal; leave a role absent to retain tensorwise scaling.

Construction order is **model placement → FP8 preparation → optional compilation
→ optimizer → runtime/DDP core**. The core reuses matching preparation and
exposes its report through `trainer.runtime.transformation_report`. Use
`trainer.model` after construction: an exact supported root projection becomes
its DO adapter, and full tracing can return a new graph root. Optimizer master
parameters retain their identity. Unsupported operations keep their ordinary
implementations; the report is not a guarantee of complete FP8 coverage.

For Triton, select `runtime.device="cuda"` and `backend="triton"`, and retain
`--batch-size-mode manual`: the FP8 path currently cannot be sized by the fake
memory estimator's data-dependent quantization checks. The reference backend
does not establish GPU performance. A device override during evaluation does
not change the saved FP8 backend: evaluate a Triton recipe on supported CUDA
hardware. See [FP8 usage and limits](../fp8/README.md)
for direct `Trainer(..., fp8=...)`, manual runtimes, autocast contexts, and
hardware requirements.

## Configure torch.compile

Model-only compilation is configured independently of trainer diagnostics:

```toml
[runtime]
execution_mode = "balanced"

[runtime.compile]
enabled = true
backend = "aot_eager"
fullgraph = false
```

This example backend exercises tracing/autograd without native kernel generation.
Use `inductor` with the appropriate compiler toolchain to measure performance.
When enabled, FP8 preparation precedes compilation; compilation happens before
optimizer creation and DDP wrapping. Runtime,
optimizer, callbacks and checkpoint writes remain eager. The first model call
pays compilation cost. See [execution and compilation](../execution/README.md).

| Compiler field | Meaning |
| --- | --- |
| `enabled` | Enable model compilation; default `false` |
| `backend` | Backend name passed to PyTorch; default `inductor` |
| `fullgraph` | Require a complete graph instead of permitting graph breaks; default `false` |
| `dynamic` | Optional boolean; omit for PyTorch's automatic shape handling |
| `mode` | Optional `default`, `reduce-overhead`, `max-autotune`, or `max-autotune-no-cudagraphs` |
| `options` | Backend options with string/boolean/finite-number values; use this or `mode`, not both |

Backend availability and option support depend on the installed PyTorch version
and platform. Enabling compilation does not imply a speedup or CUDA Graph capture.

[configs/scheduling.toml](../../configs/scheduling.toml) demonstrates LR policies
and named coefficient definitions. A custom `step_factory(schedules)` must use
those coefficients in the objective; the CLI does not invent a use for arbitrary
names. See [scheduling](../scheduling/README.md) for curve types, update/epoch
intervals, and exact resume semantics.

CUDA Graph capture remains an explicit recipe operation. Prepare FP8 and
compilation first, then follow the [CUDA Graph capture boundary](../cuda/README.md).
Enabling several execution features does not establish that their combination
supports capture or full-graph compilation; validate the chosen computation.

## Component wiring

An experiment factory receives the validated configuration and returns:

```python
trainer, train_loader, validation_loader, generators = experiment_factory(config)
```

`validation_loader` may be `None`; `generators` maps names to the generators whose
state must be saved. The supplied `build_experiment()` helper wires ordinary
components and accepts the same loss/step and metric choices as `Trainer`.

| Argument | Callable contract |
| --- | --- |
| `model_factory` | `() -> nn.Module`; the helper moves it to the selected device |
| `dataset_factory` | `() -> (train_dataset, validation_dataset_or_none)` |
| `loader_factory` | `(train_dataset, valid_dataset, *, batch_size, seed) -> (train_loader, valid_loader, generators)` |
| `optimizer_factory` | `(model, training_config) -> optimizer` |
| `scheduler_factory` | `(optimizer, training_config) -> scheduler_or_none` |
| `task_factory` | Optional `() -> task`; a task is not required for ordinary losses/steps |
| `step_factory` | Optional `(schedules) -> step_fn(model, batch)`; choose it instead of `step_fn`/`loss_fn` |
| `trainer_factory` | Accepts the model/optimizer and the helper's trainer keyword arguments |

`metric_factory`, `metrics`, `metric_inputs`, `validation_step_fn`, and an explicit
`ScheduleController` are also supported. Bind recipe settings to factories using
closures or `functools.partial`. No dynamic import registry is required.

For example, this fragment assumes caller-owned `dataset_factory()` and config:

```python
from functools import partial
from torch import nn
from torch.nn import functional as F
from deepopt.data import build_loaders
from deepopt.experiment import build_experiment

def experiment_factory(config):
    loaders = partial(build_loaders, num_workers=2,
                      resumable=config.training.checkpoint_every_steps > 0)
    return build_experiment(
        config,
        model_factory=lambda: nn.Linear(4, 1),
        dataset_factory=dataset_factory,
        loader_factory=loaders,
        loss_fn=F.mse_loss,
    )
```

The default loader helper is recognized directly and receives `resumable=True`
for configured step saving. A custom/partially bound loader factory owns that
choice itself, as above. Collation and worker options belong to the loader
factory. See [loader recovery and seeded transforms](../data/README.md).

## Runnable custom recipes

These programs bind their own data/model/task/optimizer choices:

```bash
python examples/configured/read_config.py --config configs/scheduling.toml
python examples/configured/train_regression.py --output runs/regression --epochs 2 --no-progress
python examples/configured/train_regression.py --resume runs/regression/last.pt --epochs 3 --no-progress
python examples/configured/train_multi_optimizer.py --output runs/multi --epochs 2 --no-progress
python examples/configured/train_multi_optimizer.py --resume runs/multi/last.pt --epochs 3 --no-progress
```

Fresh runs keep the scripts' normal defaults (eight regression epochs or five
multiple-optimizer epochs) when `--epochs` is omitted. Resume uses saved settings
and changes only explicit overrides. The model/step definitions must still match.

For custom reconstruction, call the Python run API with the same factory:

```python
from deepopt.run import train, evaluate

run_dir = train(config, output="runs/custom", experiment_factory=experiment_factory)
train(resume=run_dir / "last.pt", epochs=4, experiment_factory=experiment_factory)
result = evaluate(run_dir / "best.pt", experiment_factory=experiment_factory)
```

The CLI's `evaluate` command reconstructs the baseline recipe. Custom checkpoints
must be evaluated with their own factory or with a separate direct evaluator.
The config snapshot cannot recreate arbitrary Python model or loss code.

## Batching, step recovery, and distributed execution

Configured training defaults to `batch_size_mode="auto"` for fresh runs and
resume. CUDA planning needs a representative `sample_batch_fn(size)` for a custom
recipe; the built-in baseline supplies one. CPU planning needs an explicit
memory budget, otherwise it logs a skip and keeps the requested size. Use
`batch_size_mode="manual"` or CLI `--batch-size-mode manual` for fixed batching.
See [memory planning](../memory/README.md) for estimator limits.

Positive `checkpoint_every_steps` counts optimizer updates, including accumulation
windows. Step saving requires a resumable training loader. Rebatching can retain
the consumed cursor, but changes the numerical training trajectory. Worker-state
recovery and deterministic item replay have different contracts; see
[checkpoint compatibility](../checkpoints/README.md).

For multiple ranks, the same program runs on every process:

```bash
torchrun --standalone --nproc-per-node=2 -m deepopt train --config configs/ddp.toml --output runs/ddp --no-progress
```

The configured factory initializes the process group, places the model, and
shards default loaders. Batch size is per rank. Each rank enters training and
checkpoint calls; shared filesystem writes are coordinated by the run layer.
See the [distributed guide](../distributed/README.md) for metadata and cleanup.

## Generated files

| File | Meaning |
| --- | --- |
| `config.json` | Resolved settings, including the selected batch size |
| `environment.json` | Environment/device metadata for the run |
| `metrics.jsonl` | One report per completed epoch; repaired from checkpoint history on resume |
| `last.pt` | Latest complete epoch state |
| `best.pt` | Best monitored validation state, when selection is enabled |
| `last_step.pt` | Latest saved optimizer boundary when step saving is enabled |
| `batch_size.jsonl` | Start/resume planning decisions |
| `resume.jsonl` | Resume adjustment records |
| `logs/*.log` | Configuration-named diagnostics and recipe output |

Only the checkpoint variants enabled/produced by the run exist; no validation
means no best-file selection. Current writers use checkpoint format 5, with
format-4 reading support described in the [checkpoint guide](../checkpoints/README.md).
State is written atomically, but failed training/optimizer work is not rolled
back in memory. Reconstruct and restore from a durable boundary before continuing.
