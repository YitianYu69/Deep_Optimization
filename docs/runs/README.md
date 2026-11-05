# Configuration, CLI, and managed runs

[Documentation index](../README.md) · [Direct training](../training/README.md) · [Checkpoint contracts](../checkpoints/README.md)

Use `Trainer` with ready-made components when you own the surrounding program.
Use `deepopt.run.train()` when you want configuration snapshots, run directories,
checkpoint restoration and optional batch planning around the same trainer.
The CLI is an offline synthetic classification recipe; it does not discover or
import arbitrary user models from TOML.

## Run the bundled recipe

After installing the library, these commands work from any directory:

```bash
python -m deepopt inspect-config --config builtin:baseline
python -m deepopt train --config builtin:baseline --output runs/baseline --epochs 2 --no-progress
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
| `inspect-config` | Required `--config PATH_OR_PRESET`; validates a chosen TOML file or `builtin:NAME` |
| `train` | Fresh `--config PATH_OR_PRESET --output NEW_DIR`, or `--resume last.pt/last_step.pt` |
| Training overrides | `--epochs N`, `--batch-size N`, `--batch-size-mode auto/manual`, `--checkpoint-every-steps N`, `--execution-mode max_speed/balanced/analyze`, `--no-progress` |
| `evaluate` | Required `--checkpoint PATH`; optional `--device DEVICE` (an override uses FP32) |

`python -m deepopt --config PATH` remains a legacy inspection shorthand. Put
`--config` after a subcommand when using `train` or `inspect-config`.

## Choose a TOML file, dictionary, or installed preset

Users choose the configuration source explicitly. Any readable TOML file can be
selected with a relative or absolute path; its filename does not need to match a
bundled preset:

```bash
python -m deepopt inspect-config --config ./experiments/my-training.toml
python -m deepopt train --config ./experiments/my-training.toml --output runs/custom
```

The Python entry points accept the same chosen path directly, or a dictionary:

```python
from deepopt import load_config
from deepopt.run import train

config = {
    "name": "dictionary-run",
    "seed": 7,
    "training": {"epochs": 2, "batch_size": 8, "learning_rate": 0.001},
    "runtime": {"device": "cpu", "fp8": {"enabled": True}},
}
train(config, output="runs/dictionary-run", batch_size_mode="manual")

# Activate file-based defaults when you want other components to inherit them.
selected = load_config("./experiments/my-training.toml")
train(selected, output="runs/selected-config")
# Passing a path directly selects a complete source without changing defaults.
train("./experiments/my-training.toml", output="runs/selected-file")
```

Dictionary keys follow the TOML tables. Inputs are validated and copied without
adding defaults or applying overrides to the caller's dictionary. Existing
`ExperimentConfig` and `TrainingConfig` objects remain supported. Factories and
custom callbacks still receive typed settings after normalization.

## Global defaults and component snapshots

`load_config(source)` validates a complete configuration, copies it, and replaces
the process-global defaults. It accepts a dictionary, `ExperimentConfig`, TOML
path, or explicit `builtin:NAME`. It is an activation operation, not a pure file
reader. A failed load leaves the preceding defaults intact.

```python
import deepopt

deepopt.load_config("builtin:baseline")
current = deepopt.get_config()  # Independent ExperimentConfig copy.
deepopt.reset_config()         # Safe to repeat; get_config() now returns None.
```

`config_from_dict()` and `load_preset()` remain pure: they return settings without
activating them. Passing a typed configuration or TOML path directly to
`build_experiment()` or `train()` also leaves the global defaults unchanged.
Configuration loading initializes neither PyTorch/CUDA/DDP nor log files. Each
DDP process owns its own defaults and must resolve equivalent settings.

Configuration-consuming APIs resolve settings once when constructing their
owned components:

| Priority | Meaning |
| --- | --- |
| Explicit function arguments | Win over configured values, including `False` and meaningful `None` |
| Per-call `config={...}` | Recursively merges into the inherited snapshot; lists replace lists |
| Loaded global configuration | Supplies omitted settings to new components |
| Existing API defaults | Apply when a setting has not been provided |

A complete `ExperimentConfig`, TOML path, or `builtin:NAME` is an independent
source that bypasses global values. Only a per-call dictionary is a partial
override. `load_config(dictionary)` itself still requires a complete configuration;
loading a second dictionary replaces rather than merges the first global value.
Missing required information is not invented. For example, constructing an
optimizer automatically needs a configured learning rate, and `fit()` needs
epochs from its snapshot or an explicit argument.

The merged result is validated before use. If an inherited compile policy has
`mode` and an override adds `options`, the conflict raises an error. Likewise,
LR policy/gamma and execution-mode/diagnostics conflicts are not silently cleared.
`monitor=None` explicitly disables fit monitoring; omission inherits it.

`Trainer(..., config=...)` and `TrainingRuntime(..., config=...)` retain their
resolved snapshots, available through copied `.config` properties. New global
loads do not reconfigure those objects. A supplied optimizer, scheduler, or
runtime keeps its ownership and execution settings. When the optimizer is
omitted, the trainer can build it from the configured training policy.

`trainer.fit(..., config={"training": {"epochs": 5}})` uses the trainer's original
snapshot as its base, not current process defaults. Fit configuration can change
only `epochs`, `monitor`, and `monitor_mode`; changing device, optimizer, or FP8
policy requires new components. Explicit fit arguments take precedence over
these fields, and total epochs must exceed the completed count.

The shared `cuda_graph_core` reads defaults when creating a new capture.
An ordinary `CUDAGraphCore` keeps its construction snapshot, and existing capture
handles retain their settings. Capturing still requires an explicit operation.
Logging sessions snapshot inherited levels and write files only to an explicit
destination; logging outside a session uses current console defaults.

Checkpoints use the trainer's effective snapshot rather than current globals.
Resume without explicit configuration uses the saved checkpoint as authority;
loading unrelated defaults beforehand does not change the resumed computation.
See [checkpoint compatibility](../checkpoints/README.md).

## Packaged presets and local input contracts

The repository has two preset locations:

| Location | Use |
| --- | --- |
| [configs/](../../configs) | Editable TOML recipes for repository users |
| [src/deepopt/configs/](../../src/deepopt/configs) | Packaged copies available as `deepopt.configs` in wheels and source distributions |

The validation/schema API remains [deepopt.config](../../src/deepopt/config.py).
Installed presets use package resources, so loading them does not depend on the
current directory or access to this repository:

```python
from deepopt import load_config, reset_config
from deepopt.configs import list_presets, load_preset

print(list_presets())  # baseline, ddp, fp8, scheduling
typed = load_preset("fp8")
config = load_preset("fp8", as_dict=True)
config["training"]["epochs"] = 3
active = load_config("builtin:fp8")  # Explicitly activate the same preset.
reset_config()
```

Use `builtin:baseline`, `builtin:ddp`, `builtin:fp8`, or `builtin:scheduling` in
the CLI or full-experiment Python entry points. Only the explicit `builtin:`
prefix selects a packaged preset: a missing local file raises an error instead
of silently selecting a default. Each preset load returns independent data;
modifying a returned dictionary does not modify installed files. Tests keep the
repository and packaged copies in sync.

| API | Accepted configuration input |
| --- | --- |
| `load_config` | Complete dictionary, `ExperimentConfig`, selected TOML path, or `builtin:NAME`; activates defaults |
| `train`, `build_experiment`, `build_baseline_experiment`, `baseline_sample_batch` | Inherited defaults, partial full-schema dictionary, or a complete typed/file/preset source; do not activate defaults |
| `build_optimizer`, `build_scheduler` | Inherit omitted settings; preserve existing `TrainingConfig` or explicit `training` table forms |
| `compile_model` | Inherited compiler policy, `CompileConfig`, or the `runtime.compile` dictionary |
| Trainer/runtime `diagnostics` | `DiagnosticPolicy` or a dictionary of mode/default overrides |
| Trainer/runtime/core `fp8` | Existing bool, `FP8Config`, or FP8 settings dictionary |
| Synthetic `build_datasets` | `SyntheticDataSettings` or a synthetic-data options dictionary |
| Checkpoints and logging | Existing dataclass/dictionary metadata contracts; arbitrary caller metadata stays supported |

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
| `runtime.ddp` | Native DDP options: `find_unused_parameters`, `broadcast_buffers`, `gradient_as_bucket_view`, `static_graph`, optional `bucket_cap_mb` and `validation_sync_buffers` |
| `runtime.cuda_graph` | Optional `device` (`cuda`/`cuda:N`), positive `warmup`, `allow_unused_input`, `autograd`, and `capture_error_mode` (`global`/`thread_local`/`relaxed`) |
| `loader` | Optional `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, and `drop_last`; datasets, transforms, and custom samplers remain application-owned |
| `logging` | Optional console `level` and `file_level`; a logging session still needs an explicit file destination |
| `schedules` | Named coefficient curves consumed explicitly by the recipe |
| `application` | Recipe-owned JSON-compatible dataset/model/settings metadata |
| `experiment_identity` | Optional JSON-compatible semantic identity used for reconstruction/resume checks |

Unknown common fields are rejected. `application` does not impose a universal
dataset or model schema; each recipe validates the keys it consumes. TOML has no
null literal: omit optional values to use their defaults. See
[config.py](../../src/deepopt/config.py) for exact field validation.

The new DDP, CUDA Graph, loader, and logging tables default to empty dictionaries,
leaving existing API behavior intact. Persistent workers and a non-null prefetch
factor require `loader.num_workers > 0`. Rank, world size, and local GPU assignment
come from the distributed environment rather than fixed configuration fields.

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
