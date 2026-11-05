# Deep Optimization

DeepOpt trains your existing PyTorch model using your optimizer, data loaders,
and loss or step function. Use `Trainer.fit()` to control training, or use
`TrainingRuntime` inside your own loop. Tasks, model wrappers, configuration
files, and run directories are optional.

## Install and run

From the repository root, in your Python environment:

```bash
python -m pip install -e .
python examples/training/regression.py
```

Requires Python 3.11+, PyTorch `>=2.4,<3`, TorchMetrics `>=1.4,<2`, and
tqdm `>=4.66,<5`. The example runs on CPU with generated data and no downloads.
Example architectures and data shapes are recipe choices; the core does not
require a model or dataset schema.

Follow [one recipe in five steps](docs/getting-started/README.md) to add metrics,
logging, checkpoints, resume and optional batch planning without changing API families.

## Train a model

This complete CPU recipe uses generated data and writes no files:

```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from deepopt import Trainer

torch.manual_seed(7)
inputs = torch.randn(64, 3)
targets = inputs.sum(dim=1, keepdim=True)
train_loader = DataLoader(
    TensorDataset(inputs[:48], targets[:48]), batch_size=8, shuffle=True,
    generator=torch.Generator().manual_seed(17),
)
validation_loader = DataLoader(TensorDataset(inputs[48:], targets[48:]), batch_size=8)
model = nn.Linear(3, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
trainer = Trainer(model, optimizer, loss_fn=F.mse_loss)
trainer.fit(train_loader, validation_loader, epochs=2, progress_bar=False)
results = trainer.evaluate(validation_loader)
print(results)
```

Replace the model, data and loss with your own. For a complete recipe with
dataset-wide metrics, run [regression.py](examples/training/regression.py).

For a scalar loss, reported `loss` is a **mean of batch objectives**, not an
inferred sample/token mean. Unequal batches can make it differ from dataset-wide
metrics. See the [short comparison](examples/metrics/loss_reporting.py) or add an
explicit `LossResult.from_mean(loss, weight=...)` when your recipe knows the denominator.

Move the model to its device **before** constructing its optimizer, and pass
the same `device` to the trainer. Validation is optional:
`trainer.fit(train_loader, epochs=5)` trains without it. `epochs` is the total
target count, including completed epochs. Progress goes to stderr; disable it
with `progress_bar=False`. Start/epoch/end diagnostics use the shared logger.
Configured runs save logs automatically; wrap a direct recipe in
`logging_session(config, "logs")` to save its logs too. See the
[logging guide](docs/logging/README.md) and
[runnable logging recipe](examples/observability/logging_recipe.py).

For arbitrary batch structures or several loss terms, supply one function.
This fragment assumes a compatible model, optimizer and dictionary batches:

```python
def step(model, batch):
    predictions = model(batch["features"])
    targets = batch["targets"]
    loss = F.mse_loss(predictions, targets) + 0.01 * predictions.square().mean()
    return {"loss": loss, "predictions": predictions, "targets": targets}

trainer = Trainer(model, optimizer, step_fn=step)
trainer.fit(train_loader, validation_loader, epochs=5)
```

Your function computes the objective; `fit()` owns backward and optimizer
updates. Return a scalar with the reduction you want. For alternating updates
or other custom update order, use a
[manual loop](docs/training/README.md) instead.

## Set process-wide defaults

Global configuration is optional. Load a complete dictionary, TOML path, or
installed preset once, then let newly constructed components inherit it:

```python
import deepopt

deepopt.load_config({
    "name": "configured-regression",
    "seed": 7,
    "training": {"epochs": 2, "batch_size": 8, "learning_rate": 0.01},
    "runtime": {"device": "cpu", "precision": "fp32"},
})

model = nn.Linear(3, 1)
trainer = deepopt.Trainer(model, loss_fn=F.mse_loss)  # Builds the configured optimizer.
trainer.fit(train_loader, validation_loader, progress_bar=False)  # Inherits two epochs.
current = deepopt.get_config()  # An independent ExperimentConfig copy.
deepopt.reset_config()         # Existing trainer/runtime snapshots remain unchanged.
```

This fragment reuses the imports and loaders from the earlier regression recipe.
`deepopt.load_config("training.toml")` and `deepopt.load_config("builtin:fp8")`
activate file or packaged defaults. A failed load leaves the previous defaults
intact; reset is safe to repeat. Loading configuration does not initialize
PyTorch, CUDA, DDP, or a logging session. Each distributed process owns its own
configuration and must load equivalent settings.

Precedence is **explicit arguments → per-call configuration → global defaults
→ existing API defaults**. A per-call `config={...}` recursively overrides the
global snapshot; lists replace lists. A complete `ExperimentConfig`, TOML path,
or `builtin:NAME` selects an independent complete source. Explicit `False` and
meaningful `None` values are retained, such as `checks=False` and
`fit(..., monitor=None)`. Conflicting final options raise an error instead of
silently removing inherited settings.

Supplied optimizers and schedulers remain caller-owned. Global changes do not
reconfigure existing trainers, runtimes, or captured CUDA graphs. Checkpoints
record the trainer's effective snapshot; resume without an explicit override
uses the saved configuration rather than whichever global defaults are active.
`config_from_dict()` and `load_preset()` remain pure loaders: they return settings
without activating them. See the [configuration guide](docs/runs/README.md).

To start from a complete template, choose a dictionary or an annotated TOML file:

```python
config = deepopt.load_config_template()
config["training"]["epochs"] = 10
deepopt.load_config(config)

# Writes a new file, including comments for optional settings.
deepopt.write_config_template("experiments/training.toml")
```

Both template helpers leave the active configuration unchanged. The writer
creates missing parent directories and refuses to overwrite an existing file.
See the [complete template](configs/templates/complete.toml) for every supported
framework setting and examples of alternative policies.

## Add checkpoints without factories

Replace the earlier `fit()` call with this version when starting a fresh run:

```python
from deepopt import Checkpoint

checkpoint = Checkpoint("runs/my-recipe")
trainer.fit(train_loader, validation_loader, epochs=5, callbacks=[checkpoint])
```

Attach the callback from the first fit. It owns history and epoch files; ordinary
DataLoaders work. Set `every_steps=100` for a stateful loader to save every 100
optimizer updates (`0` disables step saves). Recreate the same components and
call `checkpoint.restore(trainer, train_loader, validation_loader)` to resume.
See [data loading and both batch-skip interfaces](docs/data/README.md).
To evaluate `best.pt` while keeping training weights and optimizer intact, follow the
[best-model evaluation example](docs/checkpoints/README.md#evaluate-best-weights-in-a-direct-recipe).

## FP8 training and model transformation

Enable FP8 when constructing the trainer; both the regular and DDP cores prepare
supported operations automatically. Master parameters and optimizer ownership
are preserved. With the model and loaders from the CPU recipe above:

```python
from deepopt import FP8Config

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
trainer = Trainer(model, optimizer, loss_fn=F.mse_loss,
                  fp8=FP8Config(enabled=True, backend="reference"))
trainer.fit(train_loader, validation_loader, epochs=2, progress_bar=False)
report = trainer.runtime.transformation_report
print(report)
```

The reference backend emulates FP8 quantization with FP32 matrix multiplication;
it runs on CPU or CUDA. Actual FP8 GEMM uses `backend="triton"`, an NVIDIA GPU
with compute capability 9.0 or newer, and the same FP8 dtype for all three roles.
Configured runs enable the same feature with `[runtime.fp8] enabled = true`;
see [fp8.toml](configs/fp8.toml). Manual loops can use `TrainingRuntime(..., fp8=True)`
or the combined [AMP/FP8 context](docs/fp8/README.md#combine-fp8-with-amp-in-a-context).
Native linear calls inside `deepopt.autocast(..., fp8=True)` need no transformation.

Automatic setup defaults to `trace_mode="modules"`, preserving ordinary root
methods such as `generate()`. Root projections become their DO adapters;
`"full"` returns a rewritten FX graph and `"auto"` tries tracing before fallback.
Use `trainer.model` after setup and inspect the report for partial conversion.
Explicit `transform_model(...)` remains available in the [FP8 guide](docs/fp8/README.md).

Run the offline example with `python examples/optimization/fp8_training.py`.
The [FP8 guide](docs/fp8/README.md) covers plans, supported adapters, checkpoints,
DDP, real-model tests, CUDA requirements, and future compression/sparse extension
boundaries. Activation compression and sparse execution are not implemented yet.

## Choose a guide

The [documentation index](docs/README.md) gives a reading order and a feature
map. Each topic has its own README and links to complete runnable examples.

| README | What it covers |
| --- | --- |
| [Getting started](docs/getting-started/README.md) | One recipe, progressively adding production features |
| [Data loading](docs/data/README.md) | Ordinary loaders, batch-skip sampler/loader, and worker-state resume |
| [Training APIs](docs/training/README.md) | `fit`, custom steps, callbacks, evaluation, manual loops, optional tasks and wrappers |
| [Distributed training](docs/distributed/README.md) | CPU/Gloo and CUDA/NCCL, rank metadata, sharding, accumulation and recovery |
| [Execution and compilation](docs/execution/README.md) | Diagnostic modes, update-based mode changes, transfers and model-only `torch.compile` |
| [CUDA Graphs](docs/cuda/README.md) | Model/function capture, context/decorator APIs, replay lifetime and composition with training/DDP |
| [FP8 and model transformation](docs/fp8/README.md) | FP8 autograd, full/submodule FX conversion, reference/Triton backends, model validation and extension boundaries |
| [Losses and optimization](docs/optimization/README.md) | Loss reduction and weighting, accumulation, clipping, schedulers, custom optimizers and strategies |
| [Scheduling and warmup](docs/scheduling/README.md) | Named coefficients, LR policies, update/epoch timing, compiled tensor values and exact resume |
| [Metrics](docs/metrics/README.md) | TorchMetrics, custom `Metric` subclasses, input routing, reporting and synchronization |
| [Logging](docs/logging/README.md) | vLLM-style module loggers, configuration-named files, verbosity and clean command output |
| [Configured runs and data](docs/runs/README.md) | TOML settings, factories, caller-owned datasets, CLI, artifacts and evaluation |
| [Checkpoints and resume](docs/checkpoints/README.md) | Best/last epoch and last step checkpoints, direct APIs, loader state and changed batch sizes |
| [Memory and batch sizing](docs/memory/README.md) | Parameter counts/percentages, model/optimizer GB, fake-tensor estimates and auto/manual batch sizing |
| [Observers and checks](docs/observers/README.md) | Configurable finite checks, activation/weight events and custom diagnostics |
| [Architecture and development](docs/architecture/README.md) | Code responsibilities, tests, performance checks and supported boundaries |
| [Examples](examples/README.md) | Commands, expected behavior and output files for every example |

## Run with configuration and checkpoints

The CLI supplies a synthetic classification recipe. Packaged presets work from
any directory after installation; `--config` also accepts your own TOML path:

```bash
python -m deepopt inspect-config --config builtin:baseline
python -m deepopt train --config builtin:baseline --output runs/baseline
python -m deepopt train --resume runs/baseline/last.pt --epochs 8
python -m deepopt evaluate --checkpoint runs/baseline/best.pt
```

Choose a different file with `--config ./experiments/my-training.toml`.
Python recipes can call `train(config_dict, ...)` or `train("my-training.toml", ...)`
directly. `deepopt.configs.load_preset("fp8", as_dict=True)` provides an editable
dictionary from an installed preset. The repository's [configs/](configs)
and packaged [deepopt.configs](src/deepopt/configs) are both retained; see the
[configuration input guide](docs/runs/README.md#choose-a-toml-file-dictionary-or-installed-preset).

Use a new output directory for a new run. Custom recipes use the Python
`deepopt.run.train()` API with their own experiment factory. That layer adds
logs, checkpoints, and batch-size planning around the same trainer.
Both fresh and resumed configured runs default to `batch_size_mode="auto"`;
custom recipes provide a sample factory for CUDA planning. CPU uses the
requested size unless an explicit memory budget enables planning.
Use `batch_size_mode="manual"` for a fixed choice. Direct `Trainer.fit()` uses
the loaders you supply; it does not perform automatic batch-size planning or
create files unless you attach an artifact callback. Step checkpoint cadence is
configurable using `--checkpoint-every-steps N` or `training.checkpoint_every_steps`;
the default is `0`, keeping ordinary DataLoaders compatible with epoch checkpoints.

## Current scope

Training uses a regular core for one process, or a separate DDP core for
multi-process CPU/Gloo and CUDA/NCCL execution. See the
[distributed guide](docs/distributed/README.md) for architecture, torchrun,
sharded loaders, and per-rank step/epoch recovery.

For **trainer-level automatic mixed precision (AMP)**, CPU runs FP32 with AMP
disabled; CUDA supports FP32 with AMP disabled or BF16 with AMP enabled.
**FP8 training is also supported**, through the separate DO projection and
autograd operations: the reference backend runs FP8 quantization/emulation on
CPU or CUDA, while the Triton backend executes FP8 GEMM on NVIDIA GPUs with
compute capability 9.0 or newer. Parameters retain their original floating-point
dtype. FP8 currently covers supported linear/projection operations, including
attention Q/K/V and output projections. See the [FP8 guide](docs/fp8/README.md)
for formats, scaling, transformation modes and backend requirements.

Trainer-level FP16 AMP and automatic GPU OOM retries are unsupported.
Fake-tensor memory estimates exclude kernel
workspaces, allocator overhead, metrics, and callbacks; validate your workload
on its target hardware.

Choose a runnable entry point for each execution path:

```bash
python examples/execution/execution_modes.py
python examples/execution/compile_training.py
torchrun --standalone --nproc-per-node=2 examples/distributed/ddp_training.py
python examples/cuda/graphs.py --help
```

The compile example defaults to `aot_eager`, which exercises tracing/autograd
without claiming Inductor speedups. CUDA Graph replay requires CUDA; the FP8
Triton backend requires NVIDIA compute capability 9.0 or newer. See the
[example catalog](examples/README.md) for dependencies, outputs and GPU commands.

Run the existing tests with:

```bash
python -m unittest discover -s tests -v
```

See the [development guide](docs/architecture/README.md) for targeted checks,
optional distributed-metric tests, and the CPU benchmark.
