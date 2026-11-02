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

## Start with the simplest API

Given an existing model, optimizer, and loaders producing `(inputs, targets)`:

```python
from torch.nn import functional as F
from deepopt import Trainer

trainer = Trainer(model, optimizer, loss_fn=F.mse_loss)
trainer.fit(train_loader, validation_loader, epochs=5)
results = trainer.evaluate(validation_loader)
```

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

For arbitrary batch structures or several loss terms, supply one function:

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

## Choose a guide

The [documentation index](docs/README.md) gives a reading order and a feature
map. Each topic has its own README and links to complete runnable examples.

| README | What it covers |
| --- | --- |
| [Getting started](docs/getting-started/README.md) | One recipe, progressively adding production features |
| [Data loading](docs/data/README.md) | Ordinary loaders, batch-skip sampler/loader, and worker-state resume |
| [Training APIs](docs/training/README.md) | `fit`, custom steps, callbacks, evaluation, manual loops, optional tasks and wrappers |
| [CUDA Graphs](docs/cuda/README.md) | Model/function capture, context/decorator APIs, replay lifetime and composition with training/DDP |
| [Losses and optimization](docs/optimization/README.md) | Loss reduction and weighting, accumulation, clipping, schedulers, custom optimizers and strategies |
| [Metrics](docs/metrics/README.md) | TorchMetrics, custom `Metric` subclasses, input routing, reporting and synchronization |
| [Logging](docs/logging/README.md) | vLLM-style module loggers, configuration-named files, verbosity and clean command output |
| [Configured runs and data](docs/runs/README.md) | TOML settings, factories, caller-owned datasets, CLI, artifacts and evaluation |
| [Checkpoints and resume](docs/checkpoints/README.md) | Best/last epoch and last step checkpoints, direct APIs, loader state and changed batch sizes |
| [Memory and batch sizing](docs/memory/README.md) | Parameter counts/percentages, model/optimizer GB, fake-tensor estimates and auto/manual batch sizing |
| [Observers and checks](docs/observers/README.md) | Configurable finite checks, activation/weight events and custom diagnostics |
| [Architecture and development](docs/architecture/README.md) | Code responsibilities, tests, performance checks and supported boundaries |
| [Examples](examples/README.md) | Commands, expected behavior and output files for every example |

## Run with configuration and checkpoints

The CLI supplies a synthetic classification recipe:

```bash
python -m deepopt inspect-config --config configs/baseline.toml
python -m deepopt train --config configs/baseline.toml --output runs/baseline
python -m deepopt train --resume runs/baseline/last.pt --epochs 8
python -m deepopt evaluate --checkpoint runs/baseline/best.pt
```

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
sharded loaders, and per-rank step/epoch recovery. CPU uses fp32; CUDA supports
fp32 or bf16. Fp16 and automatic GPU OOM retries are unsupported.
Fake-tensor memory estimates exclude kernel
workspaces, allocator overhead, metrics, and callbacks; validate your workload
on its target hardware.

Run the existing tests with:

```bash
python -m unittest discover -s tests -v
```

See the [development guide](docs/architecture/README.md) for targeted checks,
optional distributed-metric tests, and the CPU benchmark.
