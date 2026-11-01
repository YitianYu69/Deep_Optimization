# Runnable examples

[Project README](../README.md) · [Documentation index](../docs/README.md)

Run commands from the repository root after installing the project into your
Python environment:

```bash
python -m pip install -e .
python examples/training/regression.py
```

Alternatively, use `PYTHONPATH=src python ...` with an environment that already
has the dependencies. Every example below uses small generated CPU data and
requires no downloads. Data shapes, class counts and architectures belong to
the examples; they are not framework requirements. Reported numbers can vary
with the installed PyTorch version and environment.

Examples are grouped by the feature they teach:

| Folder | Start here for |
| --- | --- |
| `training/` | Standard losses, custom steps and optional module hooks |
| `metrics/` | Custom metrics, routing and explicit loss weighting |
| `optimization/` | Multiple optimizers, custom updates and strategies |
| `checkpointing/` | Save, resume and evaluate best weights |
| `data/` | Batch skipping and worker-aware recovery |
| `memory/` | Storage summaries, fake estimates and batch-size planning |
| `observability/` | Logging, callbacks and diagnostic observers |
| `configured/` | Configuration and reconstruction factories |

The progressive walkthrough remains at the examples root. Earlier example
scripts have moved into these folders; use the updated commands below.

## Follow one recipe end to end

The [progressive guide](../docs/getting-started/README.md) uses one script for all
stages, preserving the same model, data and functional loss:

```bash
python examples/progressive_training.py --stage basic
python examples/progressive_training.py --stage logging --output runs/progressive
python examples/progressive_training.py --stage checkpoint --output runs/progressive --epochs 2
python examples/progressive_training.py --stage resume --output runs/progressive --epochs 3
python examples/progressive_training.py --stage auto --output runs/progressive-auto
```

Basic writes no files; logging writes config-named logs; checkpoint/auto also
create `checkpoints/last.pt`, monitored `best.pt`, and `metrics.jsonl`. Resume
continues the saved run. Optional `--every-steps N` adds a resumable loader and
step saves after N optimizer updates; use the same loader choice when resuming
within an epoch. CPU auto without a budget logs a skip; supply
`--memory-budget-bytes N` to exercise fake estimates on CPU.

## Start here

| Goal | Complete script and command | What to expect |
| --- | --- | --- |
| Ordinary model + loss + metrics | [regression.py](training/regression.py): `python examples/training/regression.py` | Three completed epochs with loss/MAE; no files |
| Custom forward/loss computation | [multimodal.py](training/multimodal.py): `python examples/training/multimodal.py` | A self-supervised two-branch objective and alignment metrics; no class labels or files |
| Different optimizers for different layers | [multiple_optimizers.py](optimization/multiple_optimizers.py): `python examples/optimization/multiple_optimizers.py` | Disjoint layer routing through `MultiOptimizer` and normal `fit()` |
| Own backward/update order | [alternating_updates.py](optimization/alternating_updates.py): `python examples/optimization/alternating_updates.py` | Alternating generator/discriminator updates in a manual loop |
| Understand loss weighting | [loss_reporting.py](metrics/loss_reporting.py): `python examples/metrics/loss_reporting.py` | Unequal batches: batch loss 50 versus dataset/explicitly weighted MSE 33.33; no files |
| Weight unequal microbatches by valid tokens | [weighted_tokens.py](metrics/weighted_tokens.py): `python examples/metrics/weighted_tokens.py` | Five microbatches form two updates per epoch; three epochs produce six updates and explicitly weighted loss reports |
| Route metrics with different signatures | [metric_routing.py](metrics/metric_routing.py): `python examples/metrics/metric_routing.py` | A custom `Metric`, nested scalar names, prediction mean, and monitoring `quality/absolute/mean` |
| Observe progress and stop gracefully | [callbacks.py](observability/callbacks.py): `python examples/observability/callbacks.py` | Update/epoch callbacks, stop after epoch two, then continue the same trainer to epoch three |

These recipes create no checkpoint directories. `fit()` displays an epoch bar
on stderr unless the recipe passes `progress_bar=False`. Callback output belongs
to the recipe; some examples print JSON, while others print ordinary Python
dictionaries or concise summaries.

## Console and configuration-named log files

```bash
python examples/observability/logging_recipe.py --log-dir logs
DEEPOPT_LOGGING_LEVEL=WARNING python examples/observability/logging_recipe.py --batch-size 4
```

[logging_recipe.py](observability/logging_recipe.py) wraps a direct recipe in one `logging_session`.
It writes a config-derived `.log` file with framework lifecycle events, recipe
messages, DEBUG details, and exceptions. Changing the batch size or epoch target
changes the filename. Identical settings append to the same file. Configured
runs automatically save their logs inside `RUN_DIR/logs/`.

All example result output uses `log_output`, which preserves stdout and copies
it to the active session when present. Direct examples without a session still
create no log files. Progress bars are terminal-only. See the
[logging guide](../docs/logging/README.md) for controls and capture boundaries.

## Memory, checkpoints, and batching

| Goal | Command | Artifacts |
| --- | --- | --- |
| Count parameters and current model/optimizer storage | [model_memory.py](memory/model_memory.py): `python examples/memory/model_memory.py` | Initialization counts/percentages and GB; refresh after training shows lazy Adam state; no files |
| Estimate tensor storage from a batch | [estimate_memory.py](memory/estimate_memory.py): `python examples/memory/estimate_memory.py` | Prints JSON with initial, peak and persistent storage by device; no files |
| Resume at an epoch boundary | [checkpoint_resume.py](checkpointing/checkpoint_resume.py): `python examples/checkpointing/checkpoint_resume.py --kind epoch` | Temporary checkpoint directory; reconstructs components and changes size 4→6 |
| Resume after an optimizer update | [checkpoint_resume.py](checkpointing/checkpoint_resume.py): `python examples/checkpointing/checkpoint_resume.py --kind step` | Temporary directory; intentionally interrupts after a durable save, then continues the consumed cursor at size 6 |
| Keep the size during step resume | `python examples/checkpointing/checkpoint_resume.py --kind step --resume-batch-size 4` | Same initial/resumed size; final summary reports both epochs' sample counts |
| Auto/manual sizes on fresh and resumed runs | [batch_size_modes.py](memory/batch_size_modes.py): `python examples/memory/batch_size_modes.py` | Temporary auto and manual runs; prints decisions for start and resume |

To retain artifacts, pass a new output directory:

```bash
python examples/checkpointing/checkpoint_resume.py --kind step --output runs/checkpoint-demo
python examples/memory/batch_size_modes.py --output runs/batch-modes
```

The checkpoint example uses `Checkpoint`, then loads `best.pt` into a separate
evaluator and reports `best_epoch`, `best_validation_loss`, and
`best_validation_mse`. Best selection uses dataset-wide MSE across batch-size
changes. It does not ask
the recipe to track history or register loader generators manually.
The checkpoint example writes direct checkpoint files. The batch-sizing example
creates `auto/` and `manual/` subdirectories containing configured-run artifacts.
It supplies an explicit CPU tensor-memory budget to demonstrate estimation;
it does not search CPU RAM or use real OOM probes. Normal CPU `auto` mode without
an explicit budget keeps the requested size and logs a skip reason.

See the [checkpoint](../docs/checkpoints/README.md) and
[memory](../docs/memory/README.md) guides before applying these patterns to
worker-prefetch loaders, different input shapes, or a GPU workload.

## Focused API examples

Advanced checkpoint integration is separate:
`python examples/checkpointing/checkpoint_resume_low_level.py --kind step` manually
owns history, generator registration and saves. Prefer
[checkpoint_resume.py](checkpointing/checkpoint_resume.py) for ordinary recipes.

| Script | Commands | Demonstrates |
| --- | --- | --- |
| [training/train_functional.py](training/train_functional.py) | `python examples/training/train_functional.py --mode supervised` or `--mode custom` | A loss callable versus a custom step with multiple losses and keyword metric inputs |
| [training/train_module.py](training/train_module.py) | `python examples/training/train_module.py --mode fit`, `--mode manual`, or `--mode wrapper` | Optional module hooks, using a module in a manual loop, or ordinary forward-only wrapping |
| [optimization/train_strategies.py](optimization/train_strategies.py) | `python examples/optimization/train_strategies.py --api fit --weighting samples` | Explicit sample weights and gradient-value clipping |
| [optimization/train_strategies.py](optimization/train_strategies.py) | `python examples/optimization/train_strategies.py --api manual --weighting microbatches` | Reusing `OptimizationWindow` while owning the loop |
| [observability/train_observers.py](observability/train_observers.py) | `python examples/observability/train_observers.py --mode all` | Default checks, checks off, and selective finite/activation observers |
| [metrics/custom_metric.py](metrics/custom_metric.py) | `python examples/metrics/custom_metric.py` | Standalone registered metric states: MAE is `2.0`, then reset count is zero |
| [configured/read_config.py](configured/read_config.py) | `python examples/configured/read_config.py` | Read and validate the baseline TOML, then print selected fields |

`optimization/train_strategies.py` accepts either `--api fit|manual` with either
`--weighting samples|microbatches`. `observability/train_observers.py` also accepts
`--mode default|off|selective` individually and `--device cuda` (or `cuda:N`)
for GPU training; the default device is CPU. These scripts write no checkpoint
files. The module-hook example deliberately reuses one small dataset for both
phases to demonstrate the API; use separate splits for a real evaluation.

## Batch skipping and worker recovery

```bash
python examples/data/skip_batches.py
python examples/data/skip_batches.py --workers 2
```

[skip_batches.py](data/skip_batches.py) demonstrates both `SkipBatchSampler`
and `SkipDataLoader`: it prints remaining indices `[6,7,8,9]`, then a complete
next epoch. Neither interface claims worker-state checkpoint recovery. Both
remain unsuitable for step checkpoints; see [data loading](../docs/data/README.md).

For worker-aware step resume, first install the optional extra:

```bash
python -m pip install -e '.[workers]'
python examples/data/worker_resume.py
```

[worker_resume.py](data/worker_resume.py) uses TorchData, two workers and
explicit stochastic dataset state. It writes temporary step checkpoints,
interrupts after two updates, reconstructs components, and completes the epoch.
It is the only example requiring the optional `workers` dependency; it is not
part of the base-install smoke commands.

## Configured custom recipes

These existing examples add reconstruction factories, application-owned data
settings and managed artifacts. Unlike the direct `fit()` examples, a fresh run
requires an explicit output directory that does not already exist:

```bash
python examples/configured/train_regression.py --output runs/regression --epochs 2
python examples/configured/train_regression.py --resume runs/regression/last.pt --epochs 3

python examples/configured/train_multi_optimizer.py --output runs/multi --epochs 2
python examples/configured/train_multi_optimizer.py --resume runs/multi/last.pt --epochs 3
```

[configured/train_regression.py](configured/train_regression.py) owns its model and generated data
schema. [configured/train_multi_optimizer.py](configured/train_multi_optimizer.py) supplies a full-model
optimizer factory to route different layers. Both use CPU configuration, so the
default automatic sizing mode logs a skip unless the recipe supplies an explicit
budget. They create config/environment logs, completed-epoch metrics, batch-size
decisions, best/last epoch checkpoints, and epoch checkpoints. Step saves are opt-in using a positive configured
`checkpoint_every_steps` interval. Resume adds
decision records and continues toward the **total** requested epoch count.

The [configured-run guide](../docs/runs/README.md) explains the artifacts and
how to extend the recipe. Preserve the same reconstruction code for resume;
checkpoint state alone cannot recreate arbitrary Python models or losses.

## Adapting an example

Replace the local model, data source, and loss/step first. Add `metrics` only
when reporting needs them. Add explicit `LossResult` weights only when the
desired denominator differs from the ordinary batch-objective average. Use a
manual loop when update order changes; no Task or TrainingModule inheritance
is required for the functional recipes.

CUDA adaptation requires moving the model before optimizer construction and
using the matching runtime device; bf16 is the only supported AMP mode.
The examples here are CPU demonstrations, not GPU performance measurements.
For framework tests and the separate CPU benchmark, see
[architecture and development](../docs/architecture/README.md).
