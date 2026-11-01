# One recipe, five steps

[Project README](../../README.md) · [Complete runnable recipe](../../examples/progressive_training.py)

Start with ordinary PyTorch components. You can add logging, checkpointing,
resume and memory planning without a Task, model wrapper, or experiment factory.
Install with `python -m pip install -e .` from the repository root.

Choose either the shell commands or the Python walkthrough below. Each shell
command starts a new process. For Python or a notebook, work from the repository
root and run this setup once:

```python
from examples.progressive_training import make_components
```

This example-only helper constructs fresh PyTorch data, model, optimizer and
Trainer instances; its implementation is in the linked recipe. It is not a
required framework API. Replace it with your own component construction in a
real recipe. The Python blocks below can run in order: stages 1, 2, 3 and 5 each
start fresh, while stage 4 reconstructs components and restores stage 3. Use
unused checkpoint directories when repeating the walkthrough or switching
between the shell and Python versions.

## 1. Train

```python
trainer, train_loader, validation_loader = make_components(with_metrics=False)
trainer.fit(train_loader, validation_loader, epochs=2)
```

You own the model, optimizer and data. Move the model to its device before
creating the optimizer, and pass that device to Trainer. Validation is optional.
For a custom task, replace `loss_fn` with `step_fn(model, batch)` returning a
scalar or a dictionary containing `loss`; no subclass is needed.

Run this stage with generated CPU data:

```bash
python examples/progressive_training.py --stage basic
```

**What does the reported loss mean?** For a plain scalar, `loss` is the mean of
batch objectives. With unequal batch sizes it may differ from a dataset-wide
metric. For example, batch means 0 and 100 report 50, even if the batches contain
2 and 1 examples (dataset mean 33.33). Accumulation likewise averages plain
microbatch objectives. The framework does not guess a denominator.

## 2. Add metrics and logging

```python
from deepopt.logger import init_logger, logging_session

logger = init_logger(__name__)
settings = {"name": "progressive", "training": {"batch_size": 8, "learning_rate": 0.03}}
with logging_session(settings, "runs/progressive/logs"):
    # Fresh components; the helper adds MeanSquaredError by default.
    trainer, train_loader, validation_loader = make_components()
    trainer.fit(train_loader, validation_loader, epochs=2, monitor="mse")
    logger.info("Evaluation: %s", trainer.evaluate(validation_loader))
```

`settings` is your dictionary or dataclass of recipe settings, used to name the
log file. Construct Trainer inside the session to capture its parameter and
storage summary. `mse` is a dataset-wide metric here, independent of the default
batch-average `loss`. For arbitrary metrics, use TorchMetrics subclasses.

```bash
python examples/progressive_training.py --stage logging --output runs/progressive
```

You can keep returning ordinary scalar losses throughout this guide. If your
recipe needs sample/token-weighted accumulation or weighted loss reporting,
see [explicit weighting](../optimization/README.md#explicit-weights-and-reports).
That optional API lets the recipe supply its denominator without guessing from
the tensor shape. Run `python examples/metrics/loss_reporting.py` for a short
comparison on uneven batches.

## 3. Add checkpoints

```python
from deepopt import Checkpoint

trainer, train_loader, validation_loader = make_components()
checkpoint = Checkpoint("runs/progressive/checkpoints")
trainer.fit(train_loader, validation_loader, epochs=2,
            monitor="mse", callbacks=[checkpoint])
```

This replaces the preceding recipe's fit with a fresh run that has checkpointing
from its first epoch. Attach it from the first fit. It owns history, ordinary loader generators,
`last.pt`, optional `best.pt`, and `metrics.jsonl`; Trainer still controls only
training. Existing ordinary DataLoaders work for epoch checkpoints. Without
validation, it saves `last.pt` without selecting a best checkpoint.

```bash
python examples/progressive_training.py --stage checkpoint --output runs/progressive --epochs 2
```

For step checkpoints, choose a stateful loader and set `every_steps=N`:

```python
step_trainer, step_loader, step_validation = make_components(resumable=True)
step_checkpoint = Checkpoint("runs/step-example", every_steps=2)
step_trainer.fit(step_loader, step_validation, epochs=2,
                 monitor="mse", callbacks=[step_checkpoint])
```

N counts completed **optimizer updates**, not batches; `0` (default) disables
step saves. The loader must implement the [resume contract](../data/README.md).
Skipped intervals do not cause checkpoint-callback synchronization or writes.

## 4. Resume

Recreate the same model, optimizer, loss/step and loaders, then:

```python
trainer, train_loader, validation_loader = make_components()
checkpoint = Checkpoint("runs/progressive/checkpoints")
checkpoint.restore(trainer, train_loader, validation_loader)
trainer.fit(train_loader, validation_loader, epochs=3,
            monitor="mse", callbacks=[checkpoint])
```

The callback loads its history and selects the latest `last.pt`/`last_step.pt`,
using completed-epoch state when update counts tie. `epochs=3` means three
**total** epochs, so a checkpoint at epoch two trains one more epoch. Maintain
the monitor and recipe semantics. Checkpoints store state, not Python code.

```bash
python examples/progressive_training.py --stage resume --output runs/progressive --epochs 3
```

The direct callback defaults to manual batch sizing on restore. A supported
loader may be recreated with a new size. Automatic resume planning is explicit:
pass `batch_size_mode="auto"`, `sample_batch_fn`, and an optional budget to
`checkpoint.restore(...)`. The configured run API retains its auto default.
When the callback's `config` is omitted, saved metadata is reused and its batch
size is updated from the reconstructed loader; no configuration rewrite is needed.

## 5. Add optional batch-size planning

Before fresh training, provide a complete representative batch factory:

```python
import torch
from deepopt.memory import prepare_batch_size

def sample_batch(size):
    return torch.empty(size, 3), torch.empty(size, 1)

trainer, train_loader, validation_loader = make_components()
plan = prepare_batch_size(trainer, train_loader, validation_loader,
                          sample_batch_fn=sample_batch)
auto_checkpoint = Checkpoint("runs/progressive-auto/checkpoints")
trainer.fit(train_loader, validation_loader, epochs=2,
            monitor="mse", callbacks=[auto_checkpoint])
```

The factory describes this example's inputs; adapt it to your recipe. Planning
uses fake tensors and never retries after CUDA OOM. CUDA has a default budget;
CPU planning needs an explicit `memory_budget_bytes`, otherwise it logs a skip.
It is an estimate of tensor storage, not a guarantee about all physical memory.
If logging before planning, use a deferred logging session and bind the file
with the selected settings; the complete example demonstrates this.

```bash
python examples/progressive_training.py --stage auto --output runs/progressive-auto
```

Use an unused checkpoint directory for a fresh checkpoint/auto run. The complete
script uses the same model, data and functional loss at every stage. Stages 2–5
add a dataset-wide MSE metric. No factory or Task conversion is required.

## Evaluate the best saved model

After stage 3 or 4, load the saved weights into separate evaluation components:

```python
import torch

evaluator, _, evaluation_loader = make_components()
saved = torch.load("runs/progressive/checkpoints/best.pt", map_location="cpu", weights_only=True)
evaluator.model.load_state_dict(saved["model"])
del saved
best_results = evaluator.evaluate(evaluation_loader)
```

`trainer.evaluate()` otherwise uses its current weights. The separate evaluator
leaves the training model and optimizer intact. `Checkpoint.restore()` resumes
the last training state; loading just best-model weights is for evaluation.
See the [checkpoint recipe](../../examples/checkpointing/checkpoint_resume.py) for a
complete runnable save/resume/best-evaluation example.

Read [data loading](../data/README.md), [checkpoints](../checkpoints/README.md),
and [memory planning](../memory/README.md) only when those features are needed.
