# Regular and distributed training cores

[Documentation index](../README.md) · [Runnable DDP example](../../examples/distributed/ddp_training.py) · [Data loading](../data/README.md) · [Checkpoints](../checkpoints/README.md)

DeepOpt has two execution cores with the same runtime-facing interface:

```text
deepopt/
├── engine/
│   ├── core.py                  TrainingCore: single-process execution
│   ├── execution/runtime.py     Shared execution API and optimizer lifecycle
│   └── training/trainer.py      Shared epoch and callback orchestration
└── distributed/                Parent package for distributed cores
    ├── ddp.py                   DDPCore: DistributedDataParallel execution
    └── _process.py              Internal process-group and collective helpers
```

| Component | Responsibility |
| --- | --- |
| `TrainingCore` | Local forward/backward, local checks, local gradient divisor; no collectives |
| `DDPCore` | Native PyTorch DDP wrapping, synchronized gradients, rank-wide normalization, buffer synchronization, coordinated checks |
| `TrainingRuntime` | Select a core, move batches, apply precision policy, execute optimizer updates, track failed execution |
| `Trainer` | Epochs, accumulation windows, validation, scheduling, metrics, callbacks |
| `distributed` package | Namespace for distributed cores; currently exports only `DDPCore` |

`runtime.core` exposes the selected core. It is `TrainingCore` for a single
process and `DDPCore` when the default process group has multiple ranks. Both
cores expose the training model as `runtime.model`, so checkpoint keys do not
acquire a `module.` prefix. With automatic FP8 enabled, a root projection or
explicit full trace may replace the root; use `runtime.model` after setup.

For FP8 projections, use the [FP8 DDP example](../fp8/README.md#train-with-the-ddp-core),
which shows automatic FP8 preparation before wrapping, `Trainer`/runtime integration,
and direct `DDPCore` forward/backward calls.

## Run the Python example

From the repository root, after `python -m pip install -e .`:

```bash
# Two local CPU ranks; generated data, no downloads or files.
torchrun --standalone --nproc-per-node=2 examples/distributed/ddp_training.py

# One process per GPU; requires two CUDA devices and NCCL.
torchrun --standalone --nproc-per-node=2 examples/distributed/ddp_training.py --device cuda

# Optional checkpoint writes, then continuation to a total of three epochs.
torchrun --standalone --nproc-per-node=2 examples/distributed/ddp_training.py --output runs/ddp-example
torchrun --standalone --nproc-per-node=2 examples/distributed/ddp_training.py --output runs/ddp-example --resume --epochs 3
```

The example builds the same deterministic regression dataset on every rank,
lets `build_loaders` shard it, and runs an unwrapped model through `Trainer`.
Rank zero prints the initialization metadata, optimizer-update count, and global
validation result. Its default batch size is 8 **per rank**, with accumulation 2.
The optional callback writes `last.pt`, `best.pt`, `last_step.pt`, and
`metrics.jsonl` into the requested shared directory. Use a fresh directory for a
new run; `--resume` reconstructs the same recipe and restores the newest progress.

Running the script with ordinary `python` also works, but uses the regular
single-process core and does not validate DDP.

## Launch a configured run

```bash
torchrun --standalone --nproc-per-node=2 -m deepopt train \
  --config configs/ddp.toml --output runs/ddp

torchrun --standalone --nproc-per-node=2 -m deepopt train \
  --resume runs/ddp/last_step.pt --epochs 5
```

The [example configuration](../../configs/ddp.toml) uses `device="auto"`:
CUDA selects NCCL and one GPU per `LOCAL_RANK`; CPU selects Gloo. Choose
`device="cpu"` explicitly for CPU execution or `device="cuda"` for GPU execution.
CUDA requires a CUDA-enabled PyTorch build and enough GPUs for the local ranks.
Multi-node runs use the usual torchrun rendezvous arguments and a shared output
directory. The CLI releases process groups that DeepOpt initialized.

If local hostname resolution prevents `--standalone` from starting, omit it and
use `--nnodes=1 --master-addr=127.0.0.1 --master-port=29500` with an available
local port. The test launcher uses explicit loopback rendezvous and bounded
whole-worker-group cleanup.

`batch_size` is per rank. The nominal effective batch size is
`batch_size × world_size × accumulation_steps`. Learning rates are not changed
automatically. Partial final windows use their actual normalization weights.

## Use the Python API

```python
import gc
import traceback

import torch
from deepopt import Trainer, Checkpoint, initialize_distributed, shutdown_distributed
from deepopt.data import build_loaders

def train(context):
    train_loader, valid_loader, _ = build_loaders(
        train_dataset, valid_dataset, batch_size=16, seed=7,
        resumable=True, num_workers=2, prefetch_factor=2,
    )
    model = make_model().to(context.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optimizer, device=context.device, loss_fn=loss_fn,
                      accumulation_steps=2)
    checkpoint = Checkpoint("runs/python-ddp", every_steps=10)
    trainer.fit(train_loader, valid_loader, epochs=3, callbacks=[checkpoint])

def main():
    context = initialize_distributed("auto")
    try:
        train(context)
    except BaseException as exc:
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        gc.collect()
        shutdown_distributed()

if __name__ == "__main__":
    main()
```

Run this script with torchrun. Initialize before constructing loaders and
moving the model. Caller-initialized process groups are reused and remain
caller-owned. Pass an **unwrapped** model; `DDPCore` wraps the complete step,
so functional steps and `TrainingModule` hooks both run inside DDP's forward.
Custom steps may access the original model's attributes and submodules.

This snippet assumes your datasets, `make_model`, and `loss_fn` already exist;
the linked example is self-contained. Keep model/trainer/callback ownership in
the training function so it ends **before** process-group destruction. The final
collection releases remaining Python cycles while the native group is alive.
On errors, clearing completed traceback frames releases their local references
while preserving the exception and traceback locations. Do not retain trainers,
DDP wrappers, or callbacks in application globals past shutdown.

Initialization returns an immutable `DistributedContext`, also exported from
`deepopt`, with these fields:

| Field | Meaning |
| --- | --- |
| `device` | Resolved `torch.device` for model placement |
| `global_rank` | Process rank in the default group |
| `local_rank` | Process rank on this node, from `LOCAL_RANK` |
| `world_size` | Total number of processes in the default group |
| `local_world_size` | Number of local processes, from `LOCAL_WORLD_SIZE` |
| `backend` | Actual group backend, or `None` without a group |
| `is_initialized` | Whether a default group existed when the context was returned |
| `owns_process_group` | Whether DeepOpt created and will clean up that group |
| `is_distributed` | Convenience property: `world_size > 1` |
| `is_main_process` | Convenience property: `global_rank == 0` |

```python
context = initialize_distributed("auto")
model = make_model().to(context.device)
if context.is_main_process:
    print(context.as_dict())  # JSON-compatible metadata for logging.
```

Without a launcher or an existing group, both ranks are zero, both sizes are
one, and the backend is `None`. For an externally initialized multi-rank group,
global metadata comes from the actual group; unknown local topology is `None`
when its environment variables are absent. A CUDA device index is not inferred
to be a local process rank. Repeated initialization reuses the group and returns
a fresh snapshot. Existing snapshots do not change after shutdown.

Use `.device` where earlier code used the function's returned device directly.
The runtime also exposes its snapshot as `runtime.distributed_context`.

Native DDP options are supplied through `Trainer(..., ddp_options={...})` or
`TrainingRuntime(..., ddp_options={...})`. Supported options are
`find_unused_parameters`, `broadcast_buffers`, `gradient_as_bucket_view`,
`bucket_cap_mb`, `static_graph`, and `validation_sync_buffers`. Unused-parameter
detection defaults to false for dense models; enable it for conditional/unused
parameters. Native DDP owns training buffer synchronization (`broadcast_buffers=True`
by default); explicit false is respected. Validation buffer synchronization is
controlled separately and defaults to true. Device placement and the default
process group are managed by DeepOpt.

## Gradients and manual loops

DDP averages synchronized gradients across ranks. For supported finite loaders,
accumulation uses `no_sync()` around intermediate forward/backward pairs and
reduces on the final microbatch of each full or partial window. Unknown-length
streams retain per-backward synchronization. See [execution policies](../execution/README.md)
for explicit rank-mean, global-mean and global-sum loss semantics.
`LossResult.from_mean(value, weight=...)` supplies explicit sample/token weights;
DeepOpt normalizes by the sum of weights across all ranks and accumulated
microbatches. Plain scalar objectives retain their documented equal-weight
semantics; sample counts do not silently redefine an objective.

Manual loops use the same runtime, but the forward must enter `compute`:

```python
loss = runtime.compute(batch, lambda moved: loss_fn(model(moved[0]), moved[1]))
runtime.backward(loss)
runtime.step()
```

For a summed objective, pass its local denominator to
`runtime.step(gradient_divisor=local_weight)`; the DDP core combines denominators
across ranks. A forward performed outside `runtime.compute` is rejected at DDP
backward because it bypasses DDP's reducer setup. Higher-order synchronized
gradients (`create_graph=True`) are unsupported by native DDP; they remain
available through the regular core.

## Data, validation, and metrics

- The loader factory shards ordinary training with `DistributedSampler` and
  reshuffles it using the completed epoch number.
- Resumable loaders shard a common shuffled permutation, record each rank's
  delivered cursor, and preserve worker-prefetch and rebatching behavior. Use
  the same sampler generator seed on every rank. Worker seeds are derived
  separately per rank, so sharing a shuffle does not duplicate worker RNG streams.
- Training shards are padded to equal lengths, following PyTorch sampler
  semantics. A dataset size not divisible by world size can repeat a few
  examples. `drop_last` subsequently drops incomplete **local batches**.
- Validation is sharded without padding. With `drop_last=False`, metrics count
  each example once, including when a rank receives no validation examples. Evaluation does not
  issue per-forward DDP collectives; buffers are synchronized once beforehand.
- Built-in TorchMetrics state is reduced globally at compute. Custom metrics
  must define appropriate distributed state reductions and enable compute-time
  synchronization. All ranks must compute the same metric structure.

Custom loaders must shard their own data. Training requires the same number
of batches on every rank; unequal exhaustion raises a coordinated error rather
than entering a mismatched backward. Skip utilities remain manual index-skipping
tools, not checkpoint recovery. Worker-local random transform state is still
the dataset/loader's responsibility; use the TorchData adapter and a distributed
sampler when that state needs recovery.

## Checkpoints and side effects

Every rank must call checkpoint save/restore. Rank zero writes a single atomic
checkpoint containing shared model/optimizer state plus each rank's loader
cursor, metric progress, buffers, generators, RNGs, and strategy state.
Step and epoch resume require the same world size and compatible settings.
Step resume supports a new per-rank batch size with the resumable loader.
Saved unwrapped model weights can be evaluated in a single process.

Callbacks execute on every rank. Do **not** rank-zero-guard `Checkpoint` or other
collective operations. Guard your own noncollective output with
`deepopt.is_main_process()`. Built-in log files, progress bars, history files,
and checkpoints are written only by rank zero. A stop request on any rank is
propagated to all ranks at the epoch boundary.

Memory planning traces the regular core locally, adds a gradient-bucket storage
allowance, and selects a batch size that fits every rank's budget. NCCL workspace,
allocator fragmentation and external allocations remain outside that estimate.

## Verification and limits

The distributed tests launch two real Gloo processes and exercise weighted
updates, module hooks, exact step/epoch recovery with dropout, rank-local state,
uneven validation, malformed training lengths, and nonfinite losses. CLI tests
launch torchrun, resume a configured run, and evaluate its weights in one process.
A two-GPU NCCL variant is skipped unless two CUDA devices are available.

```bash
python -m unittest discover -s tests/distributed -v
DEEPOPT_EXTENDED_DDP=1 python -m unittest tests.scheduling.test_ddp_black_box -v
```

For persistent scheduling-worker diagnostics, set
`DEEPOPT_DDP_LOG_DIR=/path/to/logs`. Each invocation writes a distinct directory
containing rank stdout/stderr, fatal-signal Python traces, and initialization,
training, resource-release, and teardown milestones. Failures still fail the
test. CI repeats the four-rank scheduling case and uploads these logs; logging
does not add recovery, retries, or suppress native failures.

This implements fixed-world-size DDP, not FSDP, tensor/pipeline parallelism,
elastic world-size changes, or arbitrary process subgroups. Model-specific
collectives (such as custom synchronized layers) must also respect the chosen
validation and training iteration counts. Process crashes and failures inside
native collectives rely on torchrun/process-group timeout handling; Python
errors at coordinated framework boundaries are reported to the other ranks.

PyTorch references: [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
and [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html).
