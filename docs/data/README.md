# Data loading, skipping and worker resume

[Getting started](../getting-started/README.md) · [Checkpoint guide](../checkpoints/README.md)

The loader stays caller-owned. Pick the smallest interface your recipe needs:

| Need | Interface | Step-checkpoint support |
| --- | --- | --- |
| Normal training or epoch checkpoints | PyTorch `DataLoader` | No; disable step saves (the default) |
| Manually skip a known batch prefix | `SkipBatchSampler` or `SkipDataLoader` | No; skipping indices is not full state recovery |
| Exact sample cursor restore and rebatching, including prefetch | `ResumableDataLoader` | Yes; zero or multiple workers; no worker-local state recovery |
| Restore consumed batches and worker snapshots | `StatefulDataLoader` (optional TorchData adapter) | Yes; fixed batch size/worker count within an epoch |

Run these from the repository root after installing the package:

| Example | Command | Result / files |
| --- | --- | --- |
| Skip a known prefix | `python examples/data/skip_batches.py` | Logs remaining indices `[6, 7, 8, 9]`, then the full next epoch; no files |
| Replay transforms and change batch size | `python examples/data/resumable_prefetch.py` | JSON showing nine remaining samples grouped as `[3, 3, 3]` and `replay_matches: true`; no files |
| Same replay with prefetch | `python examples/data/resumable_prefetch.py --workers 2` | Spawns two workers, then restores with zero workers; no optional dependency |
| Restore worker-local RNG state | `python examples/data/worker_resume.py` | Requires `.[workers]`; temporary checkpoints are removed on exit |

All DeepOpt loaders are available from `deepopt.data`. An ordinary or skip loader
is still rejected when step checkpoints are requested. There is no silent
conversion or unsupported claim that worker state can be recovered by skipping.

After distributed process-group initialization, `build_loaders` and
`ResumableDataLoader` shard training across ranks. Training pads shards to equal
lengths; `DistributedEvalSampler` shards validation without padding, including
empty ranks. Custom loader factories own their distributed sampling. See the
[DDP guide](../distributed/README.md) for rank-local resume and batch-size rules.

## Skip using an existing batch sampler

```python
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler
from deepopt.data import SkipBatchSampler

sampler = SkipBatchSampler(
    BatchSampler(SequentialSampler(dataset), batch_size=32, drop_last=False),
    skip_batches=10,
)
loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2)
```

`SkipBatchSampler` inherits PyTorch `Sampler`. It traverses the skipped indices
without fetching or collating those samples. The skip is applied once when the
next iterator starts producing batches; later epochs begin normally. Call
`sampler.skip(N)` before another iterator to request a new skip. Before that
iterator starts, `len(sampler)` reports the remaining batches when the wrapped
sampler has a length. After the skip is consumed it describes the next full pass.

## Use the convenient loader

```python
from deepopt.data import SkipDataLoader

loader = SkipDataLoader(dataset, batch_size=32, num_workers=2, skip_batches=10)
```

`SkipDataLoader` inherits PyTorch `DataLoader`, accepts normal options including
`collate_fn` and `batch_sampler`, and uses the same skip sampler internally.
Call `loader.skip(N)` before the next iterator, never during active iteration.
It requires a map-style dataset and batching. Iterable streams cannot be skipped
by discarding their dummy indices and are rejected explicitly.

A skip counts **physical batches**, not optimizer updates. Reconstruct the same
sampler order, batch size, drop-last policy, and epoch-start sampler RNG first.
Do not infer consumed batches from optimizer-step counts when accumulating or
using partial windows. Skipping avoids fetching earlier samples; it does not
advance their random transforms or restore worker-local state. For shuffled
loading, seed/reconstruct the same epoch-start sampler and generator, rather
than reusing its end-of-epoch state.

Run both interfaces with `python examples/data/skip_batches.py`; use
`--workers 2` to run the same example with worker processes.

## Resumable sample loading with workers and prefetch

```python
import torch
from deepopt.data import ResumableDataLoader

loader = ResumableDataLoader(dataset, batch_size=32,
                             generator=torch.Generator().manual_seed(7),
                             num_workers=2, prefetch_factor=2,
                             persistent_workers=True)
```

This loader saves its permutation and the sample cursor of the last batch
returned to the caller. Worker prefetch advances a separate dispatch cursor;
queued batches do not count as consumed. Save after training has processed the
returned batches, at an optimizer boundary. On restore, unconsumed samples are
fetched again from that saved cursor. Batch size, worker count and prefetch
settings may change on restore without skipping or repeating consumed indices.
The checkpoint format remains compatible with existing zero-worker snapshots.

`loader.set_batch_size(size)` also supports an active iterator: it discards
prefetched batches and regroups the remaining samples at the consumed cursor.
Loading a checkpoint, resetting the epoch, or requesting a new epoch iterator
invalidates any older iterator. Persistent workers can be reused across epochs
and queue resets. A new epoch consumes the same shuffle-generator draws whether
workers are reused or recreated.

Use a map-style dataset and ordered delivery (`in_order=True`, the default).
Extra PyTorch [DataLoader options](https://docs.pytorch.org/docs/stable/data.html)
such as `collate_fn`, `pin_memory`, `worker_init_fn`, and
`multiprocessing_context` are accepted. `num_workers=0` remains the default;
set `prefetch_factor` only when workers are enabled. With the spawn start method,
keep dataset/collate definitions importable and run training under an
`if __name__ == "__main__":` guard.

This guarantees sample order and cursor recovery, not recovery of worker-local
random transforms, mutable dataset state, or side effects of prefetched reads.
For exact transformed values, use deterministic per-sample transforms or the
`StatefulDataLoader` adapter below with dataset state methods.

`drop_last=True` is supported on both `ResumableDataLoader` and `build_loaders`.
It discards an incomplete final batch, just as ordinary batching does. The
snapshot records this policy, which must match during step resume. The discarded
tail is committed when the last full batch is delivered, so restoring that
update with a smaller batch size cannot bring back discarded samples. Earlier
resizes regroup the remaining samples and apply the same drop policy to those
new groups. An epoch with fewer samples than its batch size yields no batches;
the trainer reports an empty-loader error.

## Worker-aware step checkpoints

For stateless, reproducible augmentations without worker RNG snapshots, wrap a
deterministic dataset with `SeededTransformDataset` and use the resumable loader:

```python
from deepopt.data import SeededTransformDataset, ResumableDataLoader

def augment(sample, *, generator):
    image, target = sample
    if torch.rand((), generator=generator) < 0.5:
        image = image.flip(-1)
    return image, target

dataset = SeededTransformDataset(dataset, augment, seed=7, transform_id="flip-v1")
loader = ResumableDataLoader(dataset, batch_size=32, num_workers=2,
                             persistent_workers=True,
                             generator=torch.Generator().manual_seed(7))
```

Every random operation in the transform must use the supplied CPU generator.
Global Python/NumPy RNG calls and randomness in the wrapped dataset are not
captured by this wrapper. Epoch/sample keys are sent with requests, so persistent
workers do not depend on mutable local epoch state. The transform configuration
and active epoch are checkpointed. `sample_seed(seed, epoch, index, view)` is
also public for recipes owning their transform generator. Use a new
`transform_id` when changing transform semantics.

This guarantees augmentation replay under that contract, not unconditional
bitwise model/optimizer replay across kernels or DDP bucket changes. See
[reproducibility limits](../execution/README.md#transfers-and-reproducibility).

Install the optional dependency:

```bash
python -m pip install -e '.[workers]'
```

```python
from deepopt import Checkpoint
from deepopt.data import StatefulDataLoader

loader = StatefulDataLoader(dataset, batch_size=32, num_workers=2)
checkpoint = Checkpoint("runs/worker-example", every_steps=100)
trainer.fit(loader, epochs=5, callbacks=[checkpoint])
```

The optional extra currently targets TorchData 0.11.x.
The adapter uses [TorchData's StatefulDataLoader](https://github.com/pytorch/data/blob/main/torchdata/stateful_dataloader/README.md)
to collect worker snapshots and restore the consumed position. Its small
compatibility adapter preserves an exhausted saved iterator once, so DeepOpt
can finalize an epoch without replaying it. Snapshots record the TorchData
version; restore requires the same version and batching/worker settings.

Stochastic dataset transforms must expose their own state. For example, a
dataset using the worker's Torch RNG can define:

```python
def state_dict(self):
    return {"rng": torch.get_rng_state()}

def load_state_dict(self, state):
    torch.set_rng_state(state["rng"])
```

Include Python/NumPy/custom RNG state too if your dataset uses it, encoded into
checkpoint-safe tensors and primitive containers. Keep `in_order=True`.
Within-epoch batch-size changes are rejected by this adapter. Fresh or completed-
epoch runs may resize a new, not-yet-iterated loader; step rebatching requires a
loader specifically capable of restoring that transformation.

`python examples/data/worker_resume.py` demonstrates two workers, dataset RNG
state, an intentional interruption after a saved update, and reconstruction.
The integration is optional: the base install, skip utilities, ordinary epoch
checkpoints and `ResumableDataLoader` (including its workers) do not import TorchData.

## Configured loader helper

`build_loaders(train_data, validation_data, *, batch_size, seed, num_workers=0,
resumable=False, **loader_options)` returns train loader, optional validation
loader, and named generators. `validation_data=None` is supported.

Ordinary loading is the default. `build_experiment` opts its default helper into
resumable loading when step saving is configured. To select workers explicitly:

```python
from functools import partial
from deepopt.data import build_loaders

loader_factory = partial(build_loaders, num_workers=2, resumable=True)
```

Pass that factory to `build_experiment`. Use `resumable=False` for worker loading
with epoch checkpoints only. A custom factory can still return any loader that
satisfies the required contract; it owns collation, batching and stream state.
`build_loaders(resumable=True)` selects `ResumableDataLoader` for every worker
count, supporting both step and epoch checkpoints, prefetch, and within-epoch
rebatching without TorchData. For worker-local dataset/RNG recovery, construct
`StatefulDataLoader` in a custom factory instead. Existing TorchData step
snapshots must still be restored with that adapter; their format differs from
the sample-cursor loader's format.

The helper shuffles training and keeps validation sequential within each rank.
Its generator mapping uses the names `"train"` and `"validation"`. Do not pass
`shuffle`, `generator`, `sampler`, or `batch_sampler` through `loader_options`;
construct a custom loader factory when those choices belong to your recipe.

When persistent workers are enabled, the helper also makes ordinary training
and validation loaders consume one base-seed draw per epoch, matching freshly
constructed loaders after an epoch restore. This preserves sampler order and
named generator progression; it does not capture worker-local transform state.

`SkipBatchSampler` and `SkipDataLoader` are manual skipping utilities, not part
of checkpoint restoration. The resumable loader already starts at its saved
sample cursor; applying an additional batch skip would discard more data.

Options passed through `loader_options`, including `drop_last`, apply to both
training and validation. Build the two loaders separately if validation should
keep its partial batch while training drops it.

## Custom collation and replay boundaries

The [prefetch example](../../examples/data/resumable_prefetch.py) combines
`SeededTransformDataset`, a module-level `collate_fn` returning a dictionary,
and restoration from batch size four to three. Snapshot the loader after the
returned batch has been processed; prefetch may have already read additional
samples, but they remain unconsumed in the saved state. The example compares
sample indices and transformed values against an uninterrupted reference.

The cursor counts consumed samples. If four samples have been delivered, the
next group of three begins at cursor four; it is not rounded to a multiple of
three. Internally, `batch_origin` records where the current grouping began so
later checkpoints can validate boundaries after repeated rebatching. Callers
use the loader's public state methods instead of editing these fields.

A loader snapshot also saves its generator and seeded-transform epoch. It
does not save a model, optimizer, accumulated gradients, or training progress.
Use [Checkpoint](../checkpoints/README.md) for complete training recovery. In
a manual epoch loop, call `loader.set_epoch(epoch)` for explicit augmentation
epoch numbering; the built-in trainer does this for you.
