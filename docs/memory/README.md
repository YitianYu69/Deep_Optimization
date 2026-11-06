# Memory estimation and batch-size planning

[Documentation index](../README.md) · [Training APIs](../training/README.md) · [Configured runs](../runs/README.md) · [Checkpoints](../checkpoints/README.md)

Use `estimate_training_memory(trainer, sample_batch)` to inspect a recipe's
tensor-storage requirements. Configured runs can use the same estimator to
select a physical batch size before fresh or resumed training. Memory planning
lives outside the trainer's epoch loop.

## Choose the entry point

| Need | API |
| --- | --- |
| Inspect current storage and estimate the persistent training baseline | `deepopt.memory.summarize_model_memory(model, optimizer)` |
| Estimate one supplied batch shape | `deepopt.memory.estimate_training_memory(trainer, sample_batch)` |
| Prepare a caller-owned loader before training | `deepopt.memory.prepare_batch_size(trainer, train_loader, ...)` |
| Automatically size a fresh configured run | `deepopt.run.train(config, output=..., sample_batch_fn=...)` |
| Automatically size a resumed configured run | `deepopt.run.train(resume=..., epochs=..., sample_batch_fn=...)` |
| Keep a specified physical batch size | `train(..., batch_size_mode="manual", batch_size=...)` |
| Restore a trainer directly with sizing | `restore_checkpoint(..., train_loader=..., sample_batch_fn=...)` |

For direct recipes, `prepare_batch_size(trainer, train_loader, validation_loader=None,
mode="auto", sample_batch_fn=..., memory_budget_bytes=...)` reads the requested
loader size, plans, applies supported changes, and returns the decision before
iteration. It is exported from `deepopt.memory`. Use `Checkpoint.restore` for
within-epoch resume sizing rather than preparing an active/resumed epoch twice.

`Trainer.fit()` and `TrainingRuntime` do not invoke automatic memory planning.
They use the loaders and components you provide. Direct recipes can call the
standalone estimator before `fit()` and choose their loader size themselves.
The configured `run.train()` path provides the planning-and-application sequence.

## Parameters, allocated storage, and the persistent baseline

Every successful `Trainer(...)` initialization logs parameter counts with their
percentage of the total, current model/optimizer storage, and estimates for dense
gradients, initialized optimizer state, and their persistent training total through
the shared logger. No additional trainer option is needed. Create the trainer inside
`logging_session(...)` to include this summary in the log file; configured runs
already establish the session before constructing the trainer.

```python
from deepopt.memory import summarize_model_memory, log_model_memory

report = summarize_model_memory(model, optimizer)  # Read-only; no log output.
assert report.model_gb == report.model_bytes / 1_000_000_000  # When storage is known.
report = log_model_memory(model, optimizer)        # Also emits INFO messages.
```

The returned `ModelMemorySummary` exposes:

| Fields | Meaning |
| --- | --- |
| `total_parameters`, `trainable_parameters`, `frozen_parameters` | Element counts of distinct registered parameters; tied parameter objects count once |
| `total_percent`, `trainable_percent`, `frozen_percent` | Count / total × 100; total is 100% for a known nonempty model; unavailable ratios are `None` and logged as N/A |
| `parameter_bytes`, `buffer_bytes` | Unique backing storage in each category, using actual tensor dtypes |
| `model_bytes`, `model_gb` | Union of registered parameter and buffer storage, including nonpersistent buffers |
| `optimizer_bytes`, `optimizer_gb` | Currently allocated tensors in optimizer state and tensor-valued parameter-group settings, excluding model allocations already counted |
| `estimated_gradient_bytes`, `estimated_gradient_gb` | Expected dense gradients for all trainable parameters, using their dtypes |
| `estimated_optimizer_bytes`, `estimated_optimizer_gb` | Existing optimizer allocations plus state expected to be allocated during initialization |
| `estimated_persistent_bytes`, `estimated_persistent_gb` | `model_bytes + estimated_gradient_bytes + estimated_optimizer_bytes`; `None` when a component is unknown |
| `notes`, `as_dict()` | Accounting limits and a serializable report including percentages/GB |

GB means **1,000,000,000 bytes**, not GiB. Shared storages and views count once;
a view retains its whole backing allocation. Because a buffer may alias a
parameter, `model_bytes` need not equal `parameter_bytes + buffer_bytes`.
`MultiOptimizer` includes its child optimizers, deduplicating shared state.
Sparse COO/compressed tensors include index and value storage.

Fresh Adam-like optimizers often have **zero allocated state** until the first
update. The current-storage figure remains zero, while the separate estimate
includes the expected lazy state. Standard `torch.optim.Adam` and `AdamW` allocate
two parameter-shaped moment tensors and one scalar step per participating
parameter tensor; AMSGrad adds a third moment tensor. Standard `SGD` allocates
one momentum buffer when momentum is enabled and no such state otherwise.
`MultiOptimizer` forecasts its supported children. Existing state/settings are
counted as allocated, and only missing state is added to the forecast.

The forecast assumes every trainable parameter receives a dense gradient. It
supports the exact standard optimizer types above; custom optimizers, unavailable
or malformed state, and meta/fake/uninitialized tensors yield `None` for affected
estimates with explanatory notes. Custom optimizer tensors outside `state` and
parameter-group settings are outside this summary. Call the helper again after
an update or checkpoint restore to refresh both measured storage and estimates.

For example, 11,181,642 trainable FP32 parameters occupy 44,726,568 bytes. If
registered model storage is 44,765,128 bytes including buffers, ordinary AdamW
adds an estimated 44,726,568 bytes of gradients and 89,453,136 bytes of moments.
The persistent baseline is **178,944,832 bytes (0.178944832 GB)**, plus scalar step
counters and any tensor-valued optimizer settings. This assumes the optimizer
owns all those parameters.

This helper does not execute the model, copy tensors, read tensor values, or
create optimizer state. Measured model/optimizer storage excludes gradients;
the persistent estimate adds their expected storage. Neither the persistent
baseline nor the measured storage includes a batch, saved activations, transient
tensors, kernel workspaces, DDP communication/bucket allowances, Python object
overhead, or allocator reservations. Meta/fake tensors have zero physical payload.
Uninitialized parameters or inaccessible storage produce `None` for affected
totals. The persistent baseline is not a batch-dependent training peak or total
process memory.

The implementation is isolated in
[`memory/statistics.py`](../../src/deepopt/memory/statistics.py), and re-exported from
`deepopt.memory`. See [model_memory.py](../../examples/memory/model_memory.py)
for frozen/trainable percentages and a before/after optimizer-state example.

## Observe CUDA allocator memory during training

Set `logging.memory_each_epoch = true` in the trainer's configuration to report
memory after each completed training/validation epoch. Alongside model/optimizer
storage and the estimated persistent baseline, CUDA runs report current allocated
and reserved memory and their observed peaks on the trainer's device.

The peak line is labeled **"CUDA peak memory since last reset"**. These are
PyTorch allocator statistics for this process on that device, including other
work in the process. They are not isolated trainer totals or per-epoch peaks:
the reporting window starts at the allocator's last statistics reset, which
application code may perform. DeepOpt reporting does not reset these counters or
synchronize CUDA. Allocations outside the PyTorch allocator, such as CUDA context
or external library allocations, remain excluded. See
[per-epoch logging](../logging/README.md#per-epoch-memory-reports) for configuration
and output policy.

## Estimate a representative computation

```python
from deepopt.memory import estimate_training_memory
from deepopt.logger import log_output

estimate = estimate_training_memory(trainer, sample_batch)
log_output(estimate.as_dict())
```

Supply the **complete batch** expected by your loss or step: inputs, targets,
masks, and any other fields. Existing tensors and meta tensors are accepted.
The sample's structure must also work with the runtime's batch-transfer logic.
No batch axis, class count, sequence length, or task reduction is inferred.

For an existing ResNet/CIFAR classification trainer that consumes batches of
256 FP32 RGB images resized to 256 × 256 and integer class targets:

```python
import torch
from deepopt.memory import estimate_training_memory
from deepopt.logger import log_output

sample_batch = (
    torch.empty((256, 3, 256, 256), dtype=torch.float32, device="meta"),
    torch.empty((256,), dtype=torch.long, device="meta"),
)
estimate = estimate_training_memory(trainer, sample_batch)
log_output({
    "peak_tensor_gb_by_device": {
        device: size / 1_000_000_000
        for device, size in estimate.peak_bytes_by_device.items()
    },
    "persistent_tensor_gb_by_device": {
        device: size / 1_000_000_000
        for device, size in estimate.persistent_bytes_by_device.items()
    },
    "assumptions": estimate.assumptions,
})
```

Match these shapes and dtypes to the actual preprocessing and loss. The meta
sample allocates no image/target payload. This is an explicit, batch-dependent
tensor-peak estimate; trainer initialization does not run it automatically.
Unsupported execution raises `MemoryEstimationError` rather than running a real
probe or reporting a guessed peak. In particular, FP8 quantization and Triton
execution may be unsupported by fake tracing; see [the limits below](#unsupported-paths-and-estimation-limits).

The estimator uses [PyTorch fake tensors](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_fake_tensor.html)
to execute tensor operations using metadata rather than model/input data
payloads. It copies the owned training components with shared parameter and
optimizer relationships preserved, enters training mode on the copy, and traces
two complete accumulation windows. This covers forward/loss computation,
backward, configured gradient processing, optimizer updates, lazy optimizer
state creation, and a following window with that state already present.

The original model, optimizer, and owned strategy state are not trained by the
estimate. Python and Torch RNG state, initialized CUDA RNG state, and an already
imported NumPy module's global RNG state are restored. Small RNG-state copies
and known scalar CPU optimizer step counters are real control-state copies;
there are no real model/input clones or real training probe batches. Ordinary
model and input tensor values remain unknown to the trace.

Run the complete [estimation example](../../examples/memory/estimate_memory.py):

```bash
python examples/memory/estimate_memory.py
```

This seeded CPU example uses meta tensors for the sample batch, so no sample
payload is allocated. Its JSON report contains byte counts and assumptions;
it creates no files. [model_memory.py](../../examples/memory/model_memory.py)
instead performs two real updates and prints the currently allocated storage,
including AdamW's newly created state. These answer different questions.

## Read the report

`TrainingMemoryEstimate` exposes these attributes; `as_dict()` converts the
report to JSON-compatible dictionaries and lists. Sizes are bytes, grouped by
device name such as `cpu` or `cuda:0`.

| Field | Meaning |
| --- | --- |
| `initial_bytes_by_device` | Live storage owned by the supplied training components before tracing; excludes the sample batch |
| `peak_bytes_by_device` | Largest observed live tensor-storage total during the fake execution, including the sample and intermediate tensors |
| `persistent_bytes_by_device` | Component-owned storage remaining after the traced updates, including initialized optimizer state; excludes the sample |
| `assumptions` | Scope and limitations recorded with the estimate |

Storage aliases, such as views and detached tensors, count once. Dead temporary
storages stop contributing. Even an unused tensor field in an already-placed
sample batch contributes while that sample remains live. Counts describe the
represented training path rather than every possible future input shape.

## Automatic and manual batch sizing

The configured runner defaults to `batch_size_mode="auto"` for both starts and
resumes. The configured `training.batch_size`, or the `batch_size=` override,
is the requested upper bound. Provide a function that constructs the complete
sample batch for a candidate size:

```python
import torch
from deepopt.run import train

def sample_batch_fn(size):
    # These dimensions belong to this example's regression recipe.
    return torch.empty(size, 3), torch.empty(size, 1)

train(config, output="runs/my-run", experiment_factory=my_recipe,
      batch_size=32, sample_batch_fn=sample_batch_fn)
train(resume="runs/my-run/last.pt", epochs=8, experiment_factory=my_recipe,
      batch_size=32, sample_batch_fn=sample_batch_fn)
```

The factory runs inside strict fake-tensor mode. Ordinary PyTorch constructors
describe the candidate's shapes without allocating the corresponding data.
The planner does not read the training loader to obtain a sample. The factory
owns image sizes, sequence lengths, layouts, optional fields, and every other
shape decision. Choose demanding representative shapes for the workload.
Keep the factory and traced computation free of I/O or external state changes.

The planner tries the requested size, then integer-halved candidates down to
one: for example, `9 → 4 → 2 → 1`. It selects the first estimated fit. This
is a bounded choice among those candidates, not an exhaustive maximum-size
search. An unsupported trace or an OOM exception aborts planning immediately;
neither causes a real training retry or another candidate attempt.

For a fixed choice, no sample factory is needed:

```python
train(config, output="runs/fixed", experiment_factory=my_recipe,
      batch_size_mode="manual", batch_size=16)
```

The baseline CLI supplies its own synthetic sample factory. Custom configured
recipes provide `sample_batch_fn` in Python. The CLI exposes `--batch-size` and
`--batch-size-mode {auto,manual}`; it does not serialize tensors or callables
through command-line arguments.

## Budget and device behavior

| Setting | Behavior |
| --- | --- |
| CUDA, automatic, no explicit budget | Compare the device's estimated tensor peak against 80% of the sum of CUDA free bytes and initially owned training-tensor bytes |
| CPU, automatic, no explicit budget | Keep the requested size and log that automatic memory planning was skipped |
| Explicit `memory_budget_bytes=N` | Use that positive integer tensor-memory budget as supplied; supports CPU demonstrations and caller-managed budgets |
| Manual | Use the requested size without estimation |

The CUDA formula is `floor(0.8 * (free_bytes + initial_training_bytes))`.
Initially owned bytes are added back because they already reduced the reported
free memory. Sample bytes are not credited, and cached reserved memory is not
treated as extra free memory. An explicit budget receives no additional 20%
reduction. Only the trainer's device is used for selecting a candidate; the
report may contain other devices as well.

The offline [batch-size modes example](../../examples/memory/batch_size_modes.py)
demonstrates auto and manual starts and resumes. It derives an explicit CPU
budget from a reference shape; it does not measure available CPU RAM.

```bash
python examples/memory/batch_size_modes.py
python examples/memory/batch_size_modes.py --output runs/batch-modes
```

Without `--output`, artifacts live in a temporary directory that is removed
when the example completes. The second command keeps both runs in a new parent
directory.

## Applying and recording the selected size

Fresh runs plan before creating their output directory or executing an update.
Resumes restore the saved components, then plan before continuing training.
The selected size is written to subsequent configuration and checkpoint files.
`batch_size.jsonl` records starts and resumes; `resume.jsonl` additionally records
resume decisions. Records contain the mode, requested and selected sizes,
status, reason, and candidate reports when planning ran. Completed-epoch
artifact repair skips an unnecessary plan and leaves `last.pt` unchanged.

When the selected size differs from the requested size, the training loader
must expose an adjustable batching contract. The built-in resumable loader
provides `set_batch_size()`. A standard `DataLoader` with its standard
`BatchSampler` can be resized before its worker iterator exists; custom loaders
need their own setter for this change. Keeping the requested size requires no
setter. Step-resume size changes additionally
require `supports_batch_size_change=True` and preservation of the consumed-data
cursor. See [the full resume contract](../checkpoints/README.md).

Validation is resized only when it used the same requested size as training;
an independently configured validation size remains unchanged. This does not
estimate validation's potentially different computation. Learning rates,
accumulation steps, and scheduler policies are not automatically rescaled.
Automatic changes are rejected for `OneCycleLR` because its fixed update
horizon needs an explicit schedule decision. Changing physical batch size also
changes the optimization trajectory even when the data cursor is preserved.

## Unsupported paths and estimation limits

The estimator raises `MemoryEstimationError` for unsupported traces. Current
limits include non-strided/sparse storage, data-dependent tensor reads or
branches, operators without suitable fake implementations, externally wrapped
`DataParallel`/DDP models, and calls from
`torch.inference_mode()`. A surrounding `torch.no_grad()` is supported: tracing
temporarily enables gradients and restores the caller's mode.

DeepOpt's `DDPCore` is supported: tracing uses an independent regular core with
no collectives, adds a gradient-bucket storage allowance, and automatic planning
selects a candidate fitting every rank's budget. Communication workspaces and
other non-tensor allocations remain outside the estimate. See
[distributed training](../distributed/README.md).

Fake sample tensors must share one strict `FakeTensorMode`; fallback kernels,
automatic real-input conversion, and real-tensor propagation must be disabled.
The planner rejects real-tensor propagation before calling the sample factory.
Failures report their cause rather than inventing output shapes.

The report excludes kernel workspaces, allocator fragmentation, CUDA context,
communication/external allocations, metrics, observers, schedulers, and Trainer
callbacks. These may require additional memory. Python code still executes;
fake-tensor mode cannot undo arbitrary closure/global mutations or external
I/O. Custom components must support independent copying.

The current [FP8 path](../fp8/README.md) performs data-dependent amax/finite-value
operations, and its Triton kernels require real CUDA execution. Do not assume
those computations can be traced by this estimator: use manual batch sizing for
the FP8 examples and measure them on the target GPU. This guide's sparse storage
accounting describes inspection only; sparse training kernels and activation
compression are not implemented by the FP8 adapters.

For ordinary models using `torch.compile`, the estimator traces an eager copy;
it does not predict compiler workspaces or CUDA Graph pools. Perform any
supported memory planning before CUDA Graph capture and choose a fixed batch
shape for replay. See [CUDA Graph capture](../cuda/README.md).

Function closures and globals are not automatically rebound to copied objects.
For example, a step that captures a separate live runtime and calls its
`observe()` method still refers to that runtime. Module or optimizer hooks
inside the traced computation also execute. Use trace-friendly computations
without those external effects, or select manual batch sizing for that recipe.

Two traced windows cannot establish the future state size of every custom
optimizer or every dynamic input path. Later shapes, branches, other processes,
and memory availability can change the real peak. Use the estimate as a tensor
budget with headroom and validate the workload on its target hardware. The
repository's CPU checks do not establish CUDA performance or a guaranteed fit.

Implementation: [memory estimator](../../src/deepopt/memory/estimation.py),
[batch-size planner](../../src/deepopt/memory/planning.py).

