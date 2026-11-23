# Compiled packed activation offload with DDP

This is an opt-in extension for DeepOpt's compiled native AutoDiff runtime.
The release qualification uses real local process groups, rank-local offload
pools and the existing DDP gradient reducer. The default offload policy remains
disabled. CPU/Gloo tests are diagnostics; H200/NCCL qualification and full VAE
throughput/overlap measurements must be recorded on the target GPUs.

The user training loop configures policy and budgets. Generated graph/runtime
code owns pack, transfer, restore and retirement operations. Application code
does not call offload lifecycle methods. DDP gradients still use the normal
reducer; offload copy callbacks do not invoke NCCL. Startup budget checks are
collective, and normal DDP/diagnostic or error coordination can also occur
during training. This does not promise zero per-update collectives.

Enabled offload disables TorchDynamo's DDP graph partitioning inside the
DDP core's scoped compilation/forward path (`optimize_ddp=False`). This keeps
whole-model history planning and forward admission atomic. Resident DDP keeps
its existing behavior, and physical DDP gradient buckets still exist. The
asynchronous copy/prefetch mechanism remains available, but a full native
backward can make parameter gradients ready later and delay DDP allreduces.
The startup receipt records this compiler scope. Target traces and timings are
needed to measure overlap and the cost; this is not a multi-GPU speedup or
zero-overhead claim.

## Qualify the installed candidate

Use the dedicated DDP qualification bundle, whose manifest pins the exact
candidate wheel, package bytes and helper/worker sources. From its root:

```bash
# Requires two visible H200 GPUs, CUDA-enabled Torch, Triton and NCCL.
bash scripts/run_h200_packed_offload_ddp.sh

# Four local ranks; one worker per visible GPU. No capacity sweep is run.
bash scripts/run_h200_packed_offload_ddp.sh --processes 4 --timeout-seconds 600
```

The launcher verifies the source/wheel pins, checks the requested visible GPUs,
then installs the wheel into a **fresh isolated target**, without replacing the
environment's DeepOpt package. It launches one bounded `torchrun` suite with
automatic retries disabled. Child output streams to the terminal and a saved
log. A timeout terminates the whole worker process group. A missing rank,
failed check, skipped CUDA path or source/identity mismatch fails qualification;
a CPU route never substitutes for an unavailable CUDA route.

Results appear under `work/packed_offload_ddp_TIMESTAMP.json`, with per-rank
metadata and a sibling ZIP containing source, JSON and logs. Package install
trees, input/weight payloads and tensor checkpoints are excluded. Tests may
create temporary checkpoints to exercise restore and the next update; those
checkpoint files are not exported. The suite qualifies its recorded contract
cases, not full-model memory savings, maximum host capacity, training quality
or communication/transfer overlap.

For an already installed candidate, pass the same manifest and its exact target:

```bash
python -B benchmarks/compiler/qualify_packed_offload_ddp.py \
  --manifest manifest.json --installed-target /absolute/pinned-target \
  --processes 2 --device cuda --output work/ddp-existing-target.json
```

An explicit `--device cpu` invokes only Gloo/testing-transport diagnostics.
`--dry-run` validates immutable pins and prints the plan without importing
Torch or allocating model/input tensors. Neither establishes CUDA execution.
The single-node launcher does not establish multi-node NCCL safety or
performance; multi-node startup metadata alone is not that evidence.

## Keep budgets per rank and account for the whole node

`runtime.packed_activation_offload.host_pinned_gb` is a **per-rank** managed
physical pool ceiling for version 2 policy. User-facing GB values use SI units:
**1 GB = 1,000,000,000 bytes**. The example below uses the user's
**80 GB per rank**, editable in the config. Four ranks therefore request 320 GB
of offload-owned arenas on their node, plus any configured pageable capacity
and other host allocations. These are configured capacity ceilings, not a
measurement of eagerly allocated host RAM. Choose the per-rank value and host
allocation for the actual job. The runtime does not assume a job size or resize
the budget automatically.

DeepOpt sizes the transport limits from the selected packed saves. The D2H
queue accommodates the largest bundle. GPU staging accommodates the largest
backward consumer's complete packed operands per configured concurrent
backward task. DeepOpt allocates staging on demand. Runtime snapshots record
the resolved byte limits under `transport_limits`. Set positive
`device_staging_gb` or `d2h_inflight_gb` values to impose caps. Use `"auto"` to
override an inherited cap.

Startup preflight gathers actual policies/topology and sums capacities by node
and shared host/job memory scope. It checks observed available RAM and reliable
cgroup/Slurm constraints before model preparation. It neither silently clamps
budgets nor reserves the reported free RAM. Its observed margin must also
accommodate model/optimizer state, Python, data-loader workers and prefetched
batches, NCCL and other processes. A job limit alone does not prove the
requested capacity is currently available. Runtime allocator-domain accounting and
process-wide Torch pinned allocator statistics have different scopes.

Use `packed_offload_selection_mode="peak"` with a bounded budget. This can
offload a selected subset and keep other saved state resident; 80 GB does not
promise complete eligible-history offload or maximum GPU-memory reduction.
`"all_eligible"` requires enough capacity on **every rank and every node** for
that exact shape and recipe. A failed capacity/admission check is an explicit
failure, not permission to pin all remaining host RAM or claim a maximum point.

## VAE configuration example

The example configuration keeps the per-GPU batch of 192, image/resize size of 158,
`torch_amp`, custom optimizer, schedules and model geometry. With four ranks
and accumulation of 1, the nominal global batch is **768**. Learning rate is not
automatically rescaled. This differs from the earlier B128/224 BF16-storage
experiment and needs its own target evidence.

The complete [DDP configuration](../../examples/vae/native_ddp_packed_offload.toml)
is also included in the bundle. Save it as a new config; keep the original
snapshot unchanged. It uses the requested 80 GB per rank and `peak` selection.
The relevant budget table is below. There is **no `runtime.cuda_graph` table**:
remove the entire old table rather than leaving
capture settings in this offload configuration.

```toml
[runtime.packed_activation_offload]
version = 2
enabled = true
policy = "demand"
host_pinned_gb = 80 # Per rank; user editable, never auto-clamped.
host_pageable_gb = 0
max_live_forwards = 1
max_backward_tasks = 1
admission = "error"
```

Offload DDP requires full-graph DeepOpt compilation and native AutoDiff.
Construct a fresh or still-uncaptured model on each rank after loading its
config. A previously executed compiled callable with DDP-split graphs cannot
be adopted into this whole-model offload scope. The normal cold
`Trainer`→`TrainingRuntime` preparation path is supported.

For full training, keep the existing custom-optimizer application and its
`torchrun` launch command. Select the candidate installation and this config.
Use the external optimizer integration for the example's `optimizer="custom"`
setting. The standard VAE entrypoint accepts AdamW and SGD.

Set the dataset paths in your local configuration and choose a fresh output
directory before launching the training application.
`load_config` is process-local: every rank loads the same config before
initializing its runtime. The actual initialized process group enables DDP;
merely including tuning keys under `runtime.ddp` does not initialize a group.
Keep the existing optimizer, checkpoint and schedule paths. Do not add
`torch.autograd.graph.saved_tensors_hooks` or user-level offload calls.

See the [DDP ownership and launch guide](../distributed/README.md),
[native AutoDiff execution scope](./compiler_native_autodiff.md) and the
[qualification worker](../../tests/distributed/_packed_offload_worker.py).
