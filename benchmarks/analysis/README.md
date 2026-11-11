# DO operator and interaction analysis

This suite combines per-operator PyTorch BF16 comparisons with the four-way
FP8/compression experiment. It benchmarks the **installed** DeepOpt library.
Copy this entire `analysis` directory to the GPU machine if the checkout is not
there. It never installs packages, downloads models, or redirects imports to a
checkout's `src` directory. `--help`, `--list-ops`, `--write-example`, and
`--dry-run` do not import Torch or initialize CUDA.

## Start with the regression

Run from the repository root using the Python environment containing the build
you intend to measure:

```bash
python benchmarks/analysis/benchmark_do_ops.py \
  --config benchmarks/analysis/configs/interaction.json \
  --observed-epoch-seconds 21 27 22 47 \
  --output work/interaction
```

Edit the shapes/settings to match your actual model. The reported epoch times
are user-supplied context: 21 seconds BF16, 27 compression only, 22 FP8 only,
47 combined. They imply an additive prediction of 28 seconds and a 19-second
interaction. These numbers are never presented as measurements made by the suite.

Each workload runs five cases, always from matched source weights and inputs:

| Case | Execution |
| --- | --- |
| `torch_bf16` | Ordinary PyTorch BF16 autocast |
| `do_bf16` | DO boundary/context, FP8 and compression disabled |
| `compression` | DO BF16 compute with saved-state compression |
| `fp8` | DO FP8, compression disabled |
| `fp8_compression` | Both enabled |

For measured wall time `T`, the interaction is:

```text
T(fp8_compression) - T(fp8) - T(compression) + T(do_bf16)
```

Positive values identify candidates for interaction regressions. The DO BF16
case controls for DO dispatch/context overhead; the plain Torch case separately
measures that overhead. Missing or failed rows prevent a complete interaction
claim. CPU results validate behavior only.

## Cover every operator and settings

```bash
# Every public operator family, index-return alias, pipeline and codec diagnostic.
python benchmarks/analysis/benchmark_do_ops.py --preset all --output work/all-ops

# Inspect the inventory and experiment count before a larger sweep.
python benchmarks/analysis/benchmark_do_ops.py --list-ops
python benchmarks/analysis/benchmark_do_ops.py --preset extended --dry-run --output work/extended

# Main-effect sweep: only one setting changes from the base configuration.
python benchmarks/analysis/benchmark_do_ops.py \
  --config benchmarks/analysis/configs/settings_sweep.json --output work/sensitivity

# Fully cross a deliberately narrowed JSON grid to test suspected interactions.
python benchmarks/analysis/benchmark_do_ops.py \
  --config your-small-grid.json --sweep cartesian --output work/crossed

# Cross all BF16/FP8 input/output pairs, scale ownership, formats and blocks.
python benchmarks/analysis/benchmark_do_ops.py \
  --config benchmarks/analysis/configs/precision_boundaries.json \
  --sweep cartesian --output work/boundaries

# Generate a complete editable settings object.
python benchmarks/analysis/benchmark_do_ops.py --write-example work/my-config.json
```

`--preset extended` applies the finite grid in [matrix.py](matrix.py) to the
regression workloads. It covers compression widths/groups, Q/K/V overrides,
normalization recentering, operand formats and independent block shapes,
current/delayed recipes and history settings, backward backend/transport,
pipeline stages, tracing/interception, binding, execution, cache/finite checks, precision boundaries,
SDPA backend, optimizer, accumulation and checkpointing. One-at-a-time coverage
does **not** test all interactions. Cartesian coverage fully crosses the declared
values; it cannot enumerate every possible size or integer hyperparameter.
`--max-experiments` rejects accidentally oversized grids before workers start.

The workload JSON supports arbitrary operator kwargs: convolution stride,
padding, groups/depthwise, dilation and output padding; normalization suffix,
affine parameters, running statistics and train/eval; pooling window/stride,
ceil mode, output size and indices; SDPA key length, KV heads, boolean/additive
mask, causality, dropout, scale and GQA. Set `layout` to `contiguous`, `strided`
or `channels_last`. Set `input_requires_grad`, `parameter_requires_grad` and
`training` explicitly when needed. Inspect [catalog.py](catalog.py) and
[workloads.py](workloads.py) for the exact construction contract.
`binding="intercept"` enables all interception families; other bindings retain
the individual `fp8.intercept_*` switches. `fp8.exclude_modules` accepts your
model's module paths. Unknown settings are rejected rather than ignored.

Use `--gradient-modes all dx dw db none` to run separate trainability probes.
They select input, weight, bias or no gradients and rerun the full operation;
their costs are not additive phases of a single backward. Parameterless
operations explicitly skip weight/bias probes. `none` measures forward/loss
without backward computation. Built-in workloads exclude illegal in-place leaf
mutation; use a model factory for a valid in-place pipeline.

## Profile the slow cases

After identifying a small set of regressions, enable separate profiling:

```bash
python benchmarks/analysis/benchmark_do_ops.py \
  --config benchmarks/analysis/configs/interaction.json --ops linear scaled_dot_product_attention \
  --profile-steps 3 --phase-steps 10 --output work/profile
```

Throughput uses warmed, uninstrumented complete updates. The default is 10 warmup
updates, then 50 updates per repeat, repeated five times. Warmup includes JIT,
autotuning and compilation. Increase these counts when variability remains high.
Both synchronized wall time and CUDA stream event intervals are retained with
repeat samples, median, p95 and standard deviation. Process order is shuffled
reproducibly. One process runs at a time to avoid GPU contention.

Separate passes collect forward/loss and backward intervals, saved tensors,
allocator peaks, annotated Chrome traces, self CPU operator time, synchronization
operations, actual GPU kernel names and launches. Name-based kernel buckets
suggest packing, decoding, scaling, GEMM and layout costs; inspect raw traces to
confirm them. Native gradient computation is visible in autograd trace ranges;
gradient probes help isolate which requested gradient introduces a cost.
Phase intervals describe a single microbatch and exclude the optimizer and
accumulation loop; complete-update timing includes all configured microbatches.

CUDA event intervals include GPU idle time caused by host submission gaps.
Kernel activity sums can overlap and exclude host gaps. Nested CPU/module ranges
must not be added to kernel times. Profiling uses the eager callable for graph
cases, explicitly labeled in the result; it does not claim to describe graph
replay's host overhead.

## Use your model and data

[example_model.py](example_model.py) is an executable, dependency-free factory
example. Replace its `build` function with your own model and representative
input/target batches; use an importable `module:function` in the workload JSON.
It receives `device=torch.device(...)` plus `factory_kwargs` and returns:

```python
{
    "model": model,                  # fresh eager torch.nn.Module
    "inputs": (input_tensor,),       # already on the chosen device
    "kwargs": {},
    "loss_fn": lambda output: loss(output, targets),
    "fingerprint_data": (targets,),  # include data captured by the loss closure
    # Optional data/transfer experiment:
    "fetch": fetch,                  # returns (inputs_tuple, kwargs_dict)
    "reset": reset,                  # rewind fetch to the identical first batch
}
```

Factories run after the same seed is set in every worker. Do not download data,
reuse mutated global models, or choose a different architecture per case.
`fetch` is timed and can include loader/host-to-device transfer; `reset` is
required when `fetch` exists. Omit both for resident-input compute experiments.
Factory-owned inputs/targets and dataset RNG must be reproducible. The initial
fingerprint covers the model, initial inputs and `fingerprint_data`; it cannot
prove that an arbitrary external loader later emits matching data.

```bash
python benchmarks/analysis/benchmark_do_ops.py \
  --config benchmarks/analysis/configs/model_ablation.json \
  --capture-workloads --output work/model-ablation
```

The ablation config enables compression in one module family at a time with FP8
unchanged. Compare those rows against both an empty family list and an explicit
all-family list: module hooks add submission overhead. MHA scopes include their
internal projections, so the attention ablation is not a pure Q/K/V-codec-only
measurement. Use the standalone SDPA comparison for that distinction.
Functional calls outside recognized module scopes remain uncompressed during
family ablations. Use transformation reports and traces to verify coverage.

`--capture-workloads` records native module shapes and calls in a separate pass
and writes `recorded-workloads.json`, which can be replayed with `--config`.
Functional-only calls, arbitrary alias strides, unpool index values and running
buffer values cannot be fully reconstructed by module hooks. The replay uses
representative data and approximates strided layout; the real-model experiment
and profiler remain the authority. With accumulation, captured counts are per
complete update; use accumulation=1 when replaying those counts. Optional
`calls_per_step` and `steps_per_epoch` produce explicitly labeled estimates of
model/epoch contributions, not measured epoch latency.

The factory example also supports `input_mode="pinned"` to measure repeated
pinned-host transfers. Connect your actual loader to `fetch` to include its
latency. Keep optimizer, EMA/logging policies and data/transfer behavior matched
when comparing with the original epoch measurements; this harness does not
silently enable your application's callbacks.

## Accuracy, memory and compatibility

- Numerical diagnostics use a fresh forward/backward before timing, reset the
  model, optimizer, delayed histories and RNG, and compare outputs, loss, input
  gradients, parameter gradients and optional optimizer updates. Integer outputs
  are checked exactly. The default samples at most 65,536 deterministic elements
  per float tensor; use `--max-numerical-elements 0` for full numerical comparison.
  Finiteness and integer indices are always checked in full. Final timed outputs
  and parameters are also checked for finiteness.
- JSON `tolerances` supplies `default` and per-operator `rtol`/`atol`. Default
  0.05/0.01 is a diagnostic threshold, not a convergence guarantee. Lossy 1/2-bit
  gradients may fail it. Raw timing ratios remain available; qualified speedups
  require matching initial fingerprints and passing the configured thresholds.
  Use `--strict-accuracy` to fail the command on numerical rejection, and verify
  real task convergence separately.
- Saved bytes include codec metadata observed by outer hooks, but nested
  attention/checkpoint hooks can hide payloads. Observed unique storage counts
  include references to existing parameters/inputs. They are not incremental
  activation memory or simultaneous peak memory. GPU allocator peaks include
  gradients/workspaces/scratch. CPU restore copies do not occupy benchmark GPU
  memory. Codec diagnostics report their payload and metadata independently.
- Delayed scaling with blocks, stateless delayed interception, host-visible
  finite checks during CUDA capture, and other known incompatible combinations
  are explicit skipped rows. Unexpected exceptions, OOMs, compilation errors,
  worker crashes and timeouts are errors with logs; they never become successes.
- Functional binding supports explicit BF16/FP8 boundaries for applicable
  activation/normalization/pooling operations. Projection outputs use
  `fp8.out_dtype`; SDPA computes natively, with FP8 projections tested in its
  pipeline. BatchNorm/GroupNorm have no caller-provided scale option. Prepared
  model precision is selected by the planner. Requested settings and effective
  execution metadata/fallbacks appear together in the results.
- Graph replay captures gradient clearing and model forward/backward; optional
  optimizer updates execute outside capture. Dynamic data fetch and module-family
  ablation scopes require eager execution. Compile mode retains DO's declared
  eager boundaries and reports Dynamo graph-break/recompile counters.
- Set `runtime.optimizer_options` to match AdamW/SGD options such as weight
  decay, betas, momentum, foreach or fused execution. Optimizer state is warmed
  before the memory pass. This runner measures a single process/device; it does
  not initialize DDP or reproduce an application's scheduler, EMA, logging or
  checkpoint callbacks. Validate those separately when reconciling epoch time.

## Results and restart

The output directory contains `plan.json`, `summary.json`, `results.csv`,
`report.md`, interruption-safe `progress.json`, and per-case request/result,
snapshot, log and optional trace files. Interactions are ranked by extra
wall milliseconds; weighted contributions are retained when call counts are
provided. Do not sum isolated contributions into a causal claim without a
confirming real-model ablation. Repeat uncertainty is descriptive; three or
more repeats and stable GPU clocks are needed for meaningful small differences.

Use `--resume` to reuse successful cases with matching benchmark code, installed
DeepOpt Python-source fingerprint, settings and measurement options. Re-run after
changing external factory code. Incomplete/failed worker results are never reused. Nonzero exit
status signals unexpected errors, or rejected accuracy with `--strict-accuracy`.

For a CPU smoke check:

```bash
python benchmarks/analysis/benchmark_do_ops.py --preset all --small \
  --device cpu --backend reference --in-process \
  --warmup 0 --iterations 1 --repeats 1 --phase-steps 0 --output work/cpu-smoke
python -m unittest tests.benchmarks.test_do_analysis_benchmark -v
```

`--in-process` is restricted to CPU validation. CUDA timing always uses fresh
worker processes for isolation. CPU/reference checks do not establish CUDA
kernel correctness, graph capture compatibility, occupancy, or throughput.
