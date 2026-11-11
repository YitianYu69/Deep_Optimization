# FP8 full-model training benchmarks

[FP8 guide](README.md) · [CUDA kernels and autotuning](cuda.md) · [Benchmark catalog](../../benchmarks/README.md)

Measure complete training updates after comparing isolated projection costs.
This guide covers the standalone installed-library training benchmark.

## Full-model training experiments

Use [benchmark_fp8_training.py](../../benchmarks/fp8/benchmark_fp8_training.py) after
projection benchmarks to locate costs in a real timm ViT training update.
Like the projection script, it imports the installed DeepOpt library, can be
copied outside the checkout, and does not modify library files or import paths.
Run directly with Python on one GPU. It requires timm, torchmetrics and a
CUDA-enabled PyTorch installation; CIFAR10 mode also requires torchvision.
No pretrained weights or datasets are downloaded.

Start with the compute experiment. BF16 runtime, prepared current FP8, and
prepared delayed FP8 run by default; the older ablation cases remain opt-in:

```bash
python benchmarks/fp8/benchmark_fp8_training.py \
  --input-mode resident --batches 512 --trace --no-fill-uninitialized-memory \
  --output results/fp8-resident
```

| Case | Precision/execution | EMA | AdamW |
|---|---|---|---|
| `bf16-native` | Native AMP and ordinary PyTorch updates | On | Native default selection |
| `bf16-runtime` | DeepOpt TrainingRuntime, FP8 disabled | On | Native default selection |
| `fp8-intercept` | FP8 runtime, native-linear interception enabled | On | Native default selection |
| `fp8-prepared` | FP8 runtime, interception disabled | On | Native default selection |
| `fp8-delayed` | Prepared delayed FP8, interception disabled | On | Native default selection |
| `bf16-no-ema` | BF16 runtime | Off | Native default selection |
| `fp8-no-ema` | FP8 runtime, interception disabled | Off | Native default selection |
| `bf16-fused-adamw` | BF16 runtime | On | Explicit `fused=True` |
| `fp8-fused-adamw` | FP8 runtime, interception disabled | On | Explicit `fused=True` |

All cases use the same model seed and initial-state SHA256, FP32 parameters and
images, BF16 AMP, CrossEntropyLoss and TorchMetrics Accuracy. Each case/batch
runs in a fresh process. Every step updates weights and clears gradients;
therefore warmed timing does not incorrectly reuse unchanged FP8 weights.
The default model is `vit_base_patch16_224`, batch 512, 10 classes, learning rate
`1e-4`, weight decay `.01` (the original script's AdamW constructor default), EMA
decay `.95`, tensorwise E4M3FN operands, three stages, full FX transformation and
`max_speed`. These settings are recorded. Deterministic execution sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` inside each worker if the variable is unset.
The script does not load the application's TOML configuration.

Then isolate host-to-device transfer using a fixed pinned CPU batch:

```bash
python benchmarks/fp8/benchmark_fp8_training.py \
  --input-mode pinned --cases bf16-runtime fp8-intercept fp8-prepared \
  --batches 512 --trace --output results/fp8-pinned
```

Finally, include real CIFAR10 loading and the original augmentation recipe:

```bash
python benchmarks/fp8/benchmark_fp8_training.py \
  --input-mode cifar10 --data-root /existing/CIFAR10/root \
  --workers 25 --prefetch-factor 2 \
  --cases bf16-runtime fp8-intercept fp8-prepared fp8-no-ema \
  --batches 512 --trace --output results/fp8-cifar10
```

CIFAR10 uses resize, horizontal flip, RandAugment, FP32 normalization and random
erasing. It uses an ordinary pinned PyTorch DataLoader with persistent workers,
not DeepOpt's resumable-loader state machinery. Incomplete batches are dropped
to keep the measured shape fixed. Test tail shapes separately with `--batches
336 272`; these are independent fixed-batch training experiments, not validation
or complete epochs. Resident mode instead reuses one synthetic GPU batch and
provides a compute ceiling. Compare precision cases within each input mode;
differences between modes are diagnostic and not an additive decomposition.

Each output directory contains an incrementally saved `summary.json` and one
subdirectory per case/batch with `result.json`, `worker.log`, GPU telemetry and
optional `training-trace.json` Chrome traces. Use a distinct output directory for
each experiment. JSON includes:

- Uninstrumented synchronized wall and CUDA-stream timing distributions and
  throughput. Defaults are one separately recorded cold update, 10 warmups, and
  three repeats of 20 updates. Setup and first-use costs are reported separately.
- A separate event/host breakdown of fetch, transfer, forward/loss, metrics,
  backward, optimizer/zero-grad and EMA. There is no per-phase synchronization.
- Warmed peak allocated/reserved memory with model, AdamW state, caches, EMA and
  the input source already resident at baseline.
- Actual FP8 projection calls, parameter names and input/output dtypes, plus
  module observations, the conversion report, and distinct autograd-saved
  storage bytes by dtype. Saved-storage totals are not simultaneous peak memory.
- CPU/CUDA operator summaries and a separate Python submission profile. CUDA
  profiling has its own warmup update; profiling is excluded from throughput.
- Installed library source SHA256, package versions, relevant environment
  flags, GPU/process snapshots and one-second GPU telemetry when nvidia-smi is
  available. Telemetry cannot establish exclusive GPU access by itself.

FP8-to-FP8 ablations suppress speedup calculations when observed FP8 projection
calls differ, preventing an interception-off case from claiming a gain by
silently leaving projections dense. All comparisons require identical initial
weight fingerprints. Cases execute sequentially; inspect sample distributions
and telemetry for clock/contention drift and repeat close comparisons in reverse
case order. `--cases` preserves the specified order.

`--steps`, `--repeats`, `--breakdown-steps`, `--profile-steps` and `--profile-rows`
control collection length. `--profile-steps 0` disables CUDA profiling;
`--no-host-profile` and `--no-telemetry` disable their respective passes.
Profiler failures are recorded explicitly. Other worker failures preserve
completed measurements where possible, leave logs, allow later cases to run,
and cause a nonzero final exit. The suite saves its partial summary after every
completed worker. Share `summary.json` first; share traces when timeline analysis
is needed.

The experiment uses a manual training-step loop, not Trainer epochs: it excludes
validation, scheduler/callback handling and checkpoint I/O. Synthetic losses are
diagnostic rather than training-quality evidence. Model, input shape, diagnostic
policy and optimizer/EMA behavior must match before comparing with epoch logs.
Component event intervals include host gaps, Python times include implicit
waits, and profiler CPU totals are nested; do not add these measurements together.

To isolate deterministic allocation filling while retaining deterministic
algorithms, compare these two runs with the same installed library:

```bash
python benchmarks/fp8/benchmark_fp8_training.py --cases bf16-runtime fp8-prepared \
  --input-mode resident --trace --output results/fill-on
python benchmarks/fp8/benchmark_fp8_training.py --cases bf16-runtime fp8-prepared \
  --input-mode resident --trace --no-fill-uninitialized-memory \
  --output results/fill-off
```

PyTorch normally initializes `empty`/`empty_strided` allocations when deterministic
algorithms and `torch.utils.deterministic.fill_uninitialized_memory` are enabled.
This can add fills before Triton overwrites outputs and scratch buffers. The
new flag changes only allocation filling; it does not change the deterministic
algorithm setting. Keep the default enabled until all consumed bytes, including
padding and split-K scratch, are verified written before use. Compare outputs
and gradients before adopting it in training. The new runs also record
`after_timing_state_sha256`: compare the same case across fill-on/fill-off runs
after the same update count. Investigate a mismatch rather than accepting a
latency improvement alone. This fingerprint is collected outside timing. See
[PyTorch's documentation](https://docs.pytorch.org/docs/2.15/deterministic.html).

Report version 2 preserves sample collection order and excludes CUDA user
annotation ranges from activity totals. It also synchronizes only within the
separate profiler pass so warmup kernels do not spill into the active capture.
Earlier raw traces can still be analyzed by selecting `kernel` events inside
complete GPU update windows; their aggregate CUDA activity totals may include
annotation ranges and warmup spillover. Uninstrumented throughput timing is
unchanged by these accounting corrections.
