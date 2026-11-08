# Delayed FP8 Linear scaling

Current-tensor scaling remains the default. Delayed scaling is an explicit
tensorwise recipe with registered numerical state, intended for comparison with
standalone Transformer Engine Linear. Its CUDA kernels and parity target require
validation on compatible hardware; no TE performance parity is claimed.

```python
from deepopt import FP8Config, DelayedScalingConfig, TrainingRuntime

settings = FP8Config(
    enabled=True, backend="triton", scaling_recipe="delayed",
    intercept_linear=False,
    delayed_scaling=DelayedScalingConfig(
        history_length=1024, amax_algorithm="max", margin=0, reduce_amax=True,
    ),
)
runtime = TrainingRuntime(model, optimizer, device="cuda", fp8=settings)
# Use runtime.model; parameter identities and optimizer ownership are preserved.
```

Use tensorwise block shapes for every operand. Blockwise delayed scaling is
rejected. Histories belong to prepared Linear/HF Conv1D/attention projections
and registered FX functional call sites. A functional adapter receives the
original weight rather than registering another Parameter. Distinct call sites
sharing a weight retain independent activation/gradient histories. Repeated
calls through one shared adapter combine observations within its boundary.
Bare intercepted `F.linear` has no persistent owner and rejects delayed scaling;
use prepared adapters or a traced functional call site.

## Numerical recipe and boundaries

Each projection owns FP32 scales and ring-buffer histories for activation,
weight and output gradient, plus int64 update/generation counters. Initial
scales are one and histories zero. Conversion uses the available scale, while
the current amax becomes an observation for a future update. `max` selects the
maximum of the history; `most_recent` selects the latest observation. The update
uses `(FP8_MAX / amax) / 2**margin` in FP32 without power-of-two rounding.
Zero or nonfinite selected amax, or an invalid new scale, preserves the previous
scale. A nonfinite history entry can keep a `max` scale unchanged until it leaves
the window. Strict finite diagnostics reject nonfinite inputs before recording
their observations; unchecked execution retains the defined history behavior.
The formula and recipe options correspond to the
[TE 2.9 recipe definitions](https://github.com/NVIDIA/TransformerEngine/blob/v2.9/transformer_engine/common/recipe/__init__.py).

With an optimizer-owned TrainingRuntime/Trainer, scales are frozen across all
microbatches until successful `runtime.step()`. Observations combine by maximum
and each active role advances once per optimizer update. Frozen weights can
reuse their quantized values within that window. `zero_grad()` abandons a
pending window, and failed runtime operations release pending observations.
Checkpoint only after an update has completed.

Standalone autocast scopes define forward boundaries. Standalone backward can
update each projection separately; an explicit `fp8_scaling_scope()` around a
forward/backward or accumulation window batches those updates. Nested scopes
inherit the active boundary and never double-advance. An explicit outer boundary
also overrides the runtime's default boundary, including when it spans steps.
Evaluation and no-grad inference freeze histories. EMA parameter swaps retain
that evaluation state and invalidate quantized weights through parameter versions.

CUDA metadata updates batch pointers and observations instead of launching a
Python history loop per operand. Reduction launches only valid observation
chunks using a flat prefix-indexed grid. State-pointer/recipe tables are cached
until storage or recipe changes; dynamic tables can be reused when their pointer
layout recurs. Caches are bounded, stream ordered and excluded from checkpoints.
Runtime state discovery and immutable recipe signatures are also reused.
Pending amax arrays are compacted after 32 MiB of retained backing storage
or a bounded observation count without advancing scales. Shared views are
charged once, including unused capacity in a cold-tuning allocation. A single
incoming conversion can transiently exceed that limit before compaction.
Warm converter allocations use the in-process winning tile; unknown/cold keys
reserve enough space for every candidate. Each invocation owns
its FP8 outputs, scratch and two-scalar scale/inverse-scale snapshot. Backward
saves the immutable inverse scale used for its operands, never a mutable scale
buffer. Streams sharing an owner are ordered; overlapping independent update
boundaries cannot mutate the same projection concurrently.

## Conversion, backward and recomputation

The delayed converter reads the high-precision input once to cast, optionally
transpose, collect amax partials, and snapshot the exact scale and inverse.
Three tile/warp candidates include the original 32x32/4-warp converter. Only
requested FP8 layouts are allocated. Applicable output-gradient preprocessing
also writes FP32 partial bias sums; bounded, ordered reductions finish the bias
gradient. Bias-only backward and current scaling keep their existing reductions.
This removes full input scans structurally; actual time savings need measurement.

Use `deepopt.fp8_checkpoint(function, *args)` for non-reentrant activation
checkpointing. It records the invocation scales and restores FP8/finite-check
scope choices during recomputation, avoiding duplicate history advancement.
It bypasses weight-cache reuse while recording/replaying so the scale tape stays
aligned. Unwrapped checkpoint recomputation is rejected with guidance rather
than silently using a new scale. Nested `fp8_checkpoint` calls are rejected;
checkpoint one enclosing region or use separate sequential checkpoint regions.
Full training CUDA graph capture is currently
unsupported; prepared GEMMs can still be measured independently.
Passive CUDA-graph configuration such as warmup/autograd options does not request
capture and is compatible with an eager delayed runtime. Rejection occurs when
capture is actually requested, or during an unsupported raw capture.

## Persistence and distributed execution

Delayed recipe identity is version 3 and includes history length, algorithm,
margin, synchronization policy and native Triton version. Current scaling keeps
version 2. Model state dictionaries persist numerical buffers and recipe metadata;
disposable FP8 caches, stream objects and workspaces are excluded. Loading or
moving an adapter invalidates cache generations. FP32 numerical buffers retain
their exact dtype when model parameters move to BF16/FP16.

DDPCore excludes numerical buffers from DDP initialization/forward and validation
buffer broadcasts. For synchronized scaling, initial numerical state follows
rank zero and one packed MAX reduction at each update boundary combines amax,
activity and nonfinite flags, including inactive projections. With
`reduce_amax=False`, histories remain rank local. DeepOpt distributed checkpoints
already gather/restore every rank's buffers; use that collective checkpoint API
to continue local histories exactly. Manual rank-zero-only model snapshots do
not capture other ranks' numerical state. Manual native DDP integrations must
provide equivalent buffer exclusions and registered, matching state order.

## Optional TE Linear comparison

Install the optional benchmark dependency with
`pip install 'deep-optimization[te-benchmark]'`, or install the pinned TE 2.9.0
build compatible with the environment's Torch/CUDA stack. TE is not required by
DeepOpt. The benchmark refuses a different installed version unless its pin is
explicitly changed with `--te-version`; target-stack compatibility is a GPU check.

```bash
python benchmarks/benchmark_fp8_te_linear.py --implementation all \
  --batches 512 336 272 --format e4m3fn --no-fill-uninitialized --output te-linear.json
python benchmarks/benchmark_fp8_te_linear.py --implementation all \
  --format hybrid --math-policy matched-fp32 --no-fill-uninitialized \
  --optimizer-updates --output te-linear-updates.json
```

The script uses installed-library imports and fresh processes/cache directories.
It compares native BF16, DeepOpt current, TE current, TE delayed and DeepOpt delayed.
QKV/FC1 use FP32 inputs; attention projection/FC2 use BF16. The ten-class head
uses the same BF16 policy in every implementation. It preserves FP32 parameter
and optimizer identity, supports requested-gradient/bias choices, records input,
upstream-gradient, initial-weight and source fingerprints, and changes allocation
filling/determinism only through explicit startup flags. Optional AdamW updates
are real updates. Missing CUDA/TE is reported as unavailable, not substituted.

Weighted projection ratios use 12 transformer blocks and one head. The 1.05 ratio
is a target, not an achieved result. TE AMP internal dX/dW and bias policies can
differ from DeepOpt even when external gradients are FP32; native mode reports
those differences. `--math-policy matched-fp32` disables AMP inside both FP8
linears, casts inputs to FP32 in a shared wrapper, uses FP32 internal forward/dX/dW
outputs and fast accumulation, and puts the same FP32 bias addition/BF16 output
cast outside both projections. TE's split-accumulator recipe overrides are set
and recorded for all three products. This controlled mode includes the extra
input casts in both timings and bypasses both native bias-fusion paths; it must
not be mixed with native-mode results. The shared ten-class head stays BF16.
Use separate runs with and without `--optimizer-updates` and explicitly disable
allocation filling when investigating kernel/runtime costs. These flags change
only the worker's startup experiment settings, never individual Linear calls.
Even matched mode needs GPU verification; accept the target only after
quantization/accumulation checks and matched
short/long full-model training. Evaluate a GEMM/backend change only after the
preprocessing comparison shows the remaining gap.

## End-to-end delayed training comparison

The training benchmark's default cases are BF16 runtime, prepared current FP8,
and prepared delayed FP8. Older loader/interception/EMA/fused-optimizer ablations
remain available explicitly with `--cases`; they are not repeated by default.
History length, algorithm and margin are forwarded into the actual runtime,
and the effective numerical recipe is recorded from that runtime.
`delayed_metadata_after_timing` reports cache hits, misses and retained table
bytes outside the timed passes so metadata transfer reuse can be audited.

```bash
python benchmarks/benchmark_fp8_training.py \
  --cases bf16-runtime fp8-prepared fp8-delayed --batches 512 336 272 \
  --history-length 1024 --amax-algorithm max --margin 0 \
  --no-fill-uninitialized-memory --output results/delayed-training
```

Every measured training step performs a real optimizer update. Keep the same
seed, model/data, EMA and diagnostics settings for all three cases. Compare native
and delayed coverage before interpreting ratios, and inspect loss trajectories;
finite sampled losses do not establish equivalent training quality.
