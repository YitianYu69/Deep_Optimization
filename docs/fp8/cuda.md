# FP8 CUDA execution and autotuning

[FP8 guide](README.md) · [Benchmark catalog](../../benchmarks/README.md) · [Validation](validation.md)

Use this guide when selecting CUDA kernels, measuring projections, or comparing
installed DeepOpt builds. For model conversion and configuration, start with the
[FP8 guide](README.md).

## Actual CUDA execution

For opt-in native backward GEMM, warmed eager launch planning, and projection
exclusions, see the [eager Linear optimization controls](eager_linear_benchmark.md#native-backward-and-selective-preparation).
These controls retain the existing forward backend and scaling default. Numerical
recipe version 5 records the dW output policy together with any native backward
executor and projection exclusions.

Use a CUDA-enabled PyTorch installation and compatible Triton exposing
`triton.tools.tensor_descriptor.TensorDescriptor`. The TMA kernel requires an
NVIDIA GPU with compute capability at least 9.0. There is no CPU fallback for a
requested Triton backend.

```bash
python examples/optimization/fp8_training.py --device cuda --backend triton
python examples/optimization/fp8_training.py --device cuda --backend triton --block-size 32 --precision bf16
```

The Triton backend supports independent `e4m3fn`/`e5m2` formats. Fused CUDA
quantization requires power-of-two block dimensions other than `[1, 1]`.
For example, activation `[1, 32]` with weight and gradient
`[32, 32]` is supported. Pipeline stages are tuned automatically by default;
an explicit integer from 1 through 5 fixes the depth.

Tensorwise scaling and blockwise scales whose reduction-axis widths are multiples
of 32 use native FP8 dot products. Other layouts use fused per-element FP32
dequantization followed by a `tf32x3` dot product. Rowwise activation and gradient
roles use the independently quantized orientations described above to retain
native FP8 reductions. Other transposes may still require the fallback when
their reduction-axis scale widths are unaligned.
The fallback preserves the stored FP8 values and scales without additional
FP8 quantization, but may have lower throughput than native FP8 dot products.
The backend packs strided/misaligned operands, pads partial tiles, and returns
FP32/BF16/FP16 results. Correctness/performance must be measured on the target GPU.

The kernel automatically uses Triton's native `@triton.autotune` to select
matrix tile sizes, warp counts, pipeline stages, grouped launch ordering, and
eligible split-K factors. The search measures at most 48 configurations,
retaining the historical tile/warp/stage choice as a measured candidate.
Compatible CUDA configurations from
[Triton's GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
are reserved in the measured set, including 128x256/256x128 FP8 tiles with
K=128 and the two-warp, five-stage small tiles. Shape-aware pruning still
excludes inappropriate sizes and incompatible scale boundaries. Up to 12
additional heuristic tile/reduction families supply narrow and split-K choices.
Candidates include small tiles along long axes to increase parallelism for
skinny products, two-warp small tiles, and larger tiles for dense products.
Shortlisting ranks estimated padded work and memory traffic, reserves narrow-axis
alternatives, and distributes the budget across reduction widths and split
factors before stage/warp variants. Candidate-list order does not affect selection.
The shortlist reserves stage-depth coverage before filling the remaining budget.
An explicit `num_stages` from 1–5 constrains the search; `"auto"` searches pipeline
depths with matching loop and compiler stages. Candidates with incompatible
reduction-scale boundaries or resource requirements are pruned. Quantization
`block_shapes` stay independent of kernel tile sizes: a 32-wide reduction-scale
block still forbids 64/128-wide native FP8 reduction tiles.

Grouped launches traverse up to eight M tiles before advancing along N to
promote L2 reuse. Row-major order (`GROUP_SIZE_M=1`) competes with grouped order
(`GROUP_SIZE_M=8`); up to eight otherwise-identical configuration pairs isolate
the scheduling choice without consuming the entire search budget. When either
output axis has a single tile, the equivalent grouped variant is omitted.
The final partial M group and split-K planes use the same coordinate mapping.
These CUDA/TMA choices do not introduce a HIP backend.

Split-K factors 2, 4, and 8 compete with unsplit launches for skinny products
(an output axis at most 32, K at least 512, and at most 262144 output elements)
and dense products with K at least 4096. This admits ViT weight gradients such
as a 768x768 output reducing across 100864 tokens. The number of split planes
is capped by a 64 MiB FP32 scratch bound, including aligned row strides.
Pruning accounts for SM count, output-grid occupancy, and at least two reduction
tiles per split. A separate reduction adds bias once and converts to the
requested output dtype. Both kernels are included in candidate timing. Each
launch overwrites its partials, so cached launches need no clearing or atomics.
Cached unsplit winners allocate no partial buffer; cached split winners allocate
only their selected number of planes.
Different reduction orders can produce small floating-point differences.

Candidate timing replays the prepared GEMM in a CUDA graph with a
20 ms repetition parameter and compares median timings. This avoids Python launch gaps
and repeated L2-cache flushing dominating the ranking of short kernels.
Packing and quantization run before this isolated timing; the training loop
does not need CUDA Graph mode. The full FP8 path still has its documented
capture limitation. The search follows the shape-aware approach used by
[Inductor's GEMM heuristics](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/heuristics/template/triton.py)
and the descriptor-based tuning in
[Triton's TMA matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html).

Packing uses only the stride alignment required by TMA. Logical descriptor
bounds handle partial tiles through zero-filled loads and discarded stores
outside the matrix. Buffers therefore do not require padding to the largest
tile in the candidate set, including on calls that reuse a cached winner.
Already aligned row-strided FP8 views are reused directly. Other layouts use
one copy/padding kernel that overwrites every destination byte; fully overwritten
buffers do not incur a preliminary zero fill.

Autotuning compiles and benchmarks candidates on the first call for each new
shape, operand/output format, scale layout, prepared row strides, bias policy,
pipeline policy, or CUDA device. Forward, input-gradient, and weight-gradient products have
different shapes; incomplete training and validation batches introduce more
shapes. These searches add first-use latency, especially in short runs with
small projections. Matching Triton calls invoke the in-process winner directly,
reusing a bounded pointer-free layout plan. Each invocation constructs fresh
descriptors bound to its current tensors; activations and outputs are never
retained in that cache. Native backward products use PyTorch's scaled-GEMM
executor instead of this Triton search. Set
`TRITON_PRINT_AUTOTUNING=1` to inspect selected configurations. A bounded search
does not guarantee the fastest possible kernel; measure on the target GPU.

On Triton runtimes supporting `cache_results`, tuning timings and winners also
persist in the native Triton cache. A launch adapter sets descriptor block shapes
for benchmarks, cached calls and warmups, so normal configurations contain no
`Config.pre_hook` that would disable Triton 3.9 result serialization. See the
[native cache implementation](https://github.com/triton-lang/triton/blob/v3.9.0/python/triton/runtime/autotuner.py).
The cache is invalidated by the kernel, target, environment, tuning key and
pruned configurations. A new key still pays compilation and search costs;
compiled-kernel caching alone is not tuning-result persistence.
Static capability/SM metadata is cached per explicit CUDA device. Descriptors,
outputs and split-K buffers remain owned by each invocation; no pointer-bound
descriptor or global scratch storage is shared across concurrent launches.

Benchmark the DeepOpt library installed in your active Python environment on
the target GPU with the [projection benchmark](../../benchmarks/fp8/benchmark_fp8_autotune.py).
For an editable installation, run `python -m pip install -e .` from the checkout
first. The script uses normal library imports and can be copied outside the
repository; it does not select a source tree or modify Python's import path.
To compare library versions, run it in the corresponding installed environments.

```bash
python benchmarks/fp8/benchmark_fp8_autotune.py
# Use an explicit pipeline depth with the installed library.
python benchmarks/fp8/benchmark_fp8_autotune.py --num-stages 3
# Dense projection with scale widths that allow K=128 native FP8 tiles.
python benchmarks/fp8/benchmark_fp8_autotune.py --batches 1024 --input-features 1024 --output-features 1024 --scale-k 128
# Isolate quantization, including independently scaled transpose views.
python benchmarks/fp8/benchmark_fp8_autotune.py --scope quantize
python benchmarks/fp8/benchmark_fp8_autotune.py --scope quantize --no-check-finite
python benchmarks/fp8/benchmark_fp8_autotune.py --scope scale --no-check-finite
python benchmarks/fp8/benchmark_fp8_autotune.py --scope dequantize
# Larger scale groups use chunked fusion rather than the unfused fallback.
python benchmarks/fp8/benchmark_fp8_autotune.py --scope quantize --batches 1024 --input-features 1024 --output-features 1024 --scale-k 256 --no-check-finite
# Actual ViT-B projection dimensions: 512 samples x 197 tokens, tensorwise.
python benchmarks/fp8/benchmark_fp8_autotune.py --preset vit --scaling tensorwise --grad-format e4m3fn --implementation both --num-stages 3 --no-check-finite --iterations 10 --output results/fp8-vit.json
# Compare an identical supported tile in two separately installed builds.
# Start with one projection shape; scales and scratch compatibility are checked.
python benchmarks/fp8/benchmark_fp8_autotune.py --scope gemm --scaling tensorwise --batches 100864 --input-features 768 --output-features 768 --gemm-config 128 128 64 8 3 1 1 --no-check-finite --output results/fp8-fixed.json
```

Use `--output PATH` to save the complete combined JSON report, including timings,
environment details, numerical errors, and autotuning diagnostics, for sharing
or later analysis. Missing parent directories are created and an existing file
is replaced after the benchmark completes successfully. The same JSON is still
printed to stdout; progress and the saved-file location go to stderr.

For a detailed H100 comparison using the same tensorwise recipe:

```bash
python benchmarks/fp8/benchmark_fp8_autotune.py \
  --preset vit --implementation both --scaling tensorwise \
  --grad-format e4m3fn --num-stages 3 --no-check-finite \
  --reuse-cache --iterations 20 --repeats 5 \
  --detailed --profile-iterations 2 --profile-rows 30 \
  --output results/fp8-detailed.json
```

`--detailed` retains the synchronized wall timings and adds measurements made
in separate passes after warmup:

- `cuda_events`: elapsed time on the current CUDA stream, including gaps while
  Python submits work. This is not a sum of kernel durations.
- `memory`: allocated/reserved baseline and peak bytes for one invocation,
  plus the extra peak above baseline. Linear gradients are cleared before the
  baseline; the result and new gradients remain alive through sampling.
  Inputs, prepared operands and existing caches are resident at baseline.
  These numbers do not measure whole-model training memory or optimizer/EMA.
- `profile`: the top CPU operators by self time, CUDA runtime calls, and raw
  CUDA activities grouped by name. Times are normalized per invocation; counts
  cover all `--profile-iterations`. `--profile-rows` limits the ranked lists,
  while CUDA activity totals include every recorded activity. CPU totals are
  nested, and CUDA activities can overlap; neither should be added to wall time.
  Missing CUDA profiler activities are explicitly marked `unavailable`.
- `cuda_graph`: event timings of graph replay for prepared GEMMs, removing
  per-operation Python dispatch and reusing captured output/scratch storage.
  Other scopes are skipped; unsupported capture reports its error. Memory and
  profiling run before this operation's graph capture. Replay is a diagnostic
  comparison, not a claim that full training can use the same graph.

The prepared BF16 GEMM path uses detached operands, matching FP8's lack of
autograd bookkeeping in this scope. Full linear measurements retain autograd.
The default input is BF16; repeat with `--input-dtype fp32` and a different
output filename to measure FP32 projection inputs, as can occur after
normalization. Input gradients follow this dtype in both implementations.
The benchmark continues to import the installed library; detailed mode does not
change its kernels or runtime policy. `--host-profile` additionally records
warmed Python submission with cProfile, separately from Kineto. Instrumented
wall time can contain synchronous waits; nested cumulative times are not additive.
CUPTI/profiler or graph failures are
recorded in the JSON so other available measurements remain usable.

By default it uses generated tensors for a `512 -> 10` CIFAR-10 classifier,
with batch sizes 2048, 848 (training tail), and 1808 (validation tail).
`--input-features`, `--output-features`, and `--batches` select other projection
shapes; `--scale-k` selects a power-of-two scale width from 32 through 1024 (default 32).
The report records group sizes, stages, and split factors for winners and trials.
`--scaling tensorwise` makes all operand block shapes `None`; the default
`blockwise` recipe is unchanged. Independent `--activation-format`, `--weight-format`,
and `--grad-format` select the actual operand formats (historical benchmark
defaults are E4M3FN/E4M3FN/E5M2). `--implementation both` runs FP8 and BF16 in
separate fresh processes. The ViT preset measures qkv, attention projection,
fc1, fc2 and head projections, including their forward/dX/dW products; body
rows equal `--batches` times `--tokens-per-sample` (default 197). It is not a
whole-model or epoch benchmark. Defaults are 512, 336 and 272 samples for ViT,
covering the training batch and training/validation tails; classifier
defaults remain 2048/848/1808.

Full-linear and BF16 accuracy reporting compares outputs and gradients with
IEEE FP32 after timing. Prepared FP8 GEMMs separately report quantization error,
accumulation plus output-cast error, and total error against tiled FP64
references. The quantized reference applies scale inverses in FP64, avoiding an
intermediate FP32 dequantization round. It bounds reference temporaries by
processing output rows and the reduction dimension in chunks. All error reports
include relative L2, maximum absolute, RMS and finite-value results.
The accumulation/output-cast category also includes FP32 scale application
rounding. It is separate from the error of choosing FP8 values and scales.
`--input-pattern cancellation`, `zeros`, and `mixed-magnitude` exercise numerical
stress cases; use both FP8 formats and irregular feature dimensions as well.
Use `--no-accuracy` only when those extra comparisons are unwanted. Candidate
records also include wall time for compilation plus benchmarking, and the worker
reports whether native tuning-result caching is enabled. These measurements
separate cold searches from warmed execution; they do not attribute data loading
or other epoch costs to autotuning.
`--scope quantize`, `--scope scale`, and `--scope dequantize` report each operation's
allocation/latency, operand shapes, and strides independently of GEMM.
Use `--scope quantize-pair --scaling tensorwise` for the actual converter that
emits both aligned layouts; ordinary `quantize` measures one layout.
`--scope bias-gradient` isolates the existing FP32 bias sum to establish whether
collecting partial bias sums during conversion would repay its added traffic.
Dequantization prepares FP8 inputs outside the timed region. Finite checks are enabled by default;
`--no-check-finite` applies the unchecked policy to all measured scopes and
requires a library version that supports the explicit check override. Use it
to compare unchecked updates; keep checks enabled when comparing strict runs.
Each implementation and measurement
scope runs in a fresh process with the same Python interpreter and an isolated Triton compilation cache by
default; `--reuse-cache` retains the normal disk cache. Reports separate first-use
latency from synchronized warmed timings, identify the imported DeepOpt package
path and installed distribution version, and record
the installed Python-source fingerprint, Triton version and numerical recipe.
They include cached winning tiles and
per-key candidate median/p20/p80 device timings when the library exposes a tuner.
Unavailable candidates are marked explicitly. Timing instrumentation runs only
during cold searches. Use `--num-stages auto` for the new default, or an integer
to compare installed versions that predate automatic stages. The full projection measurement
includes quantization, forward, backward, and gradient-buffer reset; it excludes
the convolutional backbone, optimizer, and data loading. Prequantized GEMM
measurements capture the imported library's actual autograd operands, including independently
quantized rowwise transposes, before timing. They include packing, padding,
dispatch, and split scratch allocation, so they are not kernel-only timings.
Candidate timings include split reduction but exclude allocation and packing;
compare warmed wrapper and full-projection timings as well. Neither CPU tests
nor this benchmark's existence establishes a GPU
speedup.

### Controlled native-accumulation comparisons

Use two environments with separately installed baseline and candidate builds,
with identical Torch/Triton versions. Copy the same benchmark script outside the
checkout and run it with each environment's Python. The script uses normal
library imports and records an installed-source SHA256 fingerprint, recipe,
formats, input dtype and selected configurations. A legacy library is identified
from its API; the script never injects the removed accumulation setting.

First compare one fixed configuration to isolate the accumulation change:

```bash
python benchmark_fp8_autotune.py --scope gemm --preset vit \
  --scaling tensorwise --grad-format e4m3fn --input-dtype bf16 \
  --gemm-config 128 128 64 8 3 1 1 --num-stages 3 \
  --no-check-finite --output fixed.json
```

Then omit `--gemm-config` and use `--num-stages auto` to tune each build afresh.
Use separate output files and initially omit `--reuse-cache` to keep compilation
and tuning caches isolated. The changed JIT source also gives native accumulation
its own kernel/cache identity. Repeat GEMM diagnostics with `--input-pattern
cancellation`, `mixed-magnitude` and `zeros`, both FP8 formats, and irregular
feature dimensions. Fixed configurations must satisfy each product's scale,
stage, occupancy and split-scratch constraints; invalid choices fail before
launch. Fixed mode adds a small Python validation wrapper and bypasses winner
caching; compare fixed runs with fixed runs rather than mixing timing modes.

Measure full projections with `--scope linear --implementation both --detailed
--host-profile`. Repeat with `--input-dtype fp32`; also measure `quantize-pair`,
`scale` and `bias-gradient` separately. Keep input formats, current-tensor scaling
and finite-check settings identical across runs. Projection sums and graph replay
are diagnostics, not full-model latency or training capture.

Native accumulation changes numerical behavior. Before accepting a performance
result, run matched short full-model training comparisons from the same initial
model/optimizer state, batches and augmentation seeds, with the same EMA,
diagnostics and checkpoint cadence. Compare loss and validation accuracy, startup
and steady-state timings, and allocated versus reserved memory separately; then
confirm with a longer run. Warmed weight-cache reuse does not represent a weight
update on every step. Retain the deterministic ordered split-K reduction.

Further consumer/producer fusion and head exclusions remain measurement-gated
experiments. The optional delayed recipe below implements checkpointed history
and fused bias preprocessing; its performance and training quality still need
target-hardware validation. The native accumulation, static device
metadata and selective-layout changes do not establish a GPU speedup or a
training-quality result without these target-hardware measurements.
