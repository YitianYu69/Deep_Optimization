# Benchmarking Triton TileLang and Mixed execution

Install the DeepOpt build to measure in the active environment, with the
optional compilers required by the selected routes. TileLang uses the
`tilelang` extra; the current FP8 GPU workloads target NVIDIA compute capability
9.0 or newer. Running the scripts does not install dependencies or change the
source tree used by Python. See [backend selection](../docs/architecture/kernel_backends.md).
Update the installed DeepOpt build as well as the benchmark scripts to pick up
the parallel TileLang reductions and shared timing-budget helper. An editable
installation already follows the updated source.

## Keep the runner and workers together

The coordinator launches separate family scripts. Update the benchmark folder
as a matching set instead of copying only the coordinator. For a Linear,
LayerNorm and pooling run, the required workers are
[Linear](fp8/benchmark_fp8_eager.py),
[normalization](normalization/benchmark_normalization.py) and
[pooling](pooling/benchmark_pooling.py), together with the GPU lease helpers.
The installed DeepOpt build must also contain the shared budget helpers and
backend implementation being measured. Copying benchmark scripts does not
update an installed library; an editable install follows only the source tree
it points to.

Logs such as `unrecognized arguments: --time-budget ... --no-autotune` mean the
runner and workers disagree about their CLI. Those workers did not measure
kernel performance. The coordinator now checks all selected worker flags and
required DeepOpt helper APIs before GPU discovery/admission, including in
`--dry-run`. It reports stale scripts or missing APIs without dropping flags.

Create a small coherent export with the current scripts and lease helpers:

```bash
python benchmarks/export_comparison_benchmarks.py --kernel-backends-only \
  --output work/kernel-backend-bundle
```

Optionally pass `--wheel` with a freshly built portable DeepOpt wheel. The
exporter checks every packaged DeepOpt Python source against the checkout,
includes the wheel and installation commands, and writes a source-hash
manifest. Extract the complete ZIP and follow `RUN_KERNEL_BACKENDS.txt`; do
not overlay only its coordinator. This installation preserves the environment's
Torch, Triton and TileLang packages. Use a new results directory after updating
scripts/builds because the previous run identity is different.

## Run a matched suite

The [comparison runner](benchmark_kernel_backends.py) launches separate
processes for each workload/backend/trial. The smoke preset uses small fixed
shapes; the representative preset uses larger projection, image and transformer
workloads. Start with smoke to find compilation errors and very slow paths.

```bash
python benchmarks/benchmark_kernel_backends.py --device cuda:0 --preset smoke \
  --backends triton tilelang mixed --trials 3 \
  --output work/kernel-backend-smoke

python benchmarks/benchmark_kernel_backends.py --device cuda:0 --preset representative \
  --workloads linear layer_norm pooling --backends triton tilelang \
  --warmup 10 --iterations 50 --repeats 5 --trials 1 --time-budget 1 --timeout 900 \
  --output work/kernel-backend-representative
```

The ordinary comparison disables autotuning and uses a two-second warm timing
target per variant. Requested warmup, iteration and repeat counts become caps;
slow variants use fewer calls and record every actual count. Shapes, numerical
recipes and requested independent trials stay unchanged. `--time-budget 1`
reduces the target further for an initial survey. Use `--exhaustive` to execute
all requested samples, and opt into tuning with `--autotune` when measuring
tuned schedules. Tuning and cold compilation are separate costs.

The original representative command with three trials, two backends and the
three workloads above has 18 workers and 144 timing measurements. At the default
two seconds the nominal measurement target is 288 seconds, excluding cold
compilation, setup, validation, profiling and GPU admission. This is not a job
deadline: one CUDA/JIT call cannot be interrupted by the soft measurement
budget, and each measurement keeps at least one real timed sample. The outer
worker timeout still bounds workers and cleans up their processes.

Use a new output directory for a new experiment. Reports, stdout and stderr survive
worker errors/timeouts, and the summary is updated after each worker. The
runner exits unsuccessfully when a requested workload fails or is unsupported.
Individual successful workers have durable checkpoints. Repeat exactly the
same command with `--resume` to reuse verified successful workers and retry
failed/incomplete ones. Resume checks scripts, library/build/dependency/device
identity, policy, caps and report bytes. Changed experiments or older runs
without the new identity records need a new directory; they are never silently
mixed. Progress reports worker status, elapsed time and a rough ETA. During
long calls a 15-second heartbeat includes the latest short worker log line;
`--heartbeat-interval 0` disables it without invalidating resume. Linear
also reports case-level cold/setup stages in its worker logs.
GPU workers use the existing physical-device lease manager and inherit one
lease, so family scripts do not create another coordinator. Physical devices
are exposed as logical `cuda:0`; each row records the binding and executed
command. GPU stdout/stderr share a labeled combined log; CPU logs are separate.
An existing matching lease is reused without releasing its owner's token.
On POSIX hosts a timeout stops the owned worker group before releasing the
lease, including nested Linear workers. The runner requires POSIX for this
timeout guarantee.
`--dry-run` writes the resolved matrix without launching workers. For an export,
`--benchmark-root` can point to a directory containing the copied family scripts.

Backend order is seeded per workload and rotated across trials. Each trial
starts a new process and retains the underlying benchmark's repeated samples.
For compiler workloads, `--iterations` maps to measured training steps and the
compiler reports per-step samples; its script does not use `--repeats`.
Compiler eager/compiled runs share one count plan to preserve equal optimizer
updates. Convolution variants also share counts when `--optimizer-step` is on.
The shared plan may exceed the target on a slower variant; it preserves the
meaning of the state comparison.
The default Mixed policy prefers TileLang and routes GEMM to Triton. Set an
explicit policy when comparing a different combination:

```bash
python benchmarks/benchmark_kernel_backends.py --workloads linear \
  --backends triton mixed --device cuda:0 \
  --mixed-policy '{"preferred":"tilelang","roles":{"linear.dw":"triton"}}' \
  --output work/kernel-backend-mixed-dw
```

The suite is a frozen set of representative workloads. It does not exercise
every operator option or certify correctness; use the mandatory GPU tests and
backend-specific test suite for that purpose.

## Opt-in TileLang qualification and focused runs

Existing commands retain their default validation, calibration and reporting
behavior. Add `--qualify-tilelang` to require independent numerical evidence
for the currently supported isolated current-scaling Linear, LayerNorm and
deterministic max/average pooling fixtures. Unsupported qualification recipes
fail explicitly. The mode may include Triton as an unchanged control; it does
not modify Triton execution or choose routes automatically.

Start with one affected family and one input/output/compression case:

```bash
python benchmarks/benchmark_kernel_backends.py --qualify-tilelang \
  --device cuda:0 --preset smoke --workloads layer_norm \
  --backends triton tilelang --inputs bf16 --outputs bf16 --compression off \
  --max-worker-runs 2 --time-budget 1 --trials 1 --timeout 180 \
  --output work/tilelang-qualified-norm-smoke
```

After it succeeds, repeat with `--preset representative` and a fresh output
directory. Use a separate pooling run by selecting `--workloads pooling`;
`--shape` overrides its input geometry. Linear has `--linear-shapes` and
`--linear-modes` selectors. Those coordinator controls require qualification
mode and are forwarded only to applicable workers. `--compression on` keeps
an untimed numerical baseline; it does not omit the correctness comparison.

`--max-worker-runs` caps the declared matrix before GPU admission. Opt-in
reports also persist launched attempts, including failed/crashed attempts, so
`--resume` cannot silently exceed the same batch cap. After exhaustion use a
new batch or explicitly increase the declared cap; verified checkpoint reuse
does not consume another attempt. A work-count cap is not a runtime deadline.

Custom rows carry versioned `validation` with the oracle, requested roles,
per-role metrics, limits and status. The coordinator rejects missing cases,
missing requested gradients, nonfinite/over-limit checks and required null
diagnostics. Optional unrequested fields may remain null. Failed qualification
does not contribute timing ratios. Default report/resume semantics remain
unchanged, including rejection of mismatching source/build identities.

Linear checks deterministic output samples (up to 32 × 32, with complete K
reductions) against CPU FP64 arithmetic on the actual quantized operands;
finiteness is checked on complete outputs/gradients. LayerNorm checks native
output/statistics and backward against independently decoded actual saved
state. Pooling checks native CPU values/gradients and exact compressed-route
behavior. Per-role normalized L2 limits use Torch dtype-default atol/rtol
budgets, recorded in each check; FP8 pipeline and compression approximation
errors remain separate diagnostics. These scoped checks do not establish full
operator parity, exact cross-backend bytes for every helper, or convergence.

Opt-in timing separates a cold call and uses the median of three warm pilots,
including exhaustive runs. Memory reports add live allocation baselines and
incremental peaks. Calibration and numerical work are outside throughput
samples. CPU runs exercise reference arithmetic and harness integration only.

The inexpensive compression-quality screen compares existing settings without
allocating a GPU or changing either backend's defaults:

```bash
python benchmarks/normalization/benchmark_tilelang_compression_quality.py \
  --bits 2 4 8 --group-sizes 128 256 512 \
  --distributions normal large_offset outliers \
  --output work/tilelang-compression-quality.json
```

It reports xhat and per-gradient approximation errors, moment correction and
saved bytes; it does not select a new default or qualify CUDA execution.

The TileLang-only optimized runtime uses cooperative LayerNorm row reductions,
bounded deterministic affine-gradient partials, and analytic regular pooling
windows with bounded backward traversal. Generic TileLang paths retain other
geometries and explicit Mixed auxiliary routes. H100 speed and memory savings
must be measured after installing the matching build.

When packaging an isolated staged build, the exporter also accepts
`--library-root /absolute/path/to/deepopt` with `--kernel-backends-only --wheel`.
It verifies every packaged Python source against that explicitly selected
source directory; it never changes which DeepOpt installation workers import.

## Run individual families

These programs share `--deepopt-backend triton|tilelang|mixed` and optional
`--deepopt-kernel-policy JSON`:

| Program | Measurement |
| --- | --- |
| [Eager FP8 Linear](fp8/benchmark_fp8_eager.py) | Matched tensorwise current-scaling eager Linear forward/backward, with separate diagnostic profiling. |
| [Activations](activation/benchmark_activations.py) | Activation forward/backward, saved-state compression, FP8 output and optional Linear/Conv pipelines. |
| [Normalization](normalization/benchmark_normalization.py) | BatchNorm, GroupNorm and LayerNorm, compression, output scaling and pipelines. |
| [Pooling](pooling/benchmark_pooling.py) | All existing pooling kinds selectable by `--op`, routing compression and optional pipelines. |
| [Convolution](convolution/benchmark_convolution.py) | Conv/ConvTranspose ranks 1–3, BF16/FP8 paths, compressed saved state and optional optimizer/graph/profile passes. |
| [Packing](activation_compression/benchmark_activation_pack.py) | Affine payload packing and backend-specific schedule diagnostics. |
| [Unpacking](activation_compression/benchmark_activation_unpack.py) | Reconstruction strategies consuming the same packed payload. |
| [Compiler](compiler/benchmark_compiler.py) | Eager versus compiled training, cold first step, steady-state phases and optional profiling. |

```bash
python benchmarks/activation/benchmark_activations.py --deepopt-backend tilelang \
  --shape 4096x768 --activations silu --output work/activation-tilelang.json

python benchmarks/normalization/benchmark_normalization.py --deepopt-backend mixed \
  --deepopt-kernel-policy '{"preferred":"triton","families":{"normalization":"tilelang"}}' \
  --op layer_norm --shape 4096x768 --pipeline none --output work/norm-mixed.json

python benchmarks/compiler/benchmark_compiler.py --device cuda:0 --backend deepopt \
  --deepopt-backend tilelang --fp8 --compression --output work/compiler-tilelang.json
```

Triton-specific candidate normalization variants are rejected for TileLang
routes. TileLang codecs currently use a conservative serial schedule; accepted
Triton tuning knobs must not be interpreted as an implemented TileLang tuning
search. Codec resume identities include policy and selected package versions,
so results from a different backend cannot be reused as a matching case.

## Interpret results

Family reports include `kernel_backend` policy metadata: canonical routes,
fingerprint, role overrides, execution kind and selected compiler package
versions. `route_evidence="policy_resolution"` describes configured routing;
GPU traces and operation diagnostics establish actual execution, especially
when a Mixed policy permits metadata fallback. The runner does not treat those
configured routes as a complete launch count.

Timing reports include `sampling`: requested caps, actual warmup and per-repeat
iteration counts, completed repeats, calibration/cold costs and budget status.
Every latency is divided by the actual calls. Comparisons retain both sampling
summaries and flag unequal effective counts. Budget-limited single samples are
useful for finding large slowdowns but provide weaker evidence for small speed
differences; increase the budget or use exhaustive mode for final measurements.
Codec resume accepts completed budget-limited rows only with valid count
metadata, while exhaustive mode still requires every requested sample.

The runner writes `summary.json` with the command matrix, per-worker status,
report/log paths, extracted metrics and matched ratios against Triton. A ratio
above one means higher latency or memory. It keeps unmatched rows visible and
excludes failed reports, failed exactness/parity checks and nonfinite results
from comparisons. Finite numerical diagnostics alone do not establish that an
operation meets its accuracy contract. Native baseline rows are excluded from
the custom-backend ratios. Warm wall geometric means are unweighted across
matching rows; examine individual cases before drawing conclusions.

CUDA event intervals, synchronized wall time, saved-state memory and allocator
peaks retain each family's measurement definitions. Complete pipelines include
preparation, layout work and backward according to the individual script.
Compiler reports separate the cold first step; the other scripts exclude warmup
from steady-state timing. Linear also reports synchronized cold construction
and first launch separately and checkpoints each case. Runner process wall time includes imports, compilation,
diagnostics and report writing and is not a measurement of compilation alone.
Compiler `step_phases_ms` summarizes the sum of forward/loss, backward and
optimizer phases for each sample, using CUDA stream events on GPU and a CPU
performance clock on CPU. Its ratios are reported separately from warm wall
latency ratios; phase durations do not include any work outside those markers.
Profiling and sanitizer runs belong in separate diagnostic passes.

The TileLang FP8 amax path now tiles large scale groups into at most 1,024
values per CTA, followed by bounded hierarchical merges, instead of scanning
an entire tensor in one lane. The small-group path and Triton kernels retain
their existing behavior. Frontend and CPU semantic checks cover the new
reduction; H100 execution, memory checking and speedup still need real GPU
measurement. Other conservative TileLang schedules can remain slow.

For a portability smoke run on a machine without CUDA:

```bash
python benchmarks/benchmark_kernel_backends.py --device cpu \
  --workloads activation layer_norm pooling compiler \
  --warmup 0 --iterations 1 --repeats 1 --output work/kernel-backend-cpu
```

All three selected modes execute CPU reference arithmetic in that command.
Its timings validate the harness and policy plumbing, not TileLang/Triton GPU
speed. Explicitly requested GPU-only workloads are recorded as unsupported.
No GPU performance or parity claim follows from CPU or frontend-only results.

The benchmark contract/regression suite uses the benchmark helpers on its
search path while DeepOpt comes from the selected installed environment:

```bash
PYTHONPATH="$PWD/benchmarks" python -m unittest discover -s tests/benchmarks -t . -v
```
