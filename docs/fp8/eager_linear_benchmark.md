# Eager Linear comparison: DeepOpt, PyTorch BF16 and TorchAO FP8

For a single portable benchmark covering **DeepOpt, torchao and Transformer
Engine**, see the [three-library eager benchmark](eager_three_library_benchmark.md).

[The standalone benchmark](../../benchmarks/fp8/benchmark_fp8_linear_compare.py) uses
the packages installed in the selected Python environment. It can be copied
outside the checkout. It does not modify import paths, install packages, download
data, compile the model, or fuse producers with Linear. Use one Python process,
not `torchrun`; the benchmark starts isolated workers itself.

Install DeepOpt and a TorchAO build compatible with the GPU environment's
PyTorch/CUDA stack first. A preflight records import/configuration errors rather
than substituting another implementation:

```bash
python benchmarks/fp8/benchmark_fp8_linear_compare.py --check --output fp8-linear-check
```

The comparison requires an NVIDIA SM90+ device, such as the H100 used for the
previous experiments. The local Mac can run contract/CPU integration tests but
cannot validate CUDA performance. `--torchao-version VERSION` optionally enforces
an exact installed TorchAO version. Installed package paths, versions and source
fingerprints are saved in each worker report.

## Recommended first run

```bash
python benchmarks/fp8/benchmark_fp8_linear_compare.py \
  --implementations bf16 deepopt-current deepopt-delayed torchao \
  --math-policies amp matched-fp32 \
  --batches 512 --format e4m3fn --accumulation fast \
  --optimizer-updates --no-fill-uninitialized-memory \
  --trace --output fp8-linear-comparison
```

Send `fp8-linear-comparison.zip` for analysis. It includes:

- `summary.json` with full worker results and comparisons, plus `summary.csv`.
- Per-worker logs and incrementally saved JSON under this run's unique ID.
- Per-projection Chrome profiler traces, including input shapes and diagnostic
  GEMM/quantization ranges.

The four transformer projections use the actual ViT-B dimensions and 197 tokens
per image. QKV and FC1 inputs are FP32, attention output and FC2 inputs are BF16.
Each is weighted twelve times in diagnostic sums. The ten-class head is measured
as BF16 in **every** implementation and weighted once. It is never silently
presented as FP8. All optimizer-owned parameters remain the original FP32 objects.

## What is measured

Uninstrumented, synchronized wall-clock and CUDA-stream timing covers zeroing
gradients, forward, backward, delayed-scale commit, and an actual AdamW update.
CUDA intervals include gaps while the host submits work. A separate event pass
breaks out these phases; hooks and profilers are absent from throughput timing.

The GEMM diagnostic pass records actual operand shapes, strides, scales, FP8
formats, accumulation flags where exposed, and output dtypes for forward, dX
and dW. It then replays those real prepared GEMM calls without quantization.
For each captured FP8 product it additionally runs DeepOpt Triton, native
`torch._scaled_mm(use_fast_accum=True)`, and native non-fast accumulation using
the **same FP8 buffers, scales and output dtype**. Both sides omit bias for this
controlled backend probe. Unsupported probes are recorded as errors; they do
not erase the completed throughput measurements.

Numerical checks compare forward, dX, dW and bias gradients against FP32 math
using the projection's current original parameters/input and explicit upstream
gradient, with TF32 disabled. This is an error measurement, not a convergence
test. Memory includes warmed peak allocation and unique observed saved-tensor
storage by physical dtype. Source fingerprints and sampled initial tensor
fingerprints help detect mixed environments or inputs. Tensor fingerprints use
up to 65,536 evenly spaced elements and are labeled as samples, not full hashes.

CUDA profiler totals exclude user annotation ranges to prevent double counting.
Every profiled update is synchronized separately so warmup work cannot spill
into the measured interval. Profiling is a diagnostic pass and does not provide
the primary throughput numbers. CPU cumulative times overlap and must not be
added to GPU time; Python cProfile does not cover all autograd worker threads.

## Reading the two math policies

`amp` measures the normal BF16 AMP behavior of each library. With numerical
recipe version 5, DeepOpt emits BF16 dW directly from the GEMM and autograd
accumulates it into the original FP32 parameters' gradients. dX retains the
original input dtype, which can still be FP32 where TorchAO uses BF16 internally.
Older DeepOpt builds emitted FP32 dW for FP32 parameters. Actual installed-build
dtypes are exposed in `gemm_calls`; the benchmark does not override that policy.

`matched-fp32` disables AMP inside each **FP8** projection, supplies FP32 inputs,
uses FP32 internal GEMM outputs, and applies the same external FP32 bias addition
and BF16 output cast. The BF16 reference remains BF16 AMP. Actual observed GEMM
output dtypes are reported; the flag alone does not establish a match. This mode
includes wrapper casts and bypasses native FP8 bias fusion. Current versus
delayed scaling still use different numerical recipes, and matching output
dtypes does not establish identical accumulation or rounding.

`--accumulation fast` explicitly requests fast accumulation for all three TorchAO
products to compare with DeepOpt's native Triton accumulation. To separately
measure TorchAO's normal per-role accumulation defaults, repeat with
`--accumulation torchao-default`. The actual TorchAO calls record the flags.
This option does not alter DeepOpt's implementation.

## Native backward and selective preparation

The default DeepOpt backward executor remains `inherit`, using the forward
backend. To evaluate the native scaled-GEMM backward executor on the same
installed build, repeat the AMP workload in separate output directories:

```bash
python benchmarks/fp8/benchmark_fp8_linear_compare.py \
  --math-policies amp --backward-backend inherit \
  --no-fill-uninitialized-memory --output fp8-linear-triton-backward
python benchmarks/fp8/benchmark_fp8_linear_compare.py \
  --math-policies amp --backward-backend native_fast \
  --no-fill-uninitialized-memory --output fp8-linear-native-backward
```

`native_precise` instead requests `torch._scaled_mm(use_fast_accum=False)`.
Both options keep forward on Triton, preserve the prepared FP8 operands and
inverse scales, and use the same dX/dW output policy and FP32 parameters as the
inherited backend. dX uses the original input dtype; dW follows the forward GEMM
output dtype. BF16 AMP therefore requests BF16 dW on both Triton and native.
They change accumulation and rounding; matching output dtypes does not establish
numerical equivalence or convergence. Version 2 reports record requested and
actual backends, fallback reasons, accumulation flags, numerical recipe identity,
and bounded layout-cache statistics. BF16 and TorchAO execution are unaffected.

Native backward requires NVIDIA SM90+, `backend="triton"`, and tensorwise
scaling. Products with unsupported K/N alignment, operand storage layouts, or
two E5M2 operands use Triton, as recorded in the recipe. Runtime errors from an
eligible native launch propagate instead of switching policy silently.

Projection selection is configured when preparing a full model:

```python
from deepopt import FP8Config

fp8 = FP8Config(
    enabled=True, backend="triton", scaling_recipe="delayed",
    backward_backend="native_fast", intercept_linear=False,
    exclude_modules=("blocks.*.attn.proj", "head"),
)
```

Exclusions use qualified-module-name glob patterns and apply to entire subtrees.
Every alias of an excluded module stays dense. The preparation report lists
the actual excluded paths; parameters and checkpoint names are preserved.
`<root>` selects a root projection. For native `nn.MultiheadAttention`, an
`*.out_proj` exclusion keeps only its output projection dense while QKV remains
FP8. Exclusions require `intercept_linear=False` so a global `F.linear`
interceptor cannot re-enable FP8 on an excluded projection. Dense paths follow
the surrounding AMP policy. No shape-based exclusion is enabled automatically.

Warm GEMM launches reuse bounded pointer-free layout plans and invoke the
measured configuration directly. Cold shapes still use native Triton autotuning.
Descriptors and outputs are fresh for every invocation; cached plans own no
activation, output, scale, or descriptor storage. Split-K reduction remains part
of both tuning and execution. Delayed conversion also bypasses autotuner
selection on an in-process cache hit.

The supplied measurements motivated these changes but used a different source
fingerprint. Reinstall the current checkout, confirm the recorded package path
and source hash, and collect a new AMP run before comparing results. The weighted
sum of isolated projections is not a measured full-model speedup.

## Follow-up experiments

If the first run completes, measure the two tail sizes without repeating 512:

```bash
python benchmarks/fp8/benchmark_fp8_linear_compare.py \
  --batches 336 272 --no-fill-uninitialized-memory --output fp8-linear-tails
```

For the library's default accumulation policy:

```bash
python benchmarks/fp8/benchmark_fp8_linear_compare.py \
  --accumulation torchao-default --math-policies amp \
  --output fp8-linear-default-accumulation
```

`--no-optimizer-updates` instead measures forward/backward on fixed weights.
It can enable DeepOpt weight-cache reuse and is a **different workload** from
training with real updates. Use a separate output directory and do not combine
its ratios with the update run. Weighted sums of isolated optimizer updates
are not predictions of a full-model optimizer that batches parameters.

Warm/cold disk-cache behavior is explicit: by default the installed environment's
Triton cache is reused. `--cold-cache` creates empty per-worker Triton cache
directories and may substantially increase first-use compilation time. It does
not change eager execution of the benchmark. `--no-prepared-gemms`,
`--profile-steps 0`, `--no-trace`, and `--no-host-profile` can reduce diagnostic
work, but leave their defaults enabled for the first analysis run.

Reports are saved after each major diagnostic phase and each projection. An
unavailable implementation or failed worker is explicit and produces a nonzero
final exit status. Completed results remain available. The ZIP includes only
this invocation's files even if the output directory contains older runs.
