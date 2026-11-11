# Eager FP8 Linear: DeepOpt, torchao and Transformer Engine

[benchmark_fp8_eager.py](../../benchmarks/fp8/benchmark_fp8_eager.py) compares all three
installed libraries and a PyTorch BF16 baseline in one run. You can copy the
script outside this checkout. It imports `deepopt` normally, without modifying
`sys.path`/`PYTHONPATH`, installing dependencies, or adding the checkout's `src`
directory to imports.

Use an NVIDIA SM90+ GPU and one Python environment containing CUDA-enabled
PyTorch, DeepOpt, compatible torchao, and Transformer Engine's PyTorch extension.
The TE path requires `Float8CurrentScaling`; the repository's existing TE extra
pins `transformer-engine[pytorch]==2.9.0`. Select compatible PyTorch/CUDA builds
for both external libraries. Run a single process; the script starts a fresh
worker with the same interpreter for each implementation.

```bash
python benchmarks/fp8/benchmark_fp8_eager.py \
  --implementations bf16 deepopt torchao te \
  --shapes 4096x768x2304 4096x768x768 4096x768x3072 4096x3072x768 \
  --modes forward forward-backward \
  --warmup 10 --iterations 50 --repeats 5 \
  --output fp8-eager.json
```

Shapes are **M x K x N**: input `[M, K]`, weight `[N, K]`, output `[M, N]`.
Every dimension must be a positive multiple of 16 so forward and both backward
matrix products meet the common FP8 alignment requirements. For an initial
smoke test, use `--shapes 256x768x768 --warmup 2 --iterations 2 --repeats 1`.

The default uses E4M3 for inputs, weights and upstream gradients, with
tensorwise current scaling in DeepOpt/TE and dynamic scaling in torchao. Use
`--format hybrid` for E4M3 inputs/weights and E5M2 upstream gradients across
all three libraries. DeepOpt uses Triton for forward and backward by default.
`--deepopt-backward-backend native_fast` or `native_precise` selects its existing
native backward GEMM executor while preserving Triton forward and the FP8
quantization/layout path. These options use fast or precise native accumulation,
respectively, so compare the reported numerical errors as well as timings.
The library's native executor routes layouts it cannot handle to Triton;
aligned tensorwise cases such as this benchmark's use the native path. Execution
errors are reported rather than retried with another backend.

To test whether native backward improves the reported H100 case:

```bash
python benchmarks/fp8/benchmark_fp8_eager.py \
  --implementations bf16 deepopt te \
  --shapes 4096x768x3072 --modes forward-backward \
  --deepopt-backward-backend native_precise \
  --output fp8-eager-native-backward.json
```

To measure where the eager time goes, add `--profile`:

```bash
python benchmarks/fp8/benchmark_fp8_eager.py \
  --implementations bf16 deepopt torchao te \
  --shapes 4096x768x3072 --modes forward forward-backward \
  --profile --profile-steps 10 \
  --output fp8-eager-profile.json
```

This writes the normal timing report and `fp8-eager-profile.zip`. The ZIP contains
`summary.json`, the standalone script, and this run's profiles. Each implementation
and shape/mode has:

- A Chrome/Perfetto trace with CPU operations and actual CUDA activities.
- Kernel/copy names, counts and summed CUDA activity durations per step in the
  JSON, plus the highest CPU operator costs.
- `linear.forward`, `linear.backward` and `linear.zero_grad` ranges.
- DeepOpt-only ranges for activation/weight/gradient quantization and the forward,
  input-gradient and weight-gradient GEMM calls. The JSON records the first
  step's operand shapes, strides, dtypes and requested GEMM backend.
- A separate Python `cProfile` pass, saved as both `python.txt` and
  `python.pstats`, to inspect configuration, descriptor, allocation and dispatch
  calls in the calling thread.

Use `--deepopt-backward-backend native_precise` with a different `--output` name
to profile the native-backward variant as well. `--profile-dir DIRECTORY` chooses
the parent of a unique run directory; previous runs are excluded from the ZIP.
Without `--output`, profiling uses `fp8-eager-profile.json` by default.

Throughput timing remains uninstrumented. CUDA profiling runs afterward with one
profiler warmup step, then `--profile-steps` active steps. Each diagnostic step
is synchronized so warmup or previous-step GPU work cannot enter the next step.
The Python call profile runs separately on the original operation, with explicit
synchronization outside the call profile. It covers only the calling thread;
autograd worker threads and blocking native calls limit CPU attribution.

Normal timing is in **milliseconds per step**; profile summaries use
**microseconds per step**. `cuda_activity_sum_us_per_step` sums actual recorded
kernels/copies, excluding GPU user annotation ranges and host submission gaps.
CPU range CUDA totals attribute child kernels to that range: forward/backward
contain the quantization and GEMM ranges, so their times must not be added
together. Attribution follows each actual CUDA event's correlation ID to the
runtime/driver API that launched it, then finds the enclosing CPU ranges on the
launching thread. A unique caller phase can also own launches on autograd worker
threads. This handles traces whose external IDs point only to an outer autograd
function. Its completeness is reported separately; unavailable scope times are
`null`, while the raw CUDA activity list remains available. Per-scope profiled
CPU medians and maxima expose large first-active-step profiler outliers.
GEMM scopes also include any internal packing or reduction work; use
the kernel list and trace to separate those operations. Other libraries retain
their native kernel names; the harness does not guess their categories from
names. Concurrent CUDA activities can overlap in time.

Profiler instrumentation changes execution overhead. Do not subtract a profiled
CUDA activity sum from the uninstrumented wall timing to calculate CPU overhead.
Inspect gaps and kernel durations within the trace instead. Missing CUDA/CUPTI
activity is reported as a diagnostic failure, while completed throughput results
and the independent Python profile are retained. A worker checkpoint also saves
completed timings before entering profiling.

Each case starts with identical random FP32 parameters and BF16 input values.
Forward timing runs without autograd. Forward-backward computes input, weight
and optional bias gradients using the same explicit BF16 upstream gradient;
clearing gradients with `set_to_none=True` is included. Add `--bias` to enable
bias. There is no loss reduction, optimizer update or dataset in the timing.

All calls run eagerly, without model compilation or CUDA Graph replay. Kernel
JIT compilation and autotuning can still occur during initialization/warmup.
Timed calls include quantization, layout conversion and context entry/exit.
Autocast and DeepOpt quantized-weight caching are disabled; TE is called without
its microbatch weight-cache hint. This prevents unchanged weights from skipping
quantization during repeated calls.

The table shows median synchronized wall time, median CUDA stream interval, and
BF16 wall time divided by each implementation's wall time. Values above 1 mean
the implementation is faster than the BF16 baseline. Both measurements include
host submission gaps typical of eager execution; the CUDA interval is not a sum
of kernel execution times. Synchronization occurs once per repeated group,
not once per individual Linear.

The JSON contains all timing samples, actual model types, FP8 configurations,
package versions/import paths, GPU details, and separate forward/dX/dW/db errors
against FP32 computations with TF32 disabled. The libraries retain their own
accumulation and rounding policies, so this measures their eager APIs rather
than establishing numerical equivalence or training convergence.

Missing/incompatible packages and failed cases are reported explicitly. Other
workers still run, the JSON is saved, and the parent exits nonzero if any
requested implementation fails. Non-finite results invalidate a case.

For optimizer-inclusive timings and deeper per-GEMM diagnostics, use the
[existing Linear diagnostic benchmark](eager_linear_benchmark.md).
