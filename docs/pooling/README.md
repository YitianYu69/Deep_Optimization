# Pooling with FP8 transport and activation compression

[Documentation](../README.md) · [Configuration](../configuration/README.md) · [Backward precision](../fp8/backward_precision.md) · [Design](plan.md)

Pooling supports native BF16/FP32/FP16 and scale-aware FP8 E4M3FN/E5M2 inputs,
native or FP8 outputs, and native/FP8 incoming and outgoing activation gradients.
FP8 describes storage and transport; window math and gradient reductions use
FP32. CPU FP64/reference execution supplies a numerical oracle.

| Family | Dimensions | Compression-enabled backward state |
| --- | --- | --- |
| MaxPool | 1d, 2d, 3d | Exact packed local winner offsets |
| AvgPool | 1d, 2d, 3d | No saved activation values |
| AdaptiveMaxPool | 1d, 2d, 3d | Exact packed winners and bin geometry |
| AdaptiveAvgPool | 1d, 2d, 3d | No saved activation values |
| LPPool | 1d, 2d, 3d | Quantized/packed preprocessed derivative coefficient plus output factor |
| FractionalMaxPool | 2d, 3d | Exact packed winners and original sampled geometry |
| MaxUnpool | 1d, 2d, 3d | Exact packed destination indices and required index-version guards |

The [functional module](../../src/deepopt/opt/nn/functional.py) provides native
names and `do_` counterparts for all families, including `*_with_indices`
aliases. Available exact native module types are converted during preparation;
subclasses and excluded modules retain their existing boundary. Conversion does
not consume fractional-pooling RNG or change original sample-buffer identities.
Newer native module types are optional imports on older Torch versions; the
functional LP3d expression is available even without a native LP3d wrapper.

## Configuration and APIs

```toml
[runtime.fp8]
enabled = true
backend = "triton"
backward_precision = "auto"
# intercept_pooling = true  # Omission inherits intercept_linear.

[runtime.activation_compression]
enabled = true
bits = 2
group_size = 512
```

Compression is independent of FP8. With FP8 disabled, enabled compression still
prepares BF16 pooling paths, including average pooling's saved-state elimination.
The option applies to every family. Global bits/group size control lossy LP
coefficients; exact integer routing uses its own required bit width. The norm
xhat-recentering flag does not apply to pooling.

```python
import torch
from deepopt.opt.nn import functional as doF
from deepopt.opt.policy import pooling_plan

plan = pooling_plan(
    {"enabled": True, "backend": "triton"},
    activation_compression={"enabled": True, "bits": 2, "group_size": 512},
)
x = torch.randn(2, 64, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = doF.do_max_pool2d(x, 2, plan=plan, output_dtype=torch.float8_e4m3fn)
y.backward(torch.ones(y.shape, device="cuda", dtype=torch.bfloat16))
```

Explicit calls also accept output layout and positive FP32 output-scale
multipliers on the requested FP8 scale grid. Current tensorwise/blockwise,
provided and prepared delayed scales follow the shared ABI. A bare delayed
FP8-output call needs prepared scaling ownership or explicit output scales.

Native output dtype follows input/AMP policy. In particular, finite-power LP
pooling follows CUDA AMP's FP32 power arithmetic; explicit output controls can
override the resulting output format. Prepared producer/consumer analysis chooses
FP8 outputs and dx once per train/eval call site. Functional calls, reused
modules, tuple value/index projections and the compatible global adaptive-pool
to flatten to Linear relay are included. General incompatible reshapes retain
a native boundary rather than reusing an invalid FP8 scale grid.

## Saved state and correctness

Average/adaptive-average backward retains only geometry, not input/output values
or artificial masks. Its saved activation bytes are zero. Enabling compression
selects that implementation even when FP8 is disabled.

Maximum variants pack relative winner locations losslessly. Four possible
winners normally require two bits. No affine quantization of indices is used.
Public indices remain int64 tensors. When returned or supplied by the caller,
the original tensor may remain retained as a version guard; backward uses the
packed private representation. Those guards can limit or eliminate net memory
savings and are included in reports. Arbitrary required routing widths, tail
bits and 64-bit indices are handled by the [index codec](../../src/deepopt/opt/quantization/indices.py).

LP saves the processed coefficient `p*x**(p-1)` and an output-local factor
derived from the native power/average/sign expression. It preserves native
zero guards and supports finite and infinite powers. Compressed floating
coefficients use global 1/2/4/8 bits and default group size 512; groupwise FP8
uses the input's scale-block geometry. Forward values are computed from the
original represented input, independently of saved-state compression. Infinite
powers use exact winner/sign state instead of a raw-input payload.

All pooling backward paths are first-order. Overlapping windows finish all dx
contributions before FP8 casting. Unpooling preserves overwrite semantics for
repeated destinations, not scatter-add semantics; CUDA deterministic-mode
requests follow its unsupported/determinism contract. Fractional backward reuses
the original sampled windows. Checkpoint recomputation uses RNG preservation
and the FP8 scale tape without duplicating delayed observations.

## Fusion, workspace and reporting

The [Triton backend](../../src/deepopt/opt/backends/triton/pooling/) shares window
kernels across dimensions and families:

- Local FP8 decode, reductions, output conversion and private winner packing
  share a forward program when scale ownership permits it. Backward unpacks
  routing or decodes LP coefficients inline.
- Aligned non-overlapping LP windows fuse coefficient packing into forward.
  This includes compatible flat tensorwise groups and FP8 channel blocks of
  1-by-32. Other LP geometries use a reported coefficient-preprocessing/packing
  pass without storing raw input or duplicating coefficients per window.
- Large full-domain/global reductions use bounded partial/merge/final stages.
  Final maxima carry native winner locations, then pack routing once. Native
  reduction tiling does not depend on lossy bits or compression group size.
- Cross-program FP8 scales use reduction/cast stages and FP32 output workspace.
  Generic unpool zero-fill and scatter have separate ownership phases; output
  scales/observations measure the completed scatter result. Bounds checks remain
  on device. Current/provided-scale graph paths require no diagnostic host reads
  when checks are disabled.

There is no universal one-launch claim. Execution reports expose the compression
strategy, routing width/bytes, coefficient/factor bytes, index guards, logical
saved bytes, deduplicated retained-storage bytes, scratch and forward/packing/
scale/casting/validation/backward launches. Reference counts describe reference
execution, not measured CUDA launches. CUDA profiler/allocator measurements are
the authority for physical launch/peak-memory costs.

## Measurement and acceptance

The [standalone benchmark](../../benchmarks/pooling/benchmark_pooling.py) imports
the installed package and can be copied outside this repository:

```bash
python benchmarks/pooling/benchmark_pooling.py --op max_pool2d --shape 16x64x56x56 --pipeline conv --execution both --output work/pooling.json
python benchmarks/pooling/benchmark_pooling.py --op adaptive_avg_pool2d --shape 16x64x56x56 --output-size 1 --device cpu --warmup 1 --iterations 2 --repeats 1
```

Full-step timing includes FP8 input preparation, RNG, pooling, compressed state,
output scales/casts, backward and gradient clearing. Native FP8-input baselines
also reconstruct their represented input. Native CPU/fractional baselines use
explicit FP32 pooling where Torch has BF16 gaps; reports state this. Unpool
inputs/indices are assembled before timing. Explicit scale preparation is outside
timing. Numerical compression comparisons restore RNG and numerical state, run
outside timing and report output/gradient errors and public-index agreement.
A separate eager phase profile splits forward (including input preparation)
from backward/scope flushing. CUDA events report stream intervals; CPU phase
times on CUDA describe submission, not synchronized GPU completion. This
diagnostic profile is outside the full-step timing loops, including graph runs.

```bash
python -m unittest tests.opt.test_pooling tests.opt.test_pooling_kernels tests.opt.test_pooling_integration -v
DEEPOPT_REQUIRE_CUDA_POOLING=1 python -m unittest tests.cuda.test_pooling -v
python -m unittest tests.distributed.test_fp8_delayed_scaling -v
```

CPU reference and interpreted-JIT tests do not establish CUDA compilation,
replay, bounds-sanitizer or performance acceptance. Run those on the compatible
[NVIDIA/Triton environment](../fp8/cuda.md). Delayed full-training CUDA Graph
capture retains the existing eager-update restriction. Static geometry/model
changes require the corresponding execution guards or graph recapture.
