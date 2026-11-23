# BF16 compiler execution and benchmarks

[Compiler guide](compiler.md)

The compiler recognizes native BF16 Linear modules and functional calls with FP8
disabled, including BF16 AMP with FP32 master parameters. It owns their saved
state and backward through DeepOpt's native AutoDiff. See the
[native AutoDiff guide](compiler_native_autodiff.md) for strict ownership checks,
saved bindings and custom-op coverage. Compression-only preparation gives Linear modules native
plans, independently of FP8. Prepared native activations and LayerNorm also have
compiler-owned saved-state programs using the existing numerical executors.

Generated BF16 GPU kernels are experimental and opt-in. CPU/reference correctness
does not establish CUDA compilation, speed, or model-memory savings. The native
library execution control remains the default until target-GPU qualification.

## Configuration

```toml
[runtime]
device = "cuda"
precision = "bf16"

[runtime.fp8]
enabled = false

[runtime.activation_compression]
enabled = true # false selects pure BF16
bits = 2
group_size = 1024

[runtime.compile]
enabled = true
backend = "deepopt"
dynamic = false
fullgraph = true

[runtime.compile.options]
require_full_aot = true
bf16_gemm_backend = "generated"
fuse_deepopt_activations = true
```

| Option | Values | Meaning |
| --- | --- | --- |
| `bf16_gemm_backend` | `native` (default), `generated` | Native library control or generated BF16 forward/dX/dW kernels. |
| `bf16_decode` | `materialized` (default), `tiled`, `panel` | Decode a complete saved operand, decode directly into dW GEMM tiles, or reuse a bounded decoded panel with an FP32 accumulator. Tiled/panel decode requires generated execution; panel requires the Triton dW route. |
| `bf16_backward_order` | `dx_first` (default), `dw_first` | Legal product ordering; compare latency and overlapping temporary storage. |
| `bf16_tile` | 32, 64 (default), 128 | GEMM output tile dimensions. |
| `bf16_num_stages` | 1–5, default 3 | Triton stage count; the initial TileLang implementation accepts 3. |
| `fuse_deepopt_activations` | false (default), true | Select supported BF16 and FP8 fusion patterns. Precision alone does not enable fusion. |

Select Triton/TileLang/Mixed through the existing kernel policy. Generated BF16
products require NVIDIA compute capability 8.0 or newer, rather than requiring
the FP8 path's Hopper hardware. Each target still needs hardware validation.
Mixed hard pins can reject an atomic producer-pack, epilogue or packed-read
schedule; use compatible routes or materialized decode as reported.

## Implemented patterns

- BF16 GEMM + GELU/SiLU or matching BF16 residual addition.
- Native GELU/SiLU + BF16 forward operand + packed saved input for the next Linear.
- Prepared compressed GELU/SiLU with matching codec groups/bits: jointly produce
  the BF16 operand, packed Linear input and packed activation derivative, then
  consume the derivative directly in generated dX.
- BF16 rounding/packing directly from an FP32 AMP input, with generated GEMM
  operand conversion in registers.
- dX GEMM + a native producer activation's backward derivative.
- Flat-group packed saved input decoded to BF16 inside dW GEMM tiles.
- Explicit bounded-panel dW decoding using a reused BF16 panel and an FP32
  accumulator, with final output rounding and epilogues applied once. Select
  `bf16_decode="panel"`; this experimental route has no automatic promotion.

Logical BF16 rounding boundaries are preserved within epilogues. Bias is included
in a GEMM where requested, but native Linear can already do that; it is not
counted as a newly removed operation by itself.

Linear compression changes its saved dW input. Forward uses original compute
values; dX/db do not depend on that packed save for the same upstream gradient.
Packed data is decoded to BF16 before multiplication. This is not an integer
GEMM. Activation derivative and LayerNorm xhat compression are separate existing
policies and keep their own compiler saved-state ownership.

Prepared compressed activation derivatives retain their configured codec. Their
stateful programs remain separate when codec geometry, layout or routes prevent
joint preparation. General
LayerNorm-to-GEMM fusion and an entire MLP in one kernel are not claimed.
Strided stateful activation/normalization cases use their numerical layout
executors through native-derived backward IR;
BF16 Linear supports ordinary matrix views or explicit reshape preparation.

## Measurements

Run kernel/pipeline controls first:

```sh
python benchmarks/compiler/benchmark_bf16_kernels.py \
  --device cuda:0 --shapes 4096x768x3072 4096x3072x768 \
  --backends triton --tiles 32 64 128 --profile \
  --output work/bf16-kernels.json
```

This includes complete activation/packing and materialized/tiled decode costs,
checks numerical results, uses the shared timing/GPU lease helpers, and exports
traces. Recommendations are scoped to source/hardware identity and memory budget;
they do not install defaults. CPU execution is explicitly labeled reference work.

Run matched training comparisons through fresh shared analysis workers:

```sh
python benchmarks/compiler/benchmark_bf16.py \
  --device cuda:0 --models linear mlp vit-block --batches 8 32 \
  --backends triton --trials 3 --repeats 5 --profile-steps 1 \
  --output work/bf16-training
```

Add `--models vit-base --batches 512` for the user's full ViT size after smaller
cases pass; it requires timm. Add TileLang with `--backends triton tilelang` to
qualify that implementation separately. `--dry-run` writes plans without Torch
or CUDA execution.

The training matrix compares eager, native control, generated unfused/fused,
packed-read and dW-first execution for BF16 and BF16+compression. The MLP fixture
explicitly keeps the native ATen GELU derivative to isolate projection compression.
ViT fixtures exercise their declared prepared-family policies. Cross-strategy
snapshots compare each compressed variant to eager with the same recipe, not
to an uncompressed dW oracle. Real-data quality validation is still required.

Results include effective compiler options, recipes, source identities, raw timing
samples, numerical checks, peak allocated/reserved memory and observed saved-storage
lifetimes. Aliases count once and retained graphs do not release saved storage on
first unpack. Nonpersistent saved-storage totals exclude known parameters/buffers/
optimizer state; inner hooks and private caches can limit visibility/classification.

The model matrix also runs a native `torch-inductor` runtime reference by default.
This strategy compiles only the original `torch_bf16` case; DeepOpt custom and
compressed operations are excluded from the Inductor reference. Its generated
code, including generated Triton source, FX graphs and fusion IR, is saved below
each case's `inductor/` directory in the same benchmark output. The result links
its verified code manifest. Existing DeepOpt cases and their matching-eager
correctness gates remain intact. Runtime ratios against native Inductor are
descriptive for a different compression recipe. Use `--no-inductor-reference`
only to explicitly omit this additional baseline.

For full ResNet18, ViT-B/16, EfficientNet-B0 and the original VAE V2, use the
[paired native Inductor benchmark](../../compiler_references/inductor/README.md).
It saves paired eager/Inductor full-update samples and generated code together;
opaque custom boundaries, graph breaks, missing code and absent CUDA execution
remain unqualified.

The suite checkpoints its declared jobs in `manifest.json` and publishes
qualification in `comparison.json`. Per-job paths live under
`runs/<suite_run_id>/trial-<n>/<strategy>/`; inspect the manifest for the current
invocation. Failed jobs remain recorded while independent jobs continue.
Requested profiler artifacts, complete task coverage, matching run identities
and full numerical finiteness are required by qualification.

`compiler_report.static_regions` counts selected programs. Profiler
`compiler_regions` links actual kernels to module/phase/role ranges; missing linkage
is reported as unknown. Positive profiled fusion fixtures fail qualification when
selection or execution evidence is absent. Inspect the saved traces before
attributing a runtime change to fusion. No speedup is claimed from CPU timings.
