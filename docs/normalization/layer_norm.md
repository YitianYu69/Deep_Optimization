# LayerNorm with compressed xhat and FP8 transport

[Normalization guide](README.md) · [Configuration](../configuration/README.md) · [Backward precision](../fp8/backward_precision.md) · [Design plan](layer_norm_plan.md)

LayerNorm supports BF16/native and scale-aware FP8 input/output, pre-affine
normalized-activation compression and planned FP8 activation gradients. It
normalizes the exact trailing `normalized_shape` dimensions. Affine weights and
biases have that complete suffix shape, including multidimensional shapes.
Statistics and CUDA backward reductions use FP32; CPU FP64 provides a numerical
oracle. FP8 describes stored activation/gradient values, not the statistics.

Set `runtime.activation_compression.norm_xhat_recenter = true` to enable
[fused mean correction](README.md#compressed-xhat-mean-correction) for decoded
xhat. Row packing finishes the shift inline; chunked packing reduces bounded
error/count partials. Every xhat-consuming backward path loads original rstd
and applies the same shift, including weight-only backward and FP8 dx paths.
The default is off; outputs and codec payloads stay unchanged.

## Functional and module APIs

The [functional module](../../src/deepopt/opt/nn/functional.py) exposes `layer_norm`
and `do_layer_norm`:

```python
import torch
from deepopt.opt.nn import functional as doF
from deepopt.opt.policy import normalization_plan

plan = normalization_plan(
    fp8={"enabled": True, "backend": "triton"},
    activation_compression={"enabled": True, "group_size": 512, "bits": 2},
)
x = torch.randn(32, 197, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
weight = torch.ones(768, device="cuda", requires_grad=True)
bias = torch.zeros(768, device="cuda", requires_grad=True)
y = doF.layer_norm(x, (768,), weight, bias, eps=1e-5, plan=plan)
y.backward(torch.ones_like(y))
```

An integer normalized size is also accepted. The suffix must match dimensions
exactly: `(3, 7)` is distinct from `(21,)`. LayerNorm uses population variance
and identical input-statistics rules in training/evaluation; there are no running
statistics or batch counters. `elementwise_affine=False`, module `bias=False`,
functional optional weight/bias, strided affine tensors, rank-one input,
singleton domains and empty input are supported.

Preparation replaces exact `nn.LayerNorm` with
[DOLayerNorm](../../src/deepopt/opt/nn/normalization/modules.py). Parameter objects,
shared references and state-dictionary keys are preserved. User subclasses are
not converted as exact native types. Compression-only preparation also works
with FP8 disabled. Native functional calls and explicit DO functionals can have
their own prepared call-site adapters.

```python
import torch
import deepopt
from torch import nn

model = nn.Sequential(
    nn.Linear(768, 768), nn.LayerNorm(768), nn.GELU(), nn.Linear(768, 768),
).cuda()
trainer = deepopt.Trainer(
    model, torch.optim.AdamW(model.parameters(), lr=1e-4),
    device="cuda", precision="bf16", loss_fn=nn.MSELoss(), execution_mode="max_speed",
    config={"runtime": {
        "fp8": {"enabled": True, "backend": "triton", "backward_precision": "auto"},
        "activation_compression": {"enabled": True, "bits": 2, "group_size": 512},
    }},
)
```

`intercept_normalization` controls native functional interception and inherits
`intercept_linear` when omitted. Explicit native scopes, exclusions and supplied
operation plans retain the existing policy-resolution rules.

## Output precision and scaling

Standalone calls default to native input/AMP precision. CUDA AMP LayerNorm
uses its native FP32 promotion policy. An explicit native `output_dtype`, or
the configured native output dtype, can override it. Prepared models cache FP8
outputs when a compatible consumer is found; ordinary standalone calls have no
consumer graph.

For explicit FP8 output:

```python
from deepopt.opt.nn import functional as doF

fp8_y = doF.layer_norm(
    x, (768,), weight, bias, plan=plan,
    output_dtype=torch.float8_e4m3fn, output_layout="last",
)
```

FP8 results carry physical data plus FP32 inverse scales in `FP8Activation`,
with a native logical autograd dtype. `output_layout="last"` uses
`[-1, last_dim]`; `"channels"` uses `[N*spatial, C]` and requires rank ≥ 2.
Neither transport matrix is necessarily the normalization matrix when the
normalized suffix contains multiple dimensions. Kernels map logical coordinates
without a hidden contiguous-input or full FP8 dequantization allocation.
Raw float8 values follow the unit-scale convention; externally scaled input
must carry its metadata.

Current scales measure post-affine `y`. `output_scale` can supply a contiguous,
positive finite FP32 quantization multiplier on the output scale grid, without
gradients. Delayed output scales require a prepared owner or explicit scales.
An owned delayed scale must match the requested FP8 output format; prepare
matching settings or supply explicit scales when selecting another format.
There is no general reuse of input scales for normalized output or `dx`.

## What compression saves

```text
x_hat = (x - mean) * rsqrt(variance + eps)
y = gamma * x_hat + beta
```

With compression enabled, forward quantize-packs **FP32 xhat before affine**.
It saves integer words, FP32 minimum/step metadata, inverse standard deviation,
and needed affine parameters. It retains neither raw input nor a dense xhat
shadow. Without compression, backward reconstructs xhat from the original
input representation and statistics. Bias-only backward and inference retain
no unnecessary xhat/input state.

Global `bits` supports 1/2/4/8. Group size defaults to 512 and is bounded by the
normalized row length. Groups stay within a row; tails do not affect extrema.
This is the established normalization xhat policy. FP8 transport tiles and xhat
codec groups have independent metadata; attention's precision override does
not apply. Compression does not change forward output at the same input/output
precision. Reconstructed xhat approximates `dx` and `dweight`; `dbias` is
independent of the compression. Validate convergence for your workload.

## Fusion and execution boundaries

The [LayerNorm Triton backend](../../src/deepopt/opt/backends/triton/normalization/layer/)
uses complete-row fast paths and bounded hierarchical reductions:

- Compatible small rows fuse statistics, affine output and xhat packing. Native,
  explicit/delayed-scale, and compatible local-current-scale outputs can share
  that kernel.
- Tensorwise current scales spanning multiple rows need a global amax reduction.
  The fast prepass computes statistics, packs xhat and collects post-affine amax;
  the final epilogue recomputes output and casts directly to FP8. Large rows or
  incompatible partitions use additional statistics/reduction stages.
- Backward decodes packed xhat and FP8 `dy` in compute tiles. It applies gamma
  before row reductions and reduces affine gradients over outer positions.
  Complete-row native/compatible FP8 dx paths are fused; general FP8 dx uses a
  retiled final epilogue. Large global scales or strict diagnostics can select
  the shared FP32-workspace quantization path. There is no full unpacked-xhat
  buffer and no intermediate BF16 rounding before FP8 output.
- Affine-gradient reductions use bounded fixed-order partials, rather than a
  full outer-by-feature product allocation or nondeterministic atomic updates.

`execution_info` reports forward statistics/scaling/compression stages, saved
bytes, backward core/reduction launches and workspace dtype. Gradient-slot
reports separately expose fused casts and workspace conversion stages. Counts
exclude finite diagnostics, gradient clearing and autograd accumulation. CPU
reference execution reports zero CUDA launches.

The existing precision analysis runs once per prepared call-site/train-eval
contract. LayerNorm is a scale-aware consumer and producer in both directions;
native consumers, leaf gradients and fan-out accumulation preserve native
boundaries. LayerNorm → activation → Linear supports planned FP8 transport.
Parameter gradients remain native for optimizers/DDP.

Delayed scales use frozen accumulation windows, successful-step updates, DDP
amax ownership and exact checkpoint restore. Use `deepopt.fp8_checkpoint` for
recomputation so scales are replayed without duplicate observations. Whole-step
CUDA Graph replay of eager delayed-state updates remains unsupported.

Fake/meta execution requires no tensor-value reads. Custom LayerNorm stays eager
inside compiled models; surrounding operations may compile, with existing
graph-break/fallback limits. Compatible current/fixed-scale CUDA Graph replay is
covered by hardware acceptance tests. Custom optimized autograd is first-order.

## Verification and installed-library benchmarks

```bash
python -m unittest tests.opt.test_layer_norm tests.opt.test_layer_norm_kernels -v
DEEPOPT_REQUIRE_CUDA_LAYER_NORM=1 python -m unittest tests.cuda.test_layer_norm -v

python benchmarks/normalization/benchmark_normalization.py \
  --op layer_norm --shape 32x197x768 --normalized-shape 768 \
  --pipeline linear --activation gelu --execution both --output work/layer-norm.json
python benchmarks/normalization/benchmark_normalization.py \
  --op layer_norm --shape 64x768 --normalized-shape 768 --pipeline none \
  --fp8-block 1x32 --bits 4 --outputs bf16 fp8
python benchmarks/normalization/benchmark_normalization.py \
  --op layer_norm --shape 2x3x7 --normalized-shape 3x7 --pipeline none \
  --device cpu --warmup 0 --iterations 1 --repeats 1
```

The [benchmark](../../benchmarks/normalization/benchmark_normalization.py) works
when copied outside the checkout. LayerNorm modes include native BF16 baselines,
all requested BF16/FP8 input/output combinations, compression off/on, planned
Linear/convolution pipelines, optional activations and eager/graph timing.
Input casting, statistics, xhat packing, backward decoding and output/gradient
conversion are timed. Explicit fixed-scale allocation and numerical diagnostics
are outside timing. Peaks are absolute allocator peaks; saved-byte accounting
is distinct from temporary workspace and allocation peaks.

CPU kernel-body interpretation checks real kernel indexing and arithmetic; it
cannot establish CUDA compilation, graph replay or throughput. Run the hardware
suite and benchmark on the target GPU before interpreting performance.
