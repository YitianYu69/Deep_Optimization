# Plan: fully optimized LayerNorm

[Normalization guide](README.md) · [Activation functions](../activation/elementwise.md) · [Backward precision](../fp8/backward_precision.md) · [Configuration reference](../configuration/README.md)

Status: implementation available; see the [LayerNorm API and validation guide](layer_norm.md).
This document records the design. Real CUDA compilation, replay and performance
require the hardware validation described in that guide.

## 1. Public behavior and optimization coverage

Add `layer_norm` and `do_layer_norm` to the shared
[functional module](../../src/deepopt/opt/nn/functional.py), plus a `DOLayerNorm`
adapter for exact native `nn.LayerNorm` modules. Use the existing
`normalization_plan()` and configuration sections. There is no separate
compressed-LayerNorm API or LayerNorm-specific enable flag.

Support the full applicable optimization surface from the existing DO operations:

| Area | Required behavior |
| --- | --- |
| Inputs | Native BF16 and scale-aware FP8 E4M3FN/E5M2; preserve FP32/FP16 support and provide a CPU FP64 numerical oracle. |
| Outputs | Native precision or scale-aware FP8, with explicit selection and cached automatic consumer decisions. |
| Backward | Accept native/FP8 `dy`; return native or planned FP8 `dx`. Affine parameter gradients retain their native parameter dtypes. |
| Compression | Save affine-quantized and packed **pre-affine `x_hat`**, never compressed raw input. Support global 1/2/4/8 bits and domain-local groups. |
| Scaling | Current tensorwise/blockwise FP8 scaling, explicit output scales and prepared delayed-scaling ownership. |
| Fusion | Fuse local FP8 decoding, normalization, affine math, output conversion and xhat packing where reduction domains permit it; decode packed xhat inside backward. |
| Integration | Module/functional preparation, autocast interception, FP8 wrappers, Trainer/runtime, DDP and cached forward/backward planning. |
| Execution | First-order autograd, Fake/meta, supported eager boundaries within compiled models, compatible CUDA Graph replay and checkpoint recomputation. |
| Reporting | Input/output precision, saved bytes, reduction/cast launches, workspaces, cached decisions and fallback reasons. |

Use a PyTorch-style functional signature:

```python
from deepopt.opt.nn import functional as doF
from deepopt.opt.policy import normalization_plan

# Shared precision and xhat compression policy.
plan = normalization_plan(
    fp8={"enabled": True, "backend": "triton", "backward_precision": "auto"},
    activation_compression={"enabled": True, "bits": 2, "group_size": 512},
)
# y = doF.layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, plan=plan)
```

Provide LayerNorm's explicit output precision/layout/scale controls consistently
with the activation API. Standalone execution follows the selected native/AMP
policy unless the caller requests FP8 or supplies compatible encoded input
under the defined output policy. A prepared model uses cached consumer choices.
Establish the native dtype oracle under CUDA AMP before defining the default;
do not assume every BF16 input yields a BF16 normalization output.

Preserve native LayerNorm semantics:

- `normalized_shape` accepts the native integer or shape-sequence forms. It must
  match an exact trailing suffix of the input; do not accept a matching element
  count with different dimensions.
- Normalize over every suffix dimension using population variance
  (`correction=0`), with `eps=1e-5` by default.
- Support `elementwise_affine=True/False`, module `bias=False`, and functional
  weight/bias combinations permitted by native LayerNorm. Affine tensors match
  the complete `normalized_shape`, not just the last dimension.
- Preserve parameter identities, tied/shared references, state-dictionary keys,
  training state and optimizer ownership during preparation.
- Training and evaluation use the same input statistics. LayerNorm has no running
  mean/variance or batch counter. Delayed FP8 histories still follow train/eval
  ownership separately.
- Support an input consisting solely of its normalized dimensions, singleton
  normalized domains, arbitrary outer dimensions, noncontiguous inputs and
  empty cases according to the native oracle. Handle unsupported optimized
  geometry with an explicit native boundary rather than incorrect indexing.

## 2. Geometry and saved xhat contract

For input shape `outer_shape + normalized_shape`, define:

```text
M = product(outer_shape), or 1 when outer_shape is empty
K = product(normalized_shape)
normalization matrix = [M, K]
x_hat[m, :] = (x[m, :] - mean[m]) * rsqrt(var[m] + eps)
y[m, :] = x_hat[m, :] * gamma[:] + beta[:]
```

Add a dedicated immutable LayerNorm geometry next to the existing
[normalization geometry](../../src/deepopt/opt/nn/normalization/geometry.py).
Do not force LayerNorm through the current N/C/spatial assumptions:
normalization domains, affine indexing and affine-gradient reduction axes all
differ from BatchNorm/GroupNorm.

Keep the `[M, K]` normalization view separate from the FP8 transport views
`[-1, last_dim]` and `[N*spatial, C]`. They are not generally identical when
normalizing several dimensions. Address logical coordinates and source strides
directly in kernels; do not hide an input contiguous copy or full dequantization
behind a reshape. Use 64-bit addressing where needed. Preserve native output
stride/memory-format behavior where required by the observed native contract.

Compression follows the established normalization policy:

- Compute and quantize-pack FP32 `x_hat` **before gamma and beta**.
- Use global `bits` and `group_size` (default 512), with effective group size
  bounded by `K`. Groups never mix independent normalization rows. Masked tails
  do not contribute to min/max.
- Reuse the existing affine codec, FP32 minimum/step metadata, rounding,
  constant-group behavior, low-bit ordering and zeroed tail bits.
- FP8 scale tiles describe transport values; codec groups describe normalized
  backward state. Keep their metadata independent, as in the current norm path.
  Share boundaries when compatible. A kernel requiring a common partition must
  check alignment and use a documented conversion/reduction path if it fails;
  it must not silently merge normalization domains to fit an FP8 tile.
- With compression on, save packed xhat, its codec metadata, inverse standard
  deviation and only the affine values needed by requested gradients. Retain
  neither raw input nor a dense xhat shadow in the autograd context.
- With compression off, use the existing norm convention of retaining input
  representation/statistics and reconstructing xhat during backward. An FP8
  input retains its bytes and the exact inverse-scale snapshot, not a dense copy.
- Save no backward state in inference. Bias-only backward does not need xhat;
  skip unnecessary statistics/payload retention based on requested gradients.

At the same input/output precision, compression must leave forward values
unchanged. Reconstructed xhat makes `dx` and `dweight` approximate. `dbias` uses
only upstream gradients and is independent of xhat compression. Attention's
bit-width override does not apply.

## 3. Reference math and fused CUDA kernels

Add a LayerNorm-specific autograd entry within
[normalization autograd](../../src/deepopt/opt/nn/autograd/normalization/), sharing
saved-state, finite-check and gradient-slot machinery where useful. Keep a
single functional entry that selects its precision/compression path internally.
Preserve existing BatchNorm/GroupNorm backend contracts during this extension.

Keep numerical code under the existing backend hierarchy. Add a LayerNorm
subdirectory within the [Triton normalization backend](../../src/deepopt/opt/backends/triton/normalization/)
for addressing, statistics, forward epilogues and backward reductions, plus a
matching reference implementation under the
[reference normalization backend](../../src/deepopt/opt/backends/reference/normalization/).
Reuse codec/scaling helpers instead of duplicating quantization formats.

### Forward

1. Load native values or locally decode FP8 values with their actual scale grid.
2. Compute FP32 mean and variance over the normalized suffix. Use a stable
   row reduction and bounded Welford-style combination for large rows; avoid
   cancellation-prone `E[x*x] - E[x]**2` as the general numerical algorithm.
3. Form FP32 xhat, apply optional elementwise gamma/beta, and write native or
   FP8 output. Quantize-pack xhat in that epilogue when compression is needed.
4. Observe **post-affine output `y`** for delayed output scales, not xhat.

Use a complete-row fast path where the row and scale/codec partitions fit a
bounded program. Compatible local current scales, explicit scales or frozen
delayed scales can permit normalization, output casting and packing in one
kernel. General groups/strides and large rows use bounded statistics plus
epilogue stages. Current tensorwise/global output amax may require an additional
normalized-output reduction and recomputation before final casting/packing.

The existing BatchNorm/GroupNorm forward performs FP8 casting after a dense
normalization result. LayerNorm should add a direct FP8 output epilogue where
supported, without an intermediate BF16 rounding. A staged fallback must expose
its FP32 workspace and casts in reporting. Do not describe an unavoidable global
reduction as a single launch, and do not advertise model-level Linear/GEMM fusion
as part of this change.

### Backward

For each normalization row, with `u = dy * gamma` (or `u = dy` without gamma):

```text
dx = rstd * (u - mean(u) - x_hat * mean(u * x_hat))
dgamma[k] = sum_over_outer_rows(dy[:, k] * x_hat[:, k])
dbeta[k] = sum_over_outer_rows(dy[:, k])
```

Decode incoming FP8 gradients and packed xhat directly inside reduction/compute
tiles. Use FP32 for products, row sums and affine-gradient accumulation. Apply
gamma before the input-gradient row reductions, and restore full suffix-shaped
affine gradients. Do not reuse GroupNorm's per-channel reduction axes.

Fuse the final dx expression with native/FP8 output casting, using the shared
gradient-output interface. Current scale domains that span programs may need a
reported FP32 workspace and bounded quantization stages. A fresh dx scale is
generally required; input or upstream-gradient scales are not automatically valid.

Use bounded hierarchical affine-gradient reductions rather than a full
`M*K` intermediate or unbounded atomics. Respect configured determinism and
handle `dx`-only, weight-only, bias-only, frozen-parameter and no-affine cases.
The custom optimized path follows the existing first-order autograd contract.

## 4. Preparation, precision and training integration

Extend the existing components together:

- Add `DOLayerNorm` to [normalization adapters](../../src/deepopt/opt/nn/normalization/modules.py)
  and extend functional normalization adapters with a `layer` kind.
- Register exact `nn.LayerNorm` conversion in the
  [FX registry](../../src/deepopt/opt/fx/registry.py), plus `F.layer_norm` matching
  and rewriting. Include direct DO functional calls in the
  [tracer](../../src/deepopt/opt/fx/tracer.py) and cached call-site machinery.
- Add compression-only preparation through the existing
  [native preparation](../../src/deepopt/opt/nn/prepare.py) path.
- Extend [autocast interception](../../src/deepopt/opt/_autocast.py) and supported
  [FP8 wrapper dispatch](../../src/deepopt/opt/quantization/activation_tensor.py).
  `intercept_normalization` continues to inherit `intercept_linear` when omitted.
  Preserve exclusions, explicit native scopes and native argument validation.
- Register suffix/last-axis-compatible LayerNorm capabilities in both
  [forward planning](../../src/deepopt/opt/precision/planner.py) and
  [backward planning](../../src/deepopt/opt/precision/backward.py), with accurate
  [capability declarations](../../src/deepopt/opt/precision/capabilities.py).

Analyze once in the existing preparation phase, with independent call-site and
train/eval choices. Support LayerNorm → activation → Linear and the reverse
gradient chain without redundant dense conversion where compatible. Include
LayerNorm as an FP8 consumer of preceding normalization/activation operations.
Shared LayerNorm modules must support different consumers at different calls.

LayerNorm is a real normalization boundary, not an identity relay. Choose FP8
outputs/dx only where the actual consumer can read the selected format/layout.
Explicit native-output overrides, fan-out accumulation, leaves and unsupported
layouts keep native boundaries. Rank-one LayerNorm must not be assigned an
invalid channel-matrix layout. Untraceable control flow uses the existing
conservative fallback and report; backward never performs new graph analysis.

Own forward delayed scales on prepared adapters and backward delayed scales on
captured gradient slots. Reuse accumulation-window freezing, successful-step
advancement, DDP amax synchronization, exact resume, abort cleanup and scale-tape
recomputation through `deepopt.fp8_checkpoint`. Preserve native parameter
gradients for optimizers/DDP. No LayerNorm running-statistics buffers are added.

Support Fake/meta shape and payload propagation without reading tensor values.
Keep custom kernels outside Dynamo and validate supported surrounding compilation.
Validate warmed CUDA Graph replay for current and explicit scales with changing
values. Existing eager delayed-update/full-training capture restrictions remain;
adding LayerNorm does not remove them.

## 5. Verification, benchmark and implementation order

Required coverage:

1. **Native semantics:** scalar normalized sizes and multi-dimensional suffixes;
   rank-one input, no outer dimensions, singleton domains, empty cases and
   incorrect suffix/affine shapes. Test affine on/off, independent functional
   weight/bias options, module `bias=False`, native AMP dtype behavior and
   constant/nearly constant inputs. Use an uncompressed CPU FP64 gradcheck oracle.
2. **Precision/compression:** all BF16/FP8 input-output pairs and both FP8 formats;
   native/FP8 dy and planned dx; bits 1/2/4/8, default/custom groups, tails and
   tensorwise/blockwise FP8. Inspect saved tensors to prove only xhat is packed
   and no raw input/dense xhat shadow is retained with compression enabled.
3. **Kernel structure:** noncontiguous/sliced/channel-layout inputs, multi-axis
   normalized shapes, local fast paths, large-row statistics and bounded backward
   reductions. Verify actual payloads against the affine reference and no full
   unpack allocation. Separate compression error from FP8 transport error.
4. **Integration/state:** module/native functional/direct DO calls, shared call
   sites, native consumers and fan-out; cached analysis count; every newly added
   activation in a LayerNorm → activation → Linear pipeline; Trainer accumulation,
   two-rank DDP synchronized/local histories, exact checkpoint resume and
   recomputation. Keep existing normalization/activation regressions green.
5. **Execution:** first-order gradient variants, finite diagnostics, Fake/meta,
   supported compilation, real CUDA compilation and warmed graph replay. CPU
   kernel-body interpretation complements GPU tests and cannot replace them.

Extend the [installed-library normalization benchmark](../../benchmarks/normalization/benchmark_normalization.py)
with `layer_norm`, a normalized-shape option, isolated and Linear/activation
pipeline modes, native BF16 baselines and eager/graph measurements. Include input
casting, statistics, output scaling/casting, xhat packing, backward decoding and
gradient casting in complete-step timings. Report saved bytes, peak allocation,
launches, workspaces, numerical errors and precision decisions; identify any
externally prepared inputs or explicit scales. Measure transformer-sized widths
and multi-dimensional suffixes. No speedup claim precedes GPU measurement.

Implement in this order:

1. Define geometry, validation and reference forward/backward, including the
   saved-xhat contract and native dtype oracle.
2. Add BF16/native kernels and xhat compression with bounded statistics and
   affine-gradient reductions.
3. Add FP8 input decoding, direct FP8 forward epilogues, native/FP8 dy handling
   and planned FP8 dx, with current/explicit/delayed scale ownership.
4. Wire module/functional preparation, scopes and both precision planners;
   finish checkpoint, DDP, Fake/meta, compile and graph validation.
5. Extend documentation/config explanations and benchmarks; publish measured
   hardware results separately from implementation and CPU verification.

Completion requires the full optimization matrix above. A BF16-only kernel or
an FP8 wrapper around a permanently dense path does not complete this plan.
