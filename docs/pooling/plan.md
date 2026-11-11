# Plan: pooling, adaptive pooling and unpooling

[Documentation](../README.md) · [FP8 backward precision](../fp8/backward_precision.md) · [Compression](../activation_compression/README.md) · [Configuration](../configuration/README.md)

Status: implementation available; see the [usage and validation guide](README.md).
The design below records the all-family contract. CPU/reference and interpreted
JIT validation are separate from real CUDA compile/replay/performance acceptance.

**Activation-compression support is mandatory for all 20 modules and their
functional equivalents**, including adaptive pooling and unpooling. Every
family must implement the enabled policy for both BF16 and FP8 execution;
compression cannot be omitted because the family has no floating input values
to save. Compress the actual backward state, or eliminate it when backward
needs only geometry. Do not manufacture a raw-input payload just to compress it.

## Scope and native compatibility

The installed development environment has PyTorch 2.15, whose pooling module
exports these 20 classes. Cover module calls and corresponding functional calls:

| Family | Dimensions | Count | Backward state |
| --- | --- | --- | --- |
| `MaxPool` | 1d, 2d, 3d | 3 | Exact winning locations |
| `AvgPool` | 1d, 2d, 3d | 3 | Geometry/divisors; no input values |
| `AdaptiveMaxPool` | 1d, 2d, 3d | 3 | Exact winning locations and bin geometry |
| `AdaptiveAvgPool` | 1d, 2d, 3d | 3 | Bin geometry/counts; no input values |
| `LPPool` | 1d, 2d, 3d | 3 | Preprocessed derivative factors; routing/sign state for infinite powers |
| `FractionalMaxPool` | 2d, 3d | 2 | Exact winning locations and sampled window geometry |
| `MaxUnpool` | 1d, 2d, 3d | 3 | Exact destination indices |

Inventory against [PyTorch's pooling definitions](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/pooling.py).
The repository supports `torch>=2.4,<3`; register native symbols conditionally
and test each supported version's behavior. Do not make a newer native class a
mandatory import on older installations. Version-specific LP special cases
must follow the installed native implementation.

Preserve native signatures, argument validation and dtype/AMP behavior:

- Batched and unbatched inputs, dimensional argument broadcasting, default
  stride, padding, max-pool dilation and `ceil_mode` boundary correction.
- Average-pool `count_include_pad` and `divisor_override` **where the native
  dimensional API exposes them**. Ceil-mode tails must use the correct divisor.
  See [AvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html).
- Adaptive integer/tuple output sizes and supported `None` dimensions. Include
  output sizes larger than the input: adaptive regions can overlap, and are not
  equivalent to a single fixed kernel/stride pair.
- Native maximum/index selection, ties, NaNs, signed zeros, infinities,
  dilated windows and degenerate windows. Compare against the native backend on
  the same device; do not assume a generic reduction's argmax convention matches.
- `return_indices=True` returns native-shaped **int64 spatial indices**, marked
  nondifferentiable. Floating output quantization never changes those indices.
- Fractional output-size/ratio validation, explicit `_random_samples`, ordinary
  generator behavior, module buffers and seeded reproducibility.
- Unpool output-size disambiguation, bounds checks and repeated-index behavior.
  Repeated destinations are not sum-pooling: preserve the native overwrite/
  nondeterminism contract and respect deterministic-algorithm settings. See
  [MaxUnpool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html).
- LP native power/sign/zero-gradient rules, ceil-edge behavior and supported
  finite/infinite powers. Do not silently replace native `x**p` with `abs(x)**p`.
  PyTorch 2.15 explicitly handles positive and negative infinite powers; compare
  against its functional implementation, not just a simplified norm formula.

The optimized contract covers dense floating tensors. Keep native behavior for
other supported dtypes and explicit boundaries for unsupported tensor layouts.
First-order autograd follows the existing DO contract; do not advertise new
higher-order-gradient support implicitly.

## Precision and saved-state policies

Every family must accept native BF16 and scale-aware FP8 E4M3FN/E5M2, accept
native/FP8 `dy`, and support native or planned FP8 output/`dx`. Retain existing
FP32/FP16 behavior and a CPU FP64 oracle. Native output dtype follows the
operation's native/AMP policy; explicit output controls and prepared consumer
decisions select FP8. Enabling FP8 alone must not force incompatible consumers
to receive FP8 storage.

Decode FP8 operands locally in kernels. Comparisons, sums, LP math and gradient
accumulation use FP32. FP8 here means activation/gradient storage and transport;
pooling is not a Tensor Core GEMM. A max window spanning different FP8 scales
must compare represented values, not raw FP8 codes.

Support current tensorwise/blockwise scaling, explicit output scales and
prepared delayed-scaling ownership. Compute scales/amax from the completed
pooled values or completed gradients. Input and output scale grids are separate
because pooling changes spatial shape. Any scale-preserving max shortcut must
prove compatibility with the requested dtype/grid/recipe and preserve required
amax observations; tensorwise input alone is not permission to bypass policy.

Compression remains independent of FP8 and uses the existing
`runtime.activation_compression` section. With `enabled=true`, preparation and
functional interception must select a compression-aware implementation for
**every family**, including BF16-only average pooling and unpooling. Average
pooling must not remain on a native autograd path that retains an unnecessary
input tensor. No family-specific opt-out or separate compressed API is needed.

| FP8 | Compression | Behavior |
| --- | --- | --- |
| Off | Off | Native execution with ordinary state when no optimization is needed |
| Off | On | Native arithmetic with family-specific compressed saved state |
| On | Off | FP8 transport with ordinary required backward state |
| On | On | FP8 transport plus family-specific saved-state compression, fused where possible |

The mandatory enabled-policy matrix is:

| Family | Activation-compression strategy | Backward reconstruction |
| --- | --- | --- |
| `AvgPool1d/2d/3d` | Eliminate all saved floating activations | Reconstruct window/divisor information from geometry |
| `AdaptiveAvgPool1d/2d/3d` | Eliminate all saved floating activations | Reconstruct bin bounds/counts from geometry |
| `MaxPool1d/2d/3d` | Losslessly pack private winner locations | Unpack routing inside the gradient kernel |
| `AdaptiveMaxPool1d/2d/3d` | Losslessly pack private winner locations | Unpack routing with adaptive bin geometry |
| `FractionalMaxPool2d/3d` | Losslessly pack winner locations; retain minimal sampled geometry | Unpack routing using the original sampled windows |
| `LPPool1d/2d/3d` | Quantize-pack preprocessed derivative factors; lossless routing/sign state for infinite powers | Dequantize-unpack factors or unpack discrete state inside backward |
| `MaxUnpool1d/2d/3d` | Losslessly pack destination indices for backward | Unpack destinations inside the gradient gather |

Global `bits=1/2/4/8` and `group_size` apply to lossy floating derivative
factors, following the existing BF16/tensorwise/groupwise rules below. Discrete
routing must remain exact: its packed width is determined by its value range,
not by a lossy activation-bit override. Geometry-only backward state requires
neither a quantization group nor a fictitious one-bit mask.

### Average and adaptive average

Do not save or compress raw input, output or an artificial activation mask.
Backward depends only on `dy`, the window/bin geometry and the correct divisor.
Implement activation compression as **saved-activation elimination**: the
compression-enabled autograd path saves zero input/output activation bytes and
does not retain their storage through hidden references. Recompute static
counts cheaply or keep only necessary small metadata, never a full per-input
activation shadow. This is a required zero-state implementation, not an
unsupported/no-op policy. If an existing optimized compression-off path already
saves zero activations, report zero incremental savings honestly.

### Maximum, adaptive maximum and fractional maximum

Use **lossless integer packing**, not affine quantization, for private routing:

- Save the local winning offset when the window mapping permits it. A window
  with four possible winners normally needs two bits, not a one-bit mask.
- Derive width from the number of representable locations, independent of global
  activation `bits`. Support arbitrary required widths, unequal adaptive bins,
  large indices and a no-winner sentinel where required by native geometry.
- A single possible valid winner needs no payload when geometry identifies it.
  Include padding, word ownership, tail-bit initialization and 64-bit indexing.
- Record the logical shape, window mapping and bit width in a dedicated routing
  payload. Keep the generic affine `PackedActivation` format unchanged.
- Pack directly while selecting maxima and unpack inside backward. Avoid a full
  temporary int64 index tensor for the private compressed path.
- `return_indices=True` must also support the enabled compression path: return
  ordinary public int64 indices, but use an exact packed private payload for
  backward. Preserve native mutation/version checks for the public tensor.
  If preserving that contract requires retaining its full storage as a version
  guard, account for it separately; do not disable packing or claim that storage
  was released. Public-index lifetime can limit or eliminate net memory savings.
- Fractional local offsets also require the sampled window mapping. Retain the
  minimal samples/interval metadata needed to reconstruct that mapping exactly.
  Never reconstruct it using a fresh RNG draw in backward.

Name the lossless helpers `pack_indices` / `unpack_indices`; they do not quantize.
They should sit alongside the existing boolean and affine codecs, with reference
and Triton implementations. A codec width with no storage benefit may use the
exact native integer representation and report it honestly.

### LP pooling

For finite powers, derive the backward from the installed native expression,
including its average-pool divisor and subsequent kernel-volume multiplier.
Factor it into an input-local derivative coefficient and a per-output reduction
coefficient. Save/compress the **preprocessed input coefficient**, never raw x
merely to recompute the power in backward. Preserve the zero-gradient guard.

On the finite differentiable branch, let `K` be kernel volume, `D[o]` the native
average divisor, `z[o] = sum(x[i]**p) / D[o]`, and
`s[o] = K * sign(z[o]) * relu(abs(z[o]))`. Then:

```text
a[i]  = p * x[i]**(p - 1)
b[o]  = K / D[o] * (1/p) * s[o]**(1/p - 1)
dx[i] = a[i] * sum_over_containing_windows(dy[o] * b[o])
```

Apply the native zero/nonfinite/special-power rules outside that branch. This
retains ceil-tail scaling and avoids storing one derivative per window element.

Use global affine bits 1/2/4/8 for that floating coefficient. BF16/tensorwise FP8
uses the configured group size, default 512. Apply the existing groupwise FP8
group-geometry rule when the coefficient uses that input transport mapping;
its affine minima/steps are independent of FP8 scales. Keep the small per-output
factor in FP32, like normalization's retained rstd, and report its bytes.
`norm_xhat_recenter` remains normalization-only.

Store each input coefficient once, even for overlapping windows; duplicating
it per window can consume more memory than native activations. Specialize
p=1/p=2 without losing native zero guards. Infinite-power paths share maximum/
minimum-absolute-value routing and retain the winner's derivative sign/zero
state losslessly; they do not need an approximate raw-input payload.

With compression off, retain the exact preprocessed state needed by backward.
With compression on, apply its decoded approximation consistently in every
gradient path. Forward values/scales must remain independent of compression
bits, group size and saved-factor layout.

### Unpooling

Do not retain raw floating input for backward. With compression enabled, create
an exact packed private representation of the supplied destination indices and
unpack it inside backward's gather. Do not quantize integer indices with the
affine activation codec. With compression disabled, the ordinary exact index
representation is permitted.

Preserve caller-visible int64 tensors and their mutation/version checks. Retain
the original tensor as a version guard when required by the native contract;
the backward computation must still use the compressed representation. Report
packed bytes, guard storage, externally retained index storage and total memory
separately. A packed copy plus a full retained guard does not establish net
memory savings. Do not introduce unsafe version tracking merely to reduce the
reported retained bytes.

## Kernels, fusion and deterministic backward

Implement dimension-generic addressing with specialized 1d/2d/3d launch choices.
Separate pooling geometry from the FP8 wrapper's physical matrix mapping:
unbatched inputs have a different logical channel axis. Support strides,
contiguous input and applicable channels-last/channels-last-3d layouts without
first materializing the entire BF16 input from FP8.

For adaptive output coordinate `o`, input length `I` and output length `O`, use
integer bin bounds `start=floor(o*I/O)`, `end=ceil((o+1)*I/O)` on each spatial
axis. Use safe integer arithmetic and native boundary rules.

Forward:

1. Fuse local FP8 decoding, window reduction, native/FP8 output conversion and
   private routing packing whenever tile/reduction ownership permits it.
2. One program must own each packed word, including tails. Do not introduce
   racy word updates just to claim fusion.
3. Include fast paths for ordinary non-overlapping pooling and global adaptive
   pooling to size one. Large windows use bounded hierarchical reductions;
   max reductions carry both values and exact native tie-breaking indices.
4. Complete output-scale tiles can finish FP8 scales and casting inline.
   Cross-program scales require bounded amax reduction and a casting epilogue.
5. Fuse LP preprocessing/packing when its input-group ownership aligns with
   the pooling tile. General overlap may require a separate input-coefficient
   packing pass; expose its launch and scratch cost rather than saving duplicate
   coefficients per window.
6. Generic unpool zero-fill and scatter cannot safely share an unordered kernel:
   another program's zero store could overwrite a scattered value. Use explicit
   phases unless complete output ownership is proven. Resolve collisions before
   measuring current FP8 output scales; do not approximate final amax using
   input values that may have been overwritten.
   Fuse destination-index packing into an input-owned scatter tile where packed
   word ownership permits it; otherwise expose the required packing launch.

Backward:

- Prefer input-owned gather/reduction for overlapping average/max/adaptive/LP
  windows. Decode packed state locally and finish all contributions to each dx
  before native/FP8 casting. Fractional candidates use the saved window mapping.
- Non-overlapping cases can use simpler direct mapping. Global adaptive average
  has a broadcast/divisor fast path; global adaptive max uses saved winners.
- If an atomic-scatter alternative is faster, use FP32 accumulation followed by
  a separate completed-gradient FP8 epilogue. Never cast/overwrite partial FP8
  gradients. Bound/report its workspace and honor deterministic mode.
- Unpool backward gathers `dy` at saved destinations, using the native repeated-
  index derivative contract. Do not replace unpool forward overwrites with sums.
- Deterministic execution must use a supported deterministic algorithm or match
  native deterministic-mode rejection, rather than silently use racing atomics.

No universal one-launch promise: local fusion, global scale reductions,
overlapping gradients and generic unpool have different dependency structures.
Count shared work once and report every additional launch.

## API organization and one-time planning

Put all public functional entry points in the existing
[nn functional module](../../src/deepopt/opt/nn/functional.py): native-style names
and `do_` counterparts, plus the native `*_with_indices` functional aliases.
Add `pooling_plan()` as a thin shared-policy builder. Explicit calls offer the
existing output dtype/layout/scale controls where applicable. There is no
separate compressed-pooling API.

Proposed implementation layout; directories below are to be created:

```text
src/deepopt/opt/
  nn/pooling/                 # spec, geometry, dispatch, adapters, state
  nn/autograd/pooling/        # average, maximum, power, unpool
  quantization/indices.py     # exact routing payload and pack/unpack API
  backends/pooling/           # executor selection
  backends/reference/pooling/
  backends/triton/pooling/
    common/                  # addressing, window/bin mapping, scales, reductions
    window/                  # shared regular/adaptive/fractional max/average kernels
    power/                   # finite-power factors and compression
    unpool/                  # fill/scatter, gather, validation
  backends/triton/quantization/activation/indices.py
benchmarks/pooling/
docs/pooling/
```

Extend the [conversion registry](../../src/deepopt/opt/fx/registry.py),
[FX rewriting](../../src/deepopt/opt/fx/rewriter.py),
[native preparation](../../src/deepopt/opt/nn/prepare.py),
[autocast interception](../../src/deepopt/opt/_autocast.py), and
[FP8 wrapper dispatch](../../src/deepopt/opt/quantization/activation_tensor.py).
Preserve exclusions, tied module identity, original buffers/state-dictionary keys
and native subclasses. Preparation must not consume fractional-pooling RNG.

Extend the existing [forward planner](../../src/deepopt/opt/precision/planner.py),
[backward planner](../../src/deepopt/opt/precision/backward.py) and
[capabilities](../../src/deepopt/opt/precision/capabilities.py) in their single
train/eval preparation phase:

- Register pools as scale-aware forward consumers/producers and backward
  consumers/producers. Pool-only parameterless models must also be discovered.
- Pooling backward capabilities are independent of the GEMM `backward_backend`
  selector; advertise actual pooling kernel support rather than inheriting a
  linear-specific restriction.
- Cache output precision, native/FP8 dx choices, layout and strategy constraints.
  Compute numerical scales, sampled windows and activation factors each forward;
  these values are not one-time cached analysis results.
- Treat `(values, indices)` and its `getitem` projections by type. Only the
  floating values participate in FP8 edges; never cast index edges or treat them
  as floating gradient fan-out. Preserve existing true fan-out boundaries.
- Support the common `AdaptiveAvgPool(1) -> flatten -> Linear` pipeline through
  provably compatible metadata-only shape/layout relays, including backward.
  General flattening that rearranges channels-last matrices or scale tiles
  needs a fused re-layout or a reported native boundary, not a blind scale reuse.
- Distinguish functional call sites and reused modules. No tracing or capability
  scan in later forward/backward calls. Guard changed shapes/static arguments
  through existing execution contracts; CUDA graphs require recapture when
  those guards fail.

Propose one FP8 interception override, following convolution/normalization:
`runtime.fp8.intercept_pooling: bool | None = None`, inheriting
`intercept_linear` when omitted. Explicit prepared DO modules keep their policy.
Extend exclusion validation, config normalization, templates/reference docs,
prepared-policy comparison, checkpoint checks and CUDA Graph scope fingerprints.
Do not add a separate activation-compression section or override routing width
using the lossy global activation bit count.

## Runtime and state contracts

- Save tensors through autograd's saved-tensor machinery. Inference and paths
  needing no backward state allocate no private routing/coefficient buffers;
  public requested indices still need to be returned.
- Generate fractional samples once per ordinary logical forward using native
  generator semantics and logical dtype/device, not the raw FP8 storage dtype.
  Checkpoint recomputation must reproduce that draw through
  RNG preservation and the existing FP8 scale tape, without duplicate delayed
  observations. Preserve explicit user samples and their version checks.
- Support accumulation, DDP and rank-local compression metadata. Pooling adds no
  normalization-style synchronization; existing requested FP8 amax reductions
  retain their ordinary distributed behavior.
- Current/provided-scale CUDA Graph replay must update outputs, routing,
  coefficients and native RNG state correctly on device. Audit fractional RNG
  capture support explicitly. Keep existing delayed full-training capture
  restrictions; no per-replay host value reads or hidden re-analysis.
- Supply Fake/meta shape/dtype/indices behavior and respect existing eager
  operation boundaries inside compiled models. Prefer native fallback with an
  explicit reason over silently dequantizing inside an advertised optimized path.
- Apply existing finite diagnostics to FP8 casting and lossy factor compression.
  Test native special-value/index semantics separately from checked-mode errors;
  unchecked graph paths must introduce no diagnostic host value reads.

Reports include family/dimensions, input/output/dy/dx precision, scale strategy,
planning count, private/public routing bytes and width, coefficient/factor bytes,
retained total, scratch, deterministic strategy, RNG/sample ownership, shared
fusion and extra reduction/packing/casting launches. Explain zero incremental
memory savings without confusing them with missing compression support.
Expose the requested and effective compression policy for every family, using
explicit strategies such as `eliminate_saved_activations`, `pack_routing` and
`quantize_pack_derivative`. Distinguish zero-state compression, exact packing
and lossy approximation; none should be reported as unsupported compression.
Distinguish logical payload bytes from retained storage and deduplicate aliases
when reporting physical retained memory.

## Implementation sequence and acceptance gates

1. **Contracts and oracle.** Add version-aware 20-family registration, normalized
   geometry, functional/module dispatch and CPU reference paths. Establish native
   dtype, index, divisor, adaptive-bin, LP and RNG contracts before CUDA tuning.
   Include the enabled/disabled activation-compression state contract for every
   family in this first stage.
2. **Exact routing and regular max.** Add lossless index packing and fused
   MaxPool1d/2d/3d forward/backward with BF16/FP8 inputs, outputs and dx. Integrate
   cached forward/backward planning in this first optimized vertical slice.
3. **Average and adaptive families.** Add AvgPool and both AdaptivePool families
   for all three dimensions, global fast paths and the global-pool/flatten/linear
   relay. Implement zero-saved-activation compression for both average families;
   adaptive maximum uses exact packed routing. Validate overlapping input/output
   bins and completed-gradient casting.
4. **Power and fractional families.** Add LP preprocessed-factor compression,
   all native power cases, fractional sampled-window mapping and reproducible
   RNG/checkpoint behavior. Test both public and private index paths.
5. **Unpool families.** Add native-compatible zero-fill/scatter and gradient
   gather, output-size disambiguation, bounds/duplicate handling and FP8 output
   scaling after completed stores. Verify pool-to-unpool index interoperability.
   Compression-enabled unpool backward must decode packed destinations,
   including when its public indices came from compression-enabled max pooling.
6. **System acceptance and measurements.** Finish the matrix below, add an
   installed-library benchmark and document measured fusion/workspace limits.
   Do not label all-family optimization complete while ordinary valid cases in
   a requested family still silently execute a native fallback.

Required tests:

- Every available native class/function/alias, 1d/2d/3d, batched/unbatched, tuple
  arguments, strided/channels-last input, ceil/padding/dilation/divisor cases,
  irregular adaptive bins, upsampling, empty/invalid native cases and large
  indices. Test exact indices separately from floating tolerances.
- Independent forward/backward oracles on represented FP8 input, every native/
  FP8 precision pair, both FP8 formats, all scale modes and layouts. FP8 cast
  gradients only after all overlap contributions have been included.
- Ties/NaNs/signed zeros, fully padded windows where permitted, repeated unpool
  indices, LP zero/noninteger/negative/infinite-power cases supported by that
  Torch version, fractional fixed samples and generator-state equivalence.
- Compression off/on: identical forward/scales, exact private routing gradients,
  average saved-value bytes zero, LP gradients match the decoded-factor oracle,
  all lossy bits/groups and consistent factor use. No raw-input shadow.
- Assert compression support for **each of the 20 module types and every
  functional counterpart**, in BF16 and FP8, including public-index return paths
  and unpooling. Test average-pool storage lifetimes, direct packed-state use in
  backward, index/guard byte accounting and global config propagation. Merely
  accepting or ignoring `enabled=true` is not sufficient.
- Saved-state ownership/version errors, returned-index mutation, sample-buffer
  mutation, no-grad/inference skips, parameterless/shared modules, explicit
  disable/exclusions, functional call sites and no repeated analysis.
- Actual JIT bodies and launch policies interpreted on CPU; separate real CUDA
  compile, sanitizer/bounds checks, profile-based fusion counts and changing-
  value graph/RNG replay. CPU interpretation alone is not hardware acceptance.
- Native/auto backward precision, accumulation, two-rank DDP, exact checkpoint
  resume/policy mismatch, recomputation with RNG/scale preservation, Fake/meta,
  compilation and existing linear/conv/norm/activation regressions.

Benchmark isolated forward, backward and full steps plus CNN pipelines against
matched native BF16. Include FP8 input preparation, output scaling, routing or
factor creation, overlap accumulation, backward casts and compression in timing.
Report eager/graph latency, GPU activity, retained/peak/scratch bytes, exact index
agreement and gradient error; separate preparation/warmup. Compare compression
off/on and each FP8 scale recipe. No throughput claim before CUDA measurement.
Include every family's compression strategy and its actual total-memory delta,
including public index/version-guard storage. Average pooling should demonstrate
zero saved activation storage; do not benchmark artificial retained activations
to create a compression speedup or memory reduction.
