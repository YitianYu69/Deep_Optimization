# Plan: optimized sigmoid, SiLU, GELU, tanh and ReLU6

[Current ReLU implementation](README.md) · [Activation compression](../activation_compression/README.md) · [Backward precision planning](../fp8/backward_precision.md) · [Configuration reference](../configuration/README.md)

Status: implementation available; see the [activation API and validation guide](elementwise.md).
This document records its design and acceptance criteria. Real CUDA compilation,
capture and performance require the hardware validation described in that guide.

## Required behavior

Extend the existing activation implementation to sigmoid, SiLU, GELU, tanh and
ReLU6. Each activation must support the full applicable optimization surface:

| Capability | Required for every new activation |
| --- | --- |
| Forward input | Native BF16 and scale-aware FP8, including E4M3FN/E5M2 and existing matrix layouts. Preserve existing FP32/FP16 and CPU-reference support. |
| Forward output | Native BF16 or scale-aware FP8, with explicit overrides and cached automatic consumer decisions. |
| Backward input | Native or scale-aware FP8 upstream gradients. Decode incoming scales inside the backward kernel. |
| Backward output | Native gradients or newly quantized FP8 gradients selected by the existing one-time backward planner. |
| Saved-state compression | Save preprocessed derivative information; never quantize-and-pack raw input `x` for these activations. |
| Scaling | Current tensorwise, current blockwise, explicit output scales and owned delayed scaling, within the existing scaling-policy constraints. |
| Fusion | Activation math, applicable casts and saved-state writes in the forward kernel; saved-state decoding, derivative application and applicable gradient casting in backward. |
| Integration | Functional APIs, module adapters, transformation, scoped interception, FP8 wrappers, Trainer, runtime and DDP. |
| Execution | Fake/meta handling, existing compile boundaries, compatible CUDA Graph capture/replay, finite diagnostics, checkpoint/recomputation and delayed-state ownership. |
| Reporting | Saved bytes, scale policy, conversions, workspaces, launches and planner decisions. |

There is no BF16-only or compression-only completion milestone for an individual
activation. All five must satisfy this matrix before the extension is complete.
Existing restrictions remain explicit: first-order custom backward, unsupported
fullgraph compilation across custom dispatch, and no full-training graph replay
of eager delayed-scale updates. Supporting every optimization means honoring
its contract and reporting its limits, rather than assuming every combination
can use a single launch.

## 1. Save preprocessed backward coefficients

Follow the supplied SiLU/GELU example: compute a local derivative in forward,
then save or compress that derivative. Backward evaluates
`dx = dy * saved_derivative` after local decoding. Compute coefficients from
the represented input in FP32, before BF16/FP8 output rounding. Compression
metadata is independent of forward-output and gradient FP8 scales.

For finite inputs, use these definitions:

| Activation | Forward value | Preprocessed saved state |
| --- | --- | --- |
| Sigmoid | `s = sigmoid(x)` | `d = s * (1 - s)` |
| SiLU | `y = x * s`, `s = sigmoid(x)` | `d = s + x * s * (1 - s)` |
| GELU, `approximate="none"` | `y = x * Phi(x)` | `d = Phi(x) + x * phi(x)` using the matching erf-based formula. |
| GELU, `approximate="tanh"` | `u = sqrt(2/pi) * (x + 0.044715*x^3)`, `t = tanh(u)`, `y = 0.5*x*(1+t)` | `d = 0.5*(1+t) + 0.5*x*(1-t*t)*sqrt(2/pi)*(1+3*0.044715*x*x)` |
| Tanh | `t = tanh(x)` | `d = 1 - t*t` |
| ReLU6 | `y = clamp(x, 0, 6)` | Exact boolean derivative mask for `0 < x < 6`. |

Here `Phi` is the standard normal CDF and `phi` its density. GELU defaults to
`approximate="none"`, matching the native API; preserve the chosen approximation
through preparation and backward. Do not silently replace exact GELU with its
tanh approximation. Use stable formulas and native-compatible handling of
special values; finite-input identities alone do not settle NaN/infinity rules.

ReLU6's sample kernel needs a correction: its forward must return **6** for
inputs above 6. The interior mask controls backward, not the full forward value.
Derivative values at 0 and 6 are zero. Test NaN behavior against the native
ReLU6 operation separately from the finite-input mask expression.

### Saved-state policy

- With compression disabled, smooth activations save their preprocessed
  derivative without low-bit quantization. Use FP32 on CUDA so saving does not
  introduce another BF16/FP8 rounding. Preserve FP64 in the CPU numerical oracle.
  ReLU/ReLU6 save exact boolean masks.
- With compression enabled, sigmoid/SiLU/GELU/tanh affine-quantize and pack
  **the derivative** using global `bits` (1, 2, 4 or 8). Preserve signed derivatives
  and values above one in SiLU/GELU; do not clamp or squash them to fit [0, 1].
- ReLU6, like ReLU, saves an exact one-bit mask, independent of global bits.
- BF16/tensorwise scaling uses configured `group_size`, default 512. Blockwise
  FP8 uses the operation's activation scaling rectangles in its saved matrix
  layout, including masked edge groups. Incoming-gradient scales do not redefine
  the saved coefficient's grouping.
- Use the existing affine codec's FP32 step/minimum metadata, rounding rule,
  constant-group handling and zeroed tail bits. The sample's BF16 metadata,
  forced range floor and experimental averaging/noise are not adopted.
- Do not retain raw `x`, a dense output, or a dense derivative beside a compressed
  payload in the activation's autograd context. Other model consumers may of
  course retain their own outputs. Inference allocates no backward payload.
- Reuse the existing affine codec's finite-value contract and diagnostic policy.
  Do not silently encode NaNs as ordinary finite derivative values. Uncompressed
  special-value behavior and checked compression failures need explicit tests.

Compression changes only saved backward information. At the same input/output
precision, enabling it must leave forward values unchanged. Smooth activation
gradients become approximate; ReLU/ReLU6 masks remain exact. Keep the existing
`quantize_pack` and `dequantize_unpack` naming convention.

## 2. Extend the activation architecture

Keep public functions in the existing
[functional module](../../src/deepopt/opt/nn/functional.py). Add `do_sigmoid`,
`do_silu`, `do_gelu`, `do_tanh`, `do_relu6` and their ordinary-name aliases.
Expose the same precision/layout/scale options as ReLU and preserve native
arguments such as GELU's `approximate` and supported `inplace` forms.

Add a shared `activation_plan()` helper in
[policy](../../src/deepopt/opt/policy.py), retaining `relu_plan()` compatibility.
Use the existing FP8 and activation-compression sections; no new enable flags,
bit-width sections or attention controls are needed. GELU approximation belongs
to the operation/module, not a global precision policy.

Organize the implementation by responsibility under existing directories:

- [Activation frontend](../../src/deepopt/opt/nn/activation/): reusable geometry
  and option validation, plus operation-specific module adapters. Add
  `DOSigmoid`, `DOSiLU`, `DOGELU`, `DOTanh` and `DOReLU6`.
- [Activation autograd](../../src/deepopt/opt/nn/autograd/activation/): shared
  coefficient/mask save and restore contract, with captured gradient-slot and
  finite-check policy ownership.
- [Backend dispatch](../../src/deepopt/opt/backends/activation/): select both
  backend and activation kind rather than always importing the ReLU executor.
- [Reference backend](../../src/deepopt/opt/backends/reference/activation/):
  matching forward math, derivative preprocessing, affine saved-state handling
  and a double-precision oracle.
- [Triton backend](../../src/deepopt/opt/backends/triton/activation/): shared
  addressing/casting/codec helpers, small operation-specific math modules, and
  forward/backward launch orchestration. Separate smooth-coefficient kernels
  from exact-mask kernels; avoid one growing file with all implementations.

Reuse the current affine/boolean JIT helpers instead of creating another codec
format. Keep existing ReLU entry points and scale-preserving fast paths working
through this refactor. Add only the shared abstractions needed by these six
activations.

## 3. Fuse forward and backward kernels

### Forward

For each BF16/FP8 input-output combination:

1. Load and locally decode represented input to FP32.
2. Evaluate forward output and the preprocessed derivative/mask.
3. Compute needed group metadata and quantize-and-pack the derivative, or pack
   the exact ReLU6 mask, while writing native/FP8 forward output.
4. Write inverse scales and delayed amax observations where required, without
   materializing dense intermediate `x`, `y` or derivative buffers on fused paths.

Nonlinear activations generally need **fresh output scales**. Do not copy ReLU's
scale-reuse shortcut to sigmoid, SiLU, GELU or tanh. Input scales describe decoded
input values, not these functions' outputs. ReLU6 scale reuse may be added only
where the represented clamp and native semantics remain valid; otherwise cast
with the selected output policy.

Bounded groups with compatible output-scale and compression geometry can fuse
local scale reductions and packing. Different group partitions need explicit
tiling/reduction handling. For large global groups, use bounded partial amax or
derivative min/max reductions followed by the fused output/packing kernel.
Recompute pointwise math in reduction kernels where needed rather than saving
raw input or dense derivative intermediates. Report those additional launches.

### Backward

Read native/FP8 `dy`, decode packed derivative or exact mask directly in registers,
apply it in FP32, then write native or planned FP8 `dx`. There must be no full
unpacked derivative/mask allocation. ReLU/ReLU6 retain native-compatible gating
for inactive NaN/infinite upstream gradients rather than indiscriminate `dy * 0`.
Smooth activations follow their native special-value multiplication semantics.

Reuse the existing gradient-output epilogue/workspace machinery. Smooth
derivative multiplication generally requires new gradient scales; retaining
incoming scales must not silently clip `dx`. Current tensorwise/large groups may
require a workspace and reduction/cast stage; explicit or owned delayed scales
permit direct epilogues when other constraints allow them.

Preserve supported in-place API forms, aliasing, version counters and leaf/view
errors. Save derivative state before mutation. In-place execution cannot change
dtype/layout/scales; reject incompatible requests before writing. For smooth
FP8 in-place calls, require a valid representation-preserving path rather than
assuming the original scale remains suitable. GELU has no invented in-place API.

## 4. Integrate preparation and cached precision planning

Extend [transformation](../../src/deepopt/opt/transform.py),
[native preparation](../../src/deepopt/opt/nn/prepare.py),
[scoped interception](../../src/deepopt/opt/_autocast.py) and
[FP8 activation dispatch](../../src/deepopt/opt/quantization/activation_tensor.py).
Cover exact native modules, supported `torch.nn.functional`/`torch` calls,
tensor methods and supported in-place variants. Preserve shared modules,
parameters, state-dict compatibility, training state and GELU approximation.
Do not rewrite arbitrary subclasses as though their forward behavior were known.

Update [forward planning](../../src/deepopt/opt/precision/planner.py),
[backward planning](../../src/deepopt/opt/precision/backward.py) and
[capabilities](../../src/deepopt/opt/precision/capabilities.py) together:

- Register every activation as an FP8 input consumer and eligible output producer.
- Treat each smooth activation as a real nonlinear operation. It cannot be
  traversed as an identity or scale-preserving ReLU relay.
- Cache choices by call site and train/eval path during the existing analysis.
  Shared modules can have different outputs at different calls. Backward must
  use forward-captured bindings without tracing or graph discovery.
- Select FP8 `dx` only where the backward consumer supports its format/layout.
  Keep parameter/input leaves, native operations, fan-out accumulation and
  unsupported consumers at their existing native boundaries.
- Keep normalization → activation → linear/convolution pipelines encoded where
  compatible in both directions. Preserve ordinary reconstruction boundaries
  when another consumer requires native precision.
- Retain existing interception defaults/exclusions and explicit native scopes;
  enabling compression without FP8 must also prepare/intercept all five.

## 5. Preserve state and execution contracts

Generalize ReLU's owned delayed-output path in
[delayed scaling](../../src/deepopt/opt/scaling/delayed.py) to activation kinds.
Observe post-activation `y` for forward scales and produced `dx` for gradient
scales. Derivative affine min/max metadata has separate ownership. Keep scales
frozen across accumulation, DDP amax synchronization, evaluation freezing,
abort behavior and successful-update advancement.

Make recomputation use the existing scale tape and forward-captured settings,
without duplicate amax updates. Reconstruct adapters before checkpoint restore.
Respect native-parameter gradients and DDP synchronization at accumulation
boundaries. Support module-only preparation of dynamic models and report
untraceable paths using the existing conservative fallback behavior.

Provide correct Fake/meta shapes, strides and payload metadata. Preserve the
existing eager custom-operation boundaries inside compiled models; do not claim
new fullgraph support. Validate warmed graph replay for current scaling and
explicit fixed scales, with stable buffers and changing values. Existing delayed
full-training capture restrictions remain visible in docs and errors.

## 6. Verification and completion criteria

Implement CPU-reference tests before tuning CUDA kernels, then validate each
vertical slice against native PyTorch and the shared codec:

1. **Math:** all five functions, both GELU approximations, compression off;
   native-precision forward/backward comparison and CPU FP64 gradcheck for
   first-order correctness. Exercise ReLU6 below/at/above both thresholds,
   saturation, small values, signed zero, NaNs and infinities.
2. **Compression:** bits 1/2/4/8, constant groups, masked tails, small/large and
   non-power-of-two configured groups. Match the reference quantize-pack of
   derivatives and its decoded-gradient oracle. Inspect saved tensors to prove
   raw input/dense derivative retention is absent with compression enabled.
3. **Precision:** all four BF16/FP8 forward combinations, both FP8 formats,
   tensorwise/blockwise scales, native/FP8 upstream gradients and native/FP8
   planned `dx`. Separate input/output quantization error from derivative-codec
   error; do not require compressed gradients to equal uncompressed gradients.
4. **Geometry/mutation:** scalar and empty tensors, multiple ranks/shapes,
   noncontiguous and channels-last layouts, tails, supported in-place forms,
   alias/version checks and rejected representation changes.
5. **Integration:** modules, functionals, scopes, FP8 wrappers, shared call sites,
   native consumers, fan-out boundaries, training/eval and one-time analysis;
   forward/backward linear and convolution pipelines for every activation.
6. **State/execution:** accumulation, DDP delayed amax modes, exact checkpoint
   resume, activation-checkpoint recomputation, Fake/meta, supported compile
   boundaries and actual CUDA Graph replay. ReLU regressions must remain green.
7. **Kernel structure:** real CUDA compilation and payload/memory checks, no
   dense unpack temporary, fused local/explicit-scale launch counts and correctly
   reported multi-stage global-reduction cases. CPU JIT interpretation alone
   does not establish CUDA support.

Add an installed-library activation benchmark covering all functions and GELU
variants. Measure native BF16 and each optimized precision combination with
compression off/on, forward+backward latency, retained bytes, peak memory,
launches, workspaces, gradient error and eager versus warmed replay. Include
output-scale preparation, derivative preprocessing/packing and gradient casting
inside the relevant timings; identify any externally prepared input separately.
Add pipeline measurements to expose conversion costs hidden by isolated kernels.

## Implementation order

1. Establish the common derivative/mask contract and CPU oracles; refactor shared
   activation plumbing while preserving ReLU behavior.
2. Add ReLU6 as the exact-mask vertical slice across kernels, APIs and planning.
3. Add sigmoid and tanh across the same full optimization matrix.
4. Add SiLU and both GELU variants, reusing the smooth-derivative packing path.
5. Finish cross-operation/state/capture validation and benchmark all functions;
   update the activation guide, configuration explanations and benchmark catalog.

Each slice includes the applicable state, planner and BF16/FP8 integration rather
than leaving those until after a dense-only implementation. No performance
equivalence to native kernels or other libraries is claimed before measurement.
