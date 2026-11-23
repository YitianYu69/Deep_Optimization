# Native AutoDiff compiler training

[Compiler guide](compiler.md) · [BF16 kernels](compiler_bf16.md)

DeepOpt now differentiates supported semantic regions with its own `AOTAutoDiff`
before scheduling their forward and backward kernels. BF16, compressed BF16 and
FP8 training no longer depend on the former compiler Linear/stateful backward
programs for those regions. Registered VJP rules emit inspectable backward IR;
shared numerical backends implement its GEMM, codec, activation, normalization,
pooling and attention operations.

One thin `torch.autograd.Function` bridge connects each compiled region to the
caller and returns its input/parameter gradients. Its backward binds saved
values and invokes the executable derived by DeepOpt AutoDiff. PyTorch autograd
still owns the caller's graph, leaf accumulation and DDP integration.

## Require native ownership

```toml
[runtime.compile]
enabled = true
backend = "deepopt"
dynamic = false
fullgraph = true

[runtime.compile.options]
use_native_autodiff = true
require_full_aot = true
bf16_gemm_backend = "generated"
fuse_deepopt_activations = true
```

`use_native_autodiff=true` is the normal compiler route. `require_full_aot=true`
rejects missing derivatives/lowerings and legacy/eager backward owners before
executing compiled work. Pair it with `fullgraph=true`: backend coverage alone
cannot detect a Dynamo graph break outside the captured graph.

The explicit `use_native_autodiff=false` migration control retains the former
regional/compiler programs for comparisons. It cannot satisfy strict native
ownership. `enable_aot_autodiff=false` disables compiler differentiation and is
not valid for differentiable generated regions.

Select precision and compression independently through the existing runtime
sections. BF16 generated kernels remain opt-in; native-library GEMMs remain the
default. Triton, TileLang and Mixed selection continue through the shared kernel
policy, including hard pins and explicit fusion rejection reasons.

## Training coverage

| Path | Native differentiation and state |
| --- | --- |
| BF16 Linear and AMP | Explicit forward/dX/dW/db operations, selective gradients, FP32 master parameters and logical BF16 rounding. |
| BF16 saved-input compression | Integer payload plus FP32 codec metadata; materialized or tiled dW decode; dX-first/dW-first schedules. Native empty-work gradients avoid packing. |
| FP8 Linear | Current, supported blockwise and delayed recipes; saved scale snapshots; versioned weight caches; prepared FP8 inputs and gradient transport. |
| Projection forms | Modules, captured functional calls and HF Conv1D's transposed parameter layout. |
| Activations | ReLU masks; ReLU6, sigmoid, tanh, GELU and SiLU coefficient/derivative policies; configured recomputation; native/FP8 representation edges and strided reads. |
| Normalization | Native LayerNorm exports forward statistics; prepared LayerNorm retains exact/packed xhat and moment correction; BatchNorm/GroupNorm retain their configured state and running-statistic behavior. |
| Convolutions | Native/FP8 spatial and transposed adjoints, unique-source saves, static output sizing, numeric/SAME and module padding modes. |
| Pooling | Numerical backend VJPs consume explicit routing, coefficients, factors, samples and public-index guards. Global average keeps shape-only backward state. |
| VAE primitives | Scalar clamp retains lossless branch masks; nearest 2D resize saves geometry and original scales; exp retains its actual forward coefficient dtype. |
| User operations | Registered composite decompositions or explicit saved-state/VJP contracts share eager/native recipes; pure single-result first-order explicit operations expose their backward IR. |
| Attention | Native SDPA forward/backward kernel state, math attention, masks, GQA and dropout; MHA decomposes into projections, geometry and attention before differentiation. |
| Graph composition | Multi-result VJPs, repeated/unused outputs, shared parameters, geometry/casts, reductions, matrix primitives, interleaved calls and retained graphs. |

Coverage remains subject to each operation/backend's existing dtype, shape and
device contracts. Unsupported combinations fail strict mode with their owner or
lowering reason. General symbolic shapes, higher-order/JVP differentiation,
arbitrary PyTorch operators, and extra native-MHA key/value tokens are outside
this compiler contract. The explicit FP8 non-empty validation is preserved.

Native `nn.MultiheadAttention` and the prepared DO adapter have different
interception and projection-layout behavior. Decomposition preserves those
differences, including saved-input ordering, returned weights and compression
scope; it does not introduce new approximation into the native module.

The [BF16 VAE configuration](../../examples/vae/native_bf16.toml) enables strict
fullgraph ownership and compatible convolution-source pack sharing for
MBConvVAE_V2. The [VAE guide](../../examples/vae/README.md) includes native modes
with and without compression and required CUDA qualification commands. CPU
whole-model tests cover even/odd images, exact eager/compiled outputs and
gradients, saved-state coverage, repeated updates and checkpoint resume.
CUDA numerical behavior, full-size real-data training and allocator peaks
remain separate qualification gates. See [custom operation registration](custom_operations.md)
for the reusable BlurPool decomposition and explicit user saved-state API.

## Saved values and optimization

AutoDiff exports internal forward values as well as inputs/parameters. FP8
logical saves export physical data/inverse-scale components. Explicit codec
requests export packed payload, step and minimum tensors. The backward IR
restores the required logical values or consumes packed state inside a selected
kernel. Integer payloads, statistics and scale bookkeeping receive no gradients.

Linear input compression changes its saved dW operand, while activation and
normalization codecs preserve their separate derivative/xhat policies. Saved
compression does not change the forward operand or create a straight-through
derivative through an integer quantizer.

Bindings are per invocation and use saved-tensor hooks/version checks. The
consuming backward executable clears its input slots and drops local references
after their last use. The engine's saved-variable release respects
`retain_graph=true`. Stored metadata and private weight caches remain separately
owned from transient activation storage.

Generic coefficient saves now use the consumer-specific `SavedUse` contract.
An omitted permission remains exact even when another consumer encodes the same
logical value. The resolved `SavedActivationPolicy` is immutable and captured
from the forward scope; nested `activation_compression=False` overrides an
enabled outer scope. User inputs and frozen activation coefficients can be
eligible. Parameters, persistent buffers and their geometry/cast views retain
model ownership through joint input lifting. Generic selection preserves
existing FP8 physical storage and scale domains.

Activation-compression checkpoint recipe version 2 records the multiplication
coefficient snapshot and product-round-then-reduce semantics. Exact resume
rejects older or changed numerical recipes before loading model/optimizer
state. Use the original library for an old exact resume, or load weights to
start a new run with the current recipe.

Residual reports name the logical source, ownership, dtype/shape/strides/layout,
physical components, recipe, consumer/gradient roles, rounding domain and exact
retention reason. Composite records identify their physical leaves through
`component_of`; count physical tensor identities/components once. Unknown
backend-owned auxiliary shapes remain `null`, rather than zero-byte estimates.

`codec.save_activation` and `codec.restore_activation` are typed calls with
immutable recipe/geometry specifications. Groups follow logical row-major
traversal. Scalar/rank-one explicit requests, tails, source offsets and strided
inputs share the materialized codec oracle. Restores preserve dense permutations
and channels-last layouts; overlapping/expanded or hole-containing views restore
compactly with matching IR strides. Generic selection compares expanded views
with their addressed storage and keeps them exact when encoding is larger.

Multiplication saves only the opposite coefficient needed by each gradient.
See the [multiplication contract](multiplication_saved_state.md) for eager
interception, native ownership, rounding and snapshot/version behavior.
Frozen convolution dX retains weight and geometry without saving its source;
db-only and no-grad routes save no source or weight. Requested dW state is a
separate typed `codec.save_convolution_source` (or exact
`convolution.save_source`) call with explicit channel traversal, source cast,
padding and grouping geometry.

Standalone functional/module/ATen padding and convolution padding restoration
share one exact metadata-only adjoint. Reflection/replication use a scalar-backed
shape descriptor; circular padding preserves native accumulation order,
including BF16 rounding at overlapping boundaries. These saved-state facts do
not certify the target backend's transient allocation behavior.

Attention cumulative-length and disabled-dropout RNG slots have explicit
optional presence. `None`, empty tensors and populated tensors retain their
distinct semantics and positional ABI across the saved-variable bridge.
Required state and present optional descriptors retain tensor metadata guards;
dropout still requires the selected backend's exact RNG state. The external
FlashAttention adapter keeps its independent state schema.

Replay requires a recognized pure pointwise recipe even when a custom policy
requests recomputation. Semantic calls, dropout RNG, running-statistic updates
and delayed-scaling observations are not ordinary pointwise replay. Reused
replay/encoded bindings still record every consumer and physical dependency.

### Saved-state support and qualification

| Contract | Local evidence | Remaining qualification |
| --- | --- | --- |
| Optional attention ABI | CPU controlled backend, malformed/present/absent/empty state, retained/interleaved graphs, hooks/release and external adapter tests | Actual forced CUDA kernels and complete archived H100 ViT/FP8 matrix |
| Consumer-specific residuals | Exact/encoded coexistence, owner/view provenance, immutable recipes, unique physical names, FP8 preservation, scalar/rank-one/layout/tail restoration | Target-GPU pack/restore layouts and physical-byte reconciliation |
| Multiplication | Public eager routes, reconstructed-coefficient oracle, gradient masks, promotion, native fullgraph ownership and invocation lifetimes | BF16 CUDA VAE cases; training quality is separate |
| Frozen convolution | Public fixed-buffer dX/db masks, explicit dW state, strict native graphs and existing spatial/FP8 reference regressions | CUDA numerical and transient-allocation evidence |
| Metadata-only padding | Functional, module and ATen exact VJPs, boundary accumulation, no saved source, source mutation/lifetime tests | CUDA value/allocation qualification |

Focused tests live in `tests/compiler/test_saved_activation_contracts.py`,
`test_attention_saved_state.py`, `test_native_multiplication.py`,
`test_native_padding.py`, and the corresponding eager tests. The forced-backend
fixture `tests/compiler/test_attention_saved_state_cuda.py` records actual
state metadata, imported source hashes, backend availability and pre/post source
identity. A skipped CUDA test does not qualify a backend. Newly introduced
generic approximation remains selectable through the existing compression
scope; disable it for the exact/materialized diagnostic route. First-order
limitations remain in force.

Fusion runs after differentiation. Selected patterns include BF16 activation or
residual GEMM epilogues, activation/cast plus saved-input packing, packed-read dW,
and dX plus exact or packed activation derivatives. BF16 rounding boundaries and
codec domains remain explicit. Unknown gradient layouts cannot enter the flat
pointwise ABI. Conflicting pinned routes retain separate operations.

## Inspect the evidence

```python
from deepopt.compiler import compilation_report

report = compilation_report(model).as_dict()
assert report["native_autodiff_operations"] > 0
assert not report["legacy_backward_operations"]
assert not report["runtime_fallbacks"]
for graph in report["graphs"]:
    assert not graph["eager_autograd_operations"]
    print(graph["differentiation_rules"])
    print(graph["native_autodiff_graphs"])
```

The graph evidence records derivative provenance, forward/backward operations,
saved components/bindings and gradient targets. The executable's
`backend_artifact.joint_autograd_regions` also exposes the paired graphs and
wrapper sources for direct compiler callers.

`compiled_deepopt_backward_regions` remains a compatibility counter. Use the new
native ownership fields to establish differentiation coverage. Static fusion
selection and generated source are separate from observed CUDA launches and
measured speed/memory results.

## Reproduce qualification

```sh
python -m unittest tests.compiler.test_native_autodiff -v
python -m unittest tests.compiler.test_native_autodiff_cuda -v

python benchmarks/compiler/benchmark_native_autodiff.py \
  --device cuda:0 --models linear mlp vit-block --batches 8 32 \
  --backends triton --trials 3 --repeats 5 --profile-steps 1 \
  --output work/native-autodiff-h100
```

Add `--backends triton tilelang` to qualify both implementations. Start with
smaller fixtures before `--models vit-base --batches 512`, which requires timm.
The standalone entry point includes eager, legacy compiler, native-library
control, generated unfused/fused, packed-read and alternate backward ordering.
Strict native variants require fullgraph capture and native provenance; worker
errors cannot disappear from the comparison result.

The harness preserves matched state/recipe fingerprints and records raw timing,
numerical comparisons, peak memory, saved-storage lifetimes and profiler launch
correlation. Kernel controls remain in
[benchmark_bf16_kernels.py](../../benchmarks/compiler/benchmark_bf16_kernels.py).
Compiler naming replaces diagnostic module hooks inside fullgraph execution.
The profiler omits module hooks for compiled runs so instrumentation cannot
force Dynamo to recapture `record_function` inside the graph. Phase and compiler
ranges remain available; `module_annotations_enabled` records whether eager
module hooks were used.

The output directory contains a suite manifest declaring every requested job.
It is checkpointed as jobs finish, fail or are interrupted. Each invocation uses
its own `runs/<suite_run_id>/trial-<n>/<strategy>/` directory, so a previous
invocation's results cannot fill gaps in a new run. The final comparison checks
the declared tasks and run identities before accepting numerical evidence.
Missing results, invalid snapshots, nonfinite values and failed jobs remain
qualification failures. GPU telemetry failures do not authorize GPU admission.
Uncertain worker cleanup quarantines the GPU reservation; it remains blocked
until an operator confirms the owned processes have exited and recovers the
reservation, even if the coordinator has already exited.

Compression accuracy against uncompressed BF16 is reported independently of
compiler agreement with the same compressed eager recipe. A successful
same-recipe comparison does not qualify the chosen compression policy for
training quality.

CPU/reference checks establish native differentiation and numerical/state
contracts. CUDA compilation, target-GPU fusion, runtime/memory improvements and
real-data training quality still require the pinned GPU qualification runs.
