# Choosing a custom kernel backend

DeepOpt exposes three explicit custom-kernel modes: Triton, TileLang, and
Mixed. Existing recipes keep their defaults. Installing TileLang does not
change backend selection or add a dependency to Triton and CPU workflows.

TileLang execution is experimental. Runtime and compiler implementations are
present, with CPU semantic, integration, and real TileLang frontend checks.
NVIDIA CUDA compilation, execution, capture, distributed training, and
performance qualification are still required. Current TileLang schedules
prioritize correctness and can be substantially slower than Triton.

## Installation and selection

Install the optional dependency only when using TileLang:

```sh
python -m pip install '.[tilelang]'
```

The extra pins TileLang 0.1.15. Real custom FP8 execution targets an NVIDIA
CUDA environment with compute capability 9.0 or newer and a compatible
Torch/CUDA toolchain. CPU/reference execution and configuration parsing do
not import either optional GPU compiler.

Use `runtime.kernels.backend` for custom kernels, independently of the Torch
compiler selector `runtime.compile.backend`:

```toml
[runtime]
device = "cuda:0"
precision = "bf16"

[runtime.kernels]
backend = "tilelang" # triton, tilelang, or mixed

[runtime.fp8]
enabled = true
backend = "inherit"
```

| Mode | Behavior |
| --- | --- |
| `triton` | DeepOpt custom kernels and generated regions use Triton; TileLang is not imported or queried. |
| `tilelang` | DeepOpt custom kernels and generated regions use TileLang; DeepOpt Triton implementations are not imported or launched. |
| `mixed` | An immutable routing policy selects concrete backends by family and operation role. |

Omitting the new kernel table, or using `backend="inherit"` there, preserves
legacy selection: a concrete FP8 custom backend supplies its family;
otherwise CUDA helpers use Triton. The existing FP8 `reference` default
continues to choose emulated arithmetic. A reference CUDA computation can
still need custom preparation kernels, selected by the kernel policy.

Existing `FP8Config(backend="triton")` recipes remain valid. New explicit
kernel policies should use FP8 `backend="inherit"`; conflicting concrete
FP8 and kernel choices are rejected. A concrete FP8 backend alongside Mixed
is rejected to avoid two independent routing authorities.

## Mixed routing

For example, select Triton GEMM, TileLang normalization/codecs, and TileLang
Linear weight-gradient products:

```toml
[runtime.kernels]
backend = "mixed"
preferred = "triton"
fallback = "error"

[runtime.kernels.families]
gemm = "triton"
normalization = "tilelang"
activation_codec = "tilelang"
compiler_pointwise = "tilelang"

[runtime.kernels.roles]
"linear.dw" = "tilelang"

[runtime.fp8]
enabled = true
backend = "inherit"
```

Role overrides take precedence over family overrides, then `preferred`.
Concrete routes accept only `triton` or `tilelang`. Family names cover GEMM,
quantization, scaling, activation codecs, gradients, activation, normalization,
pooling, convolution, compiler pointwise and compiler preparation. Transposed
convolution shares the convolution family. Unknown routes are rejected.
An explicit partial-config switch from Mixed to a pure mode discards inherited
Mixed-only routes; partial updates within Mixed preserve the other routes.

`fallback="error"` is the default. Optional `fallback="other"` permits a
metadata eligibility decision to select the other installed backend for an
unpinned route. Family/role overrides are hard pins. Missing selected
dependencies and compile/runtime errors remain visible; they never trigger
a retry on another backend. Mixed does not perform automatic timing searches
or choose by package installation order.

Fusion respects routing boundaries. Conflicting explicit routes for an
atomic fused kernel are rejected or split into a declared decomposition.
For example, a fused Triton activation cannot silently use a pinned TileLang
codec inside that kernel. TileLang's decomposed operations dispatch their
separate preparation/codec launches through the declared Mixed routes.

## Python and execution ownership

Pass the same tables through `config=` to `Trainer`, `TrainingRuntime`,
`autocast`, or model preparation. Standalone operations can also use a scope:

```python
from deepopt import kernel_policy_scope
from deepopt.opt.quantization.quantizer import quantize_fp8

with kernel_policy_scope("tilelang"):
    operand = quantize_fp8(cuda_matrix)
```

Prepared plans, autograd contexts and escaped FP8 wrappers retain their
policy. Backward and decoding do not depend on a later ambient scope.
Separate models can use different policies in one process; scopes are
nestable and reset on exceptions. Repreparing a model with a different
explicit policy requires reconstruction.

Generated DeepOpt kernels use the same policy. Direct compiler callers may
pass `kernel_backend="tilelang"`, or Mixed plus `kernel_policy`, in compiler
options. See the [compiler guide](../execution/compiler.md).

Checkpoints retain actual nondefault policies and compiler identities,
including compression-only and per-operation policies. Default Triton recipe
identities remain compatible. Switching a numerical policy requires a new
run or an explicit weights import, rather than strict checkpoint continuation.

Warm all TileLang specializations before CUDA capture. Unseen specialization
compilation during capture fails with an actionable warmup error. Changing
policy invalidates graph compatibility. Existing delayed-scaling full-training
capture restrictions remain in effect.

## Coverage and qualification

Implementations include FP8 GEMM, scaling/conversion, affine/Boolean/index
codecs, six runtime activations, BatchNorm2d/GroupNorm/LayerNorm, all 20 pooling
kinds, Conv/ConvTranspose ranks 1–3, gradient helpers, and generated pointwise,
backward and Linear preparation. Shared payload formats support cross-backend
consumption in Mixed mode. Tensorwise and K-aligned GEMM use FP8 matrix
instructions; arbitrary scale layouts use an explicit FP32 SIMT implementation.

TileLang currently uses conservative fixed schedules. Existing tuning fields
retain validation, while broader search and performance tuning remain
qualification work. Some fused interfaces execute as multiple launches and
large serial reductions can be slow. Generic executor diagnostics mark
`launch_counts_complete=false`; use GPU profiling for total launches and
memory, including Torch layout copies. Delayed launch-cache statistics also
differ because TileLang updates use explicit tensors instead of pointer tables.
Run real GPU qualification on the intended Linux/NVIDIA environment:

```sh
DEEPOPT_REQUIRE_TILELANG=1 python -m unittest \
  tests.opt.test_kernel_policy tests.opt.test_kernel_backend_integration \
  tests.opt.test_tilelang_primitives tests.opt.test_tilelang_primitives_ir \
  tests.opt.test_tilelang_operations tests.compiler.test_kernel_backends \
  tests.cuda.test_tilelang_backend -v
```

The required flag converts missing CUDA/TileLang prerequisites into failures.
CPU interpreters and frontend IR construction do not replace those GPU checks.
