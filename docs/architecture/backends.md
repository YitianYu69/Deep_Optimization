# Execution backend organization

[Architecture guide](README.md) · [FP8 guide](../fp8/README.md) · [Activation compression](../activation_compression/README.md)

Optimized operations separate model semantics, operand preparation, and GEMM
execution. Projection adapters and autograd live in
[opt/nn/](../../src/deepopt/opt/nn/); FP8 values, scales, caches, and compression
payloads live in [opt/quantization/](../../src/deepopt/opt/quantization/).
The [backend dispatcher](../../src/deepopt/opt/backends/__init__.py) selects the
executor for each GEMM without taking ownership of model parameters, optimizer
updates, or saved-activation lifetime.

## Functional operations and execution plans

[opt/nn/functional.py](../../src/deepopt/opt/nn/functional.py) owns callable
projection operations. Modules and autocast interception both dispatch through
`do_linear`; the execution plan selects native compute, native compute with
saved-activation compression, or FP8 compute with optional compression.
[linear_plan](../../src/deepopt/opt/policy.py) builds that policy from explicit
settings or the resolved configuration. Existing FP8 plans retain their format.

Spatial convolution uses `conv1d`, `conv2d`, and `conv3d` in the same functional
module. `conv_plan` resolves the shared policy. Geometry and adapters live in
[opt/nn/convolution/](../../src/deepopt/opt/nn/convolution/), with native and FP8
autograd implementations under
[autograd/convolution/](../../src/deepopt/opt/nn/autograd/convolution/).

The plan separates the computation dtype from saved-input storage. Enabling
compression therefore does not require a separate projection operation or a
new GEMM backend. Native and FP8 autograd implementations live under
[opt/nn/autograd/](../../src/deepopt/opt/nn/autograd/), where they retain their
respective gradient and saved-tensor contracts. Native projections use PyTorch
matmul; FP8 projections prepare quantized operands before backend dispatch.
The former standalone compression function remains a compatibility wrapper.

The same functional module exposes `scaled_dot_product_attention` (also
`do_scaled_dot_product_attention`). It uses PyTorch's native attention
operation and backend selection while compressing saved Q/K/V operands under
the plan's activation policy. Global `bits` controls projection inputs;
Q/K/V default to `max(4, bits)` unless `attention_bits` explicitly overrides
that width. Attention probabilities, masks, dropout state, outputs and
logsumexp are outside this compression policy. The Q/K/V saved-tensor
selection and lifetime policy lives in
[opt/quantization/attention.py](../../src/deepopt/opt/quantization/attention.py),
separate from the reusable affine codec and native attention execution.

## Backend families and operations

```text
opt/backends/
├── __init__.py            # Explicit gemm_nt dispatch
├── _shared/cuda.py        # Shared CUDA device and launch metadata
├── reference/gemm.py      # Dequantized FP32 reference GEMM
├── convolution/           # Lazy spatial forward/dX/dW dispatch
├── reference/convolution/ # Independently quantized convolution oracle
├── native/convolution/    # Native spatial convolution and gradients
├── native/fp8.py          # PyTorch scaled-FP8 GEMM for backward
└── triton/
    ├── convolution/      # Implicit spatial forward/dX/dW and launch metadata
    ├── activation/       # Fused ReLU, precision conversion and exact masks
    ├── gradient/         # Shared FP8 gradient epilogues, transposes and reductions
    ├── gemm/
    │   ├── fp8.py        # Triton FP8 GEMM kernels and executor
    │   ├── launch.py     # Operand layouts and launch planning
    │   └── autotune.py   # Tuning candidates and pruning
    ├── quantization/
    │   ├── fp8.py        # FP8 casts, transpose layouts and dequantization
    │   ├── delayed.py    # Delayed-scale casting and amax collection
    │   ├── launch.py     # Delayed-scale launch planning
    │   └── activation/
    │       ├── affine.py # Standalone affine 1/2/4/8-bit codec
    │       ├── boolean.py # Lossless one-bit boolean pack/unpack
    │       ├── fp8.py    # Fused FP8 cast and affine packing
    │       └── amax.py   # Amax activation codec kernels
    └── scaling/amax.py   # Amax reduction kernels
```

Each backend family contains its operation implementations. Shared CUDA
metadata belongs in [_shared/](../../src/deepopt/opt/backends/_shared/); numerical
scaling policy belongs in [opt/scaling/](../../src/deepopt/opt/scaling/).
Adding an executor means implementing its operation contract and explicit
dispatch branch, with tests for supported layouts and failure cases. There is
no automatic backend discovery or plugin registry.

## GEMM operand contract

Normalization kernels live in [triton/normalization/](../../src/deepopt/opt/backends/triton/normalization/),
with a [reference oracle](../../src/deepopt/opt/backends/reference/normalization/).
[Precision planning](../../src/deepopt/opt/precision/) chooses scale-aware FP8
edges once. [Normalization autograd](../../src/deepopt/opt/nn/autograd/normalization/)
owns packed xhat and saved statistics. See the [normalization guide](../normalization/README.md).

`gemm_nt(a, b, backend=..., out_dtype=..., bias=..., num_stages=...)` computes
`dequantize(a) @ dequantize(b).T`, followed by optional bias and output
conversion. For operands shaped `[M, K]` and `[N, K]`, the result is `[M, N]`.
Both operands use [QuantizedTensor](../../src/deepopt/opt/quantization/tensor.py):

- `data`: a nonempty 2D FP8 tensor, in E4M3FN or E5M2.
- `scale_inv`: FP32 reconstruction scales on the same device, shaped `[1, 1]`
  for tensorwise scaling or the rectangular block grid for blockwise scaling.
- `block_shape`: `None` for tensorwise scaling, or the logical rows and columns
  covered by each reconstruction scale.

The dispatcher does not quantize its inputs. Quantization and saved-activation
reconstruction happen before GEMM, preserving the required scales and layouts.
Backend implementations validate their own eligibility and output policies.

## Selection and lazy imports

Public forward configuration selects `backend="reference"` or `"triton"`.
`backward_backend="inherit"` retains that executor for backward; the optional
`"native_fast"` and `"native_precise"` choices use native scaled-FP8 GEMM for
eligible backward products, with different accumulation policies. They are
backward choices because the native executor is bias-free. Unsupported native
operand layouts or scale combinations deterministically use Triton; execution
errors are not caught and silently converted into another numerical policy.
See [native backward controls](../fp8/eager_linear_benchmark.md#native-backward-and-selective-preparation).

GPU implementations are imported when their operations execute. Importing the
package or configuring a CPU/reference recipe does not require Triton.
CPU quantization uses PyTorch reference operations. CUDA quantization uses the
shared Triton casting and reduction kernels even when the selected GEMM
executor is reference or native. GEMM backend selection therefore does not
select a separate casting implementation. The reference CUDA path is an
emulation path, with the CUDA quantization prerequisites described in the
[FP8 guide](../fp8/README.md#start-with-reference-execution).

Spatial Triton convolution gathers source positions inside each dot tile,
without an unfolded image allocation. Tensorwise and block-scaled operands use
FP8 dot products with FP32 accumulation. Blockwise reductions segment at source
scale boundaries and scale each FP32 partial; dW uses spatial stripes and
separate kernel taps. Fine source blocks can require very small padded partials.
Tensorwise pointwise operations reuse linear GEMM.
Adapters expose effective execution metadata and explain spatial fallback
from native scaled-GEMM backward selectors. See the
[convolution guide](../convolution/README.md) for layouts and measurement.

## Compatibility imports

New code should import the canonical packages above. Previous private module
paths remain available through small compatibility modules under `_legacy/`,
attached to their former package namespaces. These modules alias the canonical
module objects, so existing recipes and instrumentation share the same classes,
functions, and mutable module state. They do not duplicate kernel implementations.
Serialized class identities remain compatible with existing checkpoints.
