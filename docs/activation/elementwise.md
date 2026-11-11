# Sigmoid, SiLU, GELU, tanh and ReLU6

[Activation guide](README.md) · [Configuration reference](../configuration/README.md) · [Implementation plan](extension_plan.md)

The [functional module](../../src/deepopt/opt/nn/functional.py) provides `sigmoid`,
`silu`, `gelu`, `tanh`, `relu6` and corresponding `do_` aliases. Each supports
BF16/native or scale-aware FP8 inputs and outputs, saved-state compression and
planned FP8 activation gradients. Native FP32/FP16 inputs and CPU reference
execution are also available. CPU FP64 is the uncompressed numerical oracle.

## Use the shared policy

```python
import torch
from deepopt.opt.nn import functional as doF
from deepopt.opt.policy import activation_plan

plan = activation_plan(
    fp8={"enabled": True, "backend": "triton", "backward_precision": "auto"},
    activation_compression={"enabled": True, "bits": 2, "group_size": 512},
)
x = torch.randn(32, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = doF.gelu(x, approximate="tanh", plan=plan, output_dtype=torch.float8_e4m3fn)
y.backward(torch.ones_like(x))
```

GELU defaults to `approximate="none"` (erf-based exact GELU); `"tanh"` explicitly
selects the approximation. `relu6` clamps values to [0, 6], including values above
6. It does not zero values above the upper threshold.

All functions share ReLU's `plan`, `output_dtype`, `output_layout` (`last` or
`channels`) and `output_scale` arguments. FP8 outputs carry FP32 inverse scales
in `FP8Activation` while retaining a logical floating-point autograd dtype.
Raw float8 inputs use the existing unit-scale convention. Supply externally
scaled input through `FP8Activation`, not a bare float8 tensor.

`output_scale`, when provided, is a positive finite contiguous FP32 quantization
multiplier on the selected output scale grid. It has no gradient. Current
scaling otherwise measures post-activation values. Smooth activations compute
fresh scales; input scales are not generally valid for nonlinear outputs.

## Compression saves derivatives

Compression operates on preprocessed backward coefficients, computed in FP32
from represented input values before output rounding:

| Function | Saved coefficient |
| --- | --- |
| Sigmoid | `s * (1 - s)`, with `s = sigmoid(x)` |
| SiLU | `s * (1 + x * (1 - s))` |
| GELU | Local derivative of the selected exact or tanh formula |
| Tanh | `1 - tanh(x)**2` |
| ReLU6 | Exact threshold-backward boolean mask; false at both 0 and 6 |

Compression-off smooth execution saves unquantized FP32 coefficients (FP64 for
the CPU FP64 oracle). Compression-on saves affine integer words plus FP32
minimum/step metadata. It retains neither raw input nor a dense coefficient
shadow. SiLU/GELU coefficients can be negative or exceed one; encoding preserves
their actual group range. Their gradients are approximate when compression is
enabled. ReLU6 uses an exact one-bit mask regardless of configured bit width.
NaN threshold behavior follows native ReLU6; the finite-input interior predicate
alone does not define its NaN case.

The existing section configures every activation:

```toml
[runtime.activation_compression]
enabled = true
bits = 2
group_size = 512

[runtime.fp8]
enabled = true
backend = "triton"
backward_precision = "auto"
```

Widths 1, 2, 4 and 8 are supported. BF16 and tensorwise FP8 use `group_size`;
blockwise FP8 uses the operation's activation scaling rectangles instead. Tails
do not affect group extrema. Affine coefficient metadata is independent of
FP8 input/output scales and incoming-gradient scaling. `attention_bits` affects
only attention Q/K/V, not these coefficients. Inference saves no backward state.

At the same input/output precision, compression leaves forward values unchanged.
The finite-input contract of the affine codec applies to compressed smooth
coefficients. Strict diagnostics reject nonfinite coefficients rather than
silently encoding them as finite values; unchecked execution assumes finite
codec inputs.

## Prepared models and precision planning

Exact native modules become
[DOSigmoid, DOSiLU, DOGELU, DOTanh and DOReLU6](../../src/deepopt/opt/nn/activation/elementwise_modules.py).
Preparation preserves shared modules, training state and GELU approximation.
Full FX transformation rewrites recognized native functional calls and supported
tensor methods. Module preparation and DeepOpt scopes also route recognized
functional call sites. Explicit DO functionals participate in cached planning.
Native subclasses with custom behavior are not converted as exact native types.

```python
import torch
import deepopt
from torch import nn

model = nn.Sequential(nn.Linear(768, 3072), nn.GELU(), nn.Linear(3072, 768)).cuda()
trainer = deepopt.Trainer(
    model, torch.optim.AdamW(model.parameters(), lr=1e-4), device="cuda",
    loss_fn=nn.MSELoss(), precision="bf16", execution_mode="max_speed",
    config={"runtime": {
        "fp8": {"enabled": True, "backend": "triton", "backward_precision": "auto"},
        "activation_compression": {"enabled": True, "bits": 2},
    }},
)
```

Forward and backward decisions are analyzed once per train/eval call-site
contract. Eligible activation → Linear/convolution edges carry FP8 directly;
the reverse edges can carry newly scaled FP8 `dx`. Native consumers, leaves,
fan-out accumulation and unsupported layouts keep their existing boundaries.
An isolated functional call has no consumer graph and cannot invent a planned
FP8 gradient edge. See [backward planning](../fp8/backward_precision.md).

Supported native in-place forms retain alias/version checks. BF16 sigmoid/tanh
tensor in-place forms, SiLU and ReLU6 work without changing representation.
In-place FP8 ReLU6 can preserve its existing scale/layout/format. Smooth FP8
in-place calls that would require fresh scales are rejected before mutation;
use out-of-place execution for representation changes. GELU has no in-place API.
Only first-order custom backward is supported.

## Fusion, execution and measurement

The [Triton backend](../../src/deepopt/opt/backends/triton/activation/) fuses
activation math, output conversion and saved coefficient/mask writes. Backward
decodes packed coefficients and FP8 `dy` inside its compute tile and writes
native or planned FP8 `dx`, without a full unpacked coefficient buffer.
Explicit/delayed scales and compatible bounded local scale groups allow direct
epilogues. Large global groups and incompatible group partitions use reported
bounded reduction or FP32 workspace stages; a global reduction is not a promise
of one launch. Diagnostics can add synchronization.

Current scaling and fixed explicit scales support compatible warmed CUDA Graph
capture/replay. Owned delayed scales use post-activation amax and frozen
accumulation windows, DDP synchronization and `deepopt.fp8_checkpoint` scale-tape
recomputation. Existing eager delayed-update/full-training capture restrictions
remain. Custom activation dispatch stays outside Dynamo; surrounding operations
may compile, and planned wrappers can cause further eager fallback. Fullgraph
compilation is not supported across these boundaries.

Adapters expose `execution_info`, including activation name, saved-state bytes,
scale policy, forward/scaling/coefficient-reduction launches and expected native
backward launches. Planned gradient reports separately record fused epilogues,
workspace casts and their stages. Counts exclude diagnostics and autograd's
gradient accumulation/clearing.

The [standalone installed-library benchmark](../../benchmarks/activation/benchmark_activations.py)
covers all functions, both GELUs, precision combinations, compression, complete
forward/backward and eager/graph measurements. Preparation of operation input
FP8 values is explicitly outside timing; output scaling and derivative packing
are inside timing. Pipeline modes include input casting and planned gradients.

```bash
python benchmarks/activation/benchmark_activations.py --execution both --output work/activations.json
python benchmarks/activation/benchmark_activations.py --activations silu gelu_tanh --pipeline linear --shape 4096x768
python benchmarks/activation/benchmark_activations.py --activations relu6 tanh --pipeline conv --shape 8x64x32x32
python benchmarks/activation/benchmark_activations.py --device cpu --shape 3x33 --warmup 0 --iterations 1 --repeats 1

python -m unittest tests.opt.test_activations tests.opt.test_activation_kernels -v
DEEPOPT_REQUIRE_CUDA_ACTIVATIONS=1 python -m unittest tests.cuda.test_activations -v
```

CPU kernel-body interpretation validates addressing and arithmetic but cannot
establish real CUDA compilation, capture correctness or throughput. Run the CUDA
acceptance suite and benchmark on the intended hardware before drawing performance
conclusions.
