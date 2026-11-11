# Optimized activation functions

[Documentation](../README.md) · [Normalization](../normalization/README.md) · [Activation compression](../activation_compression/README.md)

See [sigmoid, SiLU, GELU, tanh and ReLU6](elementwise.md) for their precision,
compressed-derivative and runtime APIs. The [implementation plan](extension_plan.md)
records the design. The sections below describe ReLU's exact saved-mask path.

DeepOpt ReLU accepts BF16 and scale-aware FP8 activations, with BF16 or FP8
output. FP32/FP16 native inputs are also supported. ReLU uses an exact saved
mask for first-order backward. Enabling activation compression packs that
mask to one bit per element without changing its values or gradients.

## Functional calls

The common [functional module](../../src/deepopt/opt/nn/functional.py) exposes
`relu` and `do_relu`. `relu_plan` uses the existing independent precision and
activation-compression configuration:

```python
import torch
from deepopt.opt.nn import functional as doF
from deepopt.opt.policy import relu_plan

plan = relu_plan(
    fp8={"enabled": True, "backend": "triton"},
    activation_compression={"enabled": True},
)
inputs = torch.randn(64, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
output = doF.relu(inputs, plan=plan, output_dtype=torch.float8_e4m3fn)
output.backward(torch.ones_like(inputs))
```

FP8 output is an `FP8Activation`: FP8 data and FP32 inverse scales, with an
ordinary floating-point autograd interface. BF16 output is an ordinary tensor.
Raw float8 input follows the unit-scale numeric convention; externally scaled
FP8 values must carry their scales through `FP8Activation`.

`output_dtype` explicitly selects storage precision. With no explicit dtype,
standalone calls preserve BF16 input or compatible FP8 input. `output_layout`
selects `last` (matrix `[-1, last_dim]`) or `channels` (matrix `[N*spatial, C]`).
The selected plan supplies the FP8 format and scale-block configuration for
automatically planned outputs. Compression-only calls use `relu_plan(fp8=False,
activation_compression=True)` and preserve native compute precision.

An explicit native plan owns its compression policy. Prepared FP8 plans retain
scoped FP8/compression overrides, consistent with the projection APIs.

## Forward and backward fusion

The [Triton activation backend](../../src/deepopt/opt/backends/triton/activation/)
owns the numerical kernels. The former quantization relay is a compatibility
entry point into this implementation.

| Path | Forward work |
| --- | --- |
| BF16 → BF16 | Load, ReLU and optional saved mask |
| BF16 → FP8 | Load, ReLU, FP8 conversion and optional saved mask |
| FP8 → BF16 | Local scale decoding, ReLU, BF16 conversion and optional saved mask |
| FP8 → FP8 | ReLU and optional saved mask, preserving compatible scales or requantizing |

ReLU output and the mask are written in one fused kernel. The mask predicate
is computed before output quantization: a positive input that rounds to FP8
zero still passes its upstream gradient. BF16 forward preserves NaNs and signed
zero; the backward predicate follows PyTorch threshold-backward behavior,
including its NaN case. Inactive elements select zero instead of multiplying
by zero, so inactive NaN/infinite upstream gradients are suppressed correctly.

Without compression the saved tensor is `torch.bool`. With compression it is
a flat `int32` payload containing 32 mask bits per word, low bit first, with
zero tail bits. The mask follows the chosen matrix order; small shape/layout
metadata on the autograd context restores logical axes. Mask width is always
one bit, independently of global activation `bits`, `attention_bits` and
`group_size`. Backward retains neither the full input nor the full output.
Inference and inputs without required gradients allocate no mask.

One backward kernel reads the upstream gradient, decodes FP8 scales if needed,
extracts mask bits directly, gates the gradient, and writes its required dtype.
There is no full unpacked-mask or dense FP8 reconstruction buffer. Compatible
FP8 upstream gradients can retain their scales and physical FP8 storage;
otherwise the input's logical gradient dtype is returned. Like the existing
optimized normalization kernels, this custom path supports first-order
backward; higher-order derivatives are not supported.

Prepared models can also select fresh FP8 `dx` from BF16 upstream gradients
through the shared [backward precision planner](../fp8/backward_precision.md).
The decision is cached for the backward consumer and is independent of the
ReLU mask-compression setting.

## FP8 scales and launch counts

Compatible FP8 input reuses its inverse scales. Small current tensorwise groups
(up to 4096 values) and power-of-two blockwise groups fitting a bounded program
also compute their scales inside the fused forward kernel. Blockwise programs
cover at most 16,384 padded values. For unaligned row tails, the same kernel
assigns disjoint flat mask words to programs and reloads their predicate inputs
to avoid racing partial-word stores.

Large tensorwise or blockwise groups require bounded amax reductions before
conversion. Those reductions inspect post-ReLU values directly, without
materializing a dense ReLU result. The fused output/mask kernel follows them.
The scaling recipe is preserved; a multi-program global reduction is not
advertised as a single launch.

For a precomputed quantization multiplier, pass a contiguous FP32
`output_scale` tensor on the input device. Its shape is `[1, 1]` for tensorwise
scaling or the output matrix's rectangular scale grid. ReLU, output casting,
inverse-scale writing and mask generation then share one launch. Supplied
scales must be positive and finite. Finite checks can add reductions and host
synchronization; disable them for the documented graph/performance path.

Prepared delayed-scaling adapters use their owned scale snapshot inside the
same kernel and observe post-ReLU amax. Checkpoint replay uses the existing
scale tape and does not double-update state. Existing delayed-scaling eager
update and full-training CUDA capture restrictions still apply.

`DOReLU.execution_info` reports output precision, mask bytes, scale mode,
forward/scaling launch counts and the expected backward launch count. CUDA
counts exclude finite-check diagnostics, autograd accumulation and gradient
clearing. The CPU reference reports zero CUDA launches.

## Model preparation, scopes and in-place calls

Model preparation converts exact `nn.ReLU` modules into
[DOReLU](../../src/deepopt/opt/nn/activation/modules.py), preserving sharing,
training mode and `inplace`. Full FX conversion also supports `F.relu`,
`torch.relu`, tensor `.relu()` and their in-place variants. In module mode,
functional call-site adapters are installed when `intercept_linear` is enabled.
Within DeepOpt scopes, ordinary BF16 ReLU calls use this path when compression
is enabled; scoped FP8 interception follows `intercept_linear`. Scale-aware FP8
wrappers route ordinary ReLU calls through the same autograd entry point.

The cached precision planner analyzes ReLU consumers in the same one-time
preparation phase as normalization. Eligible linear/convolution consumers can
receive FP8 directly; native consumers receive reconstructed native values at
the boundary. Shared module call sites and train/eval paths have independent
cached choices. The Trainer does not retrace each iteration.

In-place ReLU preserves its input object and storage representation. It cannot
change dtype, FP8 scales or FP8 layout. Leaf/view mutation errors are checked
before writing, and mutations invalidate prior saved tensors through version
counters. Overlapping broadcast views are rejected. Explicit representation
conversion uses out-of-place ReLU. When an in-place FP8 result reaches a native
consumer, reconstruction occurs at that consumer boundary.

## Verification and measurement

```bash
python -m unittest tests.opt.test_relu tests.opt.test_relu_kernels -v
DEEPOPT_REQUIRE_CUDA_RELU=1 python -m unittest tests.cuda.test_relu -v

python benchmarks/activation/benchmark_relu.py \
  --shape 4096x768 --output work/relu.json
python benchmarks/activation/benchmark_relu.py \
  --shape 4096x768 --fp8-block 1x32 --output work/relu-blockwise.json
python benchmarks/activation/benchmark_relu.py \
  --device cpu --shape 3x33 --iterations 10 --output work/relu-cpu.json
```

The [standalone benchmark](../../benchmarks/activation/benchmark_relu.py) can be
copied outside the checkout and imports the installed library. It measures
all BF16/FP8 input-output combinations with compression off/on, including
output preparation, forward/backward and gradient clearing. Input FP8
preparation is outside timing. Reports include saved mask bytes, forward
quantization error, exact mask-gradient comparisons and native BF16 timing.
CUDA stream intervals include host submission gaps.

The [normalization benchmark](../../benchmarks/normalization/benchmark_normalization.py)
with `--pipeline conv` measures normalization → ReLU → convolution and includes
the ReLU execution report. It times the pipeline's input casting and full
backward, providing a broader measurement than the isolated operation.

CPU tests interpret installed JIT bodies for addressing and arithmetic checks;
they do not establish CUDA compilation or performance. Real CUDA tests cover
the precision combinations, kernel counts, masks, in-place mutation and warmed
graph replay. Enable the environment flag above to fail rather than skip when
the required CUDA prerequisites are absent.
