# Spatial convolution optimization

[Documentation index](../README.md) · [FP8](../fp8/README.md) · [Activation compression](../activation_compression/README.md) · [Benchmarks](../../benchmarks/README.md)

[Transposed Conv1d/2d/3d](./transposed.md) supports the same independent FP8
and activation-compression controls; see its guide for output sizing and
precision-planning behavior.

`conv1d`, `conv2d`, and `conv3d` use the same independent FP8 and
saved-activation compression settings as linear operations. These are spatial
convolutions with PyTorch weight layouts; the Hugging Face `Conv1D` adapter
remains a linear projection.

| FP8 | Activation compression | Computation |
| --- | --- | --- |
| Off | Off | Native PyTorch convolution |
| Off | On | Native convolution, compressed saved source for the weight gradient |
| On | Off | FP8-quantized forward, input-gradient and weight-gradient operands |
| On | On | FP8 computation with fused source casting and activation packing |

Compression leaves forward computation, input gradients and bias gradients
unchanged relative to the same path with compression disabled. It reconstructs
the saved input used for the weight gradient, so the weight gradient becomes
approximate. FP8 alone changes operands relative to native BF16. Neither
optimization promises a speedup for every convolution shape.

## Configuration and functional calls

```toml
[runtime.fp8]
enabled = true
backend = "triton"
trace_mode = "modules"

[runtime.activation_compression]
enabled = true
group_size = 512
bits = 2
```

The global `bits` and `group_size` settings apply to convolution inputs.
`attention_bits` affects only attention Q/K/V and has no effect on convolutions.
Tensorwise FP8 uses the configured compression group size. Groupwise FP8 uses
the exact activation FP8 scaling rectangles for compression, overriding
`group_size`.

The functional API mirrors PyTorch's positional convolution arguments, adding
an optional execution plan:

```python
import torch
from deepopt.opt.nn import functional as F
from deepopt.opt.policy import conv_plan

plan = conv_plan()  # Resolve the loaded configuration once.
x = torch.randn(8, 64, 56, 56, device="cuda", requires_grad=True)
weight = torch.randn(128, 64, 3, 3, device="cuda", requires_grad=True)

with torch.autocast("cuda", dtype=torch.bfloat16):
    y = F.conv2d(x, weight, stride=1, padding=1, plan=plan)
y.sum().backward()
```

`do_conv1d`, `do_conv2d`, and `do_conv3d` name the same dispatch entry points.
Use `conv_plan(False, activation_compression=True)` to select native computation
with compression explicitly. Both ordinary and unbatched inputs are accepted.
Numeric padding, `"same"`, `"valid"`, stride, dilation and groups retain their
PyTorch meanings. Module padding modes such as `reflect` pad the input before
convolution, matching native module behavior.
Complex operands use native convolution. Optimized paths require dense real
FP32, BF16, or FP16 operands and support first-order gradients. Transposed
convolution is outside this API.

## Model preparation and scoped calls

```python
import deepopt
import torch
from deepopt.opt.policy import prepare_fp8_model

model = torch.nn.Conv2d(64, 128, 3, padding=1).cuda()
settings = deepopt.FP8Config(
    enabled=True, backend="triton", trace_mode="modules", intercept_linear=False,
)
model, report = prepare_fp8_model(
    model, settings,
    activation_compression={"enabled": True, "group_size": 512, "bits": 2},
)
```

Preparation replaces supported exact `torch.nn.Conv1d`, `Conv2d` and `Conv3d`
modules with adapters sharing the original parameters. Shared modules and
state-dictionary parameter names remain shared. Module exclusions apply as
they do for linear operations; disable functional interception when selecting
module exclusions so scoped calls cannot re-enable excluded operations.
Prepare before compilation, DDP wrapping or
CUDA Graph capture. Full FX tracing can also rewrite functional convolution
calls.

Unprepared functional calls can use a `deepopt.autocast` scope. Convolution
interception follows `intercept_linear` by default; the optional
`intercept_convolution` field overrides convolution interception independently.
For example, set `intercept_linear=False, intercept_convolution=True` to
intercept convolution functions while leaving linear interception disabled.
Explicit prepared modules and execution plans do not require interception.

## Saved layout and backend behavior

The canonical source matrix is `[batch * input_spatial, input_channels]`.
Weight matrices are `[output_channels, input_channels_per_group * kernel_volume]`;
output-gradient matrices are `[batch * output_spatial, output_channels]`.
Scaling and compression refer to these matrices. Noncontiguous inputs and
channels-last tensors are converted according to their logical dimensions.
The cost of these conversions belongs in complete-step measurements.

Tensorwise pointwise convolutions with kernel size 1, stride 1, zero padding
and `groups=1` reuse the existing FP8 linear path. Block-scaled pointwise and
spatial convolutions use implicit patch
addressing: kernels generate source addresses as needed, without storing a
full unfolded activation tensor. Source casting and packing happen once per
unique source element. Padding contributes literal zero to convolution and
overlapping input-gradient contributions are accumulated. Synthetic right
padding for asymmetric `"same"` also stays literal zero after reconstruction;
affine decoding cannot turn these padding cells into activation values.

The reference backend dequantizes operands for native reference convolution;
it is a correctness implementation, not an FP8 hardware performance path.
Complex-valued inputs or parameters retain native convolution computation and
gradients, with compression disabled; execution metadata reports that fallback.
The CUDA backend has separate forward, input-gradient and weight-gradient
kernels. Both tensorwise and block-scaled operands use FP8 dot products with
FP32 accumulation. Block-scaled reductions stop at each operand's scale
boundaries; each FP8 partial product is scaled before accumulation. Saved FP8
values and their original scale grid are retained without dequantizing the dot
operands or requantizing them into another layout.

Forward fixes a spatial kernel tap and reduces input channels. Input gradients
reduce output channels for each tap. Weight gradients assign each tap to a CTA
and reduce spatial stripes, stopping at gradient-row and gathered source-row
scale boundaries. This handles independently configured activation, weight and
gradient blocks, including scale groups misaligned with convolution groups.
Small reduction segments are zero-padded to the FP8 instruction tile width.
In particular, a source block with one spatial row can limit a weight-gradient
partial to one position. Such fine groups can be substantially slower despite
using FP8 multiplication. Grouped and depthwise operations may also favor native
convolution; native FP8 arithmetic is not a throughput guarantee.

Current scaling supports tensorwise and independent activation, weight and
output-gradient block scaling. Delayed
scaling is tensorwise and needs a prepared module to own persistent scale
state; stateless functional calls cannot create delayed history implicitly.
Scale snapshots are retained for backward reconstruction. Native scaled-GEMM
backward choices apply to the pointwise linear path; general spatial
convolutions use their convolution backward implementation.

Prepared convolution adapters expose an `execution_info` snapshot after a
forward call. It reports the effective scoped precision/compression policy,
the selected forward and backward executors, pointwise eligibility and any
backward-backend fallback. Reading or changing the returned dictionary does
not change the adapter's execution plan. Delayed-scale adapters and FX call
sites reject whole-model CUDA capture before CUDA initialization; delayed
preprocessing still requires eager execution.

## Validation and measurement

CPU contracts compare compressed weight gradients against an explicit
pack/decode reference, and FP8 results against independently quantized
convolution operands. The CUDA suite validates real kernel execution and
skips explicitly when compatible hardware or Triton is missing:

```bash
python -m unittest tests.opt.test_convolution tests.opt.test_convolution_native tests.opt.test_convolution_blockwise_fp8 -v
DEEPOPT_REQUIRE_CUDA_FP8=1 python -m unittest tests.cuda.test_convolution_fp8 tests.cuda.test_convolution_blockwise_fp8 tests.cuda.test_convolution_graph -v
```

The blockwise CPU checks execute the installed JIT function bodies with a torch
interpreter, asserting FP8 dot operands and checking indexing/scaling against
independent dequantized convolution results. They do not compile CUDA. The GPU
suite additionally checks generated matrix instructions for FP8 formats and
rejects a TF32 multiplication fallback.

Run the standalone benchmark against the installed library:

```bash
python benchmarks/convolution/benchmark_convolution.py \
  --shape 8x64x56x56 --out-channels 128 --kernel 3 --padding 1 \
  --output convolution.json

python benchmarks/convolution/benchmark_convolution.py \
  --shape 2x32x16x24x24 --out-channels 64 --kernel 3 --padding 1 \
  --layout channels_last --bits 4 --output convolution3d.json

python benchmarks/convolution/benchmark_convolution.py \
  --shape 8x64x56x56 --out-channels 128 --kernel 3 --padding 1 \
  --fp8-block 32x32 --fp8-weight-block 32x32 --fp8-grad-block 32x32 \
  --output convolution-blockwise.json
```

Each selected BF16 or FP8 path runs with compression off and on. Reports
include full forward/backward wall time, CUDA event intervals, logical saved
tensor payload, peak additional allocated memory, output/gradient errors and
execution metadata distinguishing tensorwise FP8 from blockwise FP8 reduction.
Timed work includes FP8 preparation, packing, reconstruction and layout
handling. Warmup and diagnostic hooks are outside throughput timing. There is
no optimizer, `torch.compile` or CUDA Graph replay.

CUDA event intervals include stream submission gaps; they are not sums of
kernel durations. Logical saved-tensor bytes do not deduplicate aliases, and
allocator peaks include gradients and temporary workspaces. Numerical errors
describe individual steps rather than training convergence. Use matched
shapes, layouts, cuDNN settings and scaling recipes when comparing results.
