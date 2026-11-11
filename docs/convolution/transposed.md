# Transposed convolution

[Convolution guide](./README.md) · [Implementation plan](./transposed_plan.md) · [Configuration](../configuration/README.md)

`nn.ConvTranspose1d`, `nn.ConvTranspose2d`, and `nn.ConvTranspose3d` can be
prepared with the same independent FP8 and activation-compression settings as
ordinary convolution. Preparation preserves the original parameter objects,
including optimizer references. Shared modules, exclusions, training mode,
state restoration, grouped weights, bias, stride, dilation, numeric padding,
`output_padding`, and module `output_size` are supported.

```python
import torch
import deepopt
from deepopt.opt.policy import prepare_fp8_model

model = torch.nn.Sequential(
    torch.nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
    torch.nn.SiLU(),
).cuda()
settings = deepopt.FP8Config(
    enabled=True, backend="triton", trace_mode="modules",
    intercept_linear=False, scaling_recipe="current",
    backward_precision="auto",
)
model, report = prepare_fp8_model(
    model, settings,
    activation_compression={"enabled": True, "bits": 2, "group_size": 512},
)
with deepopt.autocast("cuda", dtype=torch.bfloat16, fp8=settings,
                      cache_enabled=False, check_finite=False):
    y = model(torch.randn(8, 512, 7, 7, device="cuda"))
    y.float().square().mean().backward()
```

Prepare the complete model before constructing DDP or capturing it. Existing
optimizers remain valid because parameters are shared. For a model with Python
shape guards, module conversion can succeed while precision analysis fails;
inspect `deepopt.opt.precision.precision_report(model)["fallbacks"]`. This
feature does not make arbitrary Python shape control flow traceable.

## Independent execution modes

| FP8 | Compression | Forward and saved state |
| --- | --- | --- |
| Off | Off | Native operation and native autograd |
| Off | On | Native computation, packed saved source used by dW |
| On | Off | Quantized FP8 operands, saved FP8 source used by dW |
| On | On | FP8 computation with fused source casting/packing where the recipe permits |

Compression does not change forward operands, dX operands, or db operands. It
approximates the saved input used for dW. No input is saved when dW is not
required. The BF16 path reconstructs the source to the native compute dtype;
the FP8 path reconstructs directly to FP8 and keeps the original FP8 scales.
Reconstruction currently uses one source-sized workspace for dW rather than
fusing unpacking into every overlapping matrix tile.

The existing global compression bits (1, 2, 4, or 8) and default group size 512
apply. Tensorwise FP8 follows the configured compression group size. Groupwise
FP8 follows the exact activation scaling rectangles. A source already supplied
as a compatible DO FP8 activation is consumed without another FP8 cast.

## Functional calls and precision

```python
from deepopt.opt.nn import functional as F
from deepopt.opt.policy import conv_plan

plan = conv_plan(settings, activation_compression={"enabled": True, "bits": 2})
y = F.do_conv_transpose2d(x, weight, bias, stride=2, padding=1,
                          output_padding=0, groups=1, dilation=1, plan=plan)
```

`conv_transpose1d/2d/3d` aliases are also available. Positional arguments follow
PyTorch, including the different position of `dilation` after `groups`.
`output_size` belongs to module calls; functional calls use `output_padding`.
An exact transposed module at the model root keeps its adapter and runtime
`output_size` API in every trace mode; its conversion report uses module mode.
Delayed scaling requires prepared module/call-site owners. Full FX conversion
and module-mode functional interception provide those owners.

`intercept_convolution` controls native functional transposed calls and inherits
`intercept_linear` when omitted. Explicit DO calls and prepared modules retain
their selected plans. Disable functional interception when using exclusions.

The preparation phase caches train/eval FP8 output and backward-edge decisions.
Eligible batched outputs can retain scaled FP8 storage for a following DO
operation. Otherwise the output follows the configured dense dtype, or active
autocast/input dtype when `out_dtype=None`. Unbatched calls support compute and
compression but return a dense output at the logical-layout boundary. Residual
additions, SE multiplication, and other native consumers remain boundaries.

Backward consumes compatible FP8 dY and can produce FP8 dX for eligible planned
edges. Fan-out/gradient accumulation remains conservative. Parameter gradients
use the original parameter dtype. Forward-output delayed scales have a separate
owner from input/weight scales, so they cannot overwrite each other's recipes.

## Backend implementation and limits

The native transposed weight layout is `[C_in, C_out / groups, *kernel]`.
The implicit executor uses the adjoint relationship to ordinary convolution:
transposed forward uses the ordinary input-gradient kernel; dX uses ordinary
forward; dW uses ordinary weight-gradient with the actual operands exchanged.
Each operand retains its own scale geometry and semantic quantization recipe.
Explicit bounds handle output padding admitted through dilation, even when
the inferred ordinary output would contain extra positions.

Spatial tensorwise and blockwise paths use FP8 dot operands and FP32
accumulation. They do not insert zero-filled input positions or materialize
unfolded windows. Eligible ungrouped singleton-kernel shapes use scaled GEMM.
Native backward backend selectors apply to that GEMM path; spatial gradients
use Triton, as reported by module `execution_info`.

Bias is fused into transposed spatial forward. FP8 output conversion is fused
when the cached output recipe fits the kernel epilogue. Current tensorwise
output scaling, incompatible block boundaries, or finite checking can require
a separate conversion. Weight-gradient split reductions use bounded partial
storage. These are not universal single-launch operations.

Current scaling supports the existing CUDA Graph capture workflow. The shared
`CUDAGraphCore` still rejects delayed-scaling models, which require eager
preprocessing; this feature does not change that runtime limitation.

## Verification and measurement

The tests cover native/reference values and gradients, all compression bit
widths, scale grids, explicit FP8 source reuse, unbatched inputs, output sizing,
partial gradients, fake CUDA metadata, Trainer accumulation, recomputation,
checkpoint restore, and two-rank Gloo state ownership. CPU execution of the
installed Triton bodies checks actual indexing, masks, FP8 dot operand types,
scale boundaries, bias and split reductions. It is not CUDA compilation.

Real GPU acceptance is in
[test_convolution_transpose.py](../../tests/cuda/test_convolution_transpose.py).
Run it on supported NVIDIA hardware; CUDA compilation, epilogue execution,
graph replay, and speedup remain unverified on the local CPU-only environment.

```bash
python -m unittest tests.cuda.test_convolution_transpose -v
python benchmarks/convolution/benchmark_convolution.py \
  --operation conv_transpose --shape 8x512x7x7 --out-channels 512 \
  --kernel 4 --stride 2 --padding 1 --output-padding 0 \
  --profile-dir work/transpose-profiles --output work/transpose-eager.json
python benchmarks/convolution/benchmark_convolution.py \
  --operation conv_transpose --shape 8x512x7x7 --out-channels 512 \
  --kernel 4 --stride 2 --padding 1 --cuda-graph --optimizer-step \
  --output work/transpose-graph.json
```

The benchmark uses the installed library and runs all four optimization
combinations. Timed steps include casting, compression, reconstruction and
layout handling. Graph setup/warmup is excluded; optional SGD runs outside the
captured model. Profiling is outside timed repetitions and reports kernel
counts plus identifiable spatial forward/dX/dW durations. Native/GEMM kernels,
bias reductions, casts and packing stay in a separately labeled bucket. Kernel
durations exclude CPU submission gaps; complete-step wall and CUDA-event
measurements expose that overhead. No speedup is assumed.
