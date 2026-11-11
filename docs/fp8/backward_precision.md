# Planned FP8 activation gradients

[FP8 guide](README.md) · [ReLU](../activation/README.md) · [Backend organization](../architecture/backends.md)

`backward_precision="auto"` enables one-time planning of activation-gradient
storage across DO operations. The default `"native"` retains existing output
behavior. Forward FP8 support and backward FP8 support are separate capabilities.

## Enable the policy

```toml
[runtime.fp8]
enabled = true
backend = "triton"
backward_precision = "auto"
grad_dtype = "e5m2"
```

Use the same option with model preparation or the Trainer:

```python
import torch
import deepopt
from deepopt.opt.policy import prepare_fp8_model
from deepopt.opt.precision import precision_report

settings = deepopt.FP8Config(
    enabled=True, backend="triton", backward_precision="auto", grad_dtype="e5m2",
)
model, report = prepare_fp8_model(model, settings, activation_compression=True)
print(precision_report(model)["backward_edges"])

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
trainer = deepopt.Trainer(
    model, optimizer, device="cuda", loss_fn=torch.nn.CrossEntropyLoss(),
    fp8=settings, execution_mode="max_speed",
)
```

Automatic output decisions require a prepared model. An autocast scope or a
standalone functional call alone does not supply the backward consumer graph.
Existing FP8 scopes and explicit native overrides still apply.

## Cached edge decisions

The existing preparation phase analyzes train/eval graphs once. It caches a
frozen [gradient edge plan](../../src/deepopt/opt/precision/gradient.py) for each
activation input and call site: storage precision, format, matrix layout,
scale blocks and recipe. Settings come from the backward consumer, which is
the preceding operation in forward. Shared module calls have independent slots.

Each forward snapshots its binding on the autograd context, including its
training flag and open delayed-scaling owner. Backward does not retrace or
require forward-thread ContextVars on CUDA workers. Parameterless models bind
gradient-state devices on their first tensor input without repeating analysis.

Model-input/parameter leaves, fan-out accumulation, unsupported operations,
opaque Python calls and unsupported permutations retain native gradients.
Exact identity relays can be traversed. Native leaf scopes isolate their
internal calls from outer functional call-site counters. Untraceable graphs
report a fallback and keep native outputs. Reconstruct and prepare the model
again after changing its structure or control-flow contract.

## Operation coverage

| Family | FP8 gradient input | Planned activation-gradient output |
| --- | --- | --- |
| Linear, HF Conv1D and functional projections | Reuse compatible data/scales | FP8 GEMM epilogue or current-scale workspace path |
| Conv1d/2d/3d | Reuse channel-matrix data/scales | Final spatial dX epilogue or workspace path |
| Pointwise convolution | Preserve convolution autograd boundaries while sharing Linear/GEMM math | Same transport contract as spatial convolution |
| BatchNorm2d / GroupNorm | Decode scales in local reductions/tiles | Retiled final dX kernel or workspace path |
| LayerNorm | Decode scales with suffix-shaped affine and xhat reductions | Complete-row FP8 dx, retiled epilogue or FP32 workspace path |
| ReLU | Decode locally or preserve compatible FP8 storage | Mask extraction, gating and casting together |
| Sigmoid / SiLU / GELU / tanh / ReLU6 | Decode locally with saved derivative/mask coefficients | Coefficient decoding, derivative application and fresh FP8 casting together where scale domains fit |
| Attention / MultiheadAttention | Native core boundary; projections use Linear math | Native core remains a declared boundary |
| Native compression-only operations | Native boundary | Native output |

FP8 gradients retain [scale-aware metadata](../../src/deepopt/opt/quantization/activation_tensor.py)
and logical floating-point autograd dtypes. Parameter, affine and bias gradients
retain existing dtypes, so optimizers and DDP reducers receive native tensors.
Saved-input/xhat compression and one-bit ReLU masks remain independent policies.

Linear and convolution skip ordinary incoming-gradient quantization when
compatible encoded gradients arrive. Bias reductions decode FP8 locally.
Rowwise dW still needs an independently scaled transpose of the incoming
gradient's represented values. The bounded CUDA path fuses decoding, transpose,
amax and casting. Other cases use a reported conversion stage. Producer-side
orientation pairs are not required: the consumer can perform this extra stage.

## Fusion and scaling

GEMM accumulations and normalization reductions remain FP32. Casting happens
after the complete activation gradient; split-K casts in its final sum.
Delayed scales allow direct epilogues. Current blockwise scales fuse where a
tile contains complete scale groups. ReLU and normalization retile their final
output around bounded scale blocks. GEMM prunes incompatible tile/split choices;
convolution checks channel-group alignment before selecting the epilogue.

Large current tensorwise groups or incompatible layouts use FP32 workspace and
the shared bounded reduction/cast path. No preliminary BF16 rounding is added.
This can add launches and be slower than native transport; measure the pipeline.
Finite diagnostics can select a checked workspace path and synchronize. Disable
finite checks for warmed fusion/CUDA Graph measurements.

Delayed gradient states belong to individual emitting call sites. Buffers are
registered on an existing operation without adding executable Sequential layers.
They move and checkpoint with the model. Observations use the existing update
window and distributed amax machinery; checkpoint replay does not double-update
them. Existing delayed-scaling full-training CUDA capture restrictions remain.

`backward_backend` is respected. DeepOpt's native scaled GEMM has no FP8 output
epilogue and remains an emission boundary. Spatial convolution's existing
implicit Triton fallback can use its epilogue.

`precision_report` exposes `backward_edges`, `backward_execution` and
`backward_input_execution`: input reuse, independent transpose stages, fused
outputs, extra cast/scale-reduction launches, workspace paths and fallback
reasons. Counts exclude mathematical reductions, diagnostics, accumulation and
batched delayed updates. CPU reports reference stages without claiming CUDA fusion.

## Verification and benchmarks

```bash
python -m unittest tests.opt.test_backward_precision tests.opt.test_gradient_kernels -v
python -m unittest tests.distributed.test_fp8_delayed_scaling -v
DEEPOPT_REQUIRE_CUDA_FP8=1 python -m unittest tests.cuda.test_backward_precision -v

python benchmarks/fp8/benchmark_backward_precision.py \
  --pipeline linear --output work/backward-linear.json
python benchmarks/fp8/benchmark_backward_precision.py \
  --pipeline conv2d --grad-block 1x32 --compression --output work/backward-conv.json
python benchmarks/fp8/benchmark_backward_precision.py \
  --pipeline linear --scaling-recipe delayed --output work/backward-delayed.json
```

The [benchmark](../../benchmarks/fp8/benchmark_backward_precision.py) imports the
installed library and can run outside the checkout. Matched native/auto cases
include complete forward/backward, preparation/conversions and gradient clearing.
Optimizer updates, transfers and warmup are excluded. Reports include repeated
wall/CUDA intervals, saved bytes, GPU allocation peaks, edge choices and gradient
error versus native transport. CUDA intervals include host submission gaps.

CPU tests check encoded-gradient oracles, kernel-body indexing/arithmetic,
checkpoint resume and one-time analysis. They do not validate CUDA compilation.
Run the GPU suite for epilogues, scale snapshots, profiling and graph replay.
Validate convergence before using a new gradient precision recipe in long runs.
