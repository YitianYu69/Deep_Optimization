# Normalization and planned FP8 activation edges

[Documentation](../README.md) · [FP8](../fp8/README.md) · [Activation compression](../activation_compression/README.md)

See [LayerNorm](layer_norm.md) for suffix-aware normalization, fused saved-xhat
compression and forward/backward FP8 transport. The
[extension plan](layer_norm_plan.md) records its design; the sections below
describe BatchNorm2d and GroupNorm.

Optional [compressed xhat mean correction](#compressed-xhat-mean-correction)
is produced during packing and applied inside backward decoding with explicit
original-rstd access. The [design contract](xhat_recenter_plan.md) describes the
math, fusion and acceptance requirements.

BatchNorm2d and GroupNorm accept BF16, FP32/FP16 and scale-aware FP8 input.
Statistics, inverse standard deviation and backward reductions use FP32. FP8
describes activation storage and transport, not low-precision statistics.

## Configuration and preparation

```toml
[runtime.fp8]
enabled = true
backend = "triton"
trace_mode = "modules"
intercept_linear = false

[runtime.activation_compression]
enabled = true
group_size = 512
bits = 2
norm_xhat_recenter = false
```

The existing global compression settings apply to saved normalized values;
supported widths are 1, 2, 4 and 8. Attention's override does not affect norms.
Compression alone also prepares normalization adapters when FP8 is disabled.
Affine parameters and BatchNorm buffer identities/state-dictionary keys are
preserved. BatchNorm retains training/evaluation, optional affine/bias,
`track_running_stats=False`, and cumulative averaging with `momentum=None`.
GroupNorm always uses per-sample group statistics. SyncBatchNorm is not converted.

## Compressed xhat mean correction

Set `runtime.activation_compression.norm_xhat_recenter = true` alongside
`enabled = true` to preserve the original normalized tensor's first moment in
compressed backward state. It applies to BatchNorm2d, GroupNorm and LayerNorm,
with BF16 or FP8 execution. The default is `false`; other operations ignore it.

The packing kernel decodes its exact output words in registers, sums
`decoded_xhat - original_xhat`, and records one FP32 shift per normalization
domain. A complete-domain tile finishes inline with no added launch. Domains
spanning programs use fixed-order, bounded reductions of error sums and valid
counts. There is no dense CUDA recentering pass or unpacked xhat allocation.
The CPU reference computes the same correction with ordinary tensor operations.

All backward xhat consumers subtract that shift in FP32: weighted dot products,
dx epilogues and affine-weight reductions. They explicitly load the original
saved `rstd`, which validates the domain and supplies dx scaling. It does not
perform the subtraction or rescale xhat. Unchecked zero/nonfinite rstd preserves
decoded xhat; finite diagnostics reject invalid statistics when enabled.
BatchNorm evaluation preserves the original, possibly nonzero xhat mean.
Input-only evaluation backward and bias-only backward need no correction state.

Forward values, FP8 output scales, packed words and codec metadata stay unchanged.
One FP32 shift is retained per channel for BatchNorm, sample/group for GroupNorm,
or outer row for LayerNorm. Compressed weight-only LayerNorm additionally retains
the original FP32 rstd. No new centering collective or running buffer is added.
Saved tensors retain autograd lifetime/version checking. Old configs normalize
the missing option to `false`; changing the policy requires a new training run
instead of exact checkpoint resume.

Execution reports expose `norm_xhat_recenter`, `recenter_state_bytes`,
`recenter_rstd_bytes`, `recenter_extra_rstd_bytes`, `recenter_scratch_bytes` and
`recenter_launches`. `recenter_partial_launches` counts shared packing launches
that emit partials, `recenter_partial_tiles` counts their programs, and
`recenter_finish_launches` counts the added reduction launches. Scratch bytes
count allocated partial buffers, including
intermediate reductions; they are separate from retained saved state. Shared
packing/backward launches are not counted again as extra correction launches.
Mean preservation alone does not guarantee lower total gradient error or better
convergence.

The installed-library benchmark includes metadata production, reductions and
backward loads in its timings:

```bash
python benchmarks/normalization/benchmark_normalization.py --op layer_norm --shape 32x197x768 --normalized-shape 768 --pipeline none --norm-xhat-recenter both --execution both --output work/norm-recenter.json
```

The same flag works for BatchNorm2d and GroupNorm eager measurements. Reported
mean errors use an independent CPU codec diagnostic outside timing. CUDA
acceptance, including launch-count and changing-value graph replay tests, runs
with `DEEPOPT_REQUIRE_CUDA_NORM_RECENTER=1 python -m unittest tests.cuda.test_norm_xhat_recenter -v`.
CPU reference/interpreted JIT checks do not establish CUDA compilation or speed.

```python
import torch
from torch import nn
from deepopt import Trainer, FP8Config
from deepopt.opt.precision import precision_report

model = nn.Sequential(
    nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 128, 3, padding=1),
).cuda()
trainer = Trainer(
    model, torch.optim.AdamW(model.parameters(), lr=1e-4),
    device="cuda", precision="bf16", loss_fn=nn.CrossEntropyLoss(),
    fp8=FP8Config(enabled=True, backend="triton", intercept_linear=False),
    config={"runtime": {"activation_compression": {"enabled": True, "bits": 2}}},
)
print(precision_report(trainer.model))
```

## One-time precision planning

Trainer preparation analyzes producer/consumer call sites before compilation
and DDP ownership. One preparation phase inspects train/eval graphs and caches
output choices. There is no tracing or consumer-capability analysis in later
forwards, batches or epochs. Matching explicit preparation is reused by Trainer.
Changing model structure or precision policy requires preparing a fresh model.
Batch statistics and FP8 scaling observations still update during training.

Normalization emits FP8 when a downstream prepared norm, linear or convolution
consumer can read its layout/scales. ReLU and identity relays can preserve FP8.
Unsupported consumers, incompatible layouts and untraceable paths receive
native precision. Full GraphModules contain explicit conversion nodes; ordinary
Python roots keep their class and methods, with cached per-call-site choices and
activation dispatch at native boundaries. Shared norms can have different choices
at different call sites. Standalone calls outside the planned root return dense
values. Reports record `analyses=1`, edge decisions and trace fallback reasons.
Delayed scaling retains the existing whole-model CUDA capture restriction.

## Saved xhat and fused kernels

```text
xhat = (x - mean) * rsqrt(variance + eps)
y = weight * xhat + bias
```

Compression packs **xhat before affine**, directly from FP32 normalized values.
Groups stay within a normalization domain: one channel's N×H×W values for
BatchNorm, or one sample/channel-group for GroupNorm. The configured group size
is an upper bound; short domains use shorter groups and tails are masked. FP8
input/output scales and xhat affine-codec metadata have separate meanings.

When compressed xhat is needed by backward, saved tensors contain packed words,
FP32 codec metadata, original inverse standard deviation and necessary affine
parameters. The normalization node saves no raw x or full uncompressed xhat.
Without compression it retains x/statistics and reconstructs xhat in backward.
Eval BatchNorm input-only gradients do not require xhat.

Triton forward normalizes, applies affine parameters and packs xhat in the same
tile. Statistics and backward sums use bounded hierarchical scratch reductions.
Large compression groups use partial min/max and chunked encoding. Backward
decodes xhat inside tiles without a full decoded activation allocation. Strided,
NCHW and channels-last inputs are supported. Ordinary output memory formats are
retained; FP8 outputs use the consumer's channel or last-axis matrix layout.
Output FP8 casting uses the shared quantizer after fused normalization/packing;
its cost belongs in complete-step measurements.

Decoded xhat approximates dX and dweight; dbias does not depend on xhat.
Compression does not change forward values or running-statistic updates at the
same input/output precision. GroupNorm applies channel weights before its group
backward reductions. Lower quantization error and convergence need workload
validation. First-order gradients are supported.

## Functional interfaces and storage

```python
from deepopt.opt.nn import functional as F
from deepopt.opt.policy import normalization_plan
from deepopt.opt.quantization.activation_tensor import to_fp8_activation

plan = normalization_plan(False, activation_compression=True)
y = F.group_norm(x, num_groups=8, weight=weight, bias=bias, plan=plan)
fp8_x = to_fp8_activation(x, layout="channels", block_shape=(1, 32))
y = F.group_norm(fp8_x, num_groups=8, weight=weight, bias=bias, plan=plan)
```

The functional module also exposes `batch_norm`, `do_batch_norm` and
`do_group_norm`. Batch normalization currently covers 4D BatchNorm2d.
Unprepared functional calls can use `deepopt.autocast`; optional
`intercept_normalization` inherits `intercept_linear` when omitted.

`FP8Activation` carries bytes/scales without a dense shadow. Its logical dtype
describes its autograd interface; inspect `quantized().data` or the precision
report for physical storage. Compatible linear/convolution consumers reuse its
payload without recasting input. Raw float8 tensors are unit-scale numeric
inputs; externally scaled values must carry their metadata.

## Verification and measurement

```bash
python -m unittest tests.opt.test_normalization tests.opt.test_normalization_kernels -v
DEEPOPT_REQUIRE_CUDA_FP8=1 python -m unittest tests.cuda.test_normalization -v
python benchmarks/normalization/benchmark_normalization.py \
  --op batch_norm --shape 16x64x56x56 --pipeline conv --output normalization.json
python benchmarks/normalization/benchmark_normalization.py \
  --op group_norm --shape 16x64x56x56 --groups 8 --bits 4 --pipeline none
```

The portable benchmark imports the installed library. It times BF16/FP8 input
casting, statistics, output casting, packing, reconstruction and backward, with
compression off/on. Reports include timing, saved logical bytes, allocator peaks,
precision decisions and errors versus compression off. No optimizer, compilation
or graph replay is timed. CUDA event intervals include submission gaps; saved
logical bytes do not deduplicate aliases. CPU kernel-body tests check addressing,
scales and reduction math, not CUDA compilation. Real GPU tests cover kernels,
large groups and graph replay.
