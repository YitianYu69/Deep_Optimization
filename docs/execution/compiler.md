# DeepOpt model compiler

[Execution guide](README.md) · [Native AutoDiff](compiler_native_autodiff.md) · [Operation backends](../architecture/backends.md)

The optional `deepopt` compiler lowers FX graphs to a validated semantic IR,
derives supported backward graphs with DeepOpt's native `AOTAutoDiff`, then
schedules and generates both executable graphs. BF16/FP8 projections, saved
compression, activations, normalization, spatial operations and attention reuse
shared numerical kernels. Unsupported combinations retain explicit compatibility
boundaries or fail strict native ownership checks. See the
[native AutoDiff guide](compiler_native_autodiff.md) for the current contracts.
Compilation is disabled by default.

Native BF16 and BF16 with saved-input compression now have compiler-owned Linear
programs and opt-in generated GEMMs/fusion. See the
[BF16 execution and benchmark guide](compiler_bf16.md) for configuration,
implemented patterns and GPU qualification requirements.

## Enable the compiler

Set the existing runtime compilation section in a configured recipe:

```toml
[runtime.compile]
enabled = true
backend = "deepopt"
fullgraph = false
dynamic = false
```

The ordinary setup order remains device placement, model preparation,
compilation, optimizer construction and runtime/DDP setup. Parameter objects,
the model class and state dictionary keys retain their ownership.

A complete CPU/reference example uses the direct compilation helper:

```python
import torch
from torch import nn
from deepopt import FP8Config, autocast
from deepopt.compiler import compilation_report
from deepopt.config import CompileConfig
from deepopt.engine.execution.compilation import compile_model
from deepopt.opt.policy import prepare_fp8_model

settings = FP8Config(enabled=True, backend="reference")
model, _ = prepare_fp8_model(
    nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 4)), settings,
)
compile_model(model, CompileConfig(enabled=True, backend="deepopt", dynamic=False))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
with autocast("cpu", enabled=False, fp8=settings):
    loss = model(torch.randn(4, 8)).square().mean()
loss.backward()
optimizer.step()
print(compilation_report(model).as_dict())
```

For CUDA FP8, move the model before preparation. Select custom kernels through
`runtime.kernels.backend`: `triton`, `tilelang`, or `mixed`. New FP8 recipes use
`runtime.fp8.backend = "inherit"`; legacy Triton recipes retain their behavior.
The model helper passes the same resolved policy to generated pointwise,
backward and Linear preparation programs. Mixed routes split fusion groups at
target boundaries; an explicitly pinned auxiliary route can make an atomic
activation-preparation fusion ineligible. Hardware and layout requirements
remain those of the [FP8 operations](../fp8/README.md); TileLang CUDA compilation,
capture and performance still require target-GPU qualification.

Direct compiler callers can select a target with
`BackendConfig(kernel_backend="tilelang")` or
`torch.compile(..., backend="deepopt", options={"kernel_backend": "tilelang"})`.
Mixed direct callers supply the shared validated `kernel_policy` alongside
`kernel_backend="mixed"`. Selection remains distinct from the Torch compiler
backend, and does not add FP16 AMP support to Trainer.

## Capture and backward ownership

The compilation helper lazily registers the backend and installs model-local
capture facades for prepared DO modules and prepared FX functional calls.
Dynamo records those calls with explicit model parameter and buffer operands.
The default training route differentiates semantic regions before destructive
kernel fusion. Registered VJPs produce backward IR and saved-state requirements;
one thin bridge per region connects those executables to caller autograd.
Linear lowering supports current, supported blockwise/delayed FP8 recipes,
prepared FP8 operands, gradient transport and saved-input compression. The
native-derived route does not invoke the former compiler/eager per-op backward
programs. Physical payload/scale/codec components remain explicit saved values.
Private weight caches keep their existing version checks and lifetime.

Projection modules, functional adapters, HF Conv1D and MHA decomposition share
these derivative contracts. Other prepared operation families register numerical
VJPs with explicit statistics, coefficients, routing and kernel state. Strict
mode rejects a remaining legacy/eager backward owner.

Linear capture passes module-local operands and infers output shape, dtype,
strides and gradient requirements without cloning the model. Stateful leaf
probes copy their own operation with local fake operands; specialized functional
metadata paths avoid whole-model probing. Neither path
advances live running statistics, scale histories, caches or call counters.
Deep copies of compiled models recover eager module forwards and omit the
compiled callable; reconstruction installs fresh compiler ownership.

These facades belong to this backend. Other compiler backends retain the
existing eager functional boundaries. A fresh copy can be compiled with another
backend through the same helper, which removes the prior capture facades.
An unprepared Python functional call intercepted only at runtime can still
remain eager. Use the existing full FX preparation mode when functional
operations need explicit compiler capture.

Native LayerNorm exports forward statistics to its derived VJP without rerunning
normalization. Broadcast, casts, geometry and reductions have native rules;
complex tensors and unsupported operation/policy combinations retain explicit
compatibility behavior. The compiler preserves each DO operation's configured
derivative/compression policy. Differentiation remains first order.

The pointwise kernel ABI currently accepts contiguous, equal-shape FP32,
BF16 or FP16 dense tensors on one CUDA device, with one escaping output.
Each fused intermediate rounds to its declared dtype; floating-point
multiply-add contraction is disabled. FP8 wrappers and packed saved-state
representations remain opaque and cannot enter a dense pointer kernel.
Known dense BF16 outputs of lowered Linear operations remain dense and can
participate in eligible consumers.
Native multi-output operations, including pooling indices, remain ordinary
structured values outside this fusion path.

## Options and diagnostics

Use scalar options under `[runtime.compile.options]`:

| Option | Default | Behavior |
| --- | --- | --- |
| `block_size` | `1024` | Upper bound on the generated pointwise launch block, positive power of two. The emitter derives the actual block from the partition's static element count between a 256 floor and this bound. |
| `kernel_backend` | `"triton"` for direct callers | `"triton"`, `"tilelang"`, or `"mixed"`; the model helper supplies the model/runtime kernel policy and rejects conflicting options. |
| `max_pointwise_fusion_length` | `128` | Maximum number of nodes in a pointwise partition. |
| `capture_deepopt` | `true` | Expose prepared operations through model-local facades. Disable to retain the original eager boundaries. |
| `fuse_deepopt_activations` | `false` | Experimental GELU/SiLU-to-Linear operand preparation, including affine packing when selected. Requires a sole-use contiguous CUDA activation with more than 4096 elements and the supported Linear recipe. Target-GPU qualification is pending. |
| `enable_aot_autodiff` | `true` | Connect generated training kernels to the first-order backward IR. Disabling it is rejected for differentiable generated regions. |
| `use_native_autodiff` | `true` | Derive semantic-region backward graphs before scheduling. False retains the former compiler programs as an explicit migration/benchmark control. |
| `require_full_aot` | `false` | Require native differentiation ownership and reject missing metadata, legacy callbacks and eager backward escapes. Pair with `fullgraph=true`. |
| `mutation_backend` | `"native"` | Preserve writes through original operations. `"error"` rejects writes; `"eager"` explicitly selects FX execution. `"inductor"` delegates eligible native graphs and rejects prepared DO facades. |
| `debug` | `false` | Print IR, scheduling and generated wrapper source. |
| `cache_dir` | omitted | Optional generated-source directory. The default is an external temporary cache, also configurable through `DEEPOPT_COMPILER_CACHE_DIR`. |
| `cost_model` | `"advisory"` | Static admission of pointwise partitions: `"off"`, `"advisory"` (record predictions, change nothing) or `"enforce"` (return losing partitions to their eager callable). Rejected with `require_full_aot`. |
| `cost_model_bandwidth_bytes_per_s` | omitted | Explicit streaming bandwidth in bytes per second. When omitted, bandwidth is read from declared device properties and derated to 85% of the vendor clock/bus product; an unavailable device reports `unestimated` and never declines work. An explicit value is used as given. |
| `cost_model_admit_margin` | `0.10` | Predicted speedup a fused region must exceed before it is admitted. |

Only `mode="default"` is supported. `mode` and `options` remain mutually
exclusive in runtime configuration. Unknown options fail during setup.
`dynamic=True` is rejected by the model helper; an omitted value selects
static guarded specialization. New batch shapes can trigger recompilation.
Host precision frames and compiler diagnostics remain outside the numerical
body so supported prepared graphs can use `fullgraph=true`. Compatibility mode
can still use `fullgraph=false` for unsupported caller code.

`compilation_report(model)` returns a serializable report with captured graph
counts, Triton/TileLang partitions, declared and effective kernel routes,
DO calls, compiler-owned backward counts and
operations retaining native autograd. Zero captured graphs means the backend
has not processed any work, even if forward execution succeeded. Counts are
per backend invocation, so recompilation can add entries for the same source
operation. They do not measure kernel execution counts or all Dynamo graph
breaks; use the profiler for those measurements.

`pointwise_cost_estimates` records one row per pointwise partition of the
outer schedule with the predicted fused/eager cost, launch counts and admission
decision. These rows are static predictions from declared geometry and a
bandwidth model; they are not measured time and not observed launches. A
single-operation partition is always declined because it replaces one eager
ATen launch with one Triton launch over the same traffic. Unknown geometry or an
unknown device reports `unestimated` and is never declined. Rows carry
`applied`, which is true only where `enforce` actually returned the region to
its eager callable, and `schedule_position="pre_admission"` because the index
refers to the schedule before any decline was applied.

`static_regions.predicted_declined_pointwise_regions` and
`predicted_admitted_pointwise_regions` count predictions in either mode;
`declined_pointwise_regions` counts only applied declines and stays zero in
advisory mode. Admission currently covers the outer schedule; joint AutoDiff
and backward region programs build their own schedules and are not admitted.

Pointwise kernels bake their launch geometry into the emitted source: the block
is derived from the static element count, 32-bit indexing is used below 2^31
elements, and the bounds mask is omitted when the grid divides the element count
exactly. The launcher must use the artifact's recorded `block_size`; an
unmasked store is only correct with that geometry.

New report fields distinguish `lowered_deepopt_operations`,
`compiled_deepopt_backward_regions`, `fused_deepopt_regions` and
`unsupported_deepopt_operations`. Each graph also records its program's
forward/backward steps, saved components and backend compilation phase times.
Lowered kernel-reuse programs do not count as newly generated Triton kernels.
Fusion counts describe selected templates; use a CUDA trace to establish
execution and a matched benchmark to establish performance.
Native provenance is recorded separately in `native_autodiff_operations`,
`differentiation_rules`, `native_autodiff_graphs` and
`legacy_backward_operations`. The older compiler-owned backward counter alone
does not establish which differentiation engine produced the graph.

For target-GPU qualification, compare eager and compiled ViT-block training:

```bash
python benchmarks/compiler/benchmark_compiler.py --model vit-block --device cuda \
  --precision bf16 --fp8 --compression --batch-size 512 --warmup 10 --steps 100 \
  --profile --trace-dir /tmp/deepopt-compiler-traces
```

Repeat with `--fuse-deepopt-activations` to evaluate the experimental templates,
and without `--compression` for the uncompressed pair. The block uses the actual
768-wide, 197-token projection shapes. `--model vit-base` requires timm and
benchmarks the full model. Both variants prepare projection modules only and
leave native normalization/attention semantics intact. Current tensorwise
scaling retains the global amax reduction boundary; fusion removes the
materialized activation without claiming a single launch for the entire
pipeline.

Compiled wrappers check numerical scopes and captured module configuration
before execution. If an explicit FX callable is reused with different
AMP/FP8/compression settings, training mode or operator configuration, the wrapper
selects the original FX operations and increments `runtime_fallbacks` with a
reason. This eligibility check occurs before generated work executes. Strict
AOT mode raises instead, requiring recompilation in the new scope. Runtime
kernel exceptions propagate.

Generated-source cache keys are the SHA-256 of the module source, which now
contains the per-partition launch block, index width and mask decision. The
header also records compiler ABI, Torch/Triton versions and the CUDA runtime
version. Writes are atomic and
module loads are serialized within a process. Triton retains responsibility
for device-specific JIT specialization. Checkpoints contain ordinary state
and compiler settings; generated executables are rebuilt after restoration.

## Validation and current limits

The implementation was exercised with Python 3.12.14 and PyTorch 2.15.0 on
CPU without Triton. This establishes the following reference coverage:

| Area | Validation |
| --- | --- |
| Native model compilation | Values, gradients, optimizer updates, partial batches, broadcast/strided inputs, structured outputs and native BF16 AMP. |
| Prepared DO calls | Linear and functional projections; ReLU/ReLU6, sigmoid, SiLU, GELU and tanh; spatial/transposed convolution; normalization; pooling indices; attention. |
| Numerical policy | Compression, delayed scales, planned FP8 gradient transport, train/eval transitions and overrides; backward outside autocast. |
| Training lifecycle | Accumulation, isolated fake memory estimation, parameter ownership and checkpoint/resume parity. |
| Distributed execution | Two CPU/Gloo ranks, independently averaged native and FP8 gradient/update oracles. |
| CUDA execution | Hardware tests added for pointwise values/gradients/tails, compiled callable CUDA Graph replay and prepared FP8 execution. Pending execution on target GPUs. |

CPU tests do not establish GPU throughput, CUDA Graph compatibility for every
prepared model, NCCL behavior or support across the base package's full
PyTorch version range. Validate the pinned Torch/Triton/GPU pair before using
the backend for those combinations. Delayed-scaling CUDA Graph restrictions
remain those of the existing graph runtime. General symbolic shapes,
higher-order compiler AOT gradients and compiler-owned collectives are outside
the current implementation.

Run the focused checks with:

```bash
python -m unittest discover -s tests/compiler -v
python -m unittest tests.distributed.test_compiler -v
```

The [CI workflow](../../.github/workflows/ci.yml) includes the compiler group
in CPU and optional GPU jobs. Existing compiler backends retain their own
regression tests.

## Measure performance

The [comparison benchmark](../../benchmarks/compiler/benchmark_compiler.py)
runs eager and compiled DeepOpt with the same initialization, data and
numerical policy, and checks final model state parity:

```bash
python benchmarks/compiler/benchmark_compiler.py --model mlp --fp8 --compression --steps 20
python benchmarks/compiler/benchmark_compiler.py --device cuda --precision bf16 --model residual --fp8 --compression --profile
python benchmarks/compiler/benchmark_compiler.py --device cuda --model attention --backend inductor --profile
```

It separates cold first-step latency from warm forward/loss, backward,
optimizer and whole-step latency; records variance, saved storage, peak CUDA
allocation and compiler coverage; and optionally records kernel and cast/codec
events. GPU phases use CUDA events, while whole-step and cold latency use
synchronized wall time. Saved-state measurement and profiling run outside
the timed iterations. Saved-storage counts cover tensors observed by the outer
hooks; operations with internal saved-tensor hooks can retain additional state.
Peak CUDA allocation is measured separately. CPU peak CUDA memory is reported
as null.

CPU smoke runs validate the harness. Additional fusion across DeepOpt
operation boundaries stays gated on target-GPU numerical and performance
evidence. Disable compilation or select another supported backend to return
to the previous execution path while keeping the same preparation policy.
