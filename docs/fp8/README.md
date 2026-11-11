# FP8 training and model transformation

[Documentation index](../README.md) · [Training](../training/README.md) · [Tests](../../tests/README.md) · [Example](../../examples/optimization/fp8_training.py)

DeepOpt replaces supported projections with DO modules or the shared
`do_linear` function. Parameters keep their original floating-point dtype and
identity; forward and backward quantize temporary GEMM operands. Ordinary
optimizers update the master parameters. This is first-order approximate FP8
training, not numerical differentiation through FP8 casts.

Automatic setup is available through `Trainer(..., fp8=True)`,
`TrainingRuntime(..., fp8=True)` and both core constructors. Configured runs use
`[runtime.fp8]`. Manual `transform_model(...)` remains available for applications
that want to own conversion explicitly.

Optional [saved-activation compression](../activation_compression/README.md) fuses
affine 2-bit packing with FP8 casts and reconstructs directly to FP8 during
backward. Its independent configuration also supports native BF16 linear.

For complete training examples, see the [regular engine core](#train-with-the-regular-engine-core)
and [DDP core](#train-with-the-ddp-core). Both include `Trainer` usage and direct
core calls, with a manual `TrainingRuntime` loop between those levels.

## Shared configuration defaults

Load a complete recipe once to make its FP8 policy available to subsequently
constructed trainers, runtimes, and cores:

```python
from deepopt import Trainer, load_config
from torch.nn import functional as F

load_config("builtin:fp8")
trainer = Trainer(model, loss_fn=F.mse_loss)  # Uses the loaded optimizer/FP8 settings.
trainer.fit(train_loader, epochs=2, progress_bar=False)
```

This fragment assumes your model and regression loader already exist; the preset
supplies defaults, not a model or task. Construction takes an owned snapshot.
Later `load_config()` calls do not modify that trainer's prepared operations,
optimizer, or precision. `config=None` inherits defaults, a partial `config=`
dictionary overrides selected fields, and a typed experiment/file selects a
complete source. Explicit `fp8` arguments take priority. See
[shared training configuration](../training/README.md#use-shared-configuration-defaults).

Automatic preparation and compilation still occur before a trainer-owned
optimizer and before DDP wrapping. If you supply an optimizer, its original
parameter objects and settings are retained. Bare cores and `TrainingRuntime`
keep caller-owned optimizer construction; use their exposed `.model` after
preparation. Loading a configuration does not execute kernels or capture graphs.

## Start with reference execution

```bash
python -m pip install -e .
python examples/optimization/fp8_training.py
python examples/optimization/fp8_training.py --trace-mode modules --block-size 32
```

The example trains a small MLP for three updates using generated data. It prints
the transformation report and loss, and writes no checkpoints or downloads.
`reference` quantizes to FP8, dequantizes, and multiplies in FP32. Its passing
tests validate semantics; they do not establish GPU kernel correctness or speed.

CPU scaling uses a PyTorch reference implementation for testing and does not
require Triton. CUDA scaling lazily imports the fused Triton implementation;
CUDA inputs never silently switch to the CPU reference path.
CUDA fused scaling requires power-of-two block dimensions and excludes `(1, 1)`;
CPU reference tests can use arbitrary positive block dimensions.

`--block-size 32` uses `(32, 32)` scaling grids for activation, weight, and
gradient operands. Omit it for tensorwise scaling. `--precision bf16` selects
the runtime's CUDA autocast mode; it does not change the FP8 operand format.
The example prints the actual trace strategy, matched operations, backend,
scaling grid, update count, and loss. No pretrained model package is needed.

## Configure FP8 operands

### Automatic configuration

Add this table to an experiment configuration to enable FP8 in either core:

```toml
[runtime.fp8]
enabled = true
backend = "reference"
trace_mode = "modules"
activation_dtype = "e4m3fn"
weight_dtype = "e4m3fn"
grad_dtype = "e4m3fn"
```

With no inherited FP8 policy, omitting the table leaves automatic preparation
disabled. In a partial configuration, omission inherits the existing defaults;
set `enabled = false` explicitly to disable preparation for a fresh model.
The [complete configuration example](../../configs/fp8.toml) runs on CPU. Set
`backend = "triton"` on supported CUDA hardware for actual FP8 GEMM. These
settings are independent of `runtime.precision`, which still selects
trainer-level AMP (`fp32` disables AMP; CUDA `bf16` enables BF16 AMP).

Python recipes accept `fp8=True` for defaults or
`FP8Config(enabled=True, backend="reference", ...)` for explicit settings.
Mappings use the same schema as the TOML table, including `enabled`.
Optional `block_shapes` maps `activation`, `weight` and `grad` to positive
two-dimensional shapes; roles without inherited settings use tensorwise scaling.
Partial overrides preserve unspecified inherited roles. `num_stages`
defaults to `"auto"` (search stages 1–5); an integer from 1–5 pins the depth.
`out_dtype=None` uses the active AMP/input dtype.
The weight-gradient GEMM uses that same output dtype: BF16 projections emit BF16
dW directly, FP16 projections emit FP16 dW, and explicit FP32 projections retain
FP32 dW. Input-gradient GEMMs retain the original input dtype. Autograd converts
the returned gradients to each parameter's dtype for accumulation, so FP32 master
weights and their `.grad` storage remain FP32. `grad_dtype` selects the FP8
incoming-gradient operand format, independently of the GEMM output dtype.
`backward_backend="inherit"` keeps backward on that same backend. Tensorwise
Triton recipes can explicitly select `native_fast` or `native_precise` for dX/dW;
forward retains its configured backend. Automatic preparation accepts qualified
module-name globs in `exclude_modules`; nonempty exclusions require
`intercept_linear=False`. See the [controls and benchmark procedure](eager_linear_benchmark.md#native-backward-and-selective-preparation)
for native eligibility, fallback rules, aliases and MHA output selection.

With the inherited Triton backend, FP8 uses native accumulation, retaining the chained FP32
accumulator in tensorwise GEMMs. DeepOpt no longer supplies
`max_num_imprecise_acc`. Omitting this argument follows Triton's backend default
and differs numerically from the earlier forced limit of 32; see the
[Triton semantic implementation](https://github.com/triton-lang/triton/blob/v3.9.0/python/triton/language/semantic.py#L1477).
The reference backend retains its FP32 emulation. Remove the obsolete setting
from configuration and execution plans when starting a new run; it now raises
a migration error instead of being silently ignored.

Numerical recipe version 5 records `weight_gradient_output="forward-dtype"`
for current and delayed scaling on both backends. Versions 2–4 used the parameter
dtype for dW. Native backward or projection exclusions also record the selected
executor/accumulation and excluded patterns.
Native backward also records PyTorch/CUDA versions; Triton recipes record the
Triton version. Exact resume rejects missing or changed recipes before loading
training state. Continue older FP8
runs with their original library/environment, or explicitly load model weights
into a new run. Removing an old TOML field does not migrate its numerical state.

Automatic preparation runs before DDP wrapping. Configured experiments also
prepare before compilation and optimizer creation. Direct constructors preserve
existing master parameter objects, so an optimizer created beforehand remains
valid. Use `trainer.model`, `runtime.model` or `core.model` after construction:
a root projection becomes its DO adapter, and explicit `full`/`auto` tracing can
return a GraphModule. The default `modules` mode preserves ordinary container
root methods. Inspect `runtime.transformation_report` (or the core's report) for
conversion coverage. Matching prepared models are reused during construction.
With no conflicting selected configuration, reusing an automatically prepared
model inherits its recorded policy. Explicit `fp8=None` preserves that prepared
policy. Conflicting settings, including an explicit disabled policy, require
reconstruction from the original model; disabling preparation does not undo
an earlier conversion.

### Explicit operation plans

The lower-level transformation API still accepts torch dtypes and per-role
scaling shapes directly:

```python
import torch
from deepopt.opt import transform_model

plan = {
    "fp8_training": {
        "backend": "reference",
        "fp8_dtypes": {
            "activation": torch.float8_e4m3fn,
            "weight": torch.float8_e4m3fn,
            "grad": torch.float8_e5m2,
        },
        "num_stages": "auto",
        "backward_backend": "inherit",
        "out_dtype": None,
    },
    "block_shapes": {
        "activation": None,
        "weight": None,
        "grad": None,
    },
}
model, report = transform_model(model, plan=plan, trace_mode="full")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
```

Move the source model to its device before transformation and optimizer creation.
`None` selects tensorwise scaling. A two-dimensional positive tuple such as
`(128, 128)` selects blockwise scaling; edge blocks are supported. The activation,
weight, and upstream-gradient roles can have different grids. The gradient role
controls quantization of the upstream gradient used to compute input/weight
gradients, not the dtype of optimizer-owned gradient buffers.

Each operand role independently selects E4M3FN or E5M2 on either backend.
`out_dtype=None` uses the active autocast dtype or the input dtype. Explicit
output dtypes are FP32, BF16, and FP16. Input, weight, and bias tensors must be
dense, on the same CPU/CUDA device, and use
FP32/BF16/FP16. Empty operands and second-order derivatives are unsupported.
Strict execution rejects nonfinite quantization inputs immediately. **Trainer-level AMP and FP8 operand formats are
separate choices**: CPU/CUDA can use FP32 with AMP disabled, and CUDA can enable
BF16 AMP. The FP8 plan independently selects the formats used by DO projections.

### Rowwise backward quantization

Rowwise activation or gradient grids `(1, C)` with `C > 1` quantize each needed
GEMM orientation independently from the original higher-precision tensor.
Forward uses `Q_activation(X)` and `Q_weight(W)`. Input gradients use
`Q_grad(G) @ Q_weight(W)`. For weight gradients, rowwise roles instead use
`Q_grad(G.T)` and/or `Q_activation(X.T)`; other roles transpose their previously
quantized values and scale grids. In particular, quantizing `X.T` rowwise is
intentionally different from transposing a rowwise quantization of `X`.
The CPU reference backend follows this same orientation-dependent contract.

CUDA quantization fuses block amax, scale calculation, clamping, FP8 conversion,
and inverse-scale output in one kernel for tiles with at most 16384 elements. Small
independent blocks share a CTA; input strides are read directly, including
transpose views and partial edge blocks. This avoids expanded scale grids and
full-sized FP32/contiguous input intermediates. Both FP8 formats retain
round-to-nearest-even conversion and the existing zero-amax and minimum-amax
rules. Larger scale blocks use bounded chunked reductions followed by fused
scaling/conversion; 16384 is an algorithm threshold, not a supported block-size
limit. Scale tiles are clipped to the actual input dimensions to avoid excessive
padded work. Standalone `AmaxScaling.compute` shares the same kernels in scale-only
mode and returns both scales without producing FP8 data. It reads the original
dtype and strides, with no full FP32 conversion or contiguous input copy. Layout
selection stays inside the CUDA launch wrapper. CPU emulation retains its FP32
reference tensors.

Small tensorwise inputs (at most 4096 elements) use one fused kernel. Larger
tensorwise inputs use partial amax reduction, a bounded reduction to a global
scale, and fused scaling/conversion. Very large reductions add hierarchical
partial-reduction stages. Oversized blockwise groups use the same staged
approach with a separate reduction per scale block. Current tensorwise scaling still reads the original
input twice; no global barrier or delayed-scaling recipe is introduced.

Checked fused kernels write a finite-error flag per CTA while reading the
input, then synchronously validate those flags before returning. `max_speed`,
unchecked `balanced` updates, and explicit `check_finite=False` compile out that
work and allocate no error buffer. `analyze`, checked cadence updates, and
explicit `check_finite=True` retain immediate nonfinite-input rejection.
The independent backward-check choice saved by forward remains in force.
Metadata-only FakeTensor probes allocate the fused result and scale layouts
without importing Triton or tracing the old unfused intermediates.

CUDA `QuantizedTensor.dequantize()` fuses FP8-to-FP32 conversion and inverse-scale
multiplication in one kernel. Strided values, transpose views, and noncontiguous
scale grids are supported directly; no expanded scale tensor or intermediate
FP32 value tensor is materialized. Reconstruction retains its existing behavior
in every execution mode and introduces no additional finite scans or scalar reads.

This keeps rowwise scales aligned with the reduction axis in the weight-gradient
GEMM. With activation `(1, 32)`, weight `(32, 32)`, and gradient `(1, 32)` or
`(32, 32)`, all three GEMMs select the native FP8 path. Other block grids still
need compatible reduction widths (multiples of 32); arbitrary unaligned grids
retain the dequantization fallback.

Under `torch.no_grad()` or `torch.inference_mode()`, the projection skips saved
activation orientations and other backward-only bookkeeping, even when its
parameters still have `requires_grad=True`. Training retains the independent
rowwise quantization described above, passing original transpose views directly
to the strided quantizer without dense high-precision transpose copies.
Other backward transposes can use views;
the Triton backend packs those views directly into its required aligned layout.
`QuantizedTensor.transpose()` keeps its contiguous-result default; the optional
`contiguous=False` avoids an intermediate materialization.

For native CUDA tensorwise training operands larger than 4096 values, conversion
can emit both FP8 orientations with 16-byte-aligned row strides. Both layouts
share the same scalar scale and exactly transposed quantized values. Forward
keeps only the orientation needed by backward; weight caches can retain a pair
for forward and input gradients. This avoids subsequent whole-matrix transpose
packing. Rowwise scaling still quantizes each orientation independently, and
inference does not create unused backward layouts. The size threshold is a
heuristic, not a measured hardware optimum.
When only weight gradients are required, tensorwise gradient conversion emits
only its transposed layout, avoiding an unused FP8 allocation and store. It uses
the same current-tensor amax, scalar scale, finite-check cadence and rounding.

Prepared adapters reuse quantized weights while the owning parameter's PyTorch
version, identity, storage/layout, dtype/device, FP8 format, scale blocks, finite
check policy, requested single/paired layout, and CUDA stream match. The bounded
per-adapter cache holds at most four entries; a paired entry contains two FP8
layouts. Optimizer updates, `load_state_dict`, parameter replacement, and
device/dtype changes invalidate reuse. DDP initialization clears rank-local
caches after its initial parameter broadcast. Caches are absent from `state_dict` and
are empty after a module copy or whole-module pickle; they add resident FP8
weight/scale storage outside registered parameters and buffers. Use
`module.clear_fp8_cache()` after an external write that bypasses PyTorch version
tracking, such as a `.data` write or a custom raw-pointer kernel. Setting
`cache_enabled=False` in the autocast context disables weight reuse.

## Combine FP8 with AMP in a context

`deepopt.autocast` combines native PyTorch AMP and FP8 in one context or decorator.
For an ordinary `nn.Linear`/`F.linear` computation, no transformation is needed:

```python
import torch
from deepopt import autocast

model = torch.nn.Linear(8, 3)
inputs = torch.randn(4, 8)
with autocast("cpu", enabled=False, fp8=True):
    predictions = model(inputs)
    loss = predictions.square().mean()
loss.backward()  # Uses the FP8 operands/policy saved during forward.
```

Here `enabled=False` disables **native AMP only**; `fp8=True` enables FP8 using
the selected backend (reference when no other backend is configured). For BF16
AMP plus Triton FP8 on a supported CUDA GPU:

```python
from deepopt import FP8Config, autocast

model = model.to("cuda")
inputs = inputs.to("cuda")
with autocast("cuda", enabled=True, dtype=torch.bfloat16,
              fp8=FP8Config(enabled=True, backend="triton")):
    loss = model(inputs).square().mean()
loss.backward()
```

The same context can decorate a forward function:

```python
@autocast("cpu", enabled=False, fp8=True)
def predict(model, inputs):
    return model(inputs)
```

Omitted FP8 policy inherits an outer scope, or otherwise uses the configuration
snapshot captured when the context/decorator is created. Explicit `fp8=None`
inherits only the surrounding scope; it does not activate a global FP8 policy
when no scope exists. `fp8=False` explicitly disables
FP8 in a nested scope, including on existing DO projections. Native AMP's
`enabled` and `dtype` stay independent. Exiting the scope restores both policies,
including on exceptions. Backward normally runs outside the forward context.

Omitted AMP controls use selected `runtime.precision` when present; with no
configured precision they retain PyTorch autocast defaults. An already enabled
native AMP scope retains its dtype when those controls are omitted. Explicit
`enabled` and `dtype` arguments override defaults. For a known BF16 computation,
passing both `enabled=True` and `dtype=torch.bfloat16` makes the choice explicit.
An FP8 dictionary passed to this context enables FP8 unless it includes
`enabled=False`; omitted fields inherit selected configuration. A typed
`FP8Config` supplies a complete operation policy.

The standalone context intercepts eligible floating `F.linear` calls (including
ordinary `nn.Linear`) and configures DO operations already present in the model.
It does not rewrite HF `addmm`, fused attention kernels, arbitrary `matmul`, or
the model structure. Use automatic core preparation for the full set of supported
adapters. Like native AMP, keep a context within one thread's synchronous work;
do not suspend it across interleaved async tasks or generators.

Runtime precision policies and core FP8 scopes resolve owned execution plans
once during construction. Their repeated entries only establish context-local
state; public `fp8_plan`/`validate_fp8_plan` results remain independently mutable.
Prepared adapters suspend DeepOpt's own interception inside an already-selected
operation while preserving user Torch modes. For a fully prepared model, set
`FP8Config(enabled=True, intercept_linear=False)` or the corresponding TOML field
to avoid installing the broad native-linear mode. In that mode, unconverted
native `F.linear` calls stay dense; existing DO adapters still obey nested FP8
enable/disable/override scopes. The default retains native-linear coverage.
Module exclusions are applied during model preparation, not by the standalone
autocast context. Excluded native projections follow the surrounding AMP policy.

`execution_mode` covers FP8 as well as loss/gradient observers. `max_speed` and
legacy `checks=False` runtimes skip quantization's passive finite-input scans.
`balanced` checks forward quantization on `loss_finite_every` updates and
backward quantization on `gradient_finite_every` updates (100 by default).
Explicit weight/activation inspection also enables forward checks. The
`analyze` preset checks every update; legacy/default strict execution checks
every call. Standalone scopes can explicitly use
`autocast(..., check_finite=False)`; `quantize_fp8(..., check_finite=False)` and
`AmaxScaling.compute(..., check_finite=False)` provide direct controls. Omitted
check arguments inherit the active runtime phase; explicit `True` forces an
immediate check. A forward saves the independently selected backward-check
choice, including for backward outside its scope. Enter `runtime.autocast()`
for each manual forward (a saved context object can be reentered); scheduling
is resolved before that forward. Unchecked inputs must be finite: no deferred error flag or automatic
repair is introduced. Zero-amax handling stays on the GPU and requires no
Python scalar read. Fused quantization uses the same phase-local check policy.

`runtime.autocast()` applies the runtime's configured AMP and FP8 together.
`runtime.compute()` enters it automatically and also performs device transfer
and core dispatch. For DDP, keep forwards inside `compute()`; the standalone
context by itself does not install a DDP wrapper or synchronize gradients.

CPU correctness tests exercise `eager` and `aot_eager` compiled models across
dense/FP8 context changes with `fullgraph=False`. The FP8 linear primitive is an
explicit eager boundary: plan selection, quantized-tensor metadata and custom
autograd stay outside Dynamo, while surrounding model operations can compile.
Triton still compiles and runs its own FP8 kernels. Full-graph compilation of
this path is unsupported; these checks do not establish Inductor GPU performance
or CUDA Graph capture support.

## Use a projection directly

FX transformation is optional. With the plan above, a functional projection is:

```python
from deepopt.opt.nn.functional import do_linear

x = torch.randn(4, 8, requires_grad=True)
weight = torch.nn.Parameter(torch.randn(3, 8))
bias = torch.nn.Parameter(torch.zeros(3))
output = do_linear(x, weight, bias, plan=plan)
output.square().mean().backward()
```

For an existing module, keep its parameter objects through the adapter:

```python
from deepopt.opt.nn.linear import DOLinear

original = torch.nn.Linear(8, 3)
adapted = DOLinear.from_linear(original, plan=plan)
optimizer = torch.optim.AdamW(adapted.parameters(), lr=1e-3)
```

The same pattern is available as `DOHFConv1D.from_hf_conv1d(...)` and
`DOMultiheadAttention.from_mha(...)` in their respective
[HF projection](../../src/deepopt/opt/nn/hf_conv1d.py) and
[attention](../../src/deepopt/opt/nn/attention.py) modules. Use these adapters
when the application already owns its module-selection logic; use
`transform_model` to discover supported calls in a model.

## Train with the regular engine core

FP8 is configured on the model's operations. `TrainingRuntime` supplies device
transfer, trainer-level AMP and optimizer execution, and selects
[`TrainingCore`](../../src/deepopt/engine/core.py) for one process.
`Trainer` adds epochs, accumulation, metrics and callbacks. There is no
`precision="fp8"` setting on either training core.

This complete CPU example uses reference FP8 and runs twelve optimizer updates:

```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepopt import FP8Config, Trainer, TrainingRuntime

torch.manual_seed(17)
device = torch.device("cpu")
fp8 = FP8Config(enabled=True, backend="reference")
train_loader = DataLoader(
    TensorDataset(torch.randn(24, 32), torch.randn(24, 8)),
    batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(17),
)
model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
runtime = TrainingRuntime(model, optimizer, device=device, precision="fp32", fp8=fp8)
model = runtime.model
report = runtime.transformation_report
trainer = Trainer(runtime, loss_fn=F.mse_loss)
trainer.fit(train_loader, epochs=2, progress_bar=False)
print(type(runtime.core).__name__, trainer.state.optimizer_steps)
print(report)
```

`precision="fp32"` disables **trainer-level AMP**; the converted projections
still quantize their GEMM operands to FP8. For actual CUDA FP8 GEMM, set
`device = torch.device("cuda:0")` and the FP8 setting's `backend` to `"triton"` on a
supported GPU. Choose `precision="bf16"` separately to enable CUDA BF16 AMP.

The shorter `Trainer(model, optimizer, device=device, precision="fp32", fp8=fp8,
loss_fn=F.mse_loss)` constructs this runtime internally. Passing the runtime
explicitly makes the selected core visible as `runtime.core`.

### Own the loop with TrainingRuntime

Keep the model, FP8 settings, optimizer and loader setup above, then replace
the runtime/trainer section with this loop:

```python
runtime = TrainingRuntime(model, optimizer, device=device, precision="fp32", fp8=fp8)
model = runtime.model
report = runtime.transformation_report

def compute_loss(batch):
    inputs, targets = batch
    return F.mse_loss(model(inputs), targets)

model.train()
for epoch in range(2):
    for batch in train_loader:
        runtime.zero_grad()
        loss = runtime.compute(batch, compute_loss)
        runtime.backward(loss)
        runtime.step()
```

Forward enters the core through `runtime.compute()`, which also transfers the
batch and applies AMP. FP8 forward/backward remain owned by the transformed
operations. This loop makes one optimizer update per batch; the recipe owns
epoch tracking, reporting and checkpoint calls.

### Call TrainingCore directly

For an integration that already owns transfer, precision and optimizer handling,
replace the runtime/trainer section with the bare core:

```python
from deepopt.engine import TrainingCore

core = TrainingCore(model, device, fp8=fp8)
model = core.model
report = core.transformation_report
model.train()
for epoch in range(2):
    for batch in train_loader:
        batch = tuple(tensor.to(device) for tensor in batch)
        optimizer.zero_grad(set_to_none=True)
        loss = core.compute(batch, lambda b: F.mse_loss(model(b[0]), b[1]))
        core.backward(loss)
        optimizer.step()
```

The bare core has no `step()` or AMP policy. This example keeps AMP disabled;
the application owns any autocast context, gradient normalization, diagnostics
and checkpoint lifecycle it needs. `TrainingRuntime` adds transfer, AMP,
diagnostics and optimizer handling; `Trainer` and checkpoint callbacks add
epoch and checkpoint orchestration.

## Train with the DDP core

Every rank supplies the same FP8 settings. The runtime automatically prepares
the model before wrapping it. With a multi-rank process group, `TrainingRuntime` selects
[`DDPCore`](../../src/deepopt/distributed/ddp.py) automatically. Pass the unwrapped
model: the DDP core prepares it and wraps the complete training step itself.

Save this standalone example as `fp8_ddp.py`, then launch two CPU/Gloo ranks with
`torchrun --standalone --nproc-per-node=2 fp8_ddp.py`:

```python
import gc
import json
import traceback

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from deepopt import FP8Config, Trainer, TrainingRuntime, initialize_distributed, shutdown_distributed
from deepopt.data import build_loaders
from deepopt.logger import log_output


def train(context):
    torch.manual_seed(17)  # Same source model and dataset on every rank.
    fp8 = FP8Config(enabled=True, backend="reference")
    dataset = TensorDataset(torch.randn(24, 32), torch.randn(24, 8))
    train_loader, _, _ = build_loaders(dataset, None, batch_size=4, seed=17)
    model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8)).to(context.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime = TrainingRuntime(model, optimizer, device=context.device, precision="fp32", fp8=fp8)
    model = runtime.model
    report = runtime.transformation_report
    trainer = Trainer(runtime, loss_fn=F.mse_loss)
    trainer.fit(train_loader, epochs=2, progress_bar=False)
    metrics = trainer.evaluate(train_loader)  # Every rank enters evaluation.
    log_output(json.dumps({
        "core": type(runtime.core).__name__, "world_size": context.world_size,
        "updates": trainer.state.optimizer_steps, "loss": metrics["loss"],
        "converted_modules": report.converted_modules,
    }))


def main():
    context = initialize_distributed("cpu")
    try:
        train(context)
    except BaseException as exc:
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        # Release training objects before destroying their process group.
        gc.collect()
        shutdown_distributed()


if __name__ == "__main__":
    main()
```

`build_loaders` shards the common dataset across ranks. Batch size is **per rank**:
two ranks each receive three batches per epoch, yielding six synchronized
optimizer updates over two epochs. Only rank zero prints the JSON report;
training and evaluation run on every rank. The helper's local training objects
are released before process-group shutdown, including on an exception.

For CUDA/NCCL, change initialization to `initialize_distributed("cuda")` (or
`"auto"` to select the available device), and use `context.device` as shown.
Set the FP8 backend to `"triton"` for actual GPU FP8 GEMM; keep `"reference"` for
emulation. Every participating GPU must meet the backend requirements.
`precision="bf16"` enables trainer-level BF16 AMP independently of FP8.

### Own the DDP loop or call DDPCore directly

The earlier `TrainingRuntime` loop also works inside `train(context)`: pass
`device=context.device`, retain the sharded loader and run the same number of
updates on every rank. Call `train_loader.sampler.set_epoch(epoch)` at the start
of each epoch to advance the distributed shuffle. Always call `runtime.compute()` before
`runtime.backward()` so DDP owns the forward/backward pair.

To use the bare DDP core instead, replace the runtime/trainer/report section
inside `train(context)` with:

```python
from deepopt.distributed import DDPCore

core = DDPCore(model, context.device, fp8=fp8)
model = core.model
report = core.transformation_report
model.train()
for epoch in range(2):
    train_loader.sampler.set_epoch(epoch)
    for batch in train_loader:
        batch = tuple(tensor.to(context.device) for tensor in batch)
        optimizer.zero_grad(set_to_none=True)
        loss = core.compute(batch, lambda b: F.mse_loss(model(b[0]), b[1]))
        core.backward(loss)
        optimizer.step()
```

This uses the same FP8 operations while DDP synchronizes master-parameter
gradients. With equal-size batches and a mean loss as here, native DDP already
averages gradients across ranks; do not divide the loss by world size again.
The caller owns transfer, AMP and optimizer updates, just as with the bare
regular core. For accumulation, use the runtime's
[manual accumulation helper](../optimization/README.md#accumulation-in-a-manual-loop)
to handle synchronization and normalization.

## Choose the transformation boundary

| Mode | Result | Intended use |
| --- | --- | --- |
| `full` | Rewritten `torch.fx.GraphModule` | Entire forward is symbolically traceable |
| `modules` | Original root object, with converted children | Keep root methods and dynamic Python control flow |
| `auto` | Full graph if tracing succeeds; otherwise the submodule result | Accept either return type and inspect the report |

`inplace=True` reuses source parameters. Full tracing still returns a new root
GraphModule; always use the returned model. `inplace=False` deep-copies before
conversion, so create the optimizer from the returned model. Shared registered
modules and tied parameters are preserved. Parameter registration order is kept
so optimizer checkpoint slots retain their association.

Module mode converts registered projection children directly and tries FX
rewriting inside larger children. Failed traces lead to recursive child
conversion. Operations directly in the root's Python forward are not rewritten.
A successfully traced child becomes a GraphModule, so custom methods on that
child are not automatically retained. Root methods such as HF `generate()` are
retained in module mode. Choose that mode explicitly when they are required;
successful automatic full tracing replaces the root class.

`concrete_args` specializes full tracing. An automatic request with those
arguments propagates a tracing failure rather than dropping the specialization.
FX can specialize Python decisions such as `self.training`; changing train/eval
later does not retrace inlined Python branches. Trace under the intended mode or
keep mode-dependent code outside traced regions.

The report contains `trace_mode` (the mode actually used), `original_type`,
`converted_modules`, `function_calls`, `untraced_modules`, `excluded_modules`,
and `fallback_reason`. Exclusions apply to subtrees and every shared alias;
existing FX functional projections are filtered by their module/weight owner.
Reconstruct the original eager model to exclude a previously converted adapter.
An untraced root in module mode is expected. Reports are diagnostics, not proof
that every runtime branch or every possible projection was converted. There is
no `strict` coverage policy. Conversion errors propagate; automatic fallback
handles tracing failures only.

## Optional models and custom conversion rules

Construct third-party models before creating their default registry. For example,
after installing the optional [model dependencies](#black-box-and-white-box-validation),
this builds a tiny HF model from configuration without downloading weights:

```python
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=1, n_head=4,
                    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False)
model = GPT2LMHeadModel(config)
model, report = transform_model(model, plan=plan, trace_mode="modules")
tokens = torch.randint(0, 64, (2, 8))
model(input_ids=tokens, labels=tokens, use_cache=False).loss.backward()
```

This uses the earlier CPU reference plan. Module mode retains the root HF API
and tied parameter identities; its supported projections use FP8 while attention
scores, embeddings, normalizations, and other operations keep their existing
implementations. The [full-model tests](../../tests/opt/test_full_models_black_box.py)
also exercise torchvision/timm ViTs and Llama, including checkpoint continuation.

For an application subclass with the **same linear semantics**, explicitly
register its conversion:

```python
from deepopt.opt import default_registry
from deepopt.opt.nn.linear import DOLinear

class ProjectLinear(torch.nn.Linear):
    pass

registry = default_registry()
registry.register(ProjectLinear, DOLinear, DOLinear.from_linear)
model = torch.nn.Sequential(ProjectLinear(8, 3))
model, report = transform_model(model, plan=plan, registry=registry)
```

If a subclass changes the operation, its converter must implement those semantics;
replacing it with a plain linear adapter would be incorrect. The registry chooses
converters, the matcher identifies eligible FX nodes, and the rewriter applies
those matches. Training policy belongs to the runtime/trainer, not these passes.

## Supported operations

| Source | Replacement | Notes |
| --- | --- | --- |
| `torch.nn.Linear` | `DOLinear` | Original weight and optional bias |
| `torch.nn.functional.linear` | `do_linear` | Functional graph target; parameter references stay intact |
| `torch.nn.Conv1d`, `Conv2d`, `Conv3d` | `DOConv1d`, `DOConv2d`, `DOConv3d` | Spatial convolution; independent saved-input compression |
| `torch.nn.functional.conv1d`, `conv2d`, `conv3d` | `do_conv1d`, `do_conv2d`, `do_conv3d` | Functional spatial convolution targets |
| HF Transformers `Conv1D` | `DOHFConv1D` | Keeps stored `[in, out]` weights and transposes for the linear operation |
| `torch.nn.MultiheadAttention` | `DOMultiheadAttention` | FP8 Q/K/V and output projections |

HF `Conv1D` is a linear projection; spatial convolutions use the separate
[convolution adapters](../convolution/README.md). Exact registered module types
are matched; arbitrary subclasses need an explicit conversion rule.
Construct HF models before requesting the default registry so their projection
class is loaded. Transformers is an optional dependency.

The attention adapter supports packed/separate Q/K/V weights, self/cross
attention, batch-first/sequence-first/unbatched inputs, masks, and optional
attention weights. Attention scores and value aggregation are not FP8 GEMMs:
the explicit weights path uses FP32, and the other path uses native SDPA.
Learned bias K/V tokens, `add_zero_attn`, and nested tensors are unsupported.
`is_causal=True` requires an explicit attention mask. Fully masked rows in the
explicit weights path produce nonfinite values and are rejected by quantization.

## Checkpoints and DDP

With automatic FP8 enabled, each rank reconstructs the same model/settings and
the core prepares the model before DDP wrapping. When using the explicit
`transform_model` API instead, transform before constructing `Trainer`,
`TrainingRuntime`, or native DDP.
The training and DDP cores continue to own accumulation, synchronization, and
optimizer updates. FP8 modules own the projection computation and custom
autograd, and do not introduce their own process group or optimizer.

Checkpoint normal model, optimizer, scheduler, loader and trainer state. Temporary
FP8 operands and scales saved for backward do not become model checkpoint keys.
Reconstruct with the same FP8 configuration before restoring an automatic FP8
run. Checkpoints record the enabled automatic policy and reject mismatches.
Configured runs retain it in their saved configuration. For a manually
transformed model, the `Checkpoint` callback does not serialize the caller's
Python operation plan; keep that plan in the recipe/configuration or alongside a
caller-owned checkpoint. Parameter keys and ordering remain compatible with
native model/optimizer checkpoints for the supported conversions.

The tests cover exact continuation at both step and epoch boundaries, including
partial accumulation windows, two Gloo ranks, and conditional two-GPU NCCL runs.
The FP8 path is not guaranteed CUDA-Graph-capture-safe. Strict quantization
performs a host-visible finite-error check; disabling that check alone does not establish
capture safety for policy dispatch, allocation, autotuning, or weight-cache
lifetimes. CUDA Graph support for the surrounding framework does not imply
this FP8 path can be captured.

## Actual CUDA execution

See [CUDA execution and autotuning](cuda.md#actual-cuda-execution) for hardware
requirements, scale layouts, kernel selection, and projection measurements.

## Full-model training experiments

See [full-model training benchmarks](training_benchmarks.md#full-model-training-experiments)
for matched ViT updates, memory accounting, profiling, and measurement limits.

## Black-box and white-box validation

See [validation and test environments](validation.md#black-box-and-white-box-validation)
for dependency setup, CPU reference checks, actual CUDA coverage, and required
environment switches.

<a id="future-activation-compression-and-sparse-operations"></a>

## Compression and operation extensions

Saved-activation compression is implemented as an opt-in affine 2-bit codec.
Its [configuration and representation](../activation_compression/README.md)
cover BF16, tensorwise FP8, and FP8 scaling blocks. The operation and autograd
layers own the saved payload and reconstruction; GEMM backends consume the
reconstructed operands and retain their execution responsibilities.

The existing conversion registry remains the extension boundary for additional
operations. Sparse execution is not implemented. A sparse policy needs forward
and backward products, explicit layouts, and validation before it can be
enabled; placeholder flags do not provide that behavior.

### Controlled native-accumulation comparisons

See [controlled installed-build comparisons](cuda.md#controlled-native-accumulation-comparisons)
for fixed configurations, independently tuned builds, and numerical checks.

### Optional delayed-scaling recipe

`scaling_recipe="delayed"` adds projection-owned histories, immutable invocation
scales, single-read CUDA conversion/amax collection and applicable fused bias
preprocessing. Current scaling remains the compatibility default. See the
[delayed-scaling contract and TE Linear comparison](delayed_scaling.md) for
configuration, accumulation boundaries, checkpoint recomputation, distributed
state, numerical differences and target-GPU validation.
