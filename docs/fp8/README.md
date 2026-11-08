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
FP8 uses Triton's native accumulation policy, retaining the chained FP32
accumulator in tensorwise GEMMs. DeepOpt no longer supplies
`max_num_imprecise_acc`. Omitting this argument follows Triton's backend default
and differs numerically from the earlier forced limit of 32; see the
[Triton semantic implementation](https://github.com/triton-lang/triton/blob/v3.9.0/python/triton/language/semantic.py#L1477).
The reference backend retains its FP32 emulation. Remove the obsolete setting
from configuration and execution plans when starting a new run; it now raises
a migration error instead of being silently ignored.

Checkpoints record numerical recipe version 2, the accumulation policy, and
the Triton version for native FP8. Exact resume rejects missing/changed recipes
or changed Triton versions before loading training state. Continue older FP8
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
from deepopt.opt.do_nn.functional import do_linear

x = torch.randn(4, 8, requires_grad=True)
weight = torch.nn.Parameter(torch.randn(3, 8))
bias = torch.nn.Parameter(torch.zeros(3))
output = do_linear(x, weight, bias, plan=plan)
output.square().mean().backward()
```

For an existing module, keep its parameter objects through the adapter:

```python
from deepopt.opt.do_nn.linear_module import DOLinear

original = torch.nn.Linear(8, 3)
adapted = DOLinear.from_linear(original, plan=plan)
optimizer = torch.optim.AdamW(adapted.parameters(), lr=1e-3)
```

The same pattern is available as `DOHFConv1D.from_hf_conv1d(...)` and
`DOMultiheadAttention.from_mha(...)` in their respective
[HF projection](../../src/deepopt/opt/do_nn/hf_conv1d_module.py) and
[attention](../../src/deepopt/opt/do_nn/mha_module.py) modules. Use these adapters
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
`converted_modules`, `function_calls`, `untraced_modules`, and `fallback_reason`.
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
from deepopt.opt.do_nn.linear_module import DOLinear

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
| HF Transformers `Conv1D` | `DOHFConv1D` | Keeps stored `[in, out]` weights and transposes for the linear operation |
| `torch.nn.MultiheadAttention` | `DOMultiheadAttention` | FP8 Q/K/V and output projections |

HF `Conv1D` is a linear projection, not `torch.nn.Conv1d`. The name `DOConv1D`
remains reserved for a future actual convolution adapter. Exact registered module
types are matched; arbitrary subclasses need an explicit conversion rule.
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

Use a CUDA-enabled PyTorch installation and compatible Triton exposing
`triton.tools.tensor_descriptor.TensorDescriptor`. The TMA kernel requires an
NVIDIA GPU with compute capability at least 9.0. There is no CPU fallback for a
requested Triton backend.

```bash
python examples/optimization/fp8_training.py --device cuda --backend triton
python examples/optimization/fp8_training.py --device cuda --backend triton --block-size 32 --precision bf16
```

The Triton backend supports independent `e4m3fn`/`e5m2` formats. Fused CUDA
quantization requires power-of-two block dimensions other than `[1, 1]`.
For example, activation `[1, 32]` with weight and gradient
`[32, 32]` is supported. Pipeline stages are tuned automatically by default;
an explicit integer from 1 through 5 fixes the depth.

Tensorwise scaling and blockwise scales whose reduction-axis widths are multiples
of 32 use native FP8 dot products. Other layouts use fused per-element FP32
dequantization followed by a `tf32x3` dot product. Rowwise activation and gradient
roles use the independently quantized orientations described above to retain
native FP8 reductions. Other transposes may still require the fallback when
their reduction-axis scale widths are unaligned.
The fallback preserves the stored FP8 values and scales without additional
FP8 quantization, but may have lower throughput than native FP8 dot products.
The backend packs strided/misaligned operands, pads partial tiles, and returns
FP32/BF16/FP16 results. Correctness/performance must be measured on the target GPU.

The kernel automatically uses Triton's native `@triton.autotune` to select
matrix tile sizes, warp counts, pipeline stages, grouped launch ordering, and
eligible split-K factors. The search measures at most 48 configurations,
retaining the historical tile/warp/stage choice as a measured candidate.
Compatible CUDA configurations from
[Triton's GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
are reserved in the measured set, including 128x256/256x128 FP8 tiles with
K=128 and the two-warp, five-stage small tiles. Shape-aware pruning still
excludes inappropriate sizes and incompatible scale boundaries. Up to 12
additional heuristic tile/reduction families supply narrow and split-K choices.
Candidates include small tiles along long axes to increase parallelism for
skinny products, two-warp small tiles, and larger tiles for dense products.
Shortlisting ranks estimated padded work and memory traffic, reserves narrow-axis
alternatives, and distributes the budget across reduction widths and split
factors before stage/warp variants. Candidate-list order does not affect selection.
The shortlist reserves stage-depth coverage before filling the remaining budget.
An explicit `num_stages` from 1–5 constrains the search; `"auto"` searches pipeline
depths with matching loop and compiler stages. Candidates with incompatible
reduction-scale boundaries or resource requirements are pruned. Quantization
`block_shapes` stay independent of kernel tile sizes: a 32-wide reduction-scale
block still forbids 64/128-wide native FP8 reduction tiles.

Grouped launches traverse up to eight M tiles before advancing along N to
promote L2 reuse. Row-major order (`GROUP_SIZE_M=1`) competes with grouped order
(`GROUP_SIZE_M=8`); up to eight otherwise-identical configuration pairs isolate
the scheduling choice without consuming the entire search budget. When either
output axis has a single tile, the equivalent grouped variant is omitted.
The final partial M group and split-K planes use the same coordinate mapping.
These CUDA/TMA choices do not introduce a HIP backend.

Split-K factors 2, 4, and 8 compete with unsplit launches for skinny products
(an output axis at most 32, K at least 512, and at most 262144 output elements)
and dense products with K at least 4096. This admits ViT weight gradients such
as a 768x768 output reducing across 100864 tokens. The number of split planes
is capped by a 64 MiB FP32 scratch bound, including aligned row strides.
Pruning accounts for SM count, output-grid occupancy, and at least two reduction
tiles per split. A separate reduction adds bias once and converts to the
requested output dtype. Both kernels are included in candidate timing. Each
launch overwrites its partials, so cached launches need no clearing or atomics.
Cached unsplit winners allocate no partial buffer; cached split winners allocate
only their selected number of planes.
Different reduction orders can produce small floating-point differences.

Candidate timing replays the prepared GEMM in a CUDA graph with a
20 ms repetition parameter and compares median timings. This avoids Python launch gaps
and repeated L2-cache flushing dominating the ranking of short kernels.
Packing and quantization run before this isolated timing; the training loop
does not need CUDA Graph mode. The full FP8 path still has its documented
capture limitation. The search follows the shape-aware approach used by
[Inductor's GEMM heuristics](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/heuristics/template/triton.py)
and the descriptor-based tuning in
[Triton's TMA matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html).

Packing uses only the stride alignment required by TMA. Logical descriptor
bounds handle partial tiles through zero-filled loads and discarded stores
outside the matrix. Buffers therefore do not require padding to the largest
tile in the candidate set, including on calls that reuse a cached winner.
Already aligned row-strided FP8 views are reused directly. Other layouts use
one copy/padding kernel that overwrites every destination byte; fully overwritten
buffers do not incur a preliminary zero fill.

Autotuning compiles and benchmarks candidates on the first call for each new
shape, operand/output format, scale layout, prepared row strides, bias policy,
pipeline policy, or CUDA device. Forward, input-gradient, and weight-gradient products have
different shapes; incomplete training and validation batches introduce more
shapes. These searches add first-use latency, especially in short runs with
small projections. Matching calls reuse the in-process winner. Set
`TRITON_PRINT_AUTOTUNING=1` to inspect selected configurations. A bounded search
does not guarantee the fastest possible kernel; measure on the target GPU.

On Triton runtimes supporting `cache_results`, tuning timings and winners also
persist in the native Triton cache. A launch adapter sets descriptor block shapes
for benchmarks, cached calls and warmups, so normal configurations contain no
`Config.pre_hook` that would disable Triton 3.9 result serialization. See the
[native cache implementation](https://github.com/triton-lang/triton/blob/v3.9.0/python/triton/runtime/autotuner.py).
The cache is invalidated by the kernel, target, environment, tuning key and
pruned configurations. A new key still pays compilation and search costs;
compiled-kernel caching alone is not tuning-result persistence.
Static capability/SM metadata is cached per explicit CUDA device. Descriptors,
outputs and split-K buffers remain owned by each invocation; no pointer-bound
descriptor or global scratch storage is shared across concurrent launches.

Benchmark the DeepOpt library installed in your active Python environment on
the target GPU with the [projection benchmark](../../benchmarks/benchmark_fp8_autotune.py).
For an editable installation, run `python -m pip install -e .` from the checkout
first. The script uses normal library imports and can be copied outside the
repository; it does not select a source tree or modify Python's import path.
To compare library versions, run it in the corresponding installed environments.

```bash
python benchmarks/benchmark_fp8_autotune.py
# Use an explicit pipeline depth with the installed library.
python benchmarks/benchmark_fp8_autotune.py --num-stages 3
# Dense projection with scale widths that allow K=128 native FP8 tiles.
python benchmarks/benchmark_fp8_autotune.py --batches 1024 --input-features 1024 --output-features 1024 --scale-k 128
# Isolate quantization, including independently scaled transpose views.
python benchmarks/benchmark_fp8_autotune.py --scope quantize
python benchmarks/benchmark_fp8_autotune.py --scope quantize --no-check-finite
python benchmarks/benchmark_fp8_autotune.py --scope scale --no-check-finite
python benchmarks/benchmark_fp8_autotune.py --scope dequantize
# Larger scale groups use chunked fusion rather than the unfused fallback.
python benchmarks/benchmark_fp8_autotune.py --scope quantize --batches 1024 --input-features 1024 --output-features 1024 --scale-k 256 --no-check-finite
# Actual ViT-B projection dimensions: 512 samples x 197 tokens, tensorwise.
python benchmarks/benchmark_fp8_autotune.py --preset vit --scaling tensorwise --grad-format e4m3fn --implementation both --num-stages 3 --no-check-finite --iterations 10 --output results/fp8-vit.json
# Compare an identical supported tile in two separately installed builds.
# Start with one projection shape; scales and scratch compatibility are checked.
python benchmarks/benchmark_fp8_autotune.py --scope gemm --scaling tensorwise --batches 100864 --input-features 768 --output-features 768 --gemm-config 128 128 64 8 3 1 1 --no-check-finite --output results/fp8-fixed.json
```

Use `--output PATH` to save the complete combined JSON report, including timings,
environment details, numerical errors, and autotuning diagnostics, for sharing
or later analysis. Missing parent directories are created and an existing file
is replaced after the benchmark completes successfully. The same JSON is still
printed to stdout; progress and the saved-file location go to stderr.

For a detailed H100 comparison using the same tensorwise recipe:

```bash
python benchmarks/benchmark_fp8_autotune.py \
  --preset vit --implementation both --scaling tensorwise \
  --grad-format e4m3fn --num-stages 3 --no-check-finite \
  --reuse-cache --iterations 20 --repeats 5 \
  --detailed --profile-iterations 2 --profile-rows 30 \
  --output results/fp8-detailed.json
```

`--detailed` retains the synchronized wall timings and adds measurements made
in separate passes after warmup:

- `cuda_events`: elapsed time on the current CUDA stream, including gaps while
  Python submits work. This is not a sum of kernel durations.
- `memory`: allocated/reserved baseline and peak bytes for one invocation,
  plus the extra peak above baseline. Linear gradients are cleared before the
  baseline; the result and new gradients remain alive through sampling.
  Inputs, prepared operands and existing caches are resident at baseline.
  These numbers do not measure whole-model training memory or optimizer/EMA.
- `profile`: the top CPU operators by self time, CUDA runtime calls, and raw
  CUDA activities grouped by name. Times are normalized per invocation; counts
  cover all `--profile-iterations`. `--profile-rows` limits the ranked lists,
  while CUDA activity totals include every recorded activity. CPU totals are
  nested, and CUDA activities can overlap; neither should be added to wall time.
  Missing CUDA profiler activities are explicitly marked `unavailable`.
- `cuda_graph`: event timings of graph replay for prepared GEMMs, removing
  per-operation Python dispatch and reusing captured output/scratch storage.
  Other scopes are skipped; unsupported capture reports its error. Memory and
  profiling run before this operation's graph capture. Replay is a diagnostic
  comparison, not a claim that full training can use the same graph.

The prepared BF16 GEMM path uses detached operands, matching FP8's lack of
autograd bookkeeping in this scope. Full linear measurements retain autograd.
The default input is BF16; repeat with `--input-dtype fp32` and a different
output filename to measure FP32 projection inputs, as can occur after
normalization. Input gradients follow this dtype in both implementations.
The benchmark continues to import the installed library; detailed mode does not
change its kernels or runtime policy. `--host-profile` additionally records
warmed Python submission with cProfile, separately from Kineto. Instrumented
wall time can contain synchronous waits; nested cumulative times are not additive.
CUPTI/profiler or graph failures are
recorded in the JSON so other available measurements remain usable.

By default it uses generated tensors for a `512 -> 10` CIFAR-10 classifier,
with batch sizes 2048, 848 (training tail), and 1808 (validation tail).
`--input-features`, `--output-features`, and `--batches` select other projection
shapes; `--scale-k` selects a power-of-two scale width from 32 through 1024 (default 32).
The report records group sizes, stages, and split factors for winners and trials.
`--scaling tensorwise` makes all operand block shapes `None`; the default
`blockwise` recipe is unchanged. Independent `--activation-format`, `--weight-format`,
and `--grad-format` select the actual operand formats (historical benchmark
defaults are E4M3FN/E4M3FN/E5M2). `--implementation both` runs FP8 and BF16 in
separate fresh processes. The ViT preset measures qkv, attention projection,
fc1, fc2 and head projections, including their forward/dX/dW products; body
rows equal `--batches` times `--tokens-per-sample` (default 197). It is not a
whole-model or epoch benchmark. Defaults are 512, 336 and 272 samples for ViT,
covering the training batch and training/validation tails; classifier
defaults remain 2048/848/1808.

Full-linear and BF16 accuracy reporting compares outputs and gradients with
IEEE FP32 after timing. Prepared FP8 GEMMs separately report quantization error,
accumulation plus output-cast error, and total error against tiled FP64
references. The quantized reference applies scale inverses in FP64, avoiding an
intermediate FP32 dequantization round. It bounds reference temporaries by
processing output rows and the reduction dimension in chunks. All error reports
include relative L2, maximum absolute, RMS and finite-value results.
The accumulation/output-cast category also includes FP32 scale application
rounding. It is separate from the error of choosing FP8 values and scales.
`--input-pattern cancellation`, `zeros`, and `mixed-magnitude` exercise numerical
stress cases; use both FP8 formats and irregular feature dimensions as well.
Use `--no-accuracy` only when those extra comparisons are unwanted. Candidate
records also include wall time for compilation plus benchmarking, and the worker
reports whether native tuning-result caching is enabled. These measurements
separate cold searches from warmed execution; they do not attribute data loading
or other epoch costs to autotuning.
`--scope quantize`, `--scope scale`, and `--scope dequantize` report each operation's
allocation/latency, operand shapes, and strides independently of GEMM.
Use `--scope quantize-pair --scaling tensorwise` for the actual converter that
emits both aligned layouts; ordinary `quantize` measures one layout.
`--scope bias-gradient` isolates the existing FP32 bias sum to establish whether
collecting partial bias sums during conversion would repay its added traffic.
Dequantization prepares FP8 inputs outside the timed region. Finite checks are enabled by default;
`--no-check-finite` applies the unchecked policy to all measured scopes and
requires a library version that supports the explicit check override. Use it
to compare unchecked updates; keep checks enabled when comparing strict runs.
Each implementation and measurement
scope runs in a fresh process with the same Python interpreter and an isolated Triton compilation cache by
default; `--reuse-cache` retains the normal disk cache. Reports separate first-use
latency from synchronized warmed timings, identify the imported DeepOpt package
path and installed distribution version, and record
the installed Python-source fingerprint, Triton version and numerical recipe.
They include cached winning tiles and
per-key candidate median/p20/p80 device timings when the library exposes a tuner.
Unavailable candidates are marked explicitly. Timing instrumentation runs only
during cold searches. Use `--num-stages auto` for the new default, or an integer
to compare installed versions that predate automatic stages. The full projection measurement
includes quantization, forward, backward, and gradient-buffer reset; it excludes
the convolutional backbone, optimizer, and data loading. Prequantized GEMM
measurements capture the imported library's actual autograd operands, including independently
quantized rowwise transposes, before timing. They include packing, padding,
dispatch, and split scratch allocation, so they are not kernel-only timings.
Candidate timings include split reduction but exclude allocation and packing;
compare warmed wrapper and full-projection timings as well. Neither CPU tests
nor this benchmark's existence establishes a GPU
speedup.

## Black-box and white-box validation

Install optional real-model dependencies without downloading pretrained weights:

```bash
python -m pip install -e '.[model-tests]'
DEEPOPT_REQUIRE_MODEL_TESTS=1 python -m unittest discover -s tests/opt -v
python -m unittest discover -s tests/distributed -p 'test_fp8_*.py' -v
```

For a fresh Linux/Windows CPU environment, install PyTorch and torchvision
together from the CPU wheel index **before** installing the model-test extra:

```bash
python -m pip install 'torch>=2.4,<3' 'torchvision>=0.19,<1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[model-tests]'
python -m pip check
```

Mixing CPU PyTorch with a torchvision binary from a different build channel can
cause `RuntimeError: operator torchvision::nms does not exist`. That import error
also affects timm, Transformers and TorchMetrics. Recreate the environment with
a matching pair instead of skipping the model tests or registering a fake NMS
operator. For macOS or CUDA, use the corresponding matching pair from the
[official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/).
The CPU CI job installs both wheels from the CPU index and executes real NMS
before running the test groups.

The tiny randomly initialized real architectures are torchvision ViT, timm ViT,
HF GPT-2, and HF Llama. Black-box tests compare outputs/gradients against dense or
independently quantized references, perform optimizer updates, and verify exact
checkpoint continuation. White-box tests inspect replacement identity, tied
embeddings, graph targets, saved FP8 tensors/scales, gradient dispatch, and DDP
reducer ownership. Native attention tests with quantization disabled isolate
adapter semantics from quantization error.

On a compatible GPU:

```bash
DEEPOPT_REQUIRE_MODEL_TESTS=1 DEEPOPT_REQUIRE_CUDA_FP8=1 \
  python -m unittest discover -s tests/cuda -p 'test_fp8_*.py' -v
```

CUDA tests execute actual Triton kernels against dequantized FP32 or reference
oracles, covering edge tiles, scale boundaries, strided layouts, backward,
real-model training, and conditional two-GPU NCCL resume. One GPU runs the kernel
and full-model cases; two compatible GPUs and NCCL are needed for DDP cases.
The required-environment switches turn missing prerequisites into failures.
Ordinary local discovery skips unavailable hardware/packages explicitly.
The existing opt-in GPU CI job requires two FP8-capable GPUs and model packages
so an all-skipped FP8 job cannot be mistaken for validation.

CPU/reference validation was executed in the development environment with
PyTorch 2.14.1, torchvision 0.29.1, Transformers 4.57.6 and timm 1.0.31. No CUDA
device is available in that macOS environment; actual Triton/NCCL FP8 results
remain pending a compatible GPU run. CPU passes and hardware skips are not GPU
validation or performance results.

## Future activation compression and sparse operations

The DO names and conversion registry are operation-oriented, rather than tied to
one optimization. Keep future features at the existing ownership boundaries:

| Layer | Current responsibility | Future extension location |
| --- | --- | --- |
| DO modules | Parameters, layouts, module semantics | Choose operation policy while retaining ownership and checkpoint names |
| Shared functional operation | Execute a projection under a validated plan | Compose implemented FP8, compression, and sparse operation policies |
| Custom autograd function | Save required operands and compute gradients | Encode/decode saved activations with reconstruction metadata; maintain backward lifetime |
| GEMM backend | Execute tensor/block-scaled products | Sparse layouts, sparse kernels, and forward/backward dispatch |
| FX registry/matcher/rewriter | Find and replace eligible operations | Register new DO operations without adding training logic to rewriting |

All three adapters delegate projections through the shared functional boundary.
The autograd path keeps saved-state handling separate from backend execution and
saves only tensors needed by requested gradients. Future activation compression
must preserve scales/layout/dtype metadata and autograd saved-tensor lifetime;
sparse execution must implement the corresponding backward products as well as
forward. Neither feature is enabled by placeholder flags. The current plan
validator accepts only the implemented FP8 schema; new policy fields should be
added with their implementation and tests.

### Controlled native-accumulation comparisons

Use two environments with separately installed baseline and candidate builds,
with identical Torch/Triton versions. Copy the same benchmark script outside the
checkout and run it with each environment's Python. The script uses normal
library imports and records an installed-source SHA256 fingerprint, recipe,
formats, input dtype and selected configurations. A legacy library is identified
from its API; the script never injects the removed accumulation setting.

First compare one fixed configuration to isolate the accumulation change:

```bash
python benchmark_fp8_autotune.py --scope gemm --preset vit \
  --scaling tensorwise --grad-format e4m3fn --input-dtype bf16 \
  --gemm-config 128 128 64 8 3 1 1 --num-stages 3 \
  --no-check-finite --output fixed.json
```

Then omit `--gemm-config` and use `--num-stages auto` to tune each build afresh.
Use separate output files and initially omit `--reuse-cache` to keep compilation
and tuning caches isolated. The changed JIT source also gives native accumulation
its own kernel/cache identity. Repeat GEMM diagnostics with `--input-pattern
cancellation`, `mixed-magnitude` and `zeros`, both FP8 formats, and irregular
feature dimensions. Fixed configurations must satisfy each product's scale,
stage, occupancy and split-scratch constraints; invalid choices fail before
launch. Fixed mode adds a small Python validation wrapper and bypasses winner
caching; compare fixed runs with fixed runs rather than mixing timing modes.

Measure full projections with `--scope linear --implementation both --detailed
--host-profile`. Repeat with `--input-dtype fp32`; also measure `quantize-pair`,
`scale` and `bias-gradient` separately. Keep input formats, current-tensor scaling
and finite-check settings identical across runs. Projection sums and graph replay
are diagnostics, not full-model latency or training capture.

Native accumulation changes numerical behavior. Before accepting a performance
result, run matched short full-model training comparisons from the same initial
model/optimizer state, batches and augmentation seeds, with the same EMA,
diagnostics and checkpoint cadence. Compare loss and validation accuracy, startup
and steady-state timings, and allocated versus reserved memory separately; then
confirm with a longer run. Warmed weight-cache reuse does not represent a weight
update on every step. Retain the deterministic ordered split-K reduction.

Further consumer/producer fusion, bias-sum fusion, head exclusions and delayed
scaling remain measurement-gated experiments. Delayed scaling would require a
separate recipe and checkpointed history. The native accumulation, static device
metadata and selective-layout changes do not establish a GPU speedup or a
training-quality result without these target-hardware measurements.
