# Configuration reference

[Documentation index](../README.md) · [Loading configuration and managed runs](../runs/README.md) · [Annotated TOML template](../../src/deepopt/configs/templates/complete.toml)

This reference lists every framework configuration key accepted by
[`config_from_dict()`](../../src/deepopt/config.py), including nested policies.
TOML tables and Python dictionaries use the same structure. Defaults below are
the behavior when a key is omitted, before explicit API overrides. Example values
in the complete template, such as five epochs and batch size 32, are choices for
that example rather than defaults for required fields.

## Loading and precedence

```python
import deepopt

deepopt.load_config("./experiment.toml")
settings = deepopt.get_config()  # Independent copy of the active defaults.
deepopt.reset_config()          # Existing components keep their snapshots.
```

`load_config()` accepts a complete dictionary, an `ExperimentConfig`, a TOML
path, or `"builtin:NAME"`. It validates and replaces process-local defaults;
it does not start training, initialize CUDA/DDP, or capture graphs. A failed
load preserves the previous defaults. Each distributed process loads its own
configuration. `config_from_dict()` and `deepopt.configs.load_preset()` validate
without activating global defaults.

For configuration-consuming components, precedence is:

1. Explicit API arguments, including meaningful `False` and `None`.
2. A per-call `config={...}` dictionary, recursively merged into inherited settings.
3. Loaded global defaults, or the component's existing snapshot where applicable.
4. The API's own defaults.

Lists replace lists. A complete typed configuration, path, or built-in preset
selects an independent source. In contrast, `load_config(dictionary)` requires
a complete source and replaces the global configuration; it does not merge it.
Conflicting policies still raise errors after merging. Constructed trainers and
runtimes keep their snapshots when global defaults change. An optimizer or
scheduler you supply keeps its own settings.

Unknown framework keys are errors. Store recipe-specific settings under
`application`. TOML has no null literal: omit nullable keys to use their defaults;
Python dictionaries can supply `None`. `training.monitor = false` is a special
supported TOML spelling for disabling best-checkpoint selection. Integers do not
accept booleans; numeric settings must be finite.

Generate an editable template using the installed library:

```python
from deepopt import load_config_template, write_config_template

options = load_config_template()  # Fresh dictionary; does not activate it.
write_config_template("./experiment.toml")  # Refuses to overwrite an existing file.
```

## Top-level keys

| Key | Default / accepted values | Meaning |
| --- | --- | --- |
| `name` | Required nonempty string | Experiment name used in run metadata and log identification. |
| `seed` | Required integer, 0 through 2³²−1 | Seed used by configured setup and loaders. Loading alone does not reseed a running program. |
| `training` | Required table | Training policy described below; three fields are required. |
| `runtime` | Optional table | Device, precision, diagnostics and execution policies. |
| `loader` | `{}` | Defaults consumed by the loader helper. |
| `logging` | `{}` | Console/file policy and optional epoch memory reports. |
| `schedules` | `{}` | Named coefficient schedules read by your step function. |
| `application` | `{}` | Caller-owned model, dataset, task or other recipe settings. No automatic model/dataset construction. |
| `experiment_identity` | `{}` | Caller-owned semantic identity checked on resume, such as `recipe_version` or `dataset_version`. These names are examples, not predefined fields. |

`application` and `experiment_identity` accept nested string-keyed dictionaries,
lists, strings, booleans, integers, finite floats and Python `None`. Encode paths
and dates as strings. Python callables, tensors, models and datasets stay in code.
The old top-level `data` section is rejected; recipe data options belong under
`application`, and datasets are supplied through Python factories.

## Training: `[training]`

| Key | Default / accepted values | Meaning |
| --- | --- | --- |
| `epochs` | Required integer ≥ 1 | Total target epochs, including epochs already completed when resuming. |
| `batch_size` | Required integer ≥ 1 | Physical microbatch size; per rank with DDP. Managed automatic batch planning can select a different resolved size. |
| `learning_rate` | Required number > 0 | Initial LR for the configured optimizer factory. |
| `accumulation_steps` | `1`; integer ≥ 1 | Microbatches per optimizer update, subject to final partial-window handling. |
| `optimizer` | `"adamw"`; `"adamw"`, `"sgd"`, `"custom"` | Configured optimizer choice. `custom` requires an optimizer factory. A supplied optimizer is not reconfigured. |
| `weight_decay` | `0.0`; number ≥ 0 | Weight decay passed by the configured optimizer factory. |
| `momentum` | `0.9`; number in [0, 1) | Momentum for configured SGD. |
| `lr_gamma` | `1.0`; number in (0, 1] | Legacy multiplicative LR decay at `scheduler_interval`. Must stay 1 when using `lr_schedule`. |
| `grad_clip_norm` | `None`; number > 0 when set | Maximum gradient norm for configured clipping. Omission disables configured clipping. |
| `monitor` | `"loss"`; metric name, `false`/`None` | Validation metric used for best selection. Use `false` in TOML to disable it. Timing/count fields `samples`, `batches`, `seconds`, `samples_per_second` cannot be monitors. |
| `monitor_mode` | `"min"`; `"min"` or `"max"` | Whether a smaller or larger monitored value is better. |
| `checkpoint_every_steps` | `0`; integer ≥ 0 | Optimizer-update interval for step checkpoints in managed runs. Zero disables them; a positive interval requires resumable loading. |
| `scheduler_interval` | `"epoch"`; `"epoch"` or `"step"` | LR scheduler advancement interval. `step` means completed optimizer updates, not microbatches. |
| `lr_schedule` | `None`; policy table | Native LR policy; see the next table. |
| `ema` | `{enabled = false, decay = 0.999}` | Exponential moving average policy described below. |

Direct training still needs the relevant APIs or callbacks: configuring a
checkpoint interval does not by itself create checkpoint files in an arbitrary
custom loop. See [training](../training/README.md),
[optimization](../optimization/README.md) and [checkpoints](../checkpoints/README.md).

### Learning rate: `[training.lr_schedule]`

Choose one `type`. Only fields for that type are accepted. Durations and
milestones use `training.scheduler_interval` units. All integer horizons are
positive; all listed numbers must be finite.

| `type` | Required fields | Optional fields and defaults | Behavior |
| --- | --- | --- | --- |
| `"constant"` | None | None | Keep the initial LR. |
| `"step_decay"` | `step_size`: integer ≥ 1 | `gamma = 0.1`, in (0, 1] | Multiply LR by gamma every `step_size` intervals. |
| `"exponential_decay"` | `gamma`: number in (0, 1] | None | Multiply LR by gamma at every interval. |
| `"piecewise"` | `milestones`: strictly increasing list of positive integers | `gamma = 0.1`, in (0, 1] | Multiply LR by gamma at each milestone. |
| `"linear_warmup"` | `warmup_steps`: integer ≥ 1 | `start_factor = 0.1`, in (0, 1] | Raise LR from a fraction of the initial LR to the initial LR. |
| `"cosine_decay"` | `total_steps`: integer ≥ 2 | `eta_min = 0.0`, number ≥ 0 | Cosine LR decay toward an absolute minimum LR. |
| `"warmup_cosine"` | `warmup_steps`: integer ≥ 1; `total_steps` > `warmup_steps` | `start_factor = 0.1`, in (0, 1]; `eta_min = 0.0`, number ≥ 0 | Warm up, then cosine decay. |

Do not combine a non-default `lr_gamma` with `lr_schedule`. These LR policies
are separate from named coefficient schedules under `schedules`.
See [scheduling](../scheduling/README.md) for advancement and resume semantics.

### EMA: `[training.ema]`

| Key | Default / accepted values | Meaning |
| --- | --- | --- |
| `enabled` | `false`; boolean | Update averaged parameters after optimizer steps and use them for validation. Training uses live parameters. |
| `decay` | `0.999`; number in [0, 1) | Old-shadow weight in the EMA update. |

Validation uses current model buffers. Checkpoints retain live and averaged
parameters; EMA adds parameter-shadow storage.

## Runtime: `[runtime]`

| Key | Default / accepted values | Meaning |
| --- | --- | --- |
| `device` | `"cpu"`; `"cpu"`, `"cuda"`, `"cuda:N"`, `"auto"` | Execution device. `auto` lets setup choose the available device. |
| `precision` | `"fp32"`; `"fp32"` or `"bf16"` | Trainer/runtime compute precision. BF16 selects CUDA AMP; CPU trainer execution supports FP32. FP16 training is not supported by this setting. |
| `deterministic` | `true`; boolean | Request deterministic execution in configured setup. This is independent of diagnostic mode. |
| `checks` | `true`; boolean | Legacy finite checks when no diagnostic policy is selected. |
| `execution_mode` | `None`; `"max_speed"`, `"balanced"`, `"analyze"` | Diagnostic preset. Mutually exclusive with `diagnostics`. This does not enable FP8, compilation or graph capture. |
| `diagnostics` | `None`; table | Custom diagnostic policy; see below. |
| `mode_schedule` | `None`; table | Diagnostic transitions keyed by positive optimizer-update numbers as strings, e.g. `{"100" = "analyze", "200" = "max_speed"}`. Values are the three execution modes. |
| `non_blocking` | `false`; boolean | Request nonblocking batch transfers. Pinned host memory can help CPU-to-CUDA transfers. |
| `compile` | Disabled policy | Model compilation settings below. |
| `fp8` | Disabled policy | FP8 execution settings below. |
| `activation_compression` | Disabled policy | Saved-activation compression settings below. |
| `ddp` | `{}` | Distributed wrapper options below; does not launch processes. |
| `cuda_graph` | `{}` | Defaults for explicit capture operations below; does not capture automatically. |

### Diagnostics: `[runtime.diagnostics]`

Omitted overrides inherit the selected preset. If `mode` is omitted in this
table, it is `"balanced"`.

| Key | Accepted values | Meaning |
| --- | --- | --- |
| `mode` | `"max_speed"`, `"balanced"`, `"analyze"` | Base preset. |
| `loss_finite_every` | Integer ≥ 1 or Python `None` | Check loss/report finiteness at this optimizer-update cadence; `None` disables it. |
| `gradient_finite_every` | Integer ≥ 1 or Python `None` | Gradient finite-check cadence; `None` disables it. |
| `inspect_weights` | Boolean | Enable passive weight inspection. |
| `inspect_activations` | Boolean | Enable passive activation inspection. |

| Preset | Loss interval | Gradient interval | Inspect weights | Inspect activations |
| --- | --- | --- | --- | --- |
| `max_speed` | Disabled | Disabled | `false` | `false` |
| `balanced` | `100` | `100` | `false` | `false` |
| `analyze` | `1` | `1` | `true` | `true` |

`max_speed` cannot enable passive diagnostics through overrides. `analyze`
also adds profiler ranges. A mode-schedule transition applies before the named
one-based update and remains active until the next transition; it selects that
mode's preset. Omitting both policy selectors retains legacy `checks` behavior.
TOML omission retains a preset's intervals; to explicitly disable an individual
interval, use a Python dictionary with `None`.
See [execution policies](../execution/README.md) and [observers](../observers/README.md).

### Compilation: `[runtime.compile]`

| Key | Default / accepted values | Meaning |
| --- | --- | --- |
| `enabled` | `false`; boolean | Compile the prepared model when constructing an execution core. |
| `backend` | `"inductor"`; nonempty string | Installed `torch.compile` backend, such as `inductor`, `eager` or `aot_eager`. Availability is checked when used. |
| `mode` | `None`; `"default"`, `"reduce-overhead"`, `"max-autotune"`, `"max-autotune-no-cudagraphs"` | Compiler mode. Cannot coexist with nonempty `options`. |
| `fullgraph` | `false`; boolean | Require the compiler to capture the entire model forward rather than allowing graph breaks. |
| `dynamic` | `None`; boolean when set | `None` leaves shape handling to PyTorch; `true` requests dynamic shapes; `false` specializes shapes. |
| `options` | `{}`; table | Backend-specific nonempty string keys with string, boolean, integer or finite float values. |

Preparation happens before compilation. Current custom DO dispatch/autograd
boundaries run outside Dynamo; enabling compilation does not guarantee fullgraph
support or a speedup. See [compilation](../execution/README.md).

## FP8: `[runtime.fp8]`

The FP8 policy is independent of runtime AMP precision and activation compression.
The reference backend emulates FP8 for correctness; Triton kernels require
compatible NVIDIA hardware and libraries. See [FP8](../fp8/README.md) and
[CUDA execution](../fp8/cuda.md).

| Key | Default / accepted values | Meaning |
| --- | --- | --- |
| `enabled` | `false`; boolean | Enable supported DO FP8 operations and automatic model preparation in the trainer/runtime. |
| `backend` | `"reference"`; `"reference"` or `"triton"` | Reference emulation or CUDA Triton execution. |
| `trace_mode` | `"modules"`; `"modules"`, `"full"`, `"auto"` | Preparation boundary: preserve the Python root with converted children; require a rewritten FX graph; or try full tracing and fall back to module preparation. |
| `activation_dtype` | `"e4m3fn"`; `"e4m3fn"` or `"e5m2"` | FP8 activation operand format. |
| `weight_dtype` | `"e4m3fn"`; `"e4m3fn"` or `"e5m2"` | FP8 weight operand format, chosen independently. |
| `grad_dtype` | `"e4m3fn"`; `"e4m3fn"` or `"e5m2"` | FP8 gradient operand format, chosen independently. |
| `num_stages` | `"auto"`; `"auto"` or integer 1–5 | Search Triton pipeline depths or pin a depth within autotuning. |
| `out_dtype` | `None`; `"fp32"`, `"bf16"`, `"fp16"` | Dense result dtype where this policy applies. `None` follows active AMP/input dtype. This is not an FP8-output switch. |
| `intercept_linear` | `true`; boolean | Intercept native functional linear calls inside the FP8 autocast scope. `false` leaves those calls dense while prepared DO adapters still use FP8. |
| `intercept_convolution` | `None`; boolean when set | Override native ordinary/transposed convolution interception in 1D/2D/3D; omission inherits `intercept_linear`. |
| `intercept_normalization` | `None`; boolean when set | Override native normalization interception; omission inherits `intercept_linear`. |
| `intercept_pooling` | `None`; boolean when set | Override native pooling/unpooling interception; omission inherits `intercept_linear`. Explicit DO calls and prepared modules retain their selected policy. |
| `scaling_recipe` | `"current"`; `"current"` or `"delayed"` | Compute scales from current values, or cast using saved scales and record observations for future updates. |
| `delayed_scaling` | Default recipe table below | Delayed-history policy. |
| `block_shapes` | All three operands `None` | Tensorwise or rectangular per-operand scaling; see below. |
| `backward_backend` | `"inherit"`; `"inherit"`, `"native_fast"`, `"native_precise"` | Use the selected backend or native tensorwise scaled-GEMM backward. `native_fast` enables fast accumulation; `native_precise` disables it. Native choices require `backend="triton"` and all block shapes omitted/`None`. |
| `backward_precision` | `"native"`; `"native"` or `"auto"` | Preserve native activation-gradient boundaries, or analyze supported DO backward edges once and relay FP8 gradients where compatible. Requires prepared model execution for automatic planning. |
| `exclude_modules` | `[]`; list of nonempty qualified-name glob strings | Keep selected modules dense during preparation. Nonempty exclusions require `intercept_linear=false`; explicit convolution/normalization/pooling interception cannot be true. |

Backward precision planning is cached per call site and train/eval path. Native
operations, parameter gradients, accumulation and unsupported edges remain
native; selecting `auto` does not make every tensor FP8. It is distinct from
`backward_backend`, which selects computation kernels. Normalization/ReLU forward
output planning likewise chooses FP8 only when consumers support the layout.
See [backward precision](../fp8/backward_precision.md),
[normalization](../normalization/README.md) and [ReLU](../activation/README.md).

### Scaling tiles: `[runtime.fp8.block_shapes]`

| Key | Default / accepted values | Meaning |
| --- | --- | --- |
| `activation` | `None`; two positive integers `[rows, cols]` | Activation scaling rectangle in the operation's matrix layout. |
| `weight` | `None`; two positive integers `[rows, cols]` | Weight scaling rectangle. |
| `grad` | `None`; two positive integers `[rows, cols]` | Gradient scaling rectangle. |

Omit an operand to use one scale for its whole tensor. Triton fused conversion
requires power-of-two dimensions, excluding `[1, 1]`; the CPU reference accepts
arbitrary positive dimensions. Native FP8 dot also needs compatible reduction-axis
scale widths that are multiples of 32. Other layouts, including incompatible
transposed layouts, use fused dequantization and TF32x3 arithmetic. Delayed scaling
and the native backward choices require tensorwise scaling for every operand.
See the [CUDA arithmetic rules](../fp8/cuda.md) and
[spatial convolution guide](../convolution/README.md).

### Delayed scales: `[runtime.fp8.delayed_scaling]`

| Key | Default / accepted values | Meaning |
| --- | --- | --- |
| `history_length` | `1024`; integer 1–4096 | Number of amax observations retained per owned operand. |
| `amax_algorithm` | `"max"`; `"max"` or `"most_recent"` | Select the maximum of the history or the latest observation for the next scale. |
| `margin` | `0`; integer −32 through 32 | Set the new scale to `(FP8_MAX / amax) / 2**margin`; positive values add range headroom. |
| `reduce_amax` | `true`; boolean | Synchronize amax observations across DDP ranks at update boundaries. `false` retains rank-local histories. |

Delayed scaling needs prepared state owners. Bare intercepted functional calls
cannot own the required persistent histories. With an optimizer-owned runtime,
scales remain frozen across the accumulation window and advance at successful
optimizer updates. Full-training CUDA Graph replay with delayed state updates
remains unsupported. See [delayed scaling](../fp8/delayed_scaling.md).

## Activation compression: `[runtime.activation_compression]`

| Key | Default / accepted values | Meaning |
| --- | --- | --- |
| `enabled` | `false`; boolean | Compress supported saved activations for backward. Independent of FP8 enablement. |
| `group_size` | `512`; integer ≥ 1 | Affine quantization group size for BF16 and tensorwise FP8 saved inputs. Blockwise FP8 uses its activation scale rectangles instead. |
| `bits` | `2`; integer `1`, `2`, `4` or `8` | Global saved-activation quantization precision. Lower values reduce payload size and can increase gradient error. |
| `attention_bits` | `None`; integer `1`, `2`, `4` or `8` | Override saved attention Q/K/V precision. Omission uses `max(4, bits)`. Explicit 1 or 2 bypasses that floor. |
| `norm_xhat_recenter` | `false`; boolean | Preserve each normalization domain's original xhat mean after decoding compressed backward state. Applies to BatchNorm2d, GroupNorm and LayerNorm when compression is enabled, independently of FP8. |

Attention's override affects only saved Q/K/V, not projection inputs, masks,
probabilities or outputs. Linear/convolution compression saves packed inputs;
normalization saves packed normalized `xhat` before affine parameters. Norm
groups stay within normalization domains, so `group_size` is an upper bound
there. ReLU saves an exact one-bit packed backward mask when compression is on,
regardless of global bits or group size.
ReLU6 also saves an exact one-bit mask. Sigmoid, SiLU, GELU and tanh compress
their preprocessed local derivatives with global `bits`; they do not compress raw
input. See [activation functions](../activation/elementwise.md).
LayerNorm uses the same normalization xhat policy: groups stay within its
normalized suffix and are bounded by that domain's length. It needs no separate
configuration section; see [LayerNorm](../normalization/layer_norm.md).
All [pooling families](../pooling/README.md) support this policy: average pooling
eliminates saved activation values, maximum/unpooling packs exact routing, and
LP quantize-packs preprocessed derivative factors. Routing width is determined by
its exact integer range, independently of lossy global bits. Public int64 indices
and required version guards remain accounted for in retained memory.

Norm recentering measures `mean(decoded_xhat - original_xhat)` while packing
and subtracts it inside every xhat-consuming backward kernel. Original `rstd`
validates the domain and scales dx; it does not determine the centering shift.
Forward output, FP8 output scales and codec payloads are unchanged. Retained
state grows by one FP32 shift per domain; weight-only LayerNorm also retains
rstd. The option defaults off, and checkpoint restore rejects a changed value.
See [normalization mean correction](../normalization/README.md#compressed-xhat-mean-correction).

| FP8 | Compression | Supported linear path |
| --- | --- | --- |
| Off | Off | Native execution with ordinary saved activations. |
| Off | On | BF16 execution with affine quantize-and-pack for saved inputs. |
| On | Off | FP8 execution with ordinary FP8 saved activations. |
| On | On | FP8 execution with fused casting and saved-input quantize-and-pack where supported. |

Compression approximates backward operands rather than changing the original
forward input used for computation. Fusion and temporary-workspace behavior
depend on the operation/layout; enabling both policies does not promise one
kernel for every operation. See [activation compression](../activation_compression/README.md).

## Distributed wrapper: `[runtime.ddp]`

The stored table defaults to `{}`; effective wrapper defaults are listed below.
These options do not initialize a process group or launch ranks. Batch size stays
per rank. See [distributed training](../distributed/README.md).

| Key | Effective default / accepted values | Meaning |
| --- | --- | --- |
| `find_unused_parameters` | `false`; boolean | Traverse the autograd graph to identify parameters unused by a forward. |
| `broadcast_buffers` | `true`; boolean | Broadcast model buffers during DDP training forwards. Delayed-scaling state has its own synchronization ownership. |
| `gradient_as_bucket_view` | `true`; boolean | Let parameter gradients share DDP bucket storage to reduce copies/storage. |
| `bucket_cap_mb` | PyTorch default, normally `25`; number > 0 | Gradient bucket capacity in MiB. |
| `static_graph` | PyTorch default `false`; boolean | Declare a stable used/unused parameter set and training graph for DDP. |
| `validation_sync_buffers` | `true`; boolean | Synchronize buffers for validation through DeepOpt's validation protocol, which supports unequal validation shards. |

## CUDA Graph capture defaults: `[runtime.cuda_graph]`

The stored table defaults to `{}`. These defaults are consumed by explicit
capture APIs; adding this table does not route training through a graph.

| Key | Effective default / accepted values | Meaning |
| --- | --- | --- |
| `device` | Runtime CUDA device when configured, otherwise core selection; Python `None` allowed | Explicit `"cuda"` or `"cuda:N"` capture device. |
| `warmup` | `3`; integer ≥ 1 | Warmup iterations before capture. |
| `allow_unused_input` | `false`; boolean | Permit inputs not used by captured autograd computation. |
| `autograd` | `true`; boolean | Capture an autograd-aware model/function; `false` selects forward-only capture. |
| `capture_error_mode` | `"global"`; `"global"`, `"thread_local"`, `"relaxed"` | PyTorch/CUDA capture error-detection mode. |

Streams, pools and the actual capture/close lifetime remain Python-owned.
Prepare the model before capture and validate supported shapes/state changes.
See [CUDA Graphs](../cuda/README.md).

## Data loading: `[loader]`

The table defaults to `{}` and is consumed by `build_loaders()` and the
configured loader factory. A loader you supply is not rebuilt from this table.

| Key | Effective default / accepted values | Meaning |
| --- | --- | --- |
| `num_workers` | `0`; integer ≥ 0 | Worker processes per loader/rank; zero loads in the main process. |
| `pin_memory` | `false`; boolean | Pin host tensor memory for device transfers. |
| `persistent_workers` | `false`; boolean | Retain workers across iterations/epochs. Requires `num_workers > 0`. |
| `prefetch_factor` | Omitted: DataLoader default (`None` with zero workers, normally 2 with workers); explicit integer ≥ 1 | Batches prefetched per worker. An explicit non-null value requires workers. The complete template dictionary explicitly supplies `None`. |
| `drop_last` | `false`; boolean | Drop an incomplete final batch. The helper forwards loader options to both training and validation. |

Dataset selection, transforms, `collate_fn`, custom samplers and extra loader
keywords stay in Python. `build_loaders()` selects resumable loading when the
configured step-checkpoint interval is positive unless you explicitly override
its `resumable` argument. See [data loading](../data/README.md).

## Logging: `[logging]`

The table defaults to `{}`; these are effective logger/trainer defaults.

| Key | Effective default / accepted values | Meaning |
| --- | --- | --- |
| `level` | `DEEPOPT_LOGGING_LEVEL`, otherwise `"INFO"` | Console threshold: Python logging level name or nonnegative integer. Explicit Python `None` selects environment/default behavior. The template explicitly sets `"INFO"`. |
| `file_level` | `"DEBUG"`; logging level name or nonnegative integer | File threshold, independent of console threshold. |
| `color` | `"blue"`; `black`, `red`, `green`, `yellow`, `blue`, `magenta`, `cyan`, `white`, `none` | Console color for all severities; names are case-insensitive. `none` disables it. Files have no ANSI colors. |
| `memory_each_epoch` | `false`; boolean | Emit a rank-zero memory report after each completed fit epoch. Reports do not reset allocator peaks. |

Logging environment controls are `DEEPOPT_LOGGING_LEVEL`,
`DEEPOPT_LOGGING_PREFIX` (default empty), and `DEEPOPT_LOGGING_COLOR` (default
`auto`, which colors terminals; `1` forces color and `0` disables it).
A nonempty `NO_COLOR` disables color. Configuration/API levels override the
environment when set. File output requires a logging-session destination or
managed run; loading settings alone creates no log files.
See [logging](../logging/README.md).

## Named coefficients: `[schedules.NAME]`

Choose an arbitrary nonempty coefficient name and one built-in `type` per table.
All curves also accept `interval = "update"` (default) or `"epoch"`.
Positions are one-based. Coefficients are independent of optimizer LR and are
not automatically bound to a loss; your step reads them from a
`ScheduleController`. All values must be finite.

| `type` | Required fields | Optional fields and defaults | Behavior |
| --- | --- | --- | --- |
| `"constant"` | `value`: number | `interval = "update"` | Return the same value. |
| `"linear_warmup"` | `start`, `end`: numbers; `duration`: integer ≥ 1 | `interval = "update"` | Interpolate from start to end, then hold end. |
| `"linear_decay"` | `start`, `end`: numbers; `duration`: integer ≥ 1 | `interval = "update"` | Same interpolation; caller chooses decreasing endpoints. |
| `"cosine_decay"` | `start`, `end`: numbers; `duration`: integer ≥ 1 | `interval = "update"` | Cosine interpolation, then hold end. |
| `"exponential_decay"` | `start`: number; `gamma`: number in (0, 1] | `duration = None`: integer ≥ 1 when set; `interval = "update"` | Return `start * gamma**(position - 1)`; a duration caps the position. |
| `"piecewise"` | `milestones`: strictly increasing integer list, each ≥ 2; `values`: number list of length `len(milestones) + 1` | `interval = "update"` | Select the next value at each milestone. |
| `"sequential"` | `schedules`: nonempty list of child policy tables; `durations`: one positive integer per child | `interval = "update"` | Run children in sequence with local positions, then hold the final segment's last value. Child intervals must match the parent. |

For interpolation curves, `duration=1` returns the end value immediately.
Custom schedule classes are supplied in Python. See [scheduling](../scheduling/README.md).

## Settings that remain in Python

There are no framework `[model]`, `[optimizer]`, `[attention]`, `[normalization]`,
`[memory]` or `[checkpoint]` tables. Model architecture, per-operation parameters
(such as normalization epsilon), callbacks, metrics and reconstruction factories
are Python objects. Put their serializable recipe values in `application` and
read them in your factory. Other API-level choices include:

| Choice | Where to configure it |
| --- | --- |
| Models, datasets, losses, steps, metrics, optimizer/scheduler factories | [Configured run factories](../runs/README.md) and [training APIs](../training/README.md). |
| Automatic/manual batch planning and memory budgets | [Memory planning APIs](../memory/README.md); managed runs also accept `batch_size_mode`. |
| Checkpoint paths, retention/restore calls and callbacks | [Checkpoint APIs](../checkpoints/README.md). |
| Graph capture attachment, streams and pools | [CUDA Graph APIs](../cuda/README.md). |
| Explicit operation plans and output representations | [FP8](../fp8/README.md), [normalization](../normalization/README.md) and [ReLU](../activation/README.md) functional APIs. |

The removed `runtime.fp8.max_num_imprecise_acc` option is rejected. Use the
current accumulation/backend controls above for new runs; older checkpoints
may require the original library version. Always reconstruct a checkpoint with
compatible model, numerical policies and recipe identity; see
[checkpoint compatibility](../checkpoints/README.md).
