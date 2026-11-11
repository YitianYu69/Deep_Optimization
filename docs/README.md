# Documentation

[Project README](../README.md) · [Runnable examples](../examples/README.md)

Start with your normal PyTorch model, optimizer, and loaders. You only need the
guides for features your training recipe uses; the internal component names
are not prerequisites for ordinary training.

For a first recipe, start with `Trainer`, one loss or step function, and your
PyTorch components. Add `metrics`, `logging_session`, and `Checkpoint` as needed.
`Task`, `TrainingModule`, strategies, and explicit loss-weight objects are
optional extension points; ordinary custom tasks do not require them.

## Suggested reading order

1. [Getting started](getting-started/README.md): extend one recipe through logging, checkpoints and resume.
2. [Training APIs](training/README.md): choose `fit()` or a loop you own.
3. [Metrics](metrics/README.md) and [losses and optimization](optimization/README.md):
   add reporting or change how updates are formed when your recipe needs it.
4. [Configured runs](runs/README.md) and [checkpoints](checkpoints/README.md):
   add reproducible reconstruction, saved state, and resume.
5. [Memory planning](memory/README.md) and [observers](observers/README.md):
   estimate storage, select a batch size, and configure diagnostics.

Read [architecture and development](architecture/README.md) when changing the
framework itself. Complete training programs live in
[examples/](../examples/README.md); short snippets in guides assume the named
model, loader, and optimizer already exist unless the snippet constructs them.

## Functionality map

[BatchNorm2d, GroupNorm and one-time FP8 edge planning](normalization/README.md)
cover normalized-activation compression and automatic output precision.
[LayerNorm](normalization/layer_norm.md) adds suffix-shaped normalization,
saved-xhat compression and bidirectional FP8 transport.
[Pooling](pooling/README.md) covers regular/adaptive pooling, LP/fractional
pooling and unpooling with FP8 transport and family-specific backward-state
compression. The [design plan](pooling/plan.md) records its implementation contract.

| I want to… | Read |
| --- | --- |
| Look up every configuration option, default, valid value, and interaction | [Configuration reference](configuration/README.md) |
| Follow one complete recipe from first training through resume | [Getting started](getting-started/README.md) |
| Skip a batch prefix or use worker-aware loading | [Data loading](data/README.md) |
| Train with a scalar loss or customize the whole forward/loss computation | [Training APIs](training/README.md) |
| Train across CPU/GPU ranks with separate regular and DDP cores | [Distributed training](distributed/README.md) |
| Capture models/functions or CUDA code blocks and replay with training/DDP | [CUDA Graphs](cuda/README.md) |
| Transform projections for FP8 training, preserve model methods, and validate real models | [FP8 and model transformation](fp8/README.md) |
| Optimize ordinary/transposed Conv1d, Conv2d and Conv3d with FP8 and saved-input compression | [Spatial convolutions](convolution/README.md) |
| Fuse ReLU, BF16/FP8 conversion and exact one-bit backward masks | [Fused ReLU](activation/README.md) |
| Optimize sigmoid, SiLU, GELU, tanh and ReLU6 with compressed derivatives and planned FP8 gradients | [Activation functions](activation/elementwise.md) |
| Plan FP8 activation gradients once across DO backward operations | [Backward precision](fp8/backward_precision.md) |
| Compress saved FP8 or BF16 linear activations with configurable affine 2-bit groups | [Activation compression](activation_compression/README.md) |
| Compare eager FP8 Linear, select native backward GEMM, or exclude projections | [Eager Linear benchmarks and controls](fp8/eager_linear_benchmark.md) |
| Select and tune CUDA FP8 kernels, or compare complete training updates | [CUDA and autotuning](fp8/cuda.md) · [Full-model benchmarks](fp8/training_benchmarks.md) |
| Find a standalone installed-library measurement program | [Benchmark catalog](../benchmarks/README.md) |
| Use validation, best-metric selection, progress bars, callbacks, or graceful stopping | [Training APIs](training/README.md) |
| Control backward and optimizer order, or wrap a model without using the trainer | [Training APIs](training/README.md) |
| Reuse a `Task`, `TrainingModule`, or custom batch/device-transfer method | [Training APIs](training/README.md) |
| Accumulate gradients, define token/sample weights, or report a different loss statistic | [Losses and optimization](optimization/README.md) |
| Warm up or schedule named coefficients and configure native LR policies | [Scheduling and warmup](scheduling/README.md) |
| Assign optimizers to different layers, customize optimizer construction, clip gradients, or schedule LR | [Losses and optimization](optimization/README.md) |
| Write a TorchMetrics `Metric`, route several metric inputs, or use compound scalar reports | [Metrics](metrics/README.md) |
| Understand metric lifetime, synchronization, checkpoint counters, or detached reporting | [Metrics](metrics/README.md) |
| Configure device, precision, seeds, model/data factories, or custom loaders | [Configured runs and data](runs/README.md) |
| Use the CLI, inspect TOML, evaluate a checkpoint, or inspect generated artifacts | [Configured runs and data](runs/README.md) |
| Save directly from a recipe, resume within an epoch, or change batch size without replaying data | [Checkpoints and resume](checkpoints/README.md) |
| Implement a stateful loader with workers/prefetch or understand recovery limits | [Checkpoints and resume](checkpoints/README.md) |
| Inspect parameter counts/percentages and current model/optimizer GB | [Memory and batch sizing](memory/README.md) |
| Estimate storage from a sample batch or select auto/manual batching before a fresh start or resume | [Memory and batch sizing](memory/README.md) |
| Turn numerical checks on/off or add activation, gradient, weight, or custom events | [Observers and checks](observers/README.md) |
| Select max_speed/balanced/analyze, schedule diagnostics, or inspect performance/replay contracts | [Execution policies](execution/README.md) |
| Configure model-only `torch.compile` or try an offline compilation example | [Execution and compilation](execution/README.md) · [compile recipe](../examples/execution/compile_training.py) |
| Save all framework/recipe messages to a configuration-named file | [Logging](logging/README.md) |
| Locate a module, run tests, or compare CPU execution costs | [Architecture and development](architecture/README.md) |

## Which layer should I use?

| Layer | You supply | DeepOpt controls |
| --- | --- | --- |
| `Trainer.fit()` | Ready-made model, optimizer, loaders, loss/step callable | Epochs, optional validation, automatic backward/updates, counters and callbacks |
| `TrainingRuntime` | Your loop and update algorithm | Only the execution helpers you call: transfer, autocast, backward, checks and updates |
| `deepopt.run.train()` | Configuration and a reconstruction factory | Run setup, optional batch-size planning, artifact callbacks and checkpoint restore around `fit()` |
| `python -m deepopt` | Configuration for the bundled demo recipe | The configured-run workflow using the synthetic baseline |

The CLI is a concrete demo application, not a model registry. General tasks do
not need to inherit a wrapper or implement a configuration schema. Custom
models and datasets belong to recipes.

## Execution requirements

| Feature | CPU path | Additional requirements |
| --- | --- | --- |
| Training, checkpointing, metrics, scheduling, observers | FP32 with trainer AMP disabled | CUDA supports FP32 without AMP or BF16 AMP |
| DDP | Gloo with multiple local processes | CUDA/NCCL uses one device per rank |
| Model compilation | `aot_eager` example; Inductor needs a compiler toolchain | Performance depends on backend and workload |
| CUDA Graphs | `--help` and CPU white-box checks only | CUDA device and capture-compatible operations |
| FP8 projections | Reference quantization/emulation | Triton forward and optional native scaled-GEMM backward require NVIDIA CC >= 9 and compatible libraries |
| Worker recovery | Built-in deterministic item replay | Stateful worker recovery uses the optional `workers` extra |
| Real-model validation | Tiny random ViTs/GPT-2/Llama with `model-tests` extra | No pretrained downloads; CUDA validation remains separate |

Guides describe supported behavior and boundaries. The [example catalog](../examples/README.md)
provides runnable programs, required dependencies, and expected artifacts.
Optional [activation compression](activation_compression/README.md) supports
saved FP8 and BF16 linear and [convolution](convolution/README.md) activations. Sparse execution remains an extension
point in the [FP8 guide](fp8/README.md).

## Compatibility with earlier documentation links

The previous [advanced](advanced.md), [runs](runs.md), [memory](memory.md), and
[architecture](architecture.md) pages remain as navigation pages. The topic
READMEs above are the maintained references.
