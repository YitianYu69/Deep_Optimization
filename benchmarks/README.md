# Benchmark catalog

[Project README](../README.md) · [Documentation](../docs/README.md) · [Examples](../examples/README.md)

Benchmarks are grouped by the workload they measure. Run the commands from the
repository root after installing DeepOpt in the active Python environment.
The FP8 and activation-compression programs use normal library imports and can
be copied outside the checkout to measure an installed build. They do not
install dependencies or select this repository's source tree.

## Cross-operator FP8 and compression analysis

[benchmark_do_ops.py](analysis/benchmark_do_ops.py) runs matched PyTorch BF16,
DO BF16, compression-only, FP8-only and combined cases across every public DO
operator family. It reports the extra FP8/compression interaction, numerical
errors, memory, separate profiler traces and explicit unsupported/error rows.
JSON grids support main-effect or Cartesian settings sweeps; model factories
support family ablations and recording shapes for operator replay. See the
[analysis guide](analysis/README.md) for configurations and measurement limits.

```bash
python benchmarks/analysis/benchmark_do_ops.py --preset all --dry-run
python benchmarks/analysis/benchmark_do_ops.py --config benchmarks/analysis/configs/interaction.json --output work/interaction
```

## FP8 projections and training

| Program | Measures | Guide |
| --- | --- | --- |
| [benchmark_fp8_eager.py](fp8/benchmark_fp8_eager.py) | Matched eager BF16, DeepOpt, torchao, and Transformer Engine linear forward/backward | [Three-library eager comparison](../docs/fp8/eager_three_library_benchmark.md) |
| [benchmark_fp8_linear_compare.py](fp8/benchmark_fp8_linear_compare.py) | Detailed linear correctness, native backward, delayed scaling, and profiling | [Linear comparisons and controls](../docs/fp8/eager_linear_benchmark.md) |
| [benchmark_fp8_autotune.py](fp8/benchmark_fp8_autotune.py) | Quantization, GEMM, autotuning, and fixed-kernel diagnostics | [CUDA execution and autotuning](../docs/fp8/cuda.md) |
| [benchmark_fp8_te_linear.py](fp8/benchmark_fp8_te_linear.py) | DeepOpt and Transformer Engine scaling recipes on matched projections | [Delayed scaling and TE](../docs/fp8/delayed_scaling.md) |
| [benchmark_fp8_training.py](fp8/benchmark_fp8_training.py) | Complete timm ViT updates, memory, and profiling | [Full-model training](../docs/fp8/training_benchmarks.md) |
| [benchmark_backward_precision.py](fp8/benchmark_backward_precision.py) | Native versus planned FP8 gradient transport across complete pipelines | [Backward precision](../docs/fp8/backward_precision.md) |

These programs require CUDA and compatible FP8 libraries to execute the measured
GPU paths. Their `--help` commands work without a CUDA device. Optional torchao,
Transformer Engine, and real-model packages are needed only for the selected
implementations. See each guide for supported versions and measurement limits.

```bash
python benchmarks/fp8/benchmark_fp8_eager.py --help
python benchmarks/fp8/benchmark_fp8_autotune.py --help
```

## Saved-activation compression

[benchmark_activation_compression.py](activation_compression/benchmark_activation_compression.py)
compares compression off/on for BF16 and FP8 linear steps, including packing,
reconstruction, backward, numerical differences, and saved payload bytes.
Execution requires CUDA; see the [activation-compression guide](../docs/activation_compression/README.md).

```bash
python benchmarks/activation_compression/benchmark_activation_compression.py --help
```

[benchmark_boolean_mask.py](activation_compression/benchmark_boolean_mask.py)
measures lossless one-bit boolean packing, unpacking, roundtrip, and ordinary
boolean copying for contiguous and strided masks. It supports CUDA kernel
measurements and CPU reference measurements, imports the installed library,
and reports exactness, payload bytes and repeated timings. See
[boolean mask storage](../docs/activation_compression/README.md#lossless-boolean-masks).

```bash
python benchmarks/activation_compression/benchmark_boolean_mask.py --help
```

[benchmark_activation_unpack.py](activation_compression/benchmark_activation_unpack.py)
isolates the transposed reconstruction bottleneck. It compares grouped, tiled,
and contiguous-decode-plus-copy strategies on identical packed inputs, with
exact numerical checks and eager/graph timing. Default shapes reproduce the
large ViT projection inputs. See [performance recovery](../docs/activation_compression/performance_recovery.md).

## Spatial convolutions

[benchmark_convolution.py](convolution/benchmark_convolution.py) compares native
BF16 and DeepOpt FP8 Conv1d/2d/3d with compression off/on. Use
`--operation conv_transpose` for transposed convolutions, `--output-padding`
for their output shape, `--cuda-graph` for model replay, `--optimizer-step` for
SGD updates, and `--profile-dir` for traces and spatial phase kernel durations.
Graph capture requires current scaling. Complete eager
forward/backward includes casting, packing, reconstruction and layout costs.
Reports include timing, saved payload, allocator peaks and numerical errors.
Run against the installed library on CUDA; see the
[convolution guide](../docs/convolution/README.md) for backend behavior and limits.

```bash
python benchmarks/convolution/benchmark_convolution.py --help
python benchmarks/convolution/benchmark_convolution.py --shape 8x64x56x56 --output work/convolution.json
```

## Normalization

[benchmark_normalization.py](normalization/benchmark_normalization.py) measures
BatchNorm2d/GroupNorm with BF16/FP8 input, saved-xhat compression and optional
planned FP8 consumer pipelines. See the [normalization guide](../docs/normalization/README.md).
Use `--norm-xhat-recenter both` to measure uncompressed, compressed and
compressed-with-correction variants. Timings include correction production and
backward use; reports separate retained metadata, partial-buffer scratch and
extra reduction launches. Mean-error diagnostics run outside timing.

## Pooling

The installed-library [pooling benchmark](pooling/benchmark_pooling.py) covers
all pooling families, BF16/FP8 input/output, compression off/on, native baselines,
isolated or convolution/Linear pipelines, delayed scales and eager/graph timing.
Reports include saved/retained/peak bytes, state strategies, numerical errors and
public-index agreement. See the [pooling guide](../docs/pooling/README.md).

## Activations

[benchmark_relu.py](activation/benchmark_relu.py) compares fused ReLU across
BF16/FP8 input and output, with bool or one-bit saved masks, against native
BF16 ReLU. It runs against the installed library on CPU or CUDA and reports
timing, mask bytes, quantization error and scale-launch counts. See the
[ReLU guide](../docs/activation/README.md). The normalization benchmark's
convolution pipeline also reports ReLU execution metadata.

[benchmark_activations.py](activation/benchmark_activations.py) covers sigmoid,
SiLU, exact/tanh GELU, tanh and ReLU6. It measures BF16/FP8 input-output pairs,
compression off/on, saved derivative bytes, numerical errors, peak allocation and
eager versus warmed CUDA Graph replay. Linear and convolution pipeline modes
include planned FP8 activation gradients. See the
[activation guide](../docs/activation/elementwise.md) for commands and timing boundaries.

## Trainer and scheduling costs

| Program | Measures | Guide |
| --- | --- | --- |
| [benchmark_training.py](training/benchmark_training.py) | Plain PyTorch and managed CPU training, with optional profiling | [Development and performance](../docs/architecture/README.md#autograd-and-performance) |
| [benchmark_execution.py](training/benchmark_execution.py) | Execution modes, accumulation, device transfer, and DDP | [Execution policies](../docs/execution/README.md#verification-and-benchmarks) |
| [benchmark_scheduling.py](training/benchmark_scheduling.py) | Warmup coefficient scheduling overhead | [Scheduling](../docs/scheduling/README.md) |

The CPU training benchmark accepts an explicit `--source` snapshot for contributor
comparisons and falls back to the local source tree when DeepOpt is uninstalled.
The other programs import the active installed library.

```bash
python benchmarks/training/benchmark_training.py --profile --output work/training.json
python benchmarks/training/benchmark_execution.py --device cpu --batches 24
python benchmarks/training/benchmark_scheduling.py --updates 10000 --repeats 5
```

Benchmark contract tests live in [tests/benchmarks/](../tests/benchmarks/).
The [normalization benchmark](normalization/benchmark_normalization.py) also
supports `--op layer_norm --normalized-shape ...`, native BF16 baselines,
compression and optional activation/Linear pipelines, and eager/CUDA Graph
comparisons. See [LayerNorm measurements](../docs/normalization/layer_norm.md).
Compilation, warmup, profiler runs, kernel activity, and full step wall time are
different measurements; consult the relevant guide before interpreting results.
