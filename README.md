# Deep Optimization

DeepOpt trains existing PyTorch models with your data loaders and loss or step
function. Use `Trainer.fit()` for managed training or `TrainingRuntime` inside
your own loop. Tasks, wrappers, configuration files, and run directories are
optional.

## Install and run

From the repository root, in a Python 3.11+ environment:

```bash
python -m pip install -e .
python examples/training/regression.py
```

Dependencies are PyTorch `>=2.4,<3`, TorchMetrics `>=1.4,<2`, and tqdm `>=4.66,<5`.
The example runs on CPU with generated data and no downloads.

## Train a model

This complete CPU recipe writes no files:

```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from deepopt import Trainer

torch.manual_seed(7)
inputs = torch.randn(64, 3)
targets = inputs.sum(dim=1, keepdim=True)
train_loader = DataLoader(
    TensorDataset(inputs[:48], targets[:48]), batch_size=8, shuffle=True,
    generator=torch.Generator().manual_seed(17),
)
validation_loader = DataLoader(TensorDataset(inputs[48:], targets[48:]), batch_size=8)
model = nn.Linear(3, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
trainer = Trainer(model, optimizer, loss_fn=F.mse_loss)
trainer.fit(train_loader, validation_loader, epochs=2, progress_bar=False)
results = trainer.evaluate(validation_loader)
print(results)
```

Move the model to its device before constructing its optimizer, and pass that
same device to the trainer. Validation is optional; `epochs` is the total target
including completed epochs. For arbitrary batch structures or several loss
terms, supply `step_fn`; `fit()` owns backward and updates. Use a manual loop for
custom update order. See [training APIs](docs/training/README.md).

Scalar `loss` is a mean of batch objectives. Use an explicit
`LossResult.from_mean(loss, weight=...)` for a known sample/token denominator;
see [loss reporting](docs/optimization/README.md).

Follow [one recipe in five steps](docs/getting-started/README.md) to add metrics,
logging, checkpoints, resume, and optional batch planning. Direct training saves
files only when you configure logging or attach artifact callbacks. Configured
runs handle logs and checkpoints automatically; see [logging](docs/logging/README.md)
and [checkpoints](docs/checkpoints/README.md).

## Configuration and checkpoints

`deepopt.load_config()` installs process-local defaults for new components.
Precedence is explicit arguments, per-call configuration, global defaults, then
API defaults. Existing trainers keep their snapshots; supplied optimizers and
schedulers remain caller-owned. See [configuration inputs and templates](docs/runs/README.md).
The [configuration reference](docs/configuration/README.md) lists every framework
option, its default, accepted values, and behavior.

The CLI runs a synthetic classification recipe. Installed presets work from any
directory; `--config` also accepts your own TOML path:

```bash
python -m deepopt inspect-config --config builtin:baseline
python -m deepopt train --config builtin:baseline --output runs/baseline
python -m deepopt train --resume runs/baseline/last.pt --epochs 8
python -m deepopt evaluate --checkpoint runs/baseline/best.pt
```

Use a fresh output directory for each run. Custom recipes can use
`deepopt.run.train()` with a reconstruction factory. Configured runs default to
automatic batch planning; direct `Trainer.fit()` uses the loaders you supply.
See [configured runs](docs/runs/README.md) and [memory planning](docs/memory/README.md).

## FP8 training

Supported linear/projection operations can use FP8 while preserving master
parameters and optimizer ownership. Reusing the CPU recipe's model and loaders:

```python
from deepopt import FP8Config

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
trainer = Trainer(model, optimizer, loss_fn=F.mse_loss,
                  fp8=FP8Config(enabled=True, backend="reference"))
trainer.fit(train_loader, validation_loader, epochs=2, progress_bar=False)
report = trainer.runtime.transformation_report
print(report)
```

The reference backend emulates FP8 on CPU/CUDA; actual Triton FP8 GEMM requires
NVIDIA compute capability 9.0+ and compatible Triton. Inspect the transformation
report for partial conversion and use `trainer.model` after setup. See the
[FP8 guide](docs/fp8/README.md) for formats, scaling, automatic tuning, and model methods.
Optional [native backward GEMM and projection exclusions](docs/fp8/eager_linear_benchmark.md#native-backward-and-selective-preparation)
preserve original parameter ownership. Both are explicit configuration choices;
the existing FP8 defaults remain unchanged.

## Guides and development

Start with the [documentation index](docs/README.md), [example catalog](examples/README.md),
or [benchmark catalog](benchmarks/README.md).

The [activation guide](docs/activation/elementwise.md) covers sigmoid, SiLU,
exact/tanh GELU, tanh and ReLU6 with fused BF16/FP8 conversion, compressed local
derivatives and planned FP8 gradients. ReLU/ReLU6 retain exact one-bit masks.

[LayerNorm](docs/normalization/layer_norm.md) supports normalized suffix shapes,
pre-affine xhat compression and planned FP8 forward/backward transport through
the same precision and compression settings.

| Need | Guide |
| --- | --- |
| Data, custom steps, callbacks, evaluation, or manual loops | [Training](docs/training/README.md) · [Data loading](docs/data/README.md) |
| Losses, optimizers, metrics, or warmup policies | [Optimization](docs/optimization/README.md) · [Metrics](docs/metrics/README.md) · [Scheduling](docs/scheduling/README.md) |
| Multiple devices, compilation, or graph replay | [Distributed](docs/distributed/README.md) · [Execution](docs/execution/README.md) · [CUDA Graphs](docs/cuda/README.md) |
| Diagnostics, tests, or framework changes | [Observers](docs/observers/README.md) · [Architecture and development](docs/architecture/README.md) |

Trainer AMP supports FP32 on CPU/CUDA and BF16 on CUDA. Optional
[activation compression](docs/activation_compression/README.md) reduces saved
linear activations using affine 2-bit groups. FP16 AMP, automatic GPU OOM retries,
and sparse execution are not implemented.
Memory estimates do not establish GPU fit or throughput; validate on target hardware.

Run `python -m unittest discover -s tests -v` for the tests. The
[development guide](docs/architecture/README.md) covers focused checks, shared
lint settings, preset synchronization, compatibility boundaries, and CPU benchmarks.
