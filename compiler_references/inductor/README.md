# Local TorchInductor references

Use the saved [Inductor benchmark](../../benchmarks/compiler/benchmark_inductor.py)
to inspect how native PyTorch compiles a workload. The benchmark retains the
forward and AOTAutograd backward graphs, generated code, and any runtime trace
next to that run's measurements.

```sh
python benchmarks/compiler/benchmark_inductor.py \
  --models resnet18 vit-base efficientnet-b0 vae-v2 \
  --recipes torch-bf16 --device cuda:0 --batch-size 1 --image-size 224 \
  --warmup 10 --steps 50 --profile --output compiler_references/inductor
```

CUDA captures require matching PyTorch, torchvision, Triton, and target hardware.
The VAE benchmark also uses an installed DeepOpt wheel to declare its
transparent primitives. The compared graph uses native Torch operations.
DeepOpt custom ops, compressed ops, and opaque eager boundaries disqualify it
as an Inductor reference. Native ATen, cuDNN, and cuBLAS calls can appear as
library calls in the generated wrapper.

The runner creates a local `index.json` and a new immutable directory under
`runs/` for each capture. An output directory outside this tree keeps the same
per-run layout. Imported references can have a separate local index under
`imports/`. Read those imports in place and verify the saved manifests before
using them.

```python
from pathlib import Path

from benchmarks.compiler.inductor_reference_store import verify_run

run = Path("compiler_references/inductor/runs") / run_id
result = verify_run(run)
assert result["valid"], result
```

Match the model, recipe, shape, dtype, layout, software, and GPU. Inspect both
graphs and generated wrappers, then read the trace from the same benchmark
output before interpreting timing and memory. Throughput samples exclude
compilation, warmup, profiling, and artifact export. Require repeated target
runs before promoting an optimization policy. CPU diagnostics, blocked
preflights, and code-only captures identify different evidence scopes.

The store retains code, compiler IR, and diagnostic metadata without tensor
checkpoints. Capture new conditions in a new run. Read saved generated wrappers
as data and keep the local indexes and artifacts outside the release.
