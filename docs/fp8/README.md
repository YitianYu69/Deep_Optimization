# FP8 training and model transformation

[Project README](../../README.md) · [Tests](../../tests/README.md) · [Example](../../examples/optimization/fp8_training.py)

DeepOpt replaces supported projections with DO modules or the shared
`do_linear` function. Parameters keep their original floating-point dtype and
identity; forward and backward quantize temporary GEMM operands. Ordinary
optimizers update the master parameters. This is first-order approximate FP8
training, not numerical differentiation through FP8 casts.

## Start with reference execution

```bash
python -m pip install -e .
python examples/optimization/fp8_training.py
python examples/optimization/fp8_training.py --trace-mode modules
```

The example trains a small MLP for three updates using generated data. It prints
the transformation report and loss, and writes no checkpoints or downloads.
`reference` quantizes to FP8, dequantizes, and multiplies in FP32. Its passing
tests validate semantics; they do not establish GPU kernel correctness or speed.

## Configure FP8 operands

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
        "num_stages": 3,
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

Both E4M3FN and E5M2 are accepted. `out_dtype=None` uses the active autocast dtype
or the input dtype. Explicit output dtypes are FP32, BF16, and FP16. Input,
weight, and bias tensors must be dense, on the same CPU/CUDA device, and use
FP32/BF16/FP16. Empty operands, nonfinite quantization inputs, and second-order
derivatives are unsupported. Runtime precision and FP8 operand formats are
separate choices: the trainer's CPU mode remains FP32, and CUDA supports its
existing FP32/BF16 modes.

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

Transform before constructing `Trainer`, `TrainingRuntime`, or native DDP.
Each rank reconstructs the same model and FP8 plan on its selected device.
The training and DDP cores continue to own accumulation, synchronization, and
optimizer updates. FP8 modules own the projection computation and custom
autograd, and do not introduce their own process group or optimizer.

Checkpoint normal model, optimizer, scheduler, loader and trainer state. Temporary
FP8 operands and scales saved for backward do not become model checkpoint keys.
Reconstruct the transformation with the same plan before restoring a transformed
training run. The `Checkpoint` callback does not automatically serialize the
Python FP8 plan; keep it in the recipe/configuration or save it alongside a
caller-owned checkpoint. Parameter keys and ordering remain compatible with
native model/optimizer checkpoints for the supported conversions.

The tests cover exact continuation at both step and epoch boundaries, including
partial accumulation windows, two Gloo ranks, and conditional two-GPU NCCL runs.
The FP8 path is currently not CUDA-Graph-capture-safe: scaling performs a
host-visible finite-value check. CUDA Graph support for the surrounding framework
does not imply this FP8 path can be captured.

## Actual CUDA execution

Use a CUDA-enabled PyTorch installation and compatible Triton exposing
`triton.tools.tensor_descriptor.TensorDescriptor`. The TMA kernel requires an
NVIDIA GPU with compute capability at least 9.0. There is no CPU fallback for a
requested Triton backend.

```bash
python examples/optimization/fp8_training.py --device cuda --backend triton
```

The current Triton training backend requires all three operand roles to use the
same FP8 dtype. Every non-None training block dimension must be a multiple of 32
because backward transposes the operands. Pipeline stages can be 1 through 4.
The backend packs strided/misaligned operands, pads partial tiles, and returns
FP32/BF16/FP16 results. Correctness/performance must be measured on the target GPU.

## Black-box and white-box validation

Install optional real-model dependencies without downloading pretrained weights:

```bash
python -m pip install -e '.[model-tests]'
DEEPOPT_REQUIRE_MODEL_TESTS=1 python -m unittest discover -s tests/opt -v
python -m unittest discover -s tests/distributed -p 'test_fp8_*.py' -v
```

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
