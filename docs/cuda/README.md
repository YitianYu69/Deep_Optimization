# CUDA Graphs

[Documentation index](../README.md) · [Execution policies](../execution/README.md) · [DDP](../distributed/README.md)

CUDA Graph support lives in its own sibling package:

```text
deepopt/
├── engine/core.py        # Regular forward/backward and local training policy
├── distributed/ddp.py    # DDP wrapping, synchronization and rank policy
└── cuda_graph/core.py    # CUDA capture, replay and graph lifetime
```

`CUDAGraphCore` composes with either training core. It does not own optimizers,
epochs, process groups, checkpoint counters, diagnostics, or loss reductions.
Constructing a core does not initialize CUDA. `deepopt.cuda_graph_core` is a
convenience instance using the current CUDA device; use `CUDAGraphCore(device)`
to select a device explicitly. CUDA is required at capture time; there is no
silent CPU fallback.

## Run a complete example

```bash
python examples/cuda/graphs.py --help
python examples/cuda/graphs.py
python examples/cuda/graphs.py --device cuda:1
```

The [graph example](../../examples/cuda/graphs.py) needs no dataset or downloads.
It performs three captured model-training updates, captures a differentiable
function, and checks callable inference, decorator replay, and a raw capture
block against eager tensor expressions. The help command works on CPU-only
machines; actual execution requires a CUDA-enabled PyTorch build and a CUDA GPU.
Each handle is explicitly closed after its work completes. These are correctness
examples, not timing measurements.

## Capture a model or function

Both `capture_model` and its alias `capture_callable` accept an `nn.Module` or
a callable, plus a tuple of sample tensor arguments. They capture immediately
and return a callable `CapturedCallable` handle with `replay(*args)` and `close()`.
The sample tensors supply values, shapes, strides, dtypes, devices and
`requires_grad` flags. Fresh tensors with matching metadata can be passed on
each call; the handle copies them into owned static input storage.

```python
import torch
from deepopt import CUDAGraphCore, TrainingRuntime

device = torch.device("cuda", torch.cuda.current_device())
core = CUDAGraphCore(device)
model = torch.nn.Linear(8, 2).to(device)
example = torch.randn(16, 8, device=device)
captured = core.capture_model(model, (example,))

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
runtime = TrainingRuntime(model, optimizer, device=device, execution_mode="max_speed")
prediction = runtime.compute(example, model)
runtime.backward(prediction.square().mean())
runtime.step()
captured.close()  # Restores the model's original forward.
```

Autograd capture is the default. It uses PyTorch's public
`make_graphed_callables` to capture forward and backward. A module's forward is
installed in place, so ordinary `model(...)`, `Trainer`, and custom steps that
call that module use the graph. Parameter identity and state-dict keys remain
unchanged. Capturing a submodule affects calls to that submodule; bypassing its
forward also bypasses its graph.

Plain functions remain unchanged. Call the returned handle to replay:

```python
def objective(features, weight):
    return (features @ weight).square().mean()

features = torch.randn(16, 8, device=device)
weight = torch.nn.Parameter(torch.randn(8, 2, device=device))
captured = core.capture_callable(objective, (features, weight))
loss = captured(features, weight)
loss.backward()  # Gradients flow to weight.
captured.close()
```

For a plain function, **every trainable tensor must be an explicit argument**.
Trainable tensors hidden in closures, bound methods, globals, or partials are
not discovered. Pass a module itself instead of its bound `forward` when its
registered parameters should participate in autograd. Outputs must be tensors
or tuples of tensors compatible with PyTorch's graphed-callable API; use
differentiable outputs for autograd capture across supported PyTorch versions.
Higher-order differentiation is unsupported.

The handle copies gradients returned by the captured autograd node into owned
storage before accumulation or DDP reduction. This prevents later replay from
overwriting earlier gradients, including results retained from
`torch.autograd.grad`. These copies add work outside the graph and preserve
accumulation correctness across supported PyTorch versions.

For inference, including a frozen model or a function with no differentiable
inputs, use `autograd=False`:

```python
def transform(x):
    return x.sin() * 2

captured = core.capture_model(transform, (example,), autograd=False)
output = captured.replay(example)
native_graph = captured.graph  # torch.cuda.CUDAGraph
captured.close()
```

Inference calls run under `no_grad`. Autograd handles expose `.graph = None`:
PyTorch owns the forward/backward graph pair behind its public callable API.
The handle is the replay interface for that pair. Native graphs are available
from inference handles and the context API below.

## Capture a block

The low-level context records arbitrary graph-safe CUDA work, including a
caller-owned backward or capturable optimizer update. It returns a
`CapturedGraph` with `.graph`, `.replay()`, `.captured`, and `.close()`.

```python
static_input = torch.ones(16, 8, device=device)
core.warmup(lambda: static_input.sin() * 2, iterations=3)

with core.capture() as captured:
    static_output = static_input.sin() * 2

static_input.copy_(torch.randn_like(static_input))
captured.replay()
result = static_output.clone()
captured.close()
```

A block cannot be automatically rerun for warmup. `core.warmup` runs the supplied
function on a side stream and joins the caller stream, preserving its real side
effects. The context records CUDA work; replay executes it. Keep the input and
output tensors alive and at the same addresses. The raw context does not copy,
validate, or retain arbitrary tensors used by your code. Use one capture context
per graph. An exception leaves it unusable for replay.

Optional `stream`, `pool`, and `capture_error_mode` arguments map to PyTorch's
capture controls. The default error mode is `global`. Pool sharing requires
compatible replay order and lifetimes; independent graphs use independent pools
by default. Custom CUDA generators require PyTorch's explicit graph registration;
use a caller-owned native graph directly for generator-registration workflows.

## Decorate inference code

```python
import deepopt

@deepopt.cuda_graph_core.capture
def transform(x):
    return x.square() + 1

output = transform(example)  # Warmup, capture, then first replay.
output = transform(torch.randn_like(example))  # Copies inputs and replays.
transform.close()
```

`@core.capture(warmup=5)` also works. The first call runs three warmups by
default, records one Python invocation, then replays the CUDA work. Subsequent
calls skip Python entirely. Decorated functions accept positional CUDA tensors
(or no arguments) and return a CUDA tensor, a tuple of CUDA tensors, or `None`.
Use pure inference functions: warmup effects on closures or other Python state
are not rolled back. The decorator disables gradients; use high-level autograd
capture for differentiable functions and the block API for manual training work.

## Composition and lifetime

| Concern | Owner / contract |
| --- | --- |
| Static buffers, warmup, capture, replay | CUDA Graph core |
| Local forward/backward policy | Regular training core |
| DDP reducer, gradient synchronization, rank checks | DDP core, outside the captured model |
| Transfers, precision, diagnostics, optimizer updates | Existing runtime |
| Epochs, accumulation windows, callbacks, checkpoints | Existing trainer |

For DDP, initialize the process group and select the local device, then capture
the raw model **before constructing `TrainingRuntime` or `Trainer`**. Each rank
captures its own model. The runtime then constructs `DDPCore`; graph backward
returns gradients to its ordinary reducer. Accumulation/no-sync, metrics and
checkpoints stay eager. Do not capture an already DDP-wrapped module. Full
DDP-collective capture is not managed by this API.

Do not put runtime updates or `Trainer.fit()` inside a capture block. Python
counters and diagnostics would run only once. Runtime compute/backward/update
operations reject active CUDA capture; capture the model instead. Diagnostic
modes remain independent of graph capture.

Additional contracts:

- Shapes, strides, dtype, device, input count and grad flags must match the
  samples. Keyword arguments and non-tensor input leaves are unsupported. Pad
  variable-size batches or close and recapture at a completed update boundary.
- Model capture preserves registered buffer values and the default CPU/CUDA
  RNG states during setup. Forward must not mutate inputs, parameters or arbitrary
  Python state. Plain-function closure state is caller-owned and is not restored.
- Train/eval changes, child mode changes and autocast changes use the original
  callable eagerly. Returning to the captured settings uses the graph again.
  Capture under the intended AMP/autocast precision (BF16 where supported); the
  helper disables autocast caching during setup.
- Model parameters/buffers must keep their identity, storage and metadata.
  In-place optimizer updates and ordinary `load_state_dict` copies are allowed.
  Replacing parameters, changing device/dtype, adding buffers or `assign=True`
  restores require closing and recapturing. Python attributes used in control
  flow are static: close and recapture if they change.
- No module hooks at capture time. Hooks attached afterward to the captured
  module itself run around replay; hooks on operations/submodules inside its
  graph do not run per replay. Keep diagnostics outside captured regions.
- Outputs borrow reusable storage. Clone retained inference outputs. Finish
  backward before the next forward through the same autograd capture; do not
  retain multiple outstanding forward graphs, replay concurrently, or reuse a
  backward graph for higher-order derivatives. Close only at an idle boundary.
- Captures are process-local resources. Save normal model/optimizer/runtime
  state, then reconstruct captures on resume. Do not pickle a graph handle or a
  whole patched model. Batch-size changes on resume require new captures.
- No CPU synchronization (`.item()`, CPU copies), host-dependent control flow,
  dynamic allocations unsupported by CUDA capture, or unrelated concurrent
  CUDA work during capture. Users must also coordinate external threads and
  captures; DeepOpt's guard only serializes its own capture entry points.

See the [PyTorch CUDA Graph constraints](https://docs.pytorch.org/docs/2.4/notes/cuda.html#cuda-graphs)
and [autograd-aware capture API](https://docs.pytorch.org/docs/2.4/generated/torch.cuda.make_graphed_callables.html)
for the underlying execution contract.

## CUDA tests

All additional graph tests live under [tests/cuda](../../tests/cuda/):

```bash
python -m unittest discover -s tests/cuda -p '*white_box.py' -v
python -m unittest discover -s tests/cuda -p '*black_box.py' -v
```

CUDA Graph white-box tests run on CPU with mocked CUDA boundaries. Graph
black-box tests require a CUDA GPU; the DDP test requires two GPUs and NCCL. They compare replay values,
gradients, accumulation, optimizer updates and checkpoint reconstruction with
eager PyTorch. Hardware skips do not establish GPU correctness or performance.
The opt-in CUDA CI job runs this entire section.

The same test folder also contains `test_fp8_*.py`: real Triton kernel, autograd,
ViT/Hugging Face model, and two-GPU FP8/NCCL checkpoint checks. Both FP8 black-box
and white-box tests require actual compatible hardware; they do not mock kernel
execution. See the [FP8 validation commands and prerequisites](../fp8/README.md#black-box-and-white-box-validation).
