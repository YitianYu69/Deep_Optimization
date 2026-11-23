# AccumulateGrad stream-warning investigation

Status: diagnostic tooling implemented; real GPU attribution is pending.
See the [standalone runner and usage](../../benchmarks/cuda_graph/README.md).
No library stream-management changes or warning suppression are included.

[CUDA Graph guide](./README.md) · [Configuration](../configuration/README.md)

## Objective

Identify the first producer/accumulator stream mismatch, determine whether it
originates in PyTorch or DeepOpt, and measure its correctness and performance
impact. A warning alone does not establish incorrect gradients, a failed graph
replay, or a specific amount of synchronization overhead.

Preserve the user's baseline: one-rank torchrun, BF16 AMP, current FP8 scaling,
tensorwise scales, native-fast backward GEMM, two-bit activation compression,
group size 512, and normalized-state recentering. Record effective Trainer
options, including any Python overrides of the TOML configuration.

## 1. Capture a reproducible baseline

- Record Python, PyTorch, CUDA, driver, Triton, timm, and DeepOpt versions;
  DeepOpt commit/package path; GPU model; rank/world size; and effective options.
- Record the exact capture autocast scope and subsequent Trainer scope. Include
  finite-check policy, train/eval mode, gradient requirements, and gradient
  clearing policy. Keep the original configuration unchanged for the baseline.
- Locate the first warning within model preparation, graph setup/warmup,
  capture, the first training backward, later backward calls, or the return
  from validation to training. Record both its stack and the enclosing phase.
- Use a fresh process for each experiment: PyTorch emits this warning once,
  so Python warning filters alone cannot establish its frequency or absence
  in subsequent cases.
- Start with a small model and fixed shapes; retain a separate reproduction
  using the user's timm ViT. Run plain Python and one-rank torchrun separately.

## 2. Attribute the mismatch with controlled cases

Use the same weights, batches, optimizer, loss, and precision in each matched
comparison. First use a tiny native model; then extend to the ViT only when
the relevant boundary has been isolated.

| Case | Execution | What it isolates |
| --- | --- | --- |
| A | Native BF16 eager forward/backward | Baseline autograd behavior |
| B | Direct `torch.cuda.make_graphed_callables`, BF16, manual loop | Upstream graph-helper behavior |
| C | DeepOpt `capture_model`, BF16, manual loop | DeepOpt capture wrapper and gradient ownership |
| D | Same captured BF16 model through TrainingRuntime, then Trainer | Runtime versus Trainer orchestration |
| E | DeepOpt graph + current FP8, compression off | FP8 backward integration |
| F | DeepOpt graph + current FP8 + the supplied compression settings | Compression/recentering contribution |

Do not run a complete factorial sweep initially. Change one factor after a
comparison identifies a difference. Keep compilation disabled and do not use
delayed scaling, which the current capture runtime does not support.

## 3. Trace node lifetime and stream ownership

- Inspect setup in `CapturedCallable`, PyTorch's graph helper, replay, and the
  outer eager backward. Record the calling stream's device/handle at phase
  boundaries; inspect helper-internal warmup/capture streams as well.
- An outer `torch.cuda.stream(current_stream)` context does not force
  `make_graphed_callables` to use that stream internally. Verify the installed
  helper rather than assuming the outer context determines accumulator state.
- Inspect creation/reuse of parameter accumulators and references retained by
  graph outputs, loss tensors, metrics, callbacks, hooks, or capture closures.
  Clear disposable outputs/losses between controlled iterations and compare
  against a deliberately retained-graph control. Treat `retain_graph=True`
  separately from normal first-order training.
- `zero_grad(set_to_none=True)` clears gradient tensors; it does not guarantee
  that every reference keeping an accumulator alive has been released.
- Inspect `_own_backward_gradients` independently. Its clones protect callers
  from graph-owned scratch aliases. Any diagnostic variant must retain the
  existing aliasing/correctness checks; removing this protection is not a fix.
- Where available, use autograd node-creation hooks for metadata collection.
  Feature-detect version-specific APIs. Log identifiers, names, and stream
  metadata only: retaining Node/tensor objects in diagnostic logs can itself
  extend graph lifetime and manufacture the problem being investigated.
- Calling-stream IDs do not directly prove an accumulator's internal stream.
  If public APIs cannot expose the needed ownership, provide a minimal upstream
  reproducer or use a separate debug-instrumented PyTorch build.

## 4. Establish execution, correctness, and overhead independently

- Check capture-statistics deltas during the training window: successful
  forward replay count should advance without unexpected mode/autocast
  fallback. Validation's intentional eager fallback is measured separately.
  Forward counters alone do not prove backward replay; verify backward graph
  launches with the profiler as well.
- Compare outputs, parameter gradients, optional input gradients, and optimizer
  updates with an eager reference using the same FP8/compression recipe. Compare
  BF16 against BF16; avoid attributing FP8/compression approximation to streams.
- Check repeated steps, gradient clearing, accumulation, and train/eval/train
  transitions. Include both `sample_x.requires_grad=False` as in the user's
  training setup and a separate differentiable-input case.
- Match RNG state and preserve/restore warmup-mutated buffers for comparisons.
  Store detached comparison snapshots; keep only metadata in diagnostic logs.
- Profile separate untimed and timed runs. Use CPU wall time, CUDA-event step
  time, graph replay counters, and synchronization activity from torch.profiler;
  use Nsight Systems if stream/event attribution remains unclear.
- Separate expected graph ordering and diagnostic checks from synchronization
  associated with the mismatch. Do not insert a device-wide synchronization in
  every step as a proposed fix, or treat event time as a sum of kernel durations.

## 5. Add multi-rank coverage after the single-rank case

One-rank execution selects TrainingCore, so investigate that case first. For
two GPUs, capture graph-safe model sections before DDP construction, following
the existing supported partial-capture API. Verify DDP construction/forward
stream consistency and the required stream handoffs. Do not change to capturing
DDP internals merely to remove the warning.

Compare raw PyTorch partial capture + DDP with DeepOpt DDPCore. Distinguish DDP
retaining accumulators from any mismatch already present in the single-rank
case. Compare synchronized gradients and optimizer updates as well as warnings.

## 6. Choose a remedy only after attribution

| Evidence | Action |
| --- | --- |
| Direct PyTorch reproducer shows the same mismatch | Document the upstream/version-specific behavior and provide that reproducer |
| DeepOpt-only references retain stale accumulators | Correct ownership/lifetime at the responsible boundary |
| DeepOpt-only stream transition lacks ordering | Add the specific stream/event dependency and validate its cost |
| DDP setup introduces the mismatch | Correct setup/execution stream consistency within the supported capture order |
| Handoff is intentional and correct | Document evidence and measured overhead before considering any optional warning policy |

Do not suppress the warning, change global warning/capture overrides, change
FP8 dtypes, or add broad synchronization as an unproven remedy. If upstream
behavior remains unresolved, report that explicitly rather than declaring the
warning fixed.

## 7. Deliverables and acceptance

- An installed-library diagnostic program under `benchmarks/cuda_graph/` that
  runs cases in fresh subprocesses and produces a JSON phase/version/result
  report, with optional profiler traces. No checkout path injection.
- A minimal native PyTorch reproducer if the issue survives removal of DeepOpt.
- Focused GPU regressions under `tests/cuda/`, extending the existing graph,
  Trainer, accumulation, and DDP tests. Keep warning capture version-aware and
  preserve native expected behavior rather than globally disabling warnings.
- A narrowly scoped library patch only if the experiments establish a DeepOpt
  defect; update the CUDA guide with the supported sequence and any upstream
  limitation.
- Evidence identifying the first affected phase and responsible boundary,
  correctness results across repeated updates, and measured synchronization
  cost. For an unintended DeepOpt mismatch, the fixed regression must stop
  warning without losing graph replay or introducing broad per-step barriers.

This host has no NVIDIA GPU. CPU tests can validate diagnostic/reporting code
and mocked ordering contracts; they cannot establish accumulator CUDA stream
ownership, real capture behavior, or speedup. Final attribution requires GPU
results from the user's environment or a GPU runner.

## Source references

- [PyTorch backward stream semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#stream-semantics-of-backward-passes)
- [PyTorch CUDA Graph constraints](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)
- [PyTorch graph-helper implementation](https://github.com/pytorch/pytorch/blob/main/torch/cuda/graphs.py)
- [Accumulator mismatch detection](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/input_buffer.cpp)
