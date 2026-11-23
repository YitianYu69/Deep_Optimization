# AccumulateGrad stream-warning analysis

[Benchmark index](../README.md) · [Analysis plan](../../docs/cuda/stream_warning_plan.md)

Copy [analyze_stream_warning.py](./analyze_stream_warning.py) to the GPU machine
and run it in the environment containing the installed libraries. It has no
repository imports or checkout path injection. Help requires only Python's
standard library. Execution requires PyTorch; DeepOpt cases require DeepOpt,
and ViT cases require timm.

The program diagnoses warning phase, execution path, and numerical agreement.
It does not suppress warnings or change library stream management. Native
cases do not import DeepOpt. A warning at the first training backward is labeled
`candidate.train.backward`, with update/microbatch numbers.

## Start with one GPU and the small model

Run the controller with Python. Every case and diagnostic/timing/profile pass
gets a fresh process, avoiding reuse of PyTorch's once-only warning state.

```bash
python analyze_stream_warning.py \
  --cases eager torch_graph deepopt_graph runtime trainer \
  --device cuda --steps 3 --warmup 10 --timing-steps 0 \
  --execution-mode max_speed --node-trace --output-dir stream-bf16
```

The tiny model is Linear → LayerNorm → GELU → Linear. Cases are:

| Case | Candidate execution |
| --- | --- |
| `eager` | Native eager BF16 |
| `torch_graph` | Native graphed callable, manual loop, no DeepOpt import |
| `deepopt_graph` | DeepOpt BF16 graph, manual loop |
| `runtime` | DeepOpt BF16 graph through TrainingRuntime |
| `trainer` | DeepOpt BF16 graph through Trainer |
| `fp8` | Current tensorwise FP8, native-fast backward, compression off |
| `fp8_compressed` | Same FP8 path with configured activation compression |

FP8 cases use Trainer by default, matching the reported setup. Select
`--fp8-orchestrator manual` or `runtime` to isolate that interaction. Omission
of `--cases` selects all seven. Model capture leaves loss, optimizer and outer
orchestration eager.

## Run the ViT configuration

```bash
python analyze_stream_warning.py \
  --cases fp8 fp8_compressed --model vit \
  --timm-model vit_base_patch16_224 --image-size 224 --classes 10 \
  --batch-size 8 --config config.toml --lr 1e-4 \
  --execution-mode max_speed --warmup 10 --steps 3 --timing-steps 0 \
  --launcher torchrun --ranks 1 --output-dir stream-vit
```

Start with batch size 8, then increase it to the reproducing value if memory
permits. Synthetic batch size defaults to 8 independently of the TOML training
batch size. The runner does not load CIFAR10 or launch data workers. Its local
torchrun launcher uses an independent loopback rendezvous; even one rank
initializes a process group, allowing comparison with plain Python.

Config supplies accumulation, capture warmup/unused-input defaults, compression
bits/group size/recentering, optimizer type/hyperparameters, EMA for Trainer,
checks, and diagnostic policy. Explicit CLI options override supported fields.
Only `name`, `seed`, `training`, and `runtime` are consumed from TOML;
application-owned tables are excluded from reports to avoid copying secrets.
Current tensorwise FP8 scaling, BF16 CUDA AMP, disabled compilation, and each
case's enablement are controlled by the experiment. Mode schedules are disabled
to keep the probe stationary. `--execution-mode max_speed` matches the earlier
explicit Trainer override; `legacy` honors the legacy checks switch. Loaded
and effective configurations are recorded separately.

Deterministic runs default `CUBLAS_WORKSPACE_CONFIG` to `:4096:8` before CUDA
initialization, preserving an existing setting and recording it in the report.

## Correctness and replay

Each diagnostic candidate is compared with a new eager model using the same
starting state, recipe, batches and orchestrator. Checks cover every diagnostic
output, optional input gradients, and normalized parameter gradients/weights
at the final optimizer update. Only final parameter/gradient snapshots are kept
on CPU; full ViT comparisons still need substantial host memory. Warmup-mutated
buffers and RNG state are preserved for matched comparisons.

```bash
python analyze_stream_warning.py \
  --cases torch_graph deepopt_graph runtime trainer \
  --input-grad --accumulation 2 --validation-transition \
  --execution-mode max_speed --timing-steps 0 --output-dir stream-transitions
```

Validation transition is one no-grad eval forward between updates 1 and 2,
not a full validation-loader/metrics pipeline. Its intentional eager-fallback
count is excluded from training replay checks.

Nonfinite values, numerical mismatches, worker errors/timeouts, and unexpected
DeepOpt training fallbacks produce `attention_required` and a nonzero exit code.
An upstream native graph gradient-aliasing issue can also cause a mismatch;
do not attribute every difference to streams. Defaults are rtol 0.02 and atol
0.002, recorded in the report. Inspect errors before changing tolerances.

The initial probe explicitly disabled FP8 during BF16 capture, whereas
Runtime/Trainer inherit the ambient FP8 policy when disabled. Those distinct
policies caused BF16 orchestration cases to fall back to eager. The corrected
probe matches the runtime scope; rerun earlier BF16 Runtime/Trainer results
before using them for graph performance comparisons. This correction does
not change the FP8 cases or suppress the stream warning.

## Separate timing and profiling

```bash
python analyze_stream_warning.py \
  --cases torch_graph deepopt_graph trainer fp8_compressed \
  --execution-mode max_speed --timing-warmup 5 --timing-steps 30 \
  --profile --profile-steps 3 --output-dir stream-profile
```

Timing is a separate process with a reused, preallocated synthetic GPU batch.
It excludes correctness snapshots, node tracing and setup instrumentation.
Wall/CUDA-event times include optimizer updates and orchestration; event
intervals include submission gaps. Lightweight harness phase scopes remain,
so this is not a full training throughput benchmark. There is no per-step
device-wide barrier; multi-rank timing aligns once before measurement.

Profiling is a third process exporting Chrome traces with forward, loss,
backward and optimizer ranges, plus graph-launch/synchronization API counts.
Use the ranges to distinguish forward/backward graph replay and expected
ordering from extra waits. Counts alone do not prove warning-related overhead.
Profiler times are labeled separately. Mode transitions run only in diagnostics.

## Lifetime and stream controls

- `--execution-stream side`: caller-side execution stream with boundary
  handoffs. PyTorch may still choose its own internal capture streams.
- `--retain-outputs`: deliberately retain outputs/losses to test graph lifetime.
- `--retain-graph`: manual-loop backward control; rejected for Runtime/Trainer.
- `--zero-grad zero`: manual-loop clearing control; Runtime/Trainer own clearing.
- `--gradient-clones off`: **unsafe diagnostic control**, disabling DeepOpt's
  graph-gradient ownership helper in that worker only. It can introduce
  gradient aliasing/corruption and is not a recommended workaround.
- `--node-trace`: feature-detects node-creation hooks, recording IDs/shapes and
  calling streams without retaining Node objects.

Calling-stream handles are not a direct readout of an existing accumulator's
internal stream. IDs are process-local and may be reused after destruction.
Metadata limits/truncation are reported. Later silent iterations cannot
establish warning frequency. The eager reference runs after the candidate,
so its warning absence is not independent evidence either: compare the
separately launched native cases for attribution.

## Multi-GPU after single-GPU

```bash
python analyze_stream_warning.py \
  --cases torch_graph deepopt_graph runtime trainer \
  --execution-mode max_speed --launcher torchrun --ranks 2 \
  --steps 3 --timing-steps 0 --output-dir stream-ddp
```

This is a local two-GPU/NCCL test. Capture precedes DDP construction; DDP setup
and execution share the caller-stream scope. Each rank has its own JSON,
warnings and eager comparison. The controller preserves CUDA_VISIBLE_DEVICES
and clears inherited rendezvous variables/PYTHONPATH. Run the controller once,
rather than wrapping it in an outer multi-rank torchrun.

## Read results

The controller prints a compact JSON summary. `<output-dir>/latest.json` points
to a unique invocation's `report.json`. Each case/pass has per-rank JSON and
`process.log`; profile passes also have Chrome traces. Inspect:

1. `summary[].ranks[].first_mismatch_phase`, then the warning's update,
   microbatch, stack and calling-stream handle.
2. `correctness.comparisons` for output/gradient/update agreement.
3. `training_replay` for unexpected fallback. Forward counters alone cannot
   prove backward replay; inspect the profiler trace.
4. `events`/`accumulator_creations` around the first affected phase.
5. Versions, GPU, installed-package path and pip VCS commit when available.

Native-only mode can help prepare an upstream reproducer:

```bash
python analyze_stream_warning.py --cases torch_graph --timing-steps 0 --output-dir native-only
```

CPU smoke mode uses FP32 and validates orchestration/reporting only. It makes
no CUDA or stream-correctness claim:

```bash
python analyze_stream_warning.py --device cpu --cases eager --steps 2 --timing-steps 0
```
