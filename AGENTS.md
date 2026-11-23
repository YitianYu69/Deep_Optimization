# Compiler reference workflow

For compiler, AutoDiff, fusion, code generation, scheduling or tensor-memory
work, consult [the saved Inductor reference guide](compiler_references/inductor/README.md)
and any local index at `compiler_references/inductor/index.json` before choosing
an implementation. Verify relevant saved manifests and inspect matching forward
and backward graphs, generated code and runtime traces when available.
Also inspect Inductor artifacts inside the benchmark output being analyzed;
their compiled code and runtime measurements belong to the same run. Inductor
references must use native Torch operations, excluding DeepOpt custom/compressed
operations and opaque eager fallback boundaries.

Match the model, recipe, shape, dtype/layout, software and GPU identity. A blocked
attempt or CPU diagnostic is not CUDA execution evidence. Plain Inductor uses
PyTorch AOTAutograd and does not qualify DeepOpt's native AutoDiff. Record
missing references honestly and obtain target evidence before promoting a
performance policy.

Saved reference runs are permanent and immutable. Create new runs for changed
conditions; do not overwrite historical captures or rely on temporary compiler
caches. Save code, IR and diagnostics without tensor checkpoints. Compression
testing uses `group_size=1024`.

## Learnings

Runtime-manager presence does not prove allocator ownership: a lazy unused
facade can have no plans, pools, transfers or leases and legitimately omit
physical accounting fields. Exempt only the complete frozen zero-resource
schema, while retaining strict allocator-domain and budget proof for every
initialized owner; replay real runtime snapshots including unused facades
before validating a whole-model cohort. Check frozen runtime budgets before
timed updates and stop a capacity sweep after structural evidence failures,
while preserving explicitly verified resource-limit exploration. Revalidate
completed immutable measurements offline when a reporting-only bug is fixed
instead of requiring another GPU run.

Extract model-scope offload inventory through the declared compiler graph,
native-region and `packed_activation_offload.model_scope.inventory` records;
scan every declared owner for missing or conflicting records rather than
recursively searching unrelated schedules under an arbitrary container limit.
Replay the full original-shape metadata report when fixing this handoff: a
small model does not cover a multi-million-container report. Keep completed
native execution separate from a subsequent inventory-export failure in both
worker and parent diagnostics.

Positive convolution qualification must pass `enabled=True` explicitly to
`torch.backends.cudnn.flags`; the omitted argument defaults to `False` and can
send large depthwise inputs to a 32-bit native kernel. Record actual scoped
cuDNN controls and selected backends before numerical work, while preserving
intentional disabled-backend rejection tests. A passing small CUDA case under
the disabled backend does not qualify the intended large cuDNN route.

Disable TorchDynamo reproduction by removing `TORCHDYNAMO_REPRO_AFTER` from the
worker environment and setting an already-imported `torch._dynamo.config.repro_after`
to Python `None`, with `repro_level=0`; empty strings and the string `"None"` are
invalid values even when reproduction level is zero. Test compiler-environment
changes with actual TorchInductor forward and AOTAutograd backward execution
before exporting benchmark code; fake-worker tests cannot validate these paths.
A CPU compiler check validates that diagnostic configuration only, and does not
establish full-model CUDA correctness or performance.

Saved benchmark settings can be an intermediate envelope containing both an
Inductor compile mode and DeepOpt native options; normalize the selected route
before validating configuration, including when only reading schedules.
Preserve the original receipt and strict mode/options exclusivity rather than
dropping native options or relaxing the parser. Before exporting a diagnostic
runner, exercise a real worker with both fields present in recorded settings;
planning tests and CPU inputs missing either field do not cover this handoff.

Size numerical-oracle RAM against the full original parameter geometry and
optimizer state after lazy initialization, including transport and comparison
staging. Small-model runs cannot validate this capacity contract; keep explicit
limits strict and report required versus allowed bytes before staging state.
For native saved tensors, verify that runtime shape/stride/dtype matches the
declared backward ABI on CUDA fast paths as well as CPU implementations; dense
BF16 recompute coefficients use the declared matrix layout even when the
activation input is N-D. Preserve strict metadata guards when fixing producers.

Before attributing a numerical failure to native AutoDiff or compression,
check same-route backward repeatability with the same inputs, state and
cotangents under explicitly recorded algorithm controls. Equal repeat and
cross-route error levels require operator-local investigation; repeat noise
does not justify loosening FP32 gradient tolerances. Keep full code/IR and
canonical comparisons in hashed artifacts, with compact parent receipts.

Strict deterministic CUDA replication padding can enter a Dynamo nonstrict
tracing helper even when every padding width is zero. CPU capture does not
exercise that device-specific branch. Preserve native argument validation and
full AutoDiff coverage when capturing this identity operation, and avoid
calling a replication backward kernel for its identity adjoint. Require real
CUDA capture and backward checks before treating this path as target evidence.

Checkpoint operation-recipe discovery must use an iterative worklist and
canonical bound-method identities (`__func__`, `__self__`) when traversing
compiled/precision facades, while preserving distinct numerical owners.
Benchmarks ending before the first scheduled save do not validate checkpoint
compatibility: exercise real native forward/backward, a step checkpoint,
restore and the next optimizer update. Preserve declared operation-schema
guards; do not mask traversal cycles by raising the recursion limit, omitting
recipe metadata or swallowing checkpoint failures.

When fusing activation adjoints into normalization reductions, preserve or
explicitly qualify the physical cotangent layout: standalone activation
backward writes contiguous BF16, while an expanded upstream view has zero
strides and can select different reduction code. Report any raw cotangent
layout-copy bytes separately from eliminated activation-adjoint storage.
CUDA comparisons must name each layout and tensor role, check same-route
repeatability, and retain partial-reduction comparisons plus compiled
TTIR/TTGIR/LLVM/PTX when diagnosing an FP32 mismatch. Matching scalar formulas
or a global Triton fusion warning alone does not establish equal reduction
order or justify changing numerical tolerances.

Packed offload lifecycle calls belong to generated scheduler/runtime code;
application training loops only configure policy and budgets. Pre-admit every
known region's selected history before its outer compiled forward begins, and
share host/staging budgets across regions in one execution scope. Distinguish
retained graph leases from DMA-only retirement: a subsequent update may wait
on already submitted completion work, but must not wait on a future backward
to obtain host capacity or spuriously fall back because the CPU ran ahead.

Saved-hook resident passthrough tokens must match stock `save_for_backward`
under mutation, resize and `.data` storage rebinding. A detached snapshot alone
does not follow an unchanged-version `.data` rebind of a still-live owner;
retain version/storage checks and use a weak original reference without
creating a graph ownership cycle. Exercise real generated forward/native
backward and graph abandonment; CPU lease tests do not establish CUDA safety.

Allocate restored CUDA buffers on their H2D stream: strict deterministic
uninitialized-memory filling can enqueue writes from `torch.empty`, so
allocating on another stream and copying without an allocation-ready
dependency can corrupt restored bytes. `record_stream` protects allocator
lifetime and does not order those writes. On failures, attempt all owned
cleanup and clear training state while preserving the primary exception;
attach secondary cleanup diagnostics, including panel-retirement failures.

Configured model preparation must remain idempotent across Trainer and
TrainingCore construction. `nn.Module.compile` traces `_call_impl` and resolves
the current `forward` lazily: preserve an existing numerical `_PrecisionBody`
instead of adding another policy facade after compilation, and keep ContextVar
scope mutations in the outer host facade. Test repeated preparation with a
real precision facade and native forward/backward, plus a saved step, restore
and next update; direct `compile_fx` or unwrapped AMP-only tests miss this path.

Serialized DDP tuning options do not activate distributed execution; validate
their schema and preserve them, while rejecting unsupported offload from the
actual process-group/runtime owner. Preflight every benchmark arm with the
full recorded recipe and installed candidate before expensive CUDA contracts
or measured workers. Label isolation failures by the offload arm rather than
the shared native execution route, and keep failed, unexecuted workers out of
memory/runtime comparisons.

Packed-offload DDP requires a cold, fullgraph native model and scoped
`optimize_ddp=False` at actual lazy execution. Torch DDPOptimizer partitions
otherwise reserve history independently; changing its setting does not
invalidate an already cached split specialization. Do not reset unrelated
Dynamo caches to hide that ownership gap. Keep resident DDP partitioning
separate, record the communication-overlap tradeoff, and exercise both
two-rank and four-rank gradient/checkpoint contracts. For strict multi-rank
oracles, reproduce the reducer's bucket geometry and averaging order from
independent reference gradients; bucket rebuilding can change FP32 reduction
order after checkpoint restore.

User-facing offload budgets use decimal GB aliases (1 GB = 10^9 bytes), with
80 GB per rank as the supplied default. Normalize each incoming table before
merging inherited byte descriptors; reject both units in the same explicit
table, preserve inherited positive budgets, and keep canonical checkpoint
wire formats unchanged. Do not assume a host allocation or resize user budgets.
