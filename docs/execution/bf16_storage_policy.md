# BF16 policy contract

The versioned [policy descriptor](../../src/deepopt/_bf16_policy.py) distinguishes
the existing `torch_amp` recipe from opt-in `bf16_storage`. The latter is an
experimental policy. Use the target-device qualification checks before
selecting it for a workload.

## Selection and ownership

`runtime.bf16_policy` accepts `torch_amp` (default) and `bf16_storage`. Complete
runtime configurations require `precision="bf16"` and FP8 disabled for storage
mode. Partial configuration snapshots may defer compute dtype/device selection;
the invocation must still validate effective BF16 execution before any effects.
The configuration module and descriptor import without Torch.

`deepopt.autocast(..., bf16_policy="bf16_storage")` selects eligible forwards.
Omitted controls inherit the enclosing scope; an explicit policy selects that
scope. An inner `enabled=False` suspends automatic storage rewriting without
converting explicit BF16 inputs to FP32. A later explicitly enabled nested scope
can re-enable the policy. All scope state restores on normal or exceptional exit.

Validated native operation plans retain the canonical descriptor
`{"name": "bf16_storage", "version": 1}`. Prepared models own an independent
snapshot, preserve parameter objects and reject incompatible re-preparation.
Direct functional operations resolve their owned descriptor too; a field in a
plan cannot silently select different execution. An explicitly conflicting
policy scope is rejected. Ordinary legacy/default operation plans can use an
explicit storage scope. Effective FP8 and storage together are rejected before
forward execution.

Backward retains the immutable policy selected by its own forward invocation.
Ambient autocast changes, scope exit and another graph's forward must not change
its saved-state recipe. Retained graphs and CUDA replay require the same rule.

## Storage and arithmetic

| Value or boundary | Contract |
| --- | --- |
| Eligible internal floating activation/adjoint | BF16 storage under `bf16_storage` |
| Normalization/nonlinear arithmetic and required reductions | FP32 arithmetic; GPU kernels store directly to declared outputs |
| Trainable FP32 master parameters and final parameter gradients | Preserve FP32 ownership |
| Optimizer state | Preserve optimizer-declared types and objects |
| Requested external floating leaf gradient | Preserve source dtype at that boundary |
| Explicit FP32 loss island | Preserve FP32 operands and output contract |
| Statistics and affine-codec minimum/step | Preserve declared FP32 representation |
| Masks, indices, counters, RNG, integer/FP64 diagnostics | Preserve original types and semantics |

Compression is independent: disabling it does not disable the storage policy;
selecting storage does not enable compression. Compression tests use group 1024.
Existing bit-width, layout, gradient-mask and shared-save cases remain separate
coverage axes. Backend selection and arithmetic presets are also independent.

## Saved-state classification

Native reports use saved-state schema 2. The shared [classifier](../../src/deepopt/compiler/saved_state.py)
validates physical components against their constructor ABI and codec or
normalization geometry. The [VAE gate](../../examples/vae/qualification.py) uses
this contract for both policies. Unknown schemas, contradictory memberships,
and unexplained BF16/FP32 saves with at least 1,024 elements fail compressed
qualification. A name, provenance label, or exact-retention reason alone does
not grant an exemption.

Model-state provenance and storage ownership are separate. Original parameter
and buffer aliases refer to persistent storage. Declared casts and clones of a
validated model source remain separate saved allocations, including the VAE
BlurPool kernel casts. Their actual bytes stay in nonpersistent runtime totals.
Codec payloads, FP32 step/minimum arrays, and normalization mean/rstd/corrections
also retain their actual types and bytes. Compatible shared components preserve
every consumer and appear once per physical save.

Under `torch_amp`, `exp.forward_result_coefficient` explicitly retains the actual
forward result in its forward rounding domain. The FP32 coefficient at
`[1,4,28,28]` occupies 12,544 bytes; batch 128 occupies 1,605,632 bytes. This exact
coefficient is reported separately. The exception requires the exp operator,
source identity, dtype/shape, policy, and exact consumer contract. It does not
apply to `bf16_storage`: its compression-on model exp keeps its existing packed
FP32 coefficient, and compression-off keeps its BF16 recomputation snapshot.

Report byte summaries describe declared tensors. The runtime storage ledger
establishes unique backing allocations, tied aliases, offsets, and release after
backward. Classification does not reduce memory or establish CUDA capacity.
Native Torch Inductor remains an uncompressed native-operations runtime reference
using PyTorch AOTAutograd. Its cast-preserving preset is a different arithmetic
recipe, and does not implement or qualify DeepOpt storage or native AutoDiff.

## Evidence and limits

The policy descriptor/configuration/scopes and prepared ownership are exercised
by [policy tests](../../tests/opt/test_bf16_policy.py). Norm/activation, native
AutoDiff, fusion, checkpoint/replay and full-model tests provide additional
independent gates. A same-policy eager/native failure remains a failure; a
fused/unfused agreement cannot override a failing eager comparison.

Throughput collection must follow accepted numerical execution. Saved storage,
temporary scratch, allocated/reserved peaks and process memory have different
owners and must be reported separately. Logical reference deletion is not proof
of physical allocator reuse.
