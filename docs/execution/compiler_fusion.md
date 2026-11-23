# Isolated compiler fusion qualification

[Native AutoDiff](compiler_native_autodiff.md)

Fusion runs after native AutoDiff derives the semantic backward graph. The
isolated harness compares the same numerical backend, codec, GEMM tile and
requested schedule with one transformation toggled. Library GEMM comparisons
are reported separately. Generated GEMMs and the new packed multiplication and
pack sharing options remain explicit choices; no GPU speedup is implied by a
selected rewrite or fewer launches.

## Controls

```toml
[runtime.compile.options]
require_full_aot = true
fuse_deepopt_activations = true
fusion_phases = "both" # "forward", "backward", "both", or "none"
share_compatible_packs = false
fuse_packed_multiply = false
```

`fusion_phases` controls the replacement stage while leaving differentiation
and the requested gradients unchanged. A forward-only fusion trial can still
execute a complete training update, with an unfused backward control.
`share_compatible_packs` enables CSE of explicit saved-operand pack operations;
`fuse_packed_multiply` consumes permitted packed coefficients inside a product
or bounded product/reduction. These two transformations have independent
switches and reports.

## Implemented contracts and limits

| Pattern | Replacement and retained semantics | Qualification limit |
| --- | --- | --- |
| BF16 GEMM + GELU/SiLU/residual | Existing generated GEMM epilogues preserve the BF16 accumulator-output boundary, escaped preactivation, exact or explicitly packed derivative state. | Generated route is opt-in. Native-library GEMMs remain eligible controls. |
| LayerNorm + GELU/SiLU | Existing row/chunked apply kernels receive an optional activation. Norm backward applies the exact derivative during gradient loads for dX and affine partials. | Prepared native-dense Triton norm path only; library LayerNorm and activation coefficient codecs reject visibly. |
| GroupNorm + GELU/SiLU | Existing spatial apply and backward pipelines preserve group geometry, per-channel affine gradients, xhat correction and captured output dtype, including FP32 CUDA-autocast output. | Native dense gradients; FP8 output/gradient transport and BatchNorm effects remain separate. |
| Packed coefficient × gradient | Decode to the coefficient dtype, multiply in the original promotion domain and round the original product before a broadcast reduction. Optional FP32 partials and a finishing reduction avoid a full decoded coefficient and full broadcast product. | Triton BF16/FP32, logical row-major codec, at most eight product axes. Partial scratch is capped at 8 MiB and 4096 partitions. Other exact readers can still retain a restore, which is reported. |
| Compatible explicit saved packs | One payload/step/minimum producer, remapped saved bindings and backward inputs; ordinary final liveness retains all shared components until every reader finishes. | Requires the identical SSA source, effect interval, dtype/layout/cast, traversal, codec recipe, role and correction state. Packs hidden inside opaque operator calls are not shared by this pass. |

Xhat packs, activation derivative packs and raw consumer-input packs remain
distinct. Exact readers retain their exact inputs. Unsupported layouts,
transport domains, policies and escaping contracts produce rejection records.
Pure inference does not gain derivative residuals.

The fusion pass supplies dependency names and ordering preferences for GEMM
dX replacements. The final scheduler owns placement, effect barriers and
liveness. Reports distinguish the requested, selected, post-fusion and final
dX/dW/db order; a late derivative position is not treated as evidence that
dX-first executed.

## Run paired fixtures

The entry point is
[`benchmark_native_autodiff.py`](../../benchmarks/compiler/benchmark_native_autodiff.py).
Examples below require a CUDA host for physical kernel qualification:

```bash
python benchmarks/compiler/benchmark_native_autodiff.py \
  --fixtures gemm-activation layernorm-activation \
  --phases forward backward both --activation gelu \
  --batch 8 --tokens 197 --width 768 --hidden 3072 \
  --trials 3 --profile-steps 1 --output work/fusion-transformer

python benchmarks/compiler/benchmark_native_autodiff.py \
  --fixtures groupnorm-activation --phases forward backward both \
  --activation silu --batch 1 --width 128 --groups 32 --spatial 32 32 \
  --autocast --compression on --bits 1 2 4 8 --group-size 127 \
  --trials 3 --profile-steps 1 --output work/fusion-groupnorm

python benchmarks/compiler/benchmark_native_autodiff.py \
  --fixtures packed-multiply shared-packs --phases backward both \
  --compression on --bits 1 2 4 8 --group-size 127 \
  --trials 3 --profile-steps 1 --output work/fusion-codec
```

Use `--dry-run` to inspect a proposed matrix, `--device cpu` for semantic smoke
checks, and `--gradient-mask all|input|weight|bias|none` for selective gradients.
Layouts include `contiguous`, `strided`, and spatial GroupNorm `channels-last`.
The test matrix also covers tail widths, multi-axis LayerNorm and absent
affine parameters. A shared projection fixture can truthfully be unsupported
when its implementation has no explicit compatible pack operations.

Every strategy runs in a fresh process. Paired strategies share initial model
and optimizer state, input, gradient, RNG seed, recipe and measurement policy;
their process order is randomized. Source bytes are hashed before and after
measurement and the imported package path must match the workspace. The
harness records cold compilation separately, warmed forward/backward and full
SGD-update timings, saved storage and allocator peaks. The synchronized phase
diagnostics are separate from ordinary full-update timing.

## Read the evidence

Each candidate has a graph/callsite/phase/pattern identity, a replacement
contract, implementation fingerprint, exact profiler range, required numerical
kernel roles, and final operation order. The measurement gate joins launches
to that exact range and checks its kernel symbols. An unrelated db/reduction
launch cannot qualify a GEMM or normalization candidate. Missing profiler
attribution remains unknown, including when profiling is disabled.

Pack CSE has separate `pack_sharing_decisions` and encode evidence. A selected
shared binding does not prove encoding work disappeared: its canonical range
must show exactly one encode launch. CSE does not increment kernel fusion
counts.

The harness distinguishes numerical correctness, observed CUDA execution and
performance promotion. At least three fresh-process paired trials are needed
for repeated evidence; near-noise timing differences require further trials
and review. New norm and packed-operation CUDA kernels have CPU recipe and
legality regressions but remain **CUDA unqualified** until target-device
compilation, candidate-specific traces, numerical checks and repeated timings
pass. No full-VAE benefit or batch-128 execution is claimed.
