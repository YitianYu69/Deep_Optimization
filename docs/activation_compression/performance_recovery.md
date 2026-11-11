# FP8 and activation-compression performance recovery

[Compression guide](README.md) · [Benchmark suite](../../benchmarks/analysis/README.md)

The supplied H100 ViT run measured 229.61 ms/update for the BF16 trainer,
249.59 ms for FP8 and 393.79 ms for FP8 plus 2-bit compression. The separate
profile attributed 117.66 ms/update to activation reconstruction, including
roughly 114.84 ms rebuilding transposed projection inputs. These are the
original measurements, not speedups measured for the updated implementation.

## Implementation

- FP8 input loads use floating padding in all affine packer paths, including
  chunked groups. This addresses Triton rejecting an integer zero when a planned
  activation arrives already stored in FP8.
- Matrix transposes use a two-dimensional unpack kernel. It gathers the original
  flat or rectangular compression groups across a source tile and transposes
  the decoded tile into the existing padded, aligned output layout. It creates
  no full BF16 intermediate and does not change group membership, packed words,
  metadata, rounding/saturation rules or the saved-tensor ABI. Small/nontransposed
  cases retain the grouped kernel. The initial tile policy is static; target-GPU
  measurements remain necessary before claiming it beats the old implementation.
- Module-mode transformation retains exact built-in Sequential, ModuleList and
  ModuleDict objects and converts their children. Parent forwards and checkpoint
  helpers can continue to iterate/index them, and the precision planner can
  analyze their converted children. Custom container subclasses retain the
  existing tracing path so custom functional computation is still rewritten.
- An autocast constructor resolves one authoritative owned experiment snapshot.
  Its internal sections reuse that validated snapshot rather than repeatedly
  merging and validating the entire experiment. Reusable runtime scopes still
  avoid reconstruction per update. Disabled FP8 does not install an unnecessary
  interception mode; compression can independently install its required mode.

CPU tests interpret the actual JIT bodies and test launch selection, bitwise
decoded values, indexing, tails, rectangular blocks, binary overflow handling,
container ownership and policy nesting. They do not compile CUDA or establish
GPU throughput. Dedicated CUDA regressions exercise direct FP8 input packing,
tiled/copy/grouped equivalence, graph replay and the planned GELU→Linear edge.

## Run on the H100

Install the updated build into the same environment used for the original run,
then compare identical payloads with the targeted benchmark:

```bash
python -m pip install .
python benchmarks/activation_compression/benchmark_activation_unpack.py \
  --shapes 100864x3072 100864x768 4096x768 37x65 \
  --bits 2 --group-sizes 1024 --dtypes e4m3fn \
  --execution both --output work/unpack-recovery.json
```

The controls are the old grouped transpose, contiguous decode plus an aligned
transpose copy, the tiled transpose, and production auto selection. All consume
the same packed activation and must produce identical decoded values. Eager
timing includes allocation and host dispatch; graph replay retains capture
buffers. Their allocator peaks have different meanings and are labeled.

For the real-model five-case comparison:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python benchmarks/analysis/benchmark_do_ops.py \
  --config benchmarks/analysis/configs/h100_vit_recovery.json \
  --require-fp8-plan --deterministic --fill-uninitialized-memory \
  --warmup 10 --iterations 30 --repeats 5 \
  --output work/vit-recovery
```

This factory uses timm without pretrained weights or downloads, resident images,
AdamW, EMA and the same principal FP8/compression settings. It measures all five
cases, including compression only. EMA updates run outside model graph capture
and are reset between diagnostics. `--require-fp8-plan` rejects empty/failed
plans in FP8 cases before timing. FP8 graph updates reject enabled weight caching
because Python version checks cannot refresh cached operands on graph replay.

This harness does not reproduce the original trainer's metric collection,
gradient-clone controls or process-group orchestration. Rerun the original
trainer diagnostic command as well for an exact before/after comparison. The
explicit finite-check setting here avoids host scalar reads during capture;
compare effective captured behavior, not just the requested `checks` flag.

Use a separate invocation with `--profile-steps 3` for the new kernel breakdown;
retain the uninstrumented run as the throughput result. Allocation filling is
explicit above because it was present in the supplied traces. Test a matched
`--no-fill-uninitialized-memory` comparison separately rather than crediting that
policy change to the new decoder.

## Accuracy gates

The analysis suite supports independent `gradients` and `updates` tolerance
policies, with `max_relative_l2` and `min_cosine` in addition to `rtol`/`atol`.
New default/example settings use diagnostic relative-error and cosine gates;
an explicit legacy configuration retains its supplied thresholds. The H100
configuration includes separate policies. Raw timings remain available when
accuracy is rejected. These thresholds are diagnostic budgets, not evidence
that a lossy training recipe converges. Use `--max-numerical-elements 0` for a
full numerical comparison; otherwise float errors are sampled and finiteness
and integer routing are checked in full.

Layout optimizations must match the old codec exactly. Bit width, grouping,
recentring, scaling and accumulation changes are separate numerical experiments.
For numerical precision comparisons, use original BF16 outputs and gradients;
for codec integration, also compare against the same FP8 operands with the
original decoded compressed input.

## Subsequent tuning

After CUDA correctness and end-to-end measurement, reuse the existing linear
comparison's shared-operand native-fast/native-precise/Triton probes. They
separate GEMM work from format/layout costs. Do not change the production GEMM
backend based only on isolated tensor-core time. Native forward bias handling,
eligibility and numerical differences require their own complete-path evidence.

Sweep compression groups, widths, attention overrides, recentering and scaling
only after the layout implementation is held fixed. Record allocated/reserved
peaks, scratch and retained payloads; an observed payload reduction does not
establish a full-training memory reduction. Keep delayed-scaling capture
restrictions explicit. Compare recipes on the latency/memory/accuracy frontier.

The initial performance budgets are aggregate unpacking below 15 ms/update and
combined ViT below 300 ms/update, followed by a 250 ms/update target and comparison
with the 229.61 ms BF16 baseline. None of these budgets is a verified result until
the H100 measurements pass. A faster isolated decoder is insufficient if later
casts, layout copies, planner fallbacks or optimizer work erase its gain.
