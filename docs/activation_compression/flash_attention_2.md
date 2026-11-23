# Explicit FlashAttention-2 and attention alignment

`runtime.attention.backend = "flash_attn_2"` selects the separately installed
FlashAttention-2 forward **and** backward kernels. The default, `native`, keeps
PyTorch SDPA backend selection. Backend choice is independent of FP8 projection
precision and saved-activation compression.

```toml
[runtime.attention]
backend = "flash_attn_2"

[runtime.activation_compression]
enabled = true
bits = 2
attention_bits = 4
group_size = 1024
norm_xhat_recenter = true
```

Load before model preparation, `Trainer` construction, and CUDA Graph capture.
`prepare_fp8_model` prepares attention even with FP8 disabled; `Trainer` does
this automatically. Prepared timm attention uses its SDPA branch. Standard
`nn.MultiheadAttention` is adapted preserving parameter identity/order; call it
with `need_weights=False`. Explicit selection rejects masks, causal attention
and GQA rather than silently falling back. Supported inputs are NVIDIA Ampere
or newer, BF16/FP16, nonempty BHSD with equal head counts, matching dimensions
divisible by eight up to 256, and equal K/V lengths. Noncausal cross-attention
with different Q/K lengths works. FP32 inputs are cast under BF16/FP16 AMP.
Dropout uses the installed FA2 RNG state. Higher-order gradients are unsupported.

Forward uses original Q/K/V. Saved Q/K/V are packed once in BSHD (batch,
sequence, heads, dimension) logical order; the original output, logsumexp and
RNG state stay exact. Backward reconstructs Q/K/V and invokes installed FA2
backward. This mirrors the legacy saved-state contract and remains an
approximate backward. Inference without autograd skips packing; compressed
state does not retain dense Q/K/V copies.

The production FA2 path compresses **K itself**, before any last-two-axis
transpose. It never saves or compresses K-transpose. The benchmark's explicitly
labelled shared `saved` layout is an experimental transposed-K control only.

## Installation and CUDA tests

```bash
python -m pip install packaging psutil ninja setuptools wheel einops
MAX_JOBS=4 python -m pip install flash-attn==2.8.3.post2 --no-build-isolation --no-deps
python -m unittest tests.cuda.test_flash_attention_backend -v
```

Run the tests from the checkout with the updated DeepOpt build. They cover
public-library gradient parity, compressed backward against explicit low-level
backward, partial gradients, dropout, AMP routing, and changing-input graph
replay. CPU stand-ins qualify routing/state logic, not CUDA correctness.

## Resumable gradient and optimizer alignment

Keep [benchmark_attention_alignment.py](../../benchmarks/analysis/benchmark_attention_alignment.py),
[attention_comparison.py](../../benchmarks/analysis/attention_comparison.py), and
[diagnose_training_stability.py](../../benchmarks/analysis/diagnose_training_stability.py)
together. The comparison exporter includes these files.

```bash
# Synthetic code/performance check; this does not establish convergence.
python benchmarks/analysis/benchmark_attention_alignment.py \
  --batch-size 512 --codecs current legacy_rules --graph --profile \
  --output ./attention-synthetic

# Actual pre-instability model, batch, optimizer state and extra counters.
# Run where test_optimizer.py is importable.
python benchmarks/analysis/benchmark_attention_alignment.py \
  --snapshot ./stability-snapshot.pt \
  --optimizer test_optimizer:SGD_NS_Overshoot \
  --codecs current legacy_rules --output ./attention-real
```

Resume with identical arguments plus `--resume`. Fingerprints include options,
helper/library hashes, snapshot/workload hashes and Torch/CUDA/FA2 versions.
Completed alignment and timing stages are reused and failed stages retried.
JSON and CSV are written incrementally. Alignment/eager timing survives a later
graph failure. Select backends, modes, bits or `--iterations 0` for focused runs.

Actual snapshot replays extract Q/K/V/upstream gradients into `workloads/`,
then compare manual/math, forced PyTorch Flash SDPA, and installed FA2 with
dense, four-bit and eight-bit saved values. Individual Q-only/K-only/V-only
eight-bit cases are included. `--modes shared` fixes reconstructed operands
across algorithms. `--modes production` preserves actual saving layout/dtype.
Shared `--layouts bshd saved` isolates canonical versus transposed-K grouping.
The model replay forces timm's SDPA entry point consistently for all compared
algorithms, recording each original `fused_attn` flag. The manual comparison is
DeepOpt's FP32 QK/softmax/PV implementation, not a claim that timm's own manual
implementation or the original training run executed that same code.

`legacy_rules` is an unfused Torch reference for the active old affine rules:
per-sample zero padding, half-up codes, BF16 scale/minimum and reconstruction.
It does not claim bitwise parity with old Triton or cover optional alternate
packing modes. Production keeps current FP32 metadata to separate backend and
codec effects. Shared-mode timing includes diagnostic reconstruction and
patching and must not be treated as production performance.

Reports include cosine, angle, norm ratio, parallel scale, orthogonal error,
thresholded sign disagreement, and per-head/token distributions. Zero vector
directions are undefined. Both common-dense and own-backend-dense references
are reported. Local attention holds upstream gradients fixed. Full-model
replays include propagation differences, named parameter gradients, and actual
optimizer updates. Unsupported optimizer state prevents qualification; another
optimizer is never substituted. Snapshot alignment comparisons require zero
attention dropout: equal seeds do not imply equal masks across implementations.

Extraction is bounded by `--max-workload-bytes` (8 GiB by default). Skipped calls
are reported and prevent complete qualification. Timing/profiling are separate
from CPU copies/alignment collection; `--graph --profile` profiles graph replay.
Source shape, strides, storage offsets and Q/K/V aliases are recorded before CPU
cloning and rebuilt outside timing. Synthetic inputs default to interleaved Timm
QKV storage; `--source-layout bhsd` reproduces the older contiguous synthetic
fixture, and `--source-layout bshd` selects contiguous FA2 storage. Quantization
`--layouts` and physical `--source-layout` are independent controls. Local tests
accumulate independent Q/K/V leaf gradients; full-model matrices measure the
real projection/autograd consumer and must qualify any claimed copy saving.

## Replay and configuration recovery

Construct `Trainer`, apply module overrides, then capture its prepared model
under the runtime context used by `fit`:

```python
core = deepopt.CUDAGraphCore(device, config=trainer.config)
with trainer.runtime.autocast():
    captured = core.capture_model(trainer.model, (sample_x,), autograd=True)
try:
    trainer.fit(train_loader, valid_loader, epochs=epochs)
finally:
    captured.close()
```

An explicitly disabled FP8 configuration and an inherited `None` policy have
different dispatch semantics. Rebuilding an autocast scope from a disabled
configuration can select different interception flags and make every training
call fall back. Preserve the guard; use the runtime context instead. Model
captures expose `fallback_diagnostics`, recording the first host-side policy
mismatch in each category. `statistics` remains a separate set of counters.
Forward counters do not prove backward execution; the CUDA integration test
also observes distinct forward and backward graph objects and compares updated
parameters over changing batches. Validation uses the expected mode fallback.

The training runner checks replay after every optimizer update and rejects
unexpected training policy/fill fallback. Global deterministic/fill flags are
recorded as actually applied, rather than inferred from requested settings.

## Packing and copy candidates

The default remains three independent canonical BSHD packs. The opt-in
`activation_codec_scope(attention_pack="joint")` uses one Q/K/V kernel for
ordinary CUDA BF16/FP16 operands with identical shapes/strides and groups up to
2048. Every operand retains its own affine groups and FP32 metadata. Other
layouts use the generic packer. `strided_index="bounded"` permits 32-bit source
indexing only when the complete relative address range fits; otherwise indexing
stays wide. Packed words and metadata must match the separate implementation
exactly before throughput is considered. Neither option changes K to K-transpose.

The FA2 adapter passes upstream gradients with unit last stride directly to the
extension. Dense input permutations receive matching gradient-buffer layouts;
interleaved views with gaps receive compact buffers so allocation size does not
grow to reproduce unused QKV slots. This is independent of the model graph's
gradient-ownership clones, which remain necessary for accumulator correctness.

Run the focused portable recovery bundle after installing the matching library:

```bash
bash run_attention_recovery.sh ./attention-recovery --resume
```

It runs the installed CUDA integration test first, then both packing policies'
alignment matrices, canonical packing across three source layouts/index widths,
and paired eager/graph resident-input model matrices. Models include Timm
ViT-Base and Torchvision ResNet18/EfficientNet-B0 with no pretrained weights.
All five feature cases, including plain Torch BF16, remain visible. The model
matrices use their declared AdamW/LR performance proxy and existing numerical
tolerances; they do not qualify custom-optimizer convergence or CIFAR10 epochs.
Timing holds an exclusive GPU lease; CPU supervisors may overlap preparation.
The preflight must pass before the long stages start. Failure/incomplete status
is preserved rather than reported as a complete comparison.

For a fast packing-only screen:

```bash
python benchmark_attention_pack.py --qkv-only --bits 4 8 \
  --group-sizes 256 1024 --source-layouts interleaved \
  --indexing wide bounded --output ./attention-pack.json
```

Enable `--attention-pack joint` and optionally `--pack-indexing bounded` on the
alignment or real-training runner for candidate evaluation. These paths remain
opt-in until H100 numerical and complete-step performance checks pass. The
matrix equivalents are `runtime.attention_backend="flash_attn_2"` and
`runtime.attention_pack="joint"`. The default matrix backend remains native.

## Actual CIFAR10/224 training

[benchmark_vit_attention_stability.py](../../benchmarks/analysis/benchmark_vit_attention_stability.py)
uses local CIFAR10, 224x224 images, the supplied RandAugment and RandomErasing
transforms, and `pretrained=False`. Explicit CLI values override the supplied
config; omitted fields inherit it, then documented benchmark defaults. Batch
size defaults to 512 only when no supplied setting replaces it. The same
resolved snapshot creates loaders and Trainer. Supply the actual optimizer
class and existing config to retain EMA and other settings. If the reference
uses ZerO initialization, add `--initializer zero_init:init_ZerO`. Use the same
`--dense-out-proj` setting in each comparison.

```bash
python benchmarks/analysis/benchmark_vit_attention_stability.py \
  --config work/attention.toml \
  --optimizer test_optimizer:SGD_NS_Overshoot \
  --backend flash_attn_2 --attention-bits 4 --batch-size 512 --drop-last \
  --execution-mode max_speed --seed 14 --epochs 32 \
  --snapshot-before-update 1100 --output ./vit-fa2-4bit-seed14
```

Compare `--backend manual --attention-bits 4` and
`--backend native --attention-bits 8`, with distinct output directories and
identical seed/config. Zero attention bits keeps only Q/K/V dense. Graph capture
is enabled by default; `--no-graph` selects eager. After BF16 qualification,
`--fp8` enables the current-scaling/native-fast FP8 projection recipe. Repeat
promising candidates with seeds 15 and 16.

No optimizer LR is guessed. The actual kwargs inherit `training.learning_rate`
unless explicitly supplied through `--optimizer-kwargs` or `--learning-rate`.
Use your verified LR, including any intended schedule from the config; the old
hidden `4e-4` default has been removed. Supplied group size, FP8, compile, fill,
and deterministic settings are preserved. Explicit `--no-fp8`/`--no-compression`
disable features; `--attention-bits 0` disables only attention saves. Graph runs
require `drop_last=true` and `runtime.cuda_graph.autograd=true`. Reports include
requested arguments, resolved settings, actual optimizer kwargs and execution
flags. The runner retains the model factory's GELU formula and the library's
LayerNorm correction and quantization semantics.
Custom optimizer extras are persisted with epoch checkpoints; this runner
explicitly rejects a nonzero `training.checkpoint_every_steps` instead of
silently overriding it or claiming unsupported exact step resume.

Epoch reports contain loss, accuracy, throughput, and graph statistics. Resume
with the same command plus `--resume`. Custom optimizer fields omitted from
`state_dict()` are saved before the corresponding epoch checkpoint and restored
with it. Unsupported fields reject qualification. Completion means all requested
epochs finished; inspect accuracy/loss and update alignment before calling a
candidate stable. GPU correctness and sustained training require the H100.
