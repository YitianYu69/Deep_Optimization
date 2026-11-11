# Plan: fused mean correction for compressed normalization xhat

[Normalization](README.md) · [LayerNorm](layer_norm.md) · [Compression](../activation_compression/README.md)

Status: implemented, opt-in. CPU reference and interpreted JIT verification are
available; real CUDA compilation, launch profiles and replay require running
the hardware acceptance suite. See the [usage guide](README.md#compressed-xhat-mean-correction).

## Numerical contract

Cover BatchNorm2d, GroupNorm and LayerNorm, including every backward path that
consumes saved normalized values. Retain the original forward `rstd` explicitly
for those consumers when correction is enabled. Keep BF16/FP8 transport,
first-order backward and existing finite-check/scaling policies.

`rstd` supplies normalization scale and input-gradient scaling. It does not
identify a quantization-induced mean shift. Define the actual correction using
the original pre-affine FP32 xhat and its represented decoded values:

```text
h              = original forward x_hat
h_decoded      = decode(the exact words and FP32 codec metadata being saved)
center_shift[d] = sum_valid(h_decoded - h) / valid_element_count[d]
h_corrected    = h_decoded - center_shift[d]
```

`d` denotes the **complete original normalization domain**:

| Operation | Domain |
| --- | --- |
| BatchNorm2d | One channel across the local batch and spatial dimensions |
| GroupNorm | One sample's channel group and spatial dimensions |
| LayerNorm | One outer position's complete normalized suffix |

This preserves the original xhat first moment, within FP32 reduction accuracy.
It also handles BatchNorm evaluation correctly: normalization by running
statistics can produce xhat with a nonzero mean. Preserve that forward mean
rather than forcing evaluation xhat to zero mean. No new cross-rank centering
collective is introduced; ordinary BatchNorm remains rank-local.

Sum quantization errors before dividing by the domain's valid count. Do not
average codec-group means equally when tail groups have different counts.
Do not multiply decoded xhat by `rstd` again, reconstruct raw input just to
center it, or normalize the corrected values to unit variance.

## Fusion requirement

**Produce correction metadata while quantize-packing xhat in forward; apply it
while decoding xhat in backward.** A separate dense recentering pass is not an
acceptable primary implementation.

### Forward packing

1. Use the same original xhat that the existing forward computes before affine
   weights and bias. Keep forward statistics and output arithmetic unchanged.
2. Reuse the exact existing encoder's words, FP32 minimum/step, rounding and
   binary half-step convention. Decode those words in registers to determine
   represented values; do not estimate their reconstruction from another
   quantization formula or re-read the full packed payload from global memory.
3. Accumulate FP32 `sum(h_decoded - h)` and valid counts while packing. Padding
   never contributes.
4. If a complete normalization domain fits one program, finish and save its
   scalar `center_shift` in that kernel: **zero additional launches**.
5. If a domain spans programs, write bounded partial error sums/counts. Combine
   them using a fixed-order hierarchical reduction. Fuse that finish with an
   existing later reduction only when its domain and dependency structure match;
   otherwise expose the necessary small reduction launch explicitly.

Do not attempt to combine correction finishing with statistics that must finish
before encoding: the correction depends on the final encoded representation.
The final result is one FP32 correction scalar per domain, not a dense xhat
buffer or persistent array of per-element errors.

Preserve the existing codec-independent normalization statistics order. Changing
correction/compression options must not change forward output, FP8 output scales,
packed words or codec metadata. Only backward interpretation changes.

### Backward decoding and explicit rstd loads

Extend the normalization-specific load contract to return corrected xhat and the
original rstd together. A consumer can reuse the loaded rstd for its dx epilogue.
When correction is enabled, each xhat-consuming kernel explicitly loads it:

```text
r = load(saved_rstd[d])
h = decode_saved_xhat(...)
shift = load(center_shift[d])
valid_normalizer = isfinite(r) and r > 0
h_corrected = select(valid_normalizer, h - shift, h)
```

Here rstd validates the normalization domain and supplies dx scaling; `shift`
performs centering. This gives the requested load a real numerical use in
affine-only consumers as well. Checked execution follows the existing invalid
statistics/value diagnostics. Unchecked execution preserves decoded values for
degenerate/invalid normalizers instead of manufacturing a new correction.
Define and test zero/nonfinite rstd behavior explicitly.

Load/reuse domain scalars within each tile where possible. Do not issue an
additional rstd load in final dx if the same value is already available. Physical
memory transactions and cache behavior require GPU measurement; pointer presence
alone does not prove a load survived compiler optimization.

Use **the same corrected xhat everywhere**:

- Weighted backward row/domain reductions such as `sum(u * xhat)`.
- Final native or FP8 dx epilogues.
- Weight-gradient reductions `sum(dy * xhat)`.
- Complete-row fast paths, chunked paths and retiled FP8 gradient paths.

Bias-gradient-only paths need neither xhat nor centering metadata. BatchNorm
evaluation input-only backward continues to use `dy * gamma * rstd` without
xhat. Its weight-gradient path uses the corrected xhat with preserved forward
mean. Apply correction in FP32 before gradient output casting; avoid a full
unpacked/corrected tensor allocation.

## Configuration and saved-state changes

One opt-in boolean under the existing section:

```toml
[runtime.activation_compression]
enabled = true
bits = 2
group_size = 512
norm_xhat_recenter = false
```

Default `false` preserves existing gradients and checkpoint compatibility.
With compression disabled the option has no effect. It applies only to norm
xhat, independently of FP8 enablement, and does not change Linear saved inputs,
attention Q/K/V, activation derivatives, masks, bit widths or group sizes.

Add strict schema validation, template/reference documentation, scope resolution,
operation-plan propagation and prepared-model policy checks. Missing fields in
older configurations normalize to `false`. Treat enabling the correction as a
numerical policy change for checkpoint compatibility; do not silently restore
under a different policy.

Allocate an optional normalization-only `center_shift` side-output buffer in
autograd and pass it to forward packing. Preserve the generic `PackedActivation`
format and existing non-correction backend interfaces through optional arguments.
LayerNorm's normalization state can expose the same buffer explicitly. Save
correction/rstd tensors through autograd's saved-tensor machinery for lifetime
and version checking.

Save original forward rstd whenever corrected xhat will be consumed, including
compressed weight-only LayerNorm backward, which currently can omit it. Do not
read mutable BatchNorm running statistics during backward. Inference, bias-only
backward and empty domains allocate no unnecessary correction state. Report
one additional FP32 scalar per active domain, plus any rstd storage newly needed
by an affine-only path; distinguish retained bytes from reduction scratch.

The one-time planner selects capability/layout/reduction strategy. Correction
values are computed from each forward's actual encoded data, not cached across
training steps. They are transient autograd state, not new running statistics.

## Implementation locations

Update both families rather than only the final input-gradient kernels:

- [BatchNorm/GroupNorm xhat loader](../../src/deepopt/opt/backends/triton/normalization/common.py),
  [backward reductions and native gradients](../../src/deepopt/opt/backends/triton/normalization/backward.py),
  and [FP8 dx epilogue](../../src/deepopt/opt/backends/triton/normalization/gradient.py).
- [LayerNorm xhat loader](../../src/deepopt/opt/backends/triton/normalization/layer/common.py)
  and [all LayerNorm backward kernels](../../src/deepopt/opt/backends/triton/normalization/layer/backward.py).
- [BatchNorm/GroupNorm forward packing](../../src/deepopt/opt/backends/triton/normalization/forward.py)
  and [LayerNorm row/chunked packing](../../src/deepopt/opt/backends/triton/normalization/layer/forward.py),
  with launch/reduction orchestration and CPU reference counterparts.
- [BatchNorm/GroupNorm autograd](../../src/deepopt/opt/nn/autograd/normalization/function.py)
  and [LayerNorm autograd](../../src/deepopt/opt/nn/autograd/normalization/layer.py),
  including selective retention and execution reports.

Use normalization-local helpers composed with the existing codec rather than
adding a second encoder or a separate public compressed-normalization operation.
Keep the uncompressed/default-off path free of correction work. With correction
off, compiler elimination of an unused rstd load in an affine-only kernel is
valid; the explicit load contract above applies to corrected xhat consumers.

## Training integration and reporting

Preserve all BF16/FP8 input/output and gradient combinations, both FP8 formats,
current/blockwise/explicit/delayed scales, native affine gradients and existing
FP8/native boundaries. The correction is distinct from all FP8 scale metadata.

Recomputation through `deepopt.fp8_checkpoint` reproduces the same encoded data
and correction using the scale tape without duplicate amax observations. DDP
keeps its existing gradient synchronization and optional FP8 amax collectives.
Current/fixed-scale CUDA Graph replay updates correction values in device buffers
without host reads, `.item()` or repeated analysis. Existing eager delayed-update
capture restrictions remain. Provide Fake/meta shapes without value access and
preserve existing eager boundaries inside compiled models.

Expose correction enablement, saved correction/rstd bytes, and partial/finish
launch counts. Report shared launches without double-counting them as separate
operations. The small-domain fused path must add no kernel launch; large-domain
costs must appear in complete forward/backward benchmarks.

## Acceptance tests and rollout

1. **First moment:** compare corrected decoded means with original pre-affine
   means over complete domains for all norms and bits 1/2/4/8. Include unequal
   tails, non-power-of-two groups, nonunit rstd and large domains. BatchNorm eval
   must preserve an intentionally nonzero xhat mean.
2. **Numerics:** verify dx/dweight against an independent corrected-codec oracle;
   dbias stays unchanged. Exercise weighted GroupNorm and multidimensional
   LayerNorm suffixes. For positive finite rstd, varying rstd with fixed decoded
   xhat/shift must not change the centering subtraction; it must scale dx.
3. **Consistency:** prove every xhat-consuming reduction/epilogue uses the same
   shift. Cover native/FP8 dy and dx, layouts, row fast paths and partial/retiled
   kernels. Do not combine a corrected final xhat with uncorrected dot products.
4. **Invariance/storage:** forward values, FP8 forward scales and packed codec
   payloads are unchanged when toggling the option. No raw-input/dense-xhat
   shadow appears. Verify affine-only rstd retention, inference/bias-only skips,
   empty cases, and original rstd snapshot/version behavior.
5. **Fusion:** interpret the actual JIT bodies on CPU, then compile/profile on
   CUDA. Assert zero additional launches for complete-domain packing, bounded
   reduction fallback, inline backward subtraction and no dense correction pass.
6. **System behavior:** test configuration defaults/overrides, train/eval,
   accumulation, checkpoint resume/policy mismatch, recomputation, two-rank DDP,
   Fake/meta, compilation and changing-value graph replay.
7. **Measurement:** extend normalization benchmarks with correction off/on;
   include forward correction production, partial reductions and backward loads
   inside timing. Report mean preservation, gradient error, retained/peak bytes,
   launch counts and eager/graph latency. Evaluate convergence separately;
   preserving a first moment does not guarantee lower total gradient error.

Implement schema/oracle and saved-state contracts first, then fused forward
metadata generation, then both backward families, and finally system/hardware
validation. Release the correction opt-in after default-off regressions pass.
