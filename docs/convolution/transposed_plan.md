# Transposed convolution support plan

Status: implemented, with CPU/reference and two-rank Gloo acceptance passing.
Real CUDA compilation, execution, and performance acceptance remain pending.
See the [implementation guide](./transposed.md) for usage and limits.

The shared CUDA Graph runtime still rejects delayed scaling; current scaling
has hardware graph acceptance tests. Unbatched output retains the native
logical-layout boundary. Neither limitation is treated as an FP8 speedup claim.

[Convolution documentation](./README.md) · [Configuration](../configuration/README.md)

## Scope and contract

Add `ConvTranspose1d`, `ConvTranspose2d`, and `ConvTranspose3d`, together with
the corresponding functional operations. Cover training and inference, batched
and unbatched inputs, bias, stride, numeric padding, dilation, output padding,
groups, and the module `output_size` argument. Match native validation and
output-size resolution rather than introducing different shape rules.

Preserve parameter identity, parameter names, training state, state-dict
compatibility, optimizer references, and supported memory formats. Native
fallbacks must report their reason. FP8 kernels must not silently substitute
BF16 or TF32 arithmetic for FP8 dot products.

| FP8 | Activation compression | Execution |
| --- | --- | --- |
| Off | Off | Native PyTorch operation |
| Off | On | Native compute; compress the source needed for the weight gradient |
| On | Off | FP8 forward, input-gradient, and weight-gradient operands |
| On | On | FP8 compute; fuse source casting and quantize-pack where feasible |

Accept BF16 and DO scale-carrying FP8 activations on the optimized path. Preserve
existing FP32/FP16 native behavior. Use BF16 as the normal autocast output;
support scale-carrying FP8 output when the cached precision plan selects an
eligible consumer. Backward accepts planned FP8 gradients and can emit FP8 dX
for eligible edges. Keep reductions and accumulators in FP32 where required,
and return parameter gradients in the original parameter dtype.

## 1. Geometry and public integration

- Introduce a dedicated immutable transposed-convolution geometry with input,
  output, native weight shape, rank, stride, padding, dilation, groups, and
  effective output padding. Its shape equation per spatial axis is
  `(input - 1) * stride - 2 * padding + dilation * (kernel - 1)
  + output_padding + 1`.
- Preserve native weight layout `[C_in, C_out / groups, *kernel]`; do not treat
  the weights as ordinary-convolution output-channel-first weights.
- Resolve `output_size` before dispatch, including unbatched forms. Compare
  legal and illegal combinations against native execution, including cases
  where dilation changes admissible output padding.
- Provide `do_conv_transpose1d`, `do_conv_transpose2d`, and
  `do_conv_transpose3d` in the existing public functional module. Match native
  positional arguments and add the existing optional execution-plan mechanism.
- Add module adapters and functional call-site owners. Register them with
  module conversion, FX matching/rewriting/tracing, and autocast interception.
- Extend `intercept_convolution` to include transposed convolutions. Reuse the
  existing independent FP8 and activation-compression configuration; no new
  transposed-only enable flags are needed.
- Include compression-only preparation, exclusions, delayed-scale ownership,
  weight-cache invalidation, checkpoint loading/recomputation, device moves,
  and execution reports.

## 2. Native compute and independent compression

- Build a native reference path for all three ranks before the FP8 executor.
- With compression disabled, retain the native fast path directly.
- With compression enabled, quantize-pack each unique input activation once,
  only when dW is required. Do not save expanded zeros, unfolded windows, or a
  second full input copy alongside the packed state.
- Dequantize-unpack the saved input to the native compute dtype for dW.
  Compression must leave forward, dX, and db unchanged relative to the same
  native path with compression disabled; only dW depends on reconstructed x.
- Use the global compression bit width and default group size 512. Avoid
  allocating saved activation state when only dX or db is requested.

## 3. FP8 kernels and safe reuse

Use the adjoint relationship to the existing ordinary-convolution executor.
For real-valued operands, define a virtual ordinary convolution whose input
shape is the transposed output shape and whose output shape is the transposed
input shape. Its weight tensor has exactly the native transposed weight layout.

| Transposed operation | Ordinary-convolution primitive |
| --- | --- |
| Forward `Y = conv_transpose(X, W)` | Input gradient with `X` as the virtual output gradient |
| dX | Forward convolution of `dY` with `W` |
| dW | Weight gradient with `dY` as virtual input and `X` as virtual output gradient |

Verify the geometry mapping for every supported stride/dilation/output-padding
combination. Reuse address calculation and tiled kernels through explicit
adapters; do not duplicate the full existing convolution backend or rename
activation and gradient scale roles to make a call fit.

Do not assume the ordinary convolution's inferred output shape always equals X.
For legal output-padding cases admitted through dilation, it can contain extra
positions. Carry explicit logical X bounds in the virtual geometry: mask those
extra positions as zero in the adjoint forward/dW, and crop the dX computation
to the original X shape. Add dedicated addressing when existing executor
geometry cannot express those bounds, without allocating a padded X tensor.

- Implement output-owned gather/reduction kernels to avoid an expanded
  zero-inserted input and overlapping output scatter atomics.
- Support tensorwise and all existing groupwise/blockwise FP8 recipes, current
  and delayed scaling, configured activation/weight/gradient dtypes, and
  masked tails for non-aligned channel counts.
- Keep scales attached to the actual operand and its original coordinate
  system. In particular, transposed dW swaps the semantic roles of the two
  ordinary weight-gradient operands; activation and gradient recipes must
  still apply to X and dY respectively.
- Use FP8 dot products with FP32 accumulation for blockwise spatial execution,
  applying scale products at valid reduction boundaries.
- Add bias and supported output conversion in the final compute epilogue.
  Current tensorwise output scaling may require a reduction/finalization pass;
  claim single-kernel fusion only for recipes that actually permit it.
- Reuse scaled GEMM for eligible kernel-size-one, stride-one, zero-padding,
  zero-output-padding, dilation-compatible, ungrouped shapes. Other spatial
  cases use the implicit executor, with the selected backend reported.
- Reduce dW and db safely for overlap and large batches. Use bounded split
  reductions where needed; expose temporary workspace in memory benchmarks.

## 4. Fused FP8 activation compression

- Reuse the existing affine `quantize_pack` / `dequantize_unpack` primitives.
- For BF16 input, produce the forward FP8 source and packed saved state in the
  casting/packing path, observing the existing recipe's required passes.
- For an existing FP8 source, consume its data and scales directly and pack
  that representation without a BF16 round trip or another FP8 quantization.
- In backward, reconstruct the saved activation directly into the FP8 operand
  needed by dW. Fuse unpack/dequantize into operand loads where practical;
  otherwise reconstruct each source once to a bounded reusable workspace.
- Tensorwise FP8 follows the global activation-compression group size.
  Groupwise FP8 uses the exact activation scaling groups/rectangles; groups
  must not cross the scaling boundaries.
- Retain the activation FP8 scale metadata separately from compression
  minimum/step metadata. Do not retain both packed and full-size saved data.

## 5. Cached precision planning and runtime behavior

- Add transposed operations to forward-consumer and backward-capability
  analysis. Include direct functional calls and prepared call-site owners.
- Make train/eval forward-output and backward-input precision decisions during
  preparation only. Do not trace, inspect the next operator, or decide edge
  compatibility inside the training loop.
- Honor existing conservative handling of native boundaries and fan-out.
  Residual addition and SE multiplication remain native boundaries in this
  feature; do not assume branch gradients can accumulate in FP8.
- Resolve and report graph-analysis failures. Module conversion remains useful
  when the root model's Python shape checks prevent tracing, but transposed
  support alone does not fix those checks or promise a fully connected FP8 VAE.
- Keep buffers, delayed scaling updates, cached weights, and workspaces safe
  for CUDA Graph replay. Avoid host scalar reads and allocations that depend on
  tensor values during capture/replay. Shape changes select a new capture.

## 6. Source layout

Use sibling packages matching the current convolution organization:

```text
src/deepopt/opt/
  nn/
    functional.py                       # public do_conv_transpose*d entries
    convolution_transpose/
      geometry.py
      modules.py
      dispatch.py
    autograd/convolution_transpose/
      native.py
      fp8.py
  backends/
    convolution_transpose/               # lazy backend selection/adapters
    native/convolution_transpose/
    reference/convolution_transpose/
    triton/convolution_transpose/        # transposed-specific launches/epilogues
```

Add package exports as required. Reuse existing convolution internals through
narrow helpers when semantics match. Split transposed-specific forward,
backward, and scaling/epilogue code by responsibility as it grows; avoid one
large backend file or a general convolution rewrite.

## 7. Verification and acceptance

1. Native parity for ranks 1/2/3: values and dX/dW/db, groups, depthwise-like
   shapes, bias, stride, dilation, output padding, explicit `output_size`,
   batched/unbatched inputs, supported layouts, empty-batch behavior, and
   invalid-argument errors. Include cases where only a subset needs gradients.
2. FP8 results against an explicit quantize/dequantize reference using exactly
   the same operand scales and recipes. Test tensorwise and blockwise scales,
   non-aligned dimensions, inherited FP8 input, delayed updates, and cache reuse.
3. Compression tests against the same compute path without compression:
   identical forward/dX/db, dW error and error distribution across supported
   bits/group sizes, packed-state byte counts, and absence of a hidden full x.
4. Precision-planner tests for norm/activation/transposed-conv chains, native
   boundaries, train/eval, functional interception, exclusions, failed traces,
   and FP8 backward eligibility. Verify analysis does not repeat during steps.
5. Trainer tests for accumulation, checkpoint restore/recompute, optimizer
   updates, DDP, and CUDA Graph replay across repeated training steps. Preserve
   parameter/state identity and deterministic scale state where applicable.
6. Installed-library benchmarks for eager and CUDA Graph execution: forward,
   dX, dW, complete training step, activation-state bytes, peak memory, launch
   count, and workspace. Include preparation/casting/compression time in the
   relevant phase and report the four configuration combinations separately.
7. Prioritize the supplied VAE's `ConvTranspose2d(k=4, stride=2, padding=1)`
   decoder shapes, then representative grouped 1D and volumetric 3D cases.
   Measure actual VAE decoder/end-to-end throughput without attributing native
   residual, SE, sampling, or loss overhead to the transposed kernels.

Implement in dependency order: geometry/native/compression, one complete 2D
FP8 training slice, 1D/3D and blockwise coverage, planner/runtime integration,
then GPU correctness and performance acceptance. Completion requires all three
ranks and all four configuration combinations. CPU/reference tests cannot
establish CUDA compilation, graph safety, FP8 hardware execution, or speedup.
