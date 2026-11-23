# Multiplication saved-state contract

`deepopt.opt.nn.functional.do_mul` (aliases `mul` and `multiply`) preserves
`torch.mul` forward values, broadcasting, and wrapped or zero-dimensional scalar
promotion. A DeepOpt BF16 compression scope intercepts eligible out-of-place
`*`, `torch.mul`, `torch.multiply`, `Tensor.mul`, and `Tensor.multiply` calls.
Captured multiplication is differentiated by native AutoDiff; the eager custom
autograd function is not a derivative owner inside a native compiled region.

Only the opposite operand required by a requested gradient is saved. Python
scalars use scalar metadata. Small scalar or gate tensors, FP32 coefficients and
model parameters stay exact. An ordinary BF16 activation coefficient may use
the configured affine codec when it contains at least one full codec group and
its encoded payload plus metadata is smaller than its dense state. Expanded
zero-stride views are compared with their addressed storage, so a large logical
view of a small coefficient remains exact when packing would increase state.
Native provenance additionally keeps persistent model buffers and constants
exact. A standalone eager function cannot infer whether an ordinary tensor is
a registered buffer; pass an explicit disabled plan for that coefficient when
the distinction matters.

Each invocation owns its physical payload and exact saved tensors. `x * x`
reuses a single pack within that invocation and records both gradient uses.
Separate views and separate forward calls have separate snapshots. No-grad
and inference calls allocate no backward payload.

The first-order VJP reconstructs each encoded coefficient in its original
operand dtype, multiplies by the output cotangent using PyTorch promotion,
rounds that product to the promoted dtype, and then calls `sum_to_size` before
casting to the gradient target dtype. In particular a BF16 residual-branch
product rounds before reduction into an FP32 scalar parameter gradient. A fused
FP32 product-and-reduction must preserve this boundary before replacing the
materialized route.

Packed state is a forward snapshot. Mutation through an alias after forward
does not invalidate it or change its VJP; the large original is not retained
only to reproduce a native version error. Exact saved coefficients retain
PyTorch version checks. New forwards repack the current source, and saved-tensor
hooks/offload and retained graphs use the normal autograd saved-state lifetime.
Compiled mutations remain subject to native effect checks. Higher-order/JVP
differentiation is outside this first-order contract.

In-place and `out=` calls retain PyTorch dispatch. Integer, complex, FP8,
unsupported tensor subclasses/layouts and unsupported devices use explicit
native eager paths; strict compiled coverage still requires their own supported
native lowering. `multiplication_support` exposes the eager fallback reason.
Use `activation_compression=False` to disable scope interception, or an explicit
native plan with `activation_compression.enabled=False` for the eager functional
route. Quantizer design and training-quality claims are unchanged.

The [eager regressions](../../tests/opt/test_binary_activation_saving.py) compare
VJPs with a reconstructed-coefficient oracle. The [native regressions](../../tests/compiler/test_native_multiplication.py)
compare SE, SwiGLU and residual-scale paths with the same eager recipe and
inspect coefficient bindings, consumer records and derivative ownership.
Their CUDA gate requires BF16 support independently of FP8; a CPU run does not
qualify target-GPU execution, fusion performance or allocator behavior.
