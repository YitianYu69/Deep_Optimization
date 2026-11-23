# Custom operations and saved backward state

`deepopt.register_operation` exposes two ways to extend eager execution and the
strict native compiler. Both use the same forward numerical recipe. Start with
a primitive decomposition when its existing derivatives express your operation.

```python
from deepopt import register_operation
from torch.nn import functional as F

@register_operation(name="blur_pool", decomposition=True)
def blur_pool(x, kernel, stride, padding, channels):
    padded = F.pad(x, padding, mode="reflect")
    return F.conv2d(padded, kernel.to(x.dtype), stride=stride, groups=channels)
```

This is the recipe used by [the VAE](../../examples/vae/mbconv_vae.py). Padding
requires geometry only. With a frozen kernel, convolution's input gradient
requires the kernel and geometry, so no source activation is saved. A trainable
kernel additionally needs the padded, compute-cast source for its weight
gradient, and the existing convolution policy selects its saved representation.
Model kernels and other state must be explicit tensor operands.

For the explicit VJP route, declare persistent model buffers with
`state_operands=("kernel",)`; these operands always stay exact, even if their
save requests permit approximation. `nn.Parameter` operands are recognized
automatically. A native attempt to approximate an undeclared persistent buffer
raises an error, because eager plain tensors cannot identify buffer ownership.

An opaque operation instead declares its own saved state and first-order VJP:

```python
import torch
from deepopt import SavedUse, register_operation


def square_state(ctx, inputs, output):
    if ctx.needs_grad(inputs["x"]):
        ctx.save("source", inputs["x"],
                 use=SavedUse(allow_approximation=True))
        ctx.metadata("scale", inputs["scale"])


def square_vjp(state, grad):
    return {"x": grad * state["source"] * (2 * state["scale"])}


@register_operation(
    name="scaled_square",
    fake=lambda x, scale=1: torch.empty_like(x),
    saved_state=square_state,
    vjp=square_vjp,
    schema="1",
)
def scaled_square(x, scale=1):
    return x.square() * scale
```

The VJP sees only declared state. In eager execution, exact tensors and packed
components cross `save_for_backward`, including saved-tensor hooks and their
ordinary lifetime rules. Native differentiation runs the same state callback
with the existing `BackwardTraceContext`, then traces VJP arithmetic into
inspectable backward IR. It does not retain the eager custom autograd function
as a native derivative owner.

The state context provides:

| Method | Contract |
| --- | --- |
| `ctx.needs_grad(value)` | Forward-time input gradient requirement; never predicts a later `autograd.grad` target list |
| `ctx.meta(value)` | `TensorMeta` containing shape, dtype, device, strides and gradient requirement |
| `ctx.policy` | Immutable forward-time `SavedActivationPolicy` (`enabled`, `bits`, `group_size`) |
| `ctx.save(name, value, use=SavedUse(...))` | Exact save by default; explicitly permitted coefficients use the existing effective compression policy and economic-size check |
| `ctx.metadata(name, value)` | Immutable geometry/options with no tensor payload |
| `ctx.compute(target, *args, meta=..., **kwargs)` | Pure forward auxiliary tensor, with explicit operands and nondifferentiable `TensorMeta` |
| `ctx.mask(name, value)` | Exact boolean branch decisions stored with the existing lossless one-bit mask codec |

A VJP can use ordinary supported symbolic arithmetic. For other native
backward kernels, `state.call(target, *args, meta=..., **kwargs)` executes the
same explicit callable in eager mode and emits a native backward call in
compiled mode. Supply complete single-result metadata and pass every tensor
operand explicitly. Reading a missing residual name raises an error.
`TensorMeta` is also available from the public `deepopt` namespace for typed
auxiliary-state and backward-kernel declarations.

The current explicit interface supports named nonvariadic operands, immutable
non-tensor arguments, a fresh single tensor result, pure forward preparation,
and first-order derivatives. Mutation, output aliases, declared RNG effects, hidden
tensor/module closures and conflicting decomposition/VJP registrations are
rejected. Tensor containers and multiple results need a future explicit schema;
use a decomposition where supported primitives already express them. Callbacks
are trusted Python code and must honor their declared purity and metadata.

The effective recipe is resolved for each eager forward and each guarded native
specialization. Backward can run after compression/autocast scopes exit, and
interleaved invocations retain their own state. Exact saves retain ordinary
version checks; packed coefficients own their payload and do not retain the
original source for version checking. Change `schema` when changing a custom
operation's residual/numerical contract. Referenced registered operations are
included in the model's checkpoint recipe before its first forward; exact
resume rejects a changed schema before loading model or optimizer state.
Discovery follows model methods, numerical helper functions, callable attributes,
and compiler/precision wrappers. For dynamic dispatch that cannot be inspected,
declare the possible decorated callables explicitly:

```python
model.__deepopt_operations__ = (scaled_square,)
```

Keep application-specific data and other code versions in the existing
`experiment_identity` as appropriate. The operation declaration versions its
own derivative and saved-state contract.

CPU BF16 behavior, frozen/trainable BlurPool masks, independent user operations,
lossless auxiliary masks, retained graphs and scope-exit backward are covered by
[operation registration tests](../../tests/compiler/test_operation_registration.py).
CUDA qualification remains a separate requirement.
