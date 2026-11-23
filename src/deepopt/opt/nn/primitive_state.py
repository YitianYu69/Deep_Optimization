"""Shared numerical residual contracts for clamp and nearest 2D resize.

Clamp retains a lossless predicate snapshot. Nearest resize's adjoint depends
only on geometry, including the original scale factors rather than a ratio
reconstructed from rounded output sizes.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from numbers import Real

import torch
from torch.autograd.function import once_differentiable

from ..quantization.boolean import PackedBooleanMask, pack_boolean_mask, unpack_boolean_mask


def primitive_arguments(function, args, kwargs):
    """Return only the managed, no-out cat/add ABI that we implement."""
    import operator
    from numbers import Real
    from deepopt._bf16_policy import managed_bf16_primitives

    if function in (torch.cat, torch.concat, torch.concatenate):
        if (len(args) > 2 or set(kwargs) - {"tensors", "dim"} or args and "tensors" in kwargs
                or len(args) == 2 and "dim" in kwargs):
            return None
        tensors = args[0] if args else kwargs.get("tensors")
        if not isinstance(tensors, (tuple, list)) or not tensors or any(not isinstance(t, torch.Tensor) for t in tensors):
            return None
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        if type(dim) is not int:
            return None
        operands, kind = tuple(tensors), "cat"
    elif function in (torch.add, operator.add, torch.Tensor.add, torch.Tensor.__add__):
        if (len(args) > 2 or set(kwargs) - {"input", "other", "alpha"} or kwargs.get("alpha", 1) != 1
                or any(name in kwargs for name in ("input", "other")[:len(args)])):
            return None
        operands = tuple(args[i] if len(args) > i else kwargs.get(name) for i, name in enumerate(("input", "other")))
        if any(not isinstance(value, (torch.Tensor, Real)) for value in operands):
            return None
        kind, dim = "add", None
    else:
        return None
    tensors = tuple(t for t in operands if isinstance(t, torch.Tensor))
    if (not tensors or not any(t.dtype == torch.bfloat16 for t in tensors)
            or any(t.dtype not in (torch.bfloat16, torch.float32) or t.layout != torch.strided or t.is_nested for t in tensors)
            or any(t.device != tensors[0].device for t in tensors)
            or not managed_bf16_primitives(tensors[0].device.type)):
        return None
    return kind, operands, dim


def stored_add(input, other, dtype):
    """Promoted addition with a single declared output-storage rounding."""
    from torch._subclasses.fake_tensor import FakeTensor

    tensors = tuple(t for t in (input, other) if isinstance(t, torch.Tensor))
    shape = torch.broadcast_shapes(*(t.shape for t in tensors))
    if isinstance(tensors[0], FakeTensor) or tensors[0].device.type == "meta":
        return torch.empty(shape, dtype=dtype, device=tensors[0].device)
    if tensors[0].device.type == "cuda" and all(t.dtype in (torch.bfloat16, torch.float32) for t in tensors):
        from ..backends.triton.activation.primitive_storage import add

        return add(input, other, dtype)
    return torch.add(input, other).to(dtype)


def destination_add(input, other, alpha=1):
    if alpha != 1:
        raise ValueError("The private residual-add rewrite requires alpha=1")
    return stored_add(input, other, input.dtype)


@dataclass(frozen=True)
class StorageAddState:
    shapes: tuple
    dtypes: tuple
    arithmetic_dtype: torch.dtype
    output_dtype: torch.dtype = torch.bfloat16

    def forward(self, input, other):
        return stored_add(input, other, self.output_dtype)

    def gradient(self, gradient, index):
        shape, dtype = self.shapes[index], self.dtypes[index]
        if shape is None:
            return None
        padding = gradient.ndim - len(shape)
        axes = tuple(range(padding)) + tuple(padding + axis for axis, size in enumerate(shape)
                                            if size == 1 and gradient.shape[padding + axis] != 1)
        # Reduce directly from BF16 loads into the declared accumulator. A
        # broadcast FP32 parameter does not require a full FP32 dOutput copy.
        value = gradient.sum(dim=axes, keepdim=True, dtype=self.arithmetic_dtype).reshape(shape) if axes else gradient
        return value.to(dtype)

    def backward(self, gradient, needs=(True, True)):
        return tuple(self.gradient(gradient, index) if needed else None for index, needed in enumerate(needs))


class _StorageAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other):
        tensors = (input, other)
        dtypes = tuple(t.dtype if isinstance(t, torch.Tensor) else None for t in tensors)
        arithmetic = torch.result_type(input, other)
        ctx.specification = StorageAddState(tuple(tuple(t.shape) if isinstance(t, torch.Tensor) else None for t in tensors),
                                          dtypes, arithmetic)
        return ctx.specification.forward(input, other)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return ctx.specification.backward(gradient, ctx.needs_input_grad[:2])


def model_add(input, other):
    from .._autocast import _suspend_fp8_interception

    with _suspend_fp8_interception():
        return _StorageAddFunction.apply(input, other)


def model_cat(tensors, dim=0):
    from .._autocast import _suspend_fp8_interception

    # Cast each branch before concatenation, so CUDA never materializes the
    # promoted FP32 concatenation. Cast-backward precedes expanded-token sums.
    with _suspend_fp8_interception():
        return torch.cat(tuple(t.to(torch.bfloat16) for t in tensors), dim=dim)


@dataclass(frozen=True)
class StorageExpState:
    """Model-region exp: BF16 stores, exact pre-store FP32 coefficient.

    CPU arithmetic is a semantic reference with full FP32 temporaries. CUDA
    requires a direct-store implementation, rather than claiming those
    reference temporaries satisfy the storage policy's scratch contract.
    """

    shape: tuple[int, ...]
    bf16_policy: object
    bits: int = 2
    group_size: int = 1024
    packed: bool = False

    def forward(self, x):
        from torch._subclasses.fake_tensor import FakeTensor

        if x.device.type == "cuda" and not isinstance(x, FakeTensor):
            from deepopt.kernel_policy import kernel_backend
            from ..backends.triton.activation.storage_exp import forward

            if kernel_backend("activation") != "triton":
                raise ValueError("BF16 storage model exp requires the Triton CUDA route")
            return forward(x, packed=self.packed, bits=self.bits, group_size=self.group_size)
        coefficient = x.float().exp()
        output = coefficient.to(torch.bfloat16)
        if self.packed:
            from ..quantization.activation import quantize_pack_activation

            q = quantize_pack_activation(coefficient.reshape(1, -1), self.group_size, bits=self.bits)
            return output, q.data, q.step, q.minimum
        return output, x.clone(memory_format=torch.contiguous_format)

    def backward(self, gradient, *saved):
        from torch._subclasses.fake_tensor import FakeTensor

        if gradient.device.type == "cuda" and not isinstance(gradient, FakeTensor):
            from ..backends.triton.activation.storage_exp import backward

            return backward(gradient, *saved, shape=self.shape, packed=self.packed,
                            bits=self.bits, group_size=self.group_size)
        if self.packed:
            from ..quantization.activation import PackedActivation, dequantize_unpack_activation
            from math import prod

            coefficient = dequantize_unpack_activation(PackedActivation(
                *saved, (1, prod(self.shape)), self.group_size, bits=self.bits), dtype=torch.float32).reshape(self.shape)
        else:
            coefficient = saved[0].float().exp()
        return (gradient.float() * coefficient).to(torch.bfloat16)


class _StorageExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, specification):
        values = specification.forward(x)
        ctx.specification = specification
        ctx.save_for_backward(*values[1:])
        return values[0]

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return ctx.specification.backward(gradient, *ctx.saved_tensors), None


def model_exp(input):
    """Storage-policy model exp; explicit FP32 loss operands retain native exp."""
    from deepopt._bf16_policy import current_bf16_policy
    from .._autocast import _suspend_fp8_interception, _get_activation_compression

    policy = current_bf16_policy(input.device.type)
    if not policy.storage_enabled or input.dtype != torch.bfloat16:
        with _suspend_fp8_interception():
            return torch.exp(input)
    with _suspend_fp8_interception(), torch.autocast(input.device.type, enabled=False):
        compression = _get_activation_compression(input.device.type) or {}
        state = StorageExpState(tuple(input.shape), policy, compression.get("bits", 2),
                                compression.get("group_size", 1024), bool(compression.get("enabled", False)) and input.numel() > 0)
        return _StorageExpFunction.apply(input, state)


@lru_cache(maxsize=3)
def clamp_endpoint_policy(kind="clamp"):
    """Resolve the installed native scalar-clamp endpoint convention once.

    PyTorch releases differ on the subgradient at a bound. This tiny native CPU
    capability probe runs while resolving the operation, never on model data or
    in its backward. Only immutable booleans enter the cache. Local hooks and
    dispatch suppression keep ambient diagnostics and fake tracing out of it.
    """
    with torch._C.DisableTorchFunction(), torch._C._DisableTorchDispatch(), \
            torch.inference_mode(False), torch.enable_grad(), \
            torch.autograd.graph.saved_tensors_hooks(lambda value: value, lambda value: value):
        x = torch.tensor([0., 1.], device="cpu", requires_grad=True)
        output = (x.clamp(0., 1.) if kind == "clamp" else x.clamp_min(0.)
                  if kind == "clamp_min" else x.clamp_max(1.))
        output.backward(torch.ones_like(output))
        return bool(x.grad[0].item()), bool(x.grad[1].item())


@dataclass(frozen=True)
class ScalarClampState:
    shape: tuple[int, ...]
    dtype: torch.dtype
    minimum: int | float | None
    maximum: int | float | None
    include_minimum: bool
    include_maximum: bool
    memory_format: torch.memory_format = torch.contiguous_format

    def forward(self, x):
        output = torch.clamp(x, self.minimum, self.maximum)
        predicate = torch.ones_like(x, dtype=torch.bool)
        if self.minimum is not None:
            predicate &= x >= self.minimum if self.include_minimum else x > self.minimum
        if self.maximum is not None:
            predicate &= x <= self.maximum if self.include_maximum else x < self.maximum
        return output, pack_boolean_mask(predicate).data

    def backward(self, gradient, data):
        predicate = unpack_boolean_mask(PackedBooleanMask(data, self.shape))
        # Multiplication would turn a masked NaN/Inf cotangent into NaN.
        result = torch.where(predicate, gradient, 0).to(self.dtype)
        return result.contiguous(memory_format=self.memory_format)


def scalar_clamp_state(x, minimum=None, maximum=None, *, kind="clamp"):
    if (minimum is None and maximum is None or any(
            value is not None and not isinstance(value, Real) for value in (minimum, maximum))):
        raise ValueError("The scalar clamp residual contract requires at least one scalar bound")
    include_minimum, include_maximum = clamp_endpoint_policy(kind)
    memory_format = (torch.channels_last if x.ndim == 4 and x.is_contiguous(memory_format=torch.channels_last)
                     else torch.channels_last_3d if x.ndim == 5 and x.is_contiguous(memory_format=torch.channels_last_3d)
                     else torch.contiguous_format)
    return ScalarClampState(tuple(x.shape), x.dtype, minimum, maximum,
                            include_minimum, include_maximum, memory_format)


class _ScalarClampFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, specification):
        output, mask = specification.forward(x)
        ctx.specification = specification
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return ctx.specification.backward(gradient, ctx.saved_tensors[0]), None


def scalar_clamp(x, minimum=None, maximum=None, *, kind="clamp"):
    """Eager scalar clamp with native values and an exact packed branch mask."""
    from .._autocast import _suspend_fp8_interception

    with _suspend_fp8_interception():
        if (not torch.is_grad_enabled() or not x.requires_grad
                or any(value is not None and not isinstance(value, Real) for value in (minimum, maximum))):
            return torch.clamp(x, minimum, maximum)
        return _ScalarClampFunction.apply(x, scalar_clamp_state(x, minimum, maximum, kind=kind))


@dataclass(frozen=True)
class Nearest2dPrecision:
    """Forward-selected storage contract, also used by fake inference.

    CUDA autocast promotes native nearest resize to FP32. Storage-mode BF16
    operands instead enter the native BF16 kernel directly; a cast after the
    resize would retain the activation-sized FP32 allocation.
    """

    bf16_policy: object
    disable_cuda_autocast: bool

    def call(self, function, *args, **kwargs):
        scope = torch.autocast("cuda", enabled=False) if self.disable_cuda_autocast else nullcontext()
        with scope:
            return function(*args, **kwargs)


def nearest2d_precision(dtype, device, policy=None):
    from deepopt._bf16_policy import current_bf16_policy, resolve_bf16_policy

    device_type = torch.device(device).type
    policy = current_bf16_policy(device_type) if policy is None else resolve_bf16_policy(policy)
    return Nearest2dPrecision(policy, device_type == "cuda" and dtype == torch.bfloat16 and policy.storage_enabled)


@dataclass(frozen=True)
class Nearest2dGeometry:
    input_shape: tuple[int, ...]
    output_size: tuple[int, int]
    scales: tuple[float | None, float | None]
    dtype: torch.dtype
    precision: Nearest2dPrecision
    memory_format: torch.memory_format = torch.contiguous_format
    recompute_scale_factor: bool | None = None
    input_stride: tuple[int, ...] | None = None

    def forward(self, input):
        # Keep explicit noninteger scales: rounded sizes cannot reconstruct
        # the original nearest index mapping.
        return self.precision.call(torch.ops.aten.upsample_nearest2d.default,
                                   input, self.output_size, *self.scales)

    def backward(self, gradient):
        # The native CallNode also calls this method directly. Its derivative
        # domain must survive a caller entering a new autocast scope.
        with torch.autocast(gradient.device.type, enabled=False):
            result = torch.ops.aten.upsample_nearest2d_backward.default(
                gradient, self.output_size, self.input_shape, *self.scales)
            return result.to(self.dtype).contiguous(memory_format=self.memory_format)


class _Nearest2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, function, args, kwargs, scales, recompute_scale_factor, precision):
        from torch._prims_common import suggest_memory_format

        output = precision.call(function, x, *args, **kwargs)
        ctx.geometry = Nearest2dGeometry(tuple(x.shape), tuple(output.shape[-2:]), scales,
                                         x.dtype, precision, suggest_memory_format(x), recompute_scale_factor,
                                         tuple(x.stride()))
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        from .._autocast import _suspend_fp8_interception

        with _suspend_fp8_interception(), torch.autocast(gradient.device.type, enabled=False):
            return ctx.geometry.backward(gradient), None, None, None, None, None, None


def nearest2d(input, size=None, scale_factor=None, mode="nearest", align_corners=None,
              recompute_scale_factor=None, antialias=False):
    """Eager nearest resize with native forward validation and metadata-only AD."""
    return nearest2d_call(torch.nn.functional.interpolate,
        (input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias), {})


NEAREST2D_FUNCTIONS = (torch.nn.functional.interpolate, torch._C._nn.upsample_nearest2d,
    torch.ops.aten.upsample_nearest2d.default, torch.ops.aten.upsample_nearest2d.vec)


def nearest2d_arguments(function, args, kwargs):
    """Recognize supported no-out nearest ABIs without changing validation."""
    if function not in NEAREST2D_FUNCTIONS:
        return None
    if function is torch.nn.functional.interpolate:
        names = ("input", "size", "scale_factor", "mode", "align_corners", "recompute_scale_factor", "antialias")
    elif function is torch.ops.aten.upsample_nearest2d.default:
        names = ("self", "output_size", "scales_h", "scales_w")
    else:
        names = ("input", "output_size", "scale_factors")
    if (len(args) > len(names) or set(kwargs) - set(names)
            or any(name in kwargs for name in names[:len(args)])):
        return None
    values = {name: args[index] if index < len(args) else kwargs.get(name)
              for index, name in enumerate(names)}
    x = values[names[0]]
    if (not isinstance(x, torch.Tensor) or x.device.type not in ("cpu", "cuda")
            or x.is_nested or x.layout != torch.strided or x.ndim != 4
            or x.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64)):
        return None
    recompute = values.get("recompute_scale_factor")
    if function is torch.nn.functional.interpolate:
        mode = args[3] if len(args) > 3 else kwargs.get("mode", "nearest")
        if mode != "nearest":
            return None
        factors = values["scale_factor"]
    elif function is torch.ops.aten.upsample_nearest2d.default:
        factors = (values["scales_h"], values["scales_w"])
    else:
        factors = values["scale_factors"]
    if recompute or factors is None:
        scales = (None, None)
    else:
        factors = (factors, factors) if isinstance(factors, Real) else factors
        if (not isinstance(factors, (tuple, list)) or len(factors) != 2
                or any(value is not None and not isinstance(value, Real) for value in factors)):
            return None
        scales = tuple(None if value is None else float(value) for value in factors)
    keywords = {name: value for name, value in kwargs.items() if name != names[0]}
    return x, args[1:], keywords, scales, recompute


def nearest2d_call(function, args, kwargs):
    """Use one frozen precision wrapper for functional, module and ATen calls."""
    from torch._subclasses.fake_tensor import FakeTensor
    from .._autocast import _suspend_fp8_interception

    with _suspend_fp8_interception():
        invocation = nearest2d_arguments(function, args, kwargs)
        if invocation is None:
            return function(*args, **kwargs)
        x, positional, keywords, scales, recompute = invocation
        precision = nearest2d_precision(x.dtype, x.device)
        if isinstance(x, FakeTensor) or not torch.is_grad_enabled() or not x.requires_grad:
            return precision.call(function, x, *positional, **keywords)
        return _Nearest2dFunction.apply(x, function, positional, keywords, scales, recompute, precision)
