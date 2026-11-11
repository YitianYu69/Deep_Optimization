"""Functional projections with plan-selected precision and saved activations."""

import torch
from contextlib import nullcontext
from functools import lru_cache
from torch.nn import functional as F

from ..precision.gradient import gradient_entry

from .._autocast import (_get_activation_compression, _get_bf16_compression,
                        _get_fp8_plan, _linear_arguments, _suspend_fp8_interception)
from deepopt.opt.scaling.amax import AmaxScaling
from deepopt.opt.nn.autograd.linear import _DOLinearFunction
from deepopt.config import activation_compression_config_from_dict, delayed_scaling_config_from_dict
from dataclasses import asdict

_FLOAT_DTYPES = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
)

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)

_OPERANDS = (
    "activation",
    "weight",
    "grad"
)


class _FrozenPlan(dict):
    """Owned immutable dictionaries retain the public dict/equality contract."""

    def __init__(self, values):
        dict.__init__(self, ((name, _FrozenPlan(value) if isinstance(value, dict) else value)
                            for name, value in values.items()))

    def _immutable(self, *args, **kwargs):
        raise TypeError("Validated linear plans are immutable; validate a new plan to change policy")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (dict(self),)


class _ValidatedFP8Plan(_FrozenPlan):
    def __init__(self, values):
        super().__init__(values)
        training = self["fp8_training"]
        self.delayed_signature = (tuple(training["fp8_dtypes"][role] for role in _OPERANDS),
                                  tuple(training["delayed_scaling"][name] for name in
                                        ("history_length", "amax_algorithm", "margin", "reduce_amax")))


class _ValidatedNativePlan(_FrozenPlan):
    """A native projection and its independent saved-activation policy."""


@torch.compiler.disable
@gradient_entry("relu")
def do_relu(input, inplace=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
            _scaling_state=None, _state_training=True, _planned=False, _execution_info=None):
    """ReLU with fused precision conversion and an exact bool/one-bit backward mask.

    FP8 output uses scale-aware storage. Explicit output_scale is a positive
    FP32 quantization multiplier on the selected matrix scale grid.
    """
    from .activation.dispatch import relu as dispatch_relu

    with _suspend_fp8_interception():
        return dispatch_relu(input, inplace, plan=plan, output_dtype=output_dtype,
                             output_layout=output_layout, output_scale=output_scale,
                             scaling_state=_scaling_state, state_training=_state_training,
                             planned=_planned, execution_info=_execution_info)


relu = do_relu


@torch.compiler.disable
def _activation(kind, input, inplace=False, *, approximate="none", plan=None, output_dtype=None,
                output_layout=None, output_scale=None, _scaling_state=None, _state_training=True,
                _planned=False, _execution_info=None):
    from .activation.dispatch import relu as dispatch

    if not _planned:
        from ..precision.planner import functional_normalization, normalization_output

        adapter = functional_normalization(kind)
        if adapter is not None:
            plan = adapter.plan if plan is None else plan
            output_layout = normalization_output(adapter) if output_layout is None else output_layout
            _scaling_state = adapter.fp8_scaling if _scaling_state is None else _scaling_state
            _state_training, _planned = adapter.training, True
            if _execution_info is None:
                adapter.last_execution = {}
                _execution_info = adapter.last_execution

    with _suspend_fp8_interception():
        return dispatch(input, inplace, plan=plan, output_dtype=output_dtype, output_layout=output_layout,
                        output_scale=output_scale, scaling_state=_scaling_state, state_training=_state_training,
                        planned=_planned, execution_info=_execution_info, _kind=kind, approximate=approximate)


@torch.compiler.disable
@gradient_entry("sigmoid")
def do_sigmoid(input, **options):
    """Sigmoid with fused casts and optional quantize-packed local derivatives."""
    return _activation("sigmoid", input, **options)


@torch.compiler.disable
@gradient_entry("silu")
def do_silu(input, inplace=False, **options):
    """SiLU with preprocessed derivative saving and planned FP8 gradients."""
    return _activation("silu", input, inplace, **options)


@torch.compiler.disable
@gradient_entry("gelu")
def do_gelu(input, approximate="none", **options):
    """Exact or tanh GELU with fused derivative quantize-pack saving."""
    return _activation("gelu", input, approximate=approximate, **options)


@torch.compiler.disable
@gradient_entry("tanh")
def do_tanh(input, **options):
    """Tanh with pre-output-cast derivative saving and FP8 transport."""
    return _activation("tanh", input, **options)


@torch.compiler.disable
@gradient_entry("relu6")
def do_relu6(input, inplace=False, **options):
    """Clamp to [0, 6] with an exact optional one-bit backward mask."""
    return _activation("relu6", input, inplace, **options)


sigmoid, silu, gelu, tanh, relu6 = do_sigmoid, do_silu, do_gelu, do_tanh, do_relu6


def validate_linear_plan(plan: dict) -> dict:
    """Validate a native plan or an existing FP8 operation plan.

    Native plans explicitly select ``fp8=False``. Native AMP determines their
    compute dtype; activation compression changes only the saved input for dW.
    """
    if not isinstance(plan, dict):
        raise TypeError("plan must be a dictionary")
    if "fp8" not in plan:
        return validate_fp8_plan(plan)
    if plan["fp8"] is not False:
        raise ValueError("Native linear plans require fp8=False; use linear_plan() to build an FP8 plan")
    if set(plan) - {"fp8", "activation_compression"}:
        raise ValueError("Unknown fields in native linear plan")
    compression = activation_compression_config_from_dict(plan.get("activation_compression", {}))
    return {"fp8": False, "activation_compression": asdict(compression)}


def _prepare_linear_plan(plan):
    if isinstance(plan, (_ValidatedNativePlan, _ValidatedFP8Plan)):
        return plan
    if isinstance(plan, dict) and "fp8" in plan:
        return _ValidatedNativePlan(validate_linear_plan(plan))
    return _prepare_fp8_plan(plan)


@lru_cache(maxsize=32, typed=True)
def _native_compression_plan(group_size, bits=2, attention_bits=None, norm_xhat_recenter=False):
    """Reuse scoped native policies without resolving configuration per call."""
    return _prepare_linear_plan({"fp8": False, "activation_compression": {
        "enabled": True, "group_size": group_size, "bits": bits, "attention_bits": attention_bits,
        "norm_xhat_recenter": norm_xhat_recenter,
    }})


def _prepare_fp8_plan(plan):
    """Keep the public mutable-plan API; freeze only an internal execution copy."""
    if isinstance(plan, _ValidatedFP8Plan):
        return plan
    return _ValidatedFP8Plan(validate_fp8_plan(plan))


def validate_fp8_plan(plan: dict) -> dict:
    """
    Return a validated, independently owned mutable plan with explicit defaults.
    """

    if not isinstance(plan, dict):
        raise TypeError("plan must be a dictionary")
    if set(plan) != {"fp8_training", "block_shapes"}:
        raise ValueError("plan must contain exactly 'fp8_training' and 'block_shapes'")

    training = plan["fp8_training"]
    block_shapes = plan["block_shapes"]

    if not isinstance(training, dict):
        raise TypeError("fp8_training must be a dictionary")

    allowed_fields = {
        "fp8_dtypes",
        "backend",
        "num_stages",
        "out_dtype", "scaling_recipe", "delayed_scaling", "backward_backend", "activation_compression", "backward_precision"
    }

    if "fp8_dtypes" not in training:
        raise ValueError("fp8_training requires 'fp8_dtypes'")
    if "max_num_imprecise_acc" in training:
        raise ValueError("max_num_imprecise_acc was removed; create a new plan using native Triton accumulation")
    if set(training) - allowed_fields:
        raise ValueError("Unknown fields in fp8_training")

    fp8_dtypes = training["fp8_dtypes"]
    if not isinstance(fp8_dtypes, dict):
        raise TypeError("fp8_dtypes must be a dictionary")
    if set(fp8_dtypes) != set(_OPERANDS):
        raise ValueError(
            "fp8_dtypes must define activation, weight, and grad"
        )

    for operand in _OPERANDS:
        dtype = fp8_dtypes[operand]

        if (not isinstance(dtype, torch.dtype) or dtype not in _FP8_DTYPES):
            raise ValueError(
                f"Unsupported FP8 dtype for {operand!r}"
            )

    if not isinstance(block_shapes, dict):
        raise TypeError("block_shapes must be a dictionary")
    if set(block_shapes) != set(_OPERANDS):
        raise ValueError(
            "block_shapes must define activation, weight, and grad"
        )

    for operand in _OPERANDS:
        AmaxScaling(block_shape=block_shapes[operand])

    backend = training.get("backend", "reference")
    if backend not in ("reference", "triton"):
        raise ValueError(
            "backend must be 'reference' or 'triton'"
        )

    num_stages = training.get("num_stages", "auto")

    if num_stages != "auto" and (type(num_stages) is not int or num_stages not in (1, 2, 3, 4, 5)):
        raise ValueError("num_stages must be 'auto', 1, 2, 3, 4, or 5")

    out_dtype = training.get("out_dtype")
    recipe = training.get("scaling_recipe", "current")
    if recipe not in ("current", "delayed"):
        raise ValueError("scaling_recipe must be current or delayed")
    delayed = delayed_scaling_config_from_dict(training.get("delayed_scaling", {}))
    compression = activation_compression_config_from_dict(training.get("activation_compression", {}))
    if recipe == "delayed" and any(shape is not None for shape in block_shapes.values()):
        raise ValueError("Delayed scaling requires tensorwise block_shapes")
    backward_backend = training.get("backward_backend", "inherit")
    backward_precision = training.get("backward_precision", "native")
    if backward_precision not in ("native", "auto"):
        raise ValueError("backward_precision must be native or auto")
    if backward_backend not in ("inherit", "native_fast", "native_precise"):
        raise ValueError("backward_backend must be inherit, native_fast, or native_precise")
    if backward_backend != "inherit" and (backend != "triton" or any(shape is not None for shape in block_shapes.values())):
        raise ValueError("Native FP8 backward requires backend=triton and tensorwise block_shapes")
    if out_dtype is not None:
        if (not isinstance(out_dtype, torch.dtype) or out_dtype not in _FLOAT_DTYPES):
            raise ValueError(
                "out_dtype must be FP32, BF16, FP16, or None"
            )

    return {
        "fp8_training": {
            "fp8_dtypes": dict(fp8_dtypes),
            "backend": backend,
            "num_stages": num_stages,
            "out_dtype": out_dtype,
            "scaling_recipe": recipe,
            "delayed_scaling": asdict(delayed),
            "backward_backend": backward_backend,
            "backward_precision": backward_precision,
            "activation_compression": asdict(compression),
        },
        "block_shapes": dict(block_shapes),
    }




@torch.compiler.disable
@gradient_entry("linear")
def do_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
    *,
    plan: dict | None = None, _weight_cache=None, _scaling_state=None, _state_training=True,
) -> torch.Tensor:
    """Apply linear with native or FP8 compute and optional input compression.

    ``linear_plan`` resolves the independent FP8 and activation-compression
    configuration sections into a reusable plan. Explicit native plans own
    their policy; prepared FP8 plans retain scoped AMP/FP8 overrides. Without
    a plan, an active DeepOpt scope selects the policy, otherwise PyTorch owns
    the operation. This boundary remains eager inside compiled models.
    """

    with _suspend_fp8_interception():
        return _do_linear(x, weight, bias, plan=plan, weight_cache=_weight_cache,
                          scaling_state=_scaling_state, state_training=_state_training)


def _do_linear(x, weight, bias, *, plan, weight_cache, scaling_state=None, state_training=True):
    from ..quantization.activation_tensor import FP8Activation, dequantize_activation

    if isinstance(x, FP8Activation) and x._activation_layout != "last" and x.ndim != 2:
        x = dequantize_activation(x)
    # Explicit native plans also provide the compatibility wrapper's policy
    # ownership inside an outer FP8 scope. There is one numerical entry point.
    if isinstance(plan, dict) and "fp8" in plan:
        native = _prepare_linear_plan(plan)
        return _native_linear(x, weight, bias, native["activation_compression"])
    active_plan = _get_fp8_plan(x.device.type) if isinstance(x, torch.Tensor) else None
    compression = _get_activation_compression(x.device.type) if isinstance(x, torch.Tensor) else None
    if active_plan is False:
        arguments = _linear_arguments((x, weight, bias), {})
        compression = _get_bf16_compression(x) if arguments is not None else None
        return _native_linear(x, weight, bias, compression)
    if active_plan is not None:
        if compression is None:
            # An FP8 scope can override numerical projection settings while
            # leaving a prepared adapter's independent saved-input policy
            # untouched. Ordinary intercepted F.linear passes the active plan
            # itself, so its default compression remains disabled.
            training = plan.get("fp8_training") if isinstance(plan, dict) else None
            compression = training.get("activation_compression") if isinstance(training, dict) else None
        plan = active_plan
    elif plan is None:
        arguments = _linear_arguments((x, weight, bias), {})
        compression = _get_bf16_compression(x) if arguments is not None else None
        return _native_linear(x, weight, bias, compression)
    checked_plan = _prepare_fp8_plan(plan)
    if compression is not None and checked_plan["fp8_training"]["activation_compression"] != compression:
        checked_plan = _prepare_fp8_plan({
            "fp8_training": {**checked_plan["fp8_training"], "activation_compression": compression},
            "block_shapes": dict(checked_plan["block_shapes"]),
        })
    training = checked_plan["fp8_training"]

    tensors = {
        "input" : x,
        "weight" : weight
    }

    if bias is not None:
        tensors["bias"] = bias

    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided:
            raise TypeError(f"{name} must have strided layout")
        if tensor.dtype not in _FLOAT_DTYPES:
            raise TypeError(f"{name} must have FP32, BF16, or FP16")

    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("FP8 linear supports CPU and CUDA")

    for name, tensor in tensors.items():
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on the same device as input")
    if x.ndim < 1:
        raise ValueError("Input must have at least one dimension")
    if weight.ndim != 2:
        raise ValueError("Weight must have shape [out_features, in_features]")
    if x.numel() == 0 or weight.numel() == 0:
        raise ValueError("FP8 linear currently requires non-empty input and weight")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError("Input's last dimension must match weights.shape[1]")
    if bias is not None:
        if tuple(bias.shape) != (weight.shape[0],):
            raise ValueError("Bias must have shape [out_features]")
    if training["backend"] == "triton" and x.device.type != "cuda":
        raise ValueError("The Triton backend require CUDA")

    out_dtype = training["out_dtype"]
    if out_dtype is None:
        if torch.is_autocast_enabled(x.device.type):
            out_dtype = torch.get_autocast_dtype(x.device.type)
        else:
            out_dtype = x.dtype

    if training["scaling_recipe"] == "delayed":
        if scaling_state is None:
            raise ValueError("Delayed scaling requires a prepared adapter or registered FX call site; bare F.linear is stateless")
        if scaling_state.recipe_signature != checked_plan.delayed_signature:
            raise ValueError("Delayed-scaling state does not match the active recipe; prepare a matching adapter")
    else:
        scaling_state = None
    scope = nullcontext()
    if scaling_state is not None:
        from deepopt.opt.scaling.delayed import fp8_scaling_scope

        scope = fp8_scaling_scope()
    with scope:
        return _DOLinearFunction.apply(
            x, weight, bias,
            checked_plan, out_dtype,
            torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors.values()),
            weight_cache, scaling_state, state_training and torch.is_grad_enabled(),
        )


def _native_linear(x, weight, bias, compression):
    from ..quantization.activation_tensor import dequantize_activation

    x = dequantize_activation(x)
    if compression is None or not compression["enabled"]:
        return F.linear(x, weight, bias)
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("x and weight must be tensors")
    if x.ndim < 1 or weight.ndim != 2:
        raise ValueError("x must have at least one dimension and weight must be 2D")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError("input's last dimension must match weight's input features")
    if bias is not None and (not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (weight.shape[0],)):
        raise ValueError("bias must have shape (out_features,)")
    if x.device != weight.device or (bias is not None and bias.device != x.device):
        raise ValueError("x, weight, and bias must share a device")
    if not x.is_floating_point() or not weight.is_floating_point():
        raise TypeError("x and weight must have floating point dtypes")
    # Native autograd is both exact and cheaper when no compressed dW is used.
    if (x.numel() == 0 or weight.shape[0] == 0 or not torch.is_grad_enabled()
            or not weight.requires_grad):
        return F.linear(x, weight, bias)
    from .autograd.native_linear import _NativeLinearFunction

    return _NativeLinearFunction.apply(x, weight, bias, compression["group_size"], True,
                                       compression.get("bits", 2))


def compressed_linear(x, weight, bias=None, *, group_size=512):
    """Compatibility spelling; new code supplies a plan to ``do_linear``."""
    return do_linear(x, weight, bias, plan=_native_compression_plan(group_size))


def _attention_compression(query, key, value, *, plan=None, attn_mask=None):
    """Resolve the independent Q/K/V policy without changing projection plans."""
    if not torch.is_grad_enabled() or not all(isinstance(t, torch.Tensor) for t in (query, key, value)):
        return nullcontext()
    requires_backward = (any(t.requires_grad for t in (query, key, value))
                         or isinstance(attn_mask, torch.Tensor) and attn_mask.requires_grad)
    if (not requires_backward
            or any(t.is_nested or t.layout != torch.strided or t.dtype not in _FLOAT_DTYPES
                   for t in (query, key, value))):
        return nullcontext()
    compression = _get_activation_compression(query.device.type)
    if isinstance(plan, dict) and "fp8" in plan:
        compression = _prepare_linear_plan(plan)["activation_compression"]
    elif compression is None:
        selected = plan if plan is not None else _get_fp8_plan(query.device.type)
        if selected is not None and selected is not False:
            compression = _prepare_fp8_plan(selected)["fp8_training"]["activation_compression"]
    if compression is None or not compression["enabled"]:
        return nullcontext()
    bits = compression.get("attention_bits")
    if bits is None:
        bits = max(4, compression.get("bits", 2))
    from ..quantization.attention import AttentionCompression

    return AttentionCompression(query, key, value, group_size=compression["group_size"], bits=bits,
                                attn_mask=attn_mask)


@torch.compiler.disable
def do_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                    is_causal=False, *, scale=None, enable_gqa=False, plan=None):
    """Keep native SDPA forward/backward, optionally packing its saved Q/K/V.

    Q/K/V default to at least four bits; an explicit attention_bits overrides
    that floor. Probabilities, masks, outputs and kernel statistics stay exact.
    """
    kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    if scale is not None:
        kwargs["scale"] = scale
    if enable_gqa is not False:
        kwargs["enable_gqa"] = enable_gqa
    with _suspend_fp8_interception(), _attention_compression(query, key, value, plan=plan, attn_mask=attn_mask):
        return F.scaled_dot_product_attention(query, key, value, **kwargs)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, *, scale=None, enable_gqa=False, plan=None):
    """PyTorch-style functional spelling for plan-aware native SDPA."""
    return do_scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal,
                                           scale=scale, enable_gqa=enable_gqa, plan=plan)


def linear(input, weight, bias=None, *, plan=None):
    """PyTorch-style functional spelling, including its ``input`` keyword."""
    return do_linear(input, weight, bias, plan=plan)

@torch.compiler.disable
@gradient_entry("conv1d")
def do_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *,
              plan=None, _weight_cache=None, _scaling_state=None, _state_training=True, _execution_info=None):
    """Plan-aware spatial Conv1d with native or FP8 compute and optional compression."""
    from .convolution.dispatch import convolution

    with _suspend_fp8_interception():
        return convolution(1, input, weight, bias, stride, padding, dilation, groups, plan=plan,
                           weight_cache=_weight_cache, scaling_state=_scaling_state, state_training=_state_training,
                           execution_info=_execution_info)


@torch.compiler.disable
@gradient_entry("conv2d")
def do_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *,
              plan=None, _weight_cache=None, _scaling_state=None, _state_training=True, _execution_info=None):
    """Plan-aware spatial Conv2d with native or FP8 compute and optional compression."""
    from .convolution.dispatch import convolution

    with _suspend_fp8_interception():
        return convolution(2, input, weight, bias, stride, padding, dilation, groups, plan=plan,
                           weight_cache=_weight_cache, scaling_state=_scaling_state, state_training=_state_training,
                           execution_info=_execution_info)


@torch.compiler.disable
@gradient_entry("conv3d")
def do_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *,
              plan=None, _weight_cache=None, _scaling_state=None, _state_training=True, _execution_info=None):
    """Plan-aware spatial Conv3d with native or FP8 compute and optional compression."""
    from .convolution.dispatch import convolution

    with _suspend_fp8_interception():
        return convolution(3, input, weight, bias, stride, padding, dilation, groups, plan=plan,
                           weight_cache=_weight_cache, scaling_state=_scaling_state, state_training=_state_training,
                           execution_info=_execution_info)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, plan=None):
    return do_conv1d(input, weight, bias, stride, padding, dilation, groups, plan=plan)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, plan=None):
    return do_conv2d(input, weight, bias, stride, padding, dilation, groups, plan=plan)


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, plan=None):
    return do_conv3d(input, weight, bias, stride, padding, dilation, groups, plan=plan)



@torch.compiler.disable
@gradient_entry("conv_transpose1d")
def do_conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, *,
                         plan=None, _weight_cache=None, _scaling_state=None, _state_training=True,
                         _execution_info=None, _output_slot=None):
    """Transposed Conv1d with independent FP8 compute and activation compression."""
    if _weight_cache is None:
        from ..precision.planner import functional_normalization

        owner = functional_normalization("conv_transpose1d")
        if owner is not None:
            return owner(input, weight, bias, stride, padding, output_padding, groups, dilation)
    from .convolution_transpose.dispatch import convolution_transpose

    with _suspend_fp8_interception():
        return convolution_transpose(1, input, weight, bias, stride, padding, output_padding, groups, dilation,
                                     plan=plan, weight_cache=_weight_cache, scaling_state=_scaling_state,
                                     state_training=_state_training, execution_info=_execution_info,
                                     output_slot=_output_slot)


def conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, *, plan=None):
    return do_conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation, plan=plan)



@torch.compiler.disable
@gradient_entry("conv_transpose2d")
def do_conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, *,
                         plan=None, _weight_cache=None, _scaling_state=None, _state_training=True,
                         _execution_info=None, _output_slot=None):
    """Transposed Conv2d with independent FP8 compute and activation compression."""
    if _weight_cache is None:
        from ..precision.planner import functional_normalization

        owner = functional_normalization("conv_transpose2d")
        if owner is not None:
            return owner(input, weight, bias, stride, padding, output_padding, groups, dilation)
    from .convolution_transpose.dispatch import convolution_transpose

    with _suspend_fp8_interception():
        return convolution_transpose(2, input, weight, bias, stride, padding, output_padding, groups, dilation,
                                     plan=plan, weight_cache=_weight_cache, scaling_state=_scaling_state,
                                     state_training=_state_training, execution_info=_execution_info,
                                     output_slot=_output_slot)


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, *, plan=None):
    return do_conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation, plan=plan)



@torch.compiler.disable
@gradient_entry("conv_transpose3d")
def do_conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, *,
                         plan=None, _weight_cache=None, _scaling_state=None, _state_training=True,
                         _execution_info=None, _output_slot=None):
    """Transposed Conv3d with independent FP8 compute and activation compression."""
    if _weight_cache is None:
        from ..precision.planner import functional_normalization

        owner = functional_normalization("conv_transpose3d")
        if owner is not None:
            return owner(input, weight, bias, stride, padding, output_padding, groups, dilation)
    from .convolution_transpose.dispatch import convolution_transpose

    with _suspend_fp8_interception():
        return convolution_transpose(3, input, weight, bias, stride, padding, output_padding, groups, dilation,
                                     plan=plan, weight_cache=_weight_cache, scaling_state=_scaling_state,
                                     state_training=_state_training, execution_info=_execution_info,
                                     output_slot=_output_slot)


def conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, *, plan=None):
    return do_conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation, plan=plan)


@torch.compiler.disable
@gradient_entry("batch_norm")
def do_batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False,
                  momentum=0.1, eps=1e-5, *, plan=None, _output_layout=None, _scaling_state=None,
                  _state_training=True, _execution_info=None):
    from .normalization.dispatch import normalization

    with _suspend_fp8_interception():
        return normalization("batch", input, weight, bias, running_mean=running_mean, running_var=running_var,
                             training=training, momentum=momentum, eps=eps, plan=plan,
                             output_layout=_output_layout, scaling_state=_scaling_state, state_training=_state_training,
                             execution_info=_execution_info)


@torch.compiler.disable
@gradient_entry("group_norm")
def do_group_norm(input, num_groups, weight=None, bias=None, eps=1e-5, *, plan=None,
                  _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None):
    from .normalization.dispatch import normalization

    with _suspend_fp8_interception():
        return normalization("group", input, weight, bias, groups=num_groups, eps=eps, plan=plan,
                             output_layout=_output_layout, scaling_state=_scaling_state, state_training=_state_training,
                             execution_info=_execution_info)


@torch.compiler.disable
@gradient_entry("layer_norm")
def do_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5, *, plan=None,
                  output_dtype=None, output_layout=None, output_scale=None, _output_layout=None,
                  _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """Suffix LayerNorm with saved-xhat compression and native/FP8 transport."""
    from .normalization.layer_dispatch import layer_norm as dispatch
    if not _planned:
        from ..precision.planner import functional_normalization, normalization_output

        adapter = functional_normalization("layer")
        if adapter is not None:
            plan = adapter.plan if plan is None else plan
            _output_layout = normalization_output(adapter) if _output_layout is None else _output_layout
            _scaling_state = adapter.fp8_scaling if _scaling_state is None else _scaling_state
            _state_training = adapter.training
            if _execution_info is None:
                adapter.last_execution = {}
                _execution_info = adapter.last_execution
    with _suspend_fp8_interception():
        return dispatch(input, normalized_shape, weight, bias, eps, plan=plan, output_dtype=output_dtype,
                        output_layout=output_layout, output_scale=output_scale, planned_layout=_output_layout,
                        scaling_state=_scaling_state, state_training=_state_training, execution_info=_execution_info)


layer_norm = do_layer_norm


def _pool(kind, input, args, kwargs, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
          _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    from .pooling.dispatch import pooling

    if not _planned:
        from ..precision.planner import functional_normalization, normalization_output

        adapter = functional_normalization(kind)
        if adapter is not None:
            plan = adapter.plan if plan is None else plan
            _output_layout = normalization_output(adapter) if _output_layout is None else _output_layout
            _scaling_state = adapter.fp8_scaling if _scaling_state is None else _scaling_state
            _state_training = adapter.training
            if _execution_info is None:
                adapter.last_execution = {}
                _execution_info = adapter.last_execution
    with _suspend_fp8_interception():
        return pooling(kind, input, args, kwargs, plan=plan, output_dtype=output_dtype, output_layout=output_layout,
                       output_scale=output_scale, planned_layout=_output_layout, scaling_state=_scaling_state,
                       state_training=_state_training, execution_info=_execution_info)


def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False,
               momentum=0.1, eps=1e-5, *, plan=None):
    return do_batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps, plan=plan)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5, *, plan=None):
    return do_group_norm(input, num_groups, weight, bias, eps, plan=plan)


@torch.compiler.disable
@gradient_entry('max_pool1d')
def do_max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """max_pool1d with scale-aware precision and family-specific saved-state compression."""
    return _pool('max_pool1d', input, (kernel_size, stride, padding, dilation, ceil_mode, return_indices,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_max_pool1d._deepopt_pool_kind = 'max_pool1d'
max_pool1d = do_max_pool1d


@torch.compiler.disable
@gradient_entry('max_pool1d')
def do_max_pool1d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """max_pool1d_with_indices with scale-aware precision and family-specific saved-state compression."""
    return _pool('max_pool1d', input, (kernel_size, stride, padding, dilation, ceil_mode, True,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_max_pool1d_with_indices._deepopt_pool_kind = 'max_pool1d'
do_max_pool1d_with_indices._deepopt_pool_indices = True
max_pool1d_with_indices = do_max_pool1d_with_indices


@torch.compiler.disable
@gradient_entry('avg_pool1d')
def do_avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """avg_pool1d with scale-aware precision and family-specific saved-state compression."""
    return _pool('avg_pool1d', input, (kernel_size, stride, padding, ceil_mode, count_include_pad,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_avg_pool1d._deepopt_pool_kind = 'avg_pool1d'
avg_pool1d = do_avg_pool1d


@torch.compiler.disable
@gradient_entry('adaptive_max_pool1d')
def do_adaptive_max_pool1d(input, output_size, return_indices=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """adaptive_max_pool1d with scale-aware precision and family-specific saved-state compression."""
    return _pool('adaptive_max_pool1d', input, (output_size, return_indices,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_adaptive_max_pool1d._deepopt_pool_kind = 'adaptive_max_pool1d'
adaptive_max_pool1d = do_adaptive_max_pool1d


@torch.compiler.disable
@gradient_entry('adaptive_max_pool1d')
def do_adaptive_max_pool1d_with_indices(input, output_size, return_indices=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """adaptive_max_pool1d_with_indices with scale-aware precision and family-specific saved-state compression."""
    return _pool('adaptive_max_pool1d', input, (output_size, True,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_adaptive_max_pool1d_with_indices._deepopt_pool_kind = 'adaptive_max_pool1d'
do_adaptive_max_pool1d_with_indices._deepopt_pool_indices = True
adaptive_max_pool1d_with_indices = do_adaptive_max_pool1d_with_indices


@torch.compiler.disable
@gradient_entry('adaptive_avg_pool1d')
def do_adaptive_avg_pool1d(input, output_size, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """adaptive_avg_pool1d with scale-aware precision and family-specific saved-state compression."""
    return _pool('adaptive_avg_pool1d', input, (output_size,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_adaptive_avg_pool1d._deepopt_pool_kind = 'adaptive_avg_pool1d'
adaptive_avg_pool1d = do_adaptive_avg_pool1d


@torch.compiler.disable
@gradient_entry('lp_pool1d')
def do_lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """lp_pool1d with scale-aware precision and family-specific saved-state compression."""
    return _pool('lp_pool1d', input, (norm_type, kernel_size, stride, ceil_mode,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_lp_pool1d._deepopt_pool_kind = 'lp_pool1d'
lp_pool1d = do_lp_pool1d


@torch.compiler.disable
@gradient_entry('max_unpool1d')
def do_max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """max_unpool1d with scale-aware precision and family-specific saved-state compression."""
    return _pool('max_unpool1d', input, (indices, kernel_size, stride, padding, output_size,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_max_unpool1d._deepopt_pool_kind = 'max_unpool1d'
max_unpool1d = do_max_unpool1d


@torch.compiler.disable
@gradient_entry('max_pool2d')
def do_max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """max_pool2d with scale-aware precision and family-specific saved-state compression."""
    return _pool('max_pool2d', input, (kernel_size, stride, padding, dilation, ceil_mode, return_indices,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_max_pool2d._deepopt_pool_kind = 'max_pool2d'
max_pool2d = do_max_pool2d


@torch.compiler.disable
@gradient_entry('max_pool2d')
def do_max_pool2d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """max_pool2d_with_indices with scale-aware precision and family-specific saved-state compression."""
    return _pool('max_pool2d', input, (kernel_size, stride, padding, dilation, ceil_mode, True,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_max_pool2d_with_indices._deepopt_pool_kind = 'max_pool2d'
do_max_pool2d_with_indices._deepopt_pool_indices = True
max_pool2d_with_indices = do_max_pool2d_with_indices


@torch.compiler.disable
@gradient_entry('avg_pool2d')
def do_avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """avg_pool2d with scale-aware precision and family-specific saved-state compression."""
    return _pool('avg_pool2d', input, (kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_avg_pool2d._deepopt_pool_kind = 'avg_pool2d'
avg_pool2d = do_avg_pool2d


@torch.compiler.disable
@gradient_entry('adaptive_max_pool2d')
def do_adaptive_max_pool2d(input, output_size, return_indices=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """adaptive_max_pool2d with scale-aware precision and family-specific saved-state compression."""
    return _pool('adaptive_max_pool2d', input, (output_size, return_indices,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_adaptive_max_pool2d._deepopt_pool_kind = 'adaptive_max_pool2d'
adaptive_max_pool2d = do_adaptive_max_pool2d


@torch.compiler.disable
@gradient_entry('adaptive_max_pool2d')
def do_adaptive_max_pool2d_with_indices(input, output_size, return_indices=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """adaptive_max_pool2d_with_indices with scale-aware precision and family-specific saved-state compression."""
    return _pool('adaptive_max_pool2d', input, (output_size, True,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_adaptive_max_pool2d_with_indices._deepopt_pool_kind = 'adaptive_max_pool2d'
do_adaptive_max_pool2d_with_indices._deepopt_pool_indices = True
adaptive_max_pool2d_with_indices = do_adaptive_max_pool2d_with_indices


@torch.compiler.disable
@gradient_entry('adaptive_avg_pool2d')
def do_adaptive_avg_pool2d(input, output_size, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """adaptive_avg_pool2d with scale-aware precision and family-specific saved-state compression."""
    return _pool('adaptive_avg_pool2d', input, (output_size,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_adaptive_avg_pool2d._deepopt_pool_kind = 'adaptive_avg_pool2d'
adaptive_avg_pool2d = do_adaptive_avg_pool2d


@torch.compiler.disable
@gradient_entry('lp_pool2d')
def do_lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """lp_pool2d with scale-aware precision and family-specific saved-state compression."""
    return _pool('lp_pool2d', input, (norm_type, kernel_size, stride, ceil_mode,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_lp_pool2d._deepopt_pool_kind = 'lp_pool2d'
lp_pool2d = do_lp_pool2d


@torch.compiler.disable
@gradient_entry('max_unpool2d')
def do_max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """max_unpool2d with scale-aware precision and family-specific saved-state compression."""
    return _pool('max_unpool2d', input, (indices, kernel_size, stride, padding, output_size,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_max_unpool2d._deepopt_pool_kind = 'max_unpool2d'
max_unpool2d = do_max_unpool2d


@torch.compiler.disable
@gradient_entry('fractional_max_pool2d')
def do_fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """fractional_max_pool2d with scale-aware precision and family-specific saved-state compression."""
    return _pool('fractional_max_pool2d', input, (kernel_size, output_size, output_ratio, return_indices, _random_samples,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_fractional_max_pool2d._deepopt_pool_kind = 'fractional_max_pool2d'
fractional_max_pool2d = do_fractional_max_pool2d


@torch.compiler.disable
@gradient_entry('fractional_max_pool2d')
def do_fractional_max_pool2d_with_indices(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """fractional_max_pool2d_with_indices with scale-aware precision and family-specific saved-state compression."""
    return _pool('fractional_max_pool2d', input, (kernel_size, output_size, output_ratio, True, _random_samples,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_fractional_max_pool2d_with_indices._deepopt_pool_kind = 'fractional_max_pool2d'
do_fractional_max_pool2d_with_indices._deepopt_pool_indices = True
fractional_max_pool2d_with_indices = do_fractional_max_pool2d_with_indices


@torch.compiler.disable
@gradient_entry('max_pool3d')
def do_max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """max_pool3d with scale-aware precision and family-specific saved-state compression."""
    return _pool('max_pool3d', input, (kernel_size, stride, padding, dilation, ceil_mode, return_indices,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_max_pool3d._deepopt_pool_kind = 'max_pool3d'
max_pool3d = do_max_pool3d


@torch.compiler.disable
@gradient_entry('max_pool3d')
def do_max_pool3d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """max_pool3d_with_indices with scale-aware precision and family-specific saved-state compression."""
    return _pool('max_pool3d', input, (kernel_size, stride, padding, dilation, ceil_mode, True,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_max_pool3d_with_indices._deepopt_pool_kind = 'max_pool3d'
do_max_pool3d_with_indices._deepopt_pool_indices = True
max_pool3d_with_indices = do_max_pool3d_with_indices


@torch.compiler.disable
@gradient_entry('avg_pool3d')
def do_avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """avg_pool3d with scale-aware precision and family-specific saved-state compression."""
    return _pool('avg_pool3d', input, (kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_avg_pool3d._deepopt_pool_kind = 'avg_pool3d'
avg_pool3d = do_avg_pool3d


@torch.compiler.disable
@gradient_entry('adaptive_max_pool3d')
def do_adaptive_max_pool3d(input, output_size, return_indices=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """adaptive_max_pool3d with scale-aware precision and family-specific saved-state compression."""
    return _pool('adaptive_max_pool3d', input, (output_size, return_indices,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_adaptive_max_pool3d._deepopt_pool_kind = 'adaptive_max_pool3d'
adaptive_max_pool3d = do_adaptive_max_pool3d


@torch.compiler.disable
@gradient_entry('adaptive_max_pool3d')
def do_adaptive_max_pool3d_with_indices(input, output_size, return_indices=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """adaptive_max_pool3d_with_indices with scale-aware precision and family-specific saved-state compression."""
    return _pool('adaptive_max_pool3d', input, (output_size, True,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_adaptive_max_pool3d_with_indices._deepopt_pool_kind = 'adaptive_max_pool3d'
do_adaptive_max_pool3d_with_indices._deepopt_pool_indices = True
adaptive_max_pool3d_with_indices = do_adaptive_max_pool3d_with_indices


@torch.compiler.disable
@gradient_entry('adaptive_avg_pool3d')
def do_adaptive_avg_pool3d(input, output_size, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """adaptive_avg_pool3d with scale-aware precision and family-specific saved-state compression."""
    return _pool('adaptive_avg_pool3d', input, (output_size,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_adaptive_avg_pool3d._deepopt_pool_kind = 'adaptive_avg_pool3d'
adaptive_avg_pool3d = do_adaptive_avg_pool3d


@torch.compiler.disable
@gradient_entry('lp_pool3d')
def do_lp_pool3d(input, norm_type, kernel_size, stride=None, ceil_mode=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """lp_pool3d with scale-aware precision and family-specific saved-state compression."""
    return _pool('lp_pool3d', input, (norm_type, kernel_size, stride, ceil_mode,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_lp_pool3d._deepopt_pool_kind = 'lp_pool3d'
lp_pool3d = do_lp_pool3d


@torch.compiler.disable
@gradient_entry('max_unpool3d')
def do_max_unpool3d(input, indices, kernel_size, stride=None, padding=0, output_size=None, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """max_unpool3d with scale-aware precision and family-specific saved-state compression."""
    return _pool('max_unpool3d', input, (indices, kernel_size, stride, padding, output_size,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_max_unpool3d._deepopt_pool_kind = 'max_unpool3d'
max_unpool3d = do_max_unpool3d


@torch.compiler.disable
@gradient_entry('fractional_max_pool3d')
def do_fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """fractional_max_pool3d with scale-aware precision and family-specific saved-state compression."""
    return _pool('fractional_max_pool3d', input, (kernel_size, output_size, output_ratio, return_indices, _random_samples,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_fractional_max_pool3d._deepopt_pool_kind = 'fractional_max_pool3d'
fractional_max_pool3d = do_fractional_max_pool3d


@torch.compiler.disable
@gradient_entry('fractional_max_pool3d')
def do_fractional_max_pool3d_with_indices(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
                    _output_layout=None, _scaling_state=None, _state_training=True, _execution_info=None, _planned=False):
    """fractional_max_pool3d_with_indices with scale-aware precision and family-specific saved-state compression."""
    return _pool('fractional_max_pool3d', input, (kernel_size, output_size, output_ratio, True, _random_samples,), {}, plan=plan, output_dtype=output_dtype,
                 output_layout=output_layout, output_scale=output_scale, _output_layout=_output_layout,
                 _scaling_state=_scaling_state, _state_training=_state_training, _execution_info=_execution_info, _planned=_planned)


do_fractional_max_pool3d_with_indices._deepopt_pool_kind = 'fractional_max_pool3d'
do_fractional_max_pool3d_with_indices._deepopt_pool_indices = True
fractional_max_pool3d_with_indices = do_fractional_max_pool3d_with_indices


__all__ = ["linear", "do_linear", "scaled_dot_product_attention", "do_scaled_dot_product_attention",
           "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
           "do_conv_transpose1d", "do_conv_transpose2d", "do_conv_transpose3d",
           "conv1d", "conv2d", "conv3d", "do_conv1d", "do_conv2d", "do_conv3d",
           "batch_norm", "do_batch_norm", "group_norm", "do_group_norm",
           "layer_norm", "do_layer_norm",
           "validate_linear_plan", "validate_fp8_plan"]


__all__ += ['max_pool1d', 'do_max_pool1d', 'max_pool1d_with_indices', 'do_max_pool1d_with_indices', 'avg_pool1d', 'do_avg_pool1d', 'adaptive_max_pool1d', 'do_adaptive_max_pool1d', 'adaptive_max_pool1d_with_indices', 'do_adaptive_max_pool1d_with_indices', 'adaptive_avg_pool1d', 'do_adaptive_avg_pool1d', 'lp_pool1d', 'do_lp_pool1d', 'max_unpool1d', 'do_max_unpool1d', 'max_pool2d', 'do_max_pool2d', 'max_pool2d_with_indices', 'do_max_pool2d_with_indices', 'avg_pool2d', 'do_avg_pool2d', 'adaptive_max_pool2d', 'do_adaptive_max_pool2d', 'adaptive_max_pool2d_with_indices', 'do_adaptive_max_pool2d_with_indices', 'adaptive_avg_pool2d', 'do_adaptive_avg_pool2d', 'lp_pool2d', 'do_lp_pool2d', 'max_unpool2d', 'do_max_unpool2d', 'fractional_max_pool2d', 'do_fractional_max_pool2d', 'fractional_max_pool2d_with_indices', 'do_fractional_max_pool2d_with_indices', 'max_pool3d', 'do_max_pool3d', 'max_pool3d_with_indices', 'do_max_pool3d_with_indices', 'avg_pool3d', 'do_avg_pool3d', 'adaptive_max_pool3d', 'do_adaptive_max_pool3d', 'adaptive_max_pool3d_with_indices', 'do_adaptive_max_pool3d_with_indices', 'adaptive_avg_pool3d', 'do_adaptive_avg_pool3d', 'lp_pool3d', 'do_lp_pool3d', 'max_unpool3d', 'do_max_unpool3d', 'fractional_max_pool3d', 'do_fractional_max_pool3d', 'fractional_max_pool3d_with_indices', 'do_fractional_max_pool3d_with_indices']
