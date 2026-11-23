"""Scoped AMP, FP8 linear execution, and saved-activation compression."""

from contextlib import ContextDecorator, ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from collections.abc import Mapping
from copy import copy, deepcopy
from dataclasses import asdict
from numbers import Real

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode, _get_current_function_mode, _pop_mode_temporarily
from deepopt._config_resolution import _UNSET, resolve_config
from deepopt._bf16_policy import (bf16_policy_scope, current_bf16_policy,
                                 resolve_bf16_policy, active_bf16_policy, bf16_rewrite_scope)
from deepopt._activation_memory import (ActivationMemoryConfig, activation_memory_scope,
                                       current_activation_memory, validate_activation_memory_execution)
from deepopt.config import ActivationCompressionConfig, FP8Config
from ._execution import finite_check_scope


# Values are normalized operation plans or False (explicit dense override).
# Missing device keys mean that an existing DO operation keeps its own plan.
_active_fp8 = ContextVar("deepopt_active_fp8", default={})
_active_interception = ContextVar("deepopt_fp8_linear_interception", default={})
_active_convolution_interception = ContextVar("deepopt_fp8_convolution_interception", default={})
_active_pooling_interception = ContextVar("deepopt_fp8_pooling_interception", default={})
_active_normalization_interception = ContextVar("deepopt_fp8_normalization_interception", default={})
_active_compression = ContextVar("deepopt_activation_compression", default={})
_active_attention = ContextVar("deepopt_attention_backend", default={})
_linear_mode_active = ContextVar("deepopt_linear_mode_active", default=False)
_autocast_entries = ContextVar("deepopt_autocast_entries", default=())
_FLOAT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
_RELU_FUNCTIONS = (F.relu, torch.relu, torch.Tensor.relu, torch.relu_, torch.Tensor.relu_)
_MUL_FUNCTIONS = (torch.mul, torch.multiply, torch.Tensor.mul, torch.Tensor.multiply,
                  torch.Tensor.__mul__, torch.Tensor.__rmul__)
_CLAMP_FUNCTIONS = {getattr(namespace, name): "clamp" if name == "clip" else name
    for namespace in (torch, torch.Tensor) for name in ("clamp", "clip", "clamp_min", "clamp_max")}


def _get_fp8_plan(device_type):
    return _active_fp8.get().get(device_type)


def _get_activation_compression(device_type):
    return _active_compression.get().get(device_type)


def _get_activation_memory(device_type):
    return current_activation_memory(device_type)


def _get_attention_backend(device_type):
    return _active_attention.get().get(device_type)


def _get_bf16_compression(x):
    compression = _get_activation_compression(x.device.type)
    storage = current_bf16_policy(x.device.type).storage_enabled
    if (compression is None or not compression["enabled"]) and not storage:
        return None
    compute_dtype = (torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type)
                     else x.dtype)
    if compute_dtype != torch.bfloat16:
        return None
    return compression if compression is not None else asdict(ActivationCompressionConfig())


def _native_policy_plan(compression, device_type):
    from .nn.functional import _native_compression_plan

    return _native_compression_plan(
        compression["group_size"], compression.get("bits", 2), compression.get("attention_bits"),
        compression.get("norm_xhat_recenter", False), enabled=compression["enabled"],
        bf16_policy=current_bf16_policy(device_type).name)


def _linear_arguments(args, kwargs):
    """Leave malformed calls to PyTorch, including duplicate/unknown keywords."""
    names = ("input", "weight", "bias")
    if len(args) > 3 or set(kwargs) - set(names) or any(name in kwargs for name in names[:len(args)]):
        return None
    values = tuple(args[index] if index < len(args) else kwargs.get(name)
                   for index, name in enumerate(names))
    x, weight, bias = values
    tensors = (x, weight) if bias is None else values
    if any(not isinstance(value, torch.Tensor) or value.is_nested
           or value.layout != torch.strided or value.dtype not in _FLOAT_DTYPES for value in tensors):
        return None
    if x.device.type not in ("cpu", "cuda") or any(value.device != x.device for value in tensors):
        return None
    if x.ndim < 1 or weight.ndim != 2 or not x.numel() or not weight.numel():
        return None
    return values


class _FP8LinearMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if not torch.compiler.is_compiling():
            from .nn.primitive_state import primitive_arguments, model_add, model_cat

            primitive = primitive_arguments(func, args, kwargs)
            if primitive is not None:
                kind, values, dim = primitive
                return model_cat(values, dim) if kind == "cat" else model_add(*values)
        if (not torch.compiler.is_compiling() and func in (torch.exp, torch.Tensor.exp)
                and len(args) <= 1 and not set(kwargs) - {"input"}
                and not (args and "input" in kwargs)):
            x = args[0] if args else kwargs.get("input")
            if (type(x) in (torch.Tensor, torch.nn.Parameter, FakeTensor)
                    and x.device.type in ("cpu", "cuda") and x.dtype == torch.bfloat16
                    and not x.is_nested and x.layout == torch.strided
                    and current_bf16_policy(x.device.type).storage_enabled):
                from .nn.primitive_state import model_exp

                return model_exp(x)
        # Native compiler lowering owns these captured primitive derivatives.
        # Eager saved-state adapters must stay outside Dynamo's fullgraph body.
        if not torch.compiler.is_compiling() and func in _CLAMP_FUNCTIONS:
            kind = _CLAMP_FUNCTIONS[func]
            names = ("input", "min", "max") if kind == "clamp" else ("input", "min" if kind == "clamp_min" else "max")
            if (len(args) <= len(names) and not set(kwargs) - set(names)
                    and not any(name in kwargs for name in names[:len(args)])):
                values = {name: args[index] if index < len(args) else kwargs.get(name)
                          for index, name in enumerate(names)}
                x, minimum, maximum = values.get("input"), values.get("min"), values.get("max")
                if (type(x) in (torch.Tensor, torch.nn.Parameter, FakeTensor) and x.device.type in ("cpu", "cuda")
                        and not x.is_nested and x.layout == torch.strided
                        and x.dtype in (*_FLOAT_DTYPES, torch.float64)
                        and any(value is not None for value in (minimum, maximum))
                        and all(value is None or isinstance(value, Real) for value in (minimum, maximum))
                        and _get_bf16_compression(x) is not None):
                    from .nn.primitive_state import scalar_clamp

                    return scalar_clamp(x, minimum, maximum, kind=kind)
        if not torch.compiler.is_compiling():
            from .nn.primitive_state import nearest2d_arguments, nearest2d_call

            resize = nearest2d_arguments(func, args, kwargs)
            if resize is not None:
                x = resize[0]
                if (_get_bf16_compression(x) is not None
                        or current_bf16_policy(x.device.type).storage_enabled):
                    return nearest2d_call(func, args, kwargs)
        if func in _MUL_FUNCTIONS and not torch.compiler.is_compiling():
            names = ("input", "other")
            if (len(args) <= 2 and not set(kwargs) - set(names)
                    and not any(name in kwargs for name in names[:len(args)])):
                values = tuple(args[index] if index < len(args) else kwargs.get(name)
                               for index, name in enumerate(names))
                from .nn.binary import _policy_tensor, multiplication_support

                if multiplication_support(*values) is None:
                    tensor = _policy_tensor(values)
                    if _get_bf16_compression(tensor) is not None:
                        from .nn.functional import do_mul

                        return do_mul(*values)
        if func is F.pad and not torch.compiler.is_compiling():
            x = args[0] if args else kwargs.get("input")
            if (type(x) in (torch.Tensor, FakeTensor) and x.device.type in ("cpu", "cuda") and not x.is_nested
                    and x.layout == torch.strided and x.dtype in (*_FLOAT_DTYPES, torch.float64)
                    and _get_bf16_compression(x) is not None):
                from .nn.functional import do_pad

                return do_pad(*args, **kwargs)
        from .nn.activation.spec import arguments

        activation = arguments(func, args, kwargs)
        if activation is not None:
            kind, x, options = activation
            if (isinstance(x, torch.Tensor) and x.device.type in ("cpu", "cuda") and not x.is_nested
                    and x.layout == torch.strided and x.dtype in (*_FLOAT_DTYPES, torch.float8_e4m3fn, torch.float8_e5m2)):
                from .precision.planner import functional_normalization

                adapter = functional_normalization(kind)
                if adapter is not None:
                    with _suspend_fp8_interception():
                        return adapter(x, **options)
                plan = _get_fp8_plan(x.device.type)
                compression = _get_bf16_compression(x)
                if (plan is not None and plan is not False and _active_interception.get().get(x.device.type, True)
                        or compression is not None):
                    from .nn.functional import _activation
                    from .precision.gradient import functional_gradient

                    with functional_gradient(kind):
                        return _activation(kind, x, plan=plan if plan is not False else None, **options)
        from .nn.pooling.spec import call_kind, FORCE_INDICES

        pool_kind = call_kind(func)
        if pool_kind is not None:
            x = args[0] if args else kwargs.get("input")
            if isinstance(x, torch.Tensor) and x.device.type in ("cpu", "cuda"):
                from .precision.planner import functional_normalization
                from .nn import functional as doF

                adapter = functional_normalization(pool_kind)
                if func in FORCE_INDICES:
                    from .nn.pooling.spec import SPECS

                    bound = SPECS[pool_kind].arguments(args[1:], {k: v for k, v in kwargs.items() if k != "input"})
                    bound["return_indices"] = True
                    args, kwargs = (x,), bound
                if adapter is not None:
                    with _suspend_fp8_interception():
                        return adapter(*args, **kwargs)
                plan, compression = _get_fp8_plan(x.device.type), _get_bf16_compression(x)
                fp8 = plan is not None and plan is not False and _active_pooling_interception.get().get(x.device.type, True)
                if fp8 or compression is not None:
                    selected = plan if fp8 else _native_policy_plan(compression, x.device.type)
                    return getattr(doF, "do_" + pool_kind)(*args, **kwargs, plan=selected)
        if func in _RELU_FUNCTIONS:
            relu_names = ("input", "inplace") if func is F.relu else ("input",)
            if (len(args) > len(relu_names) or set(kwargs) - set(relu_names)
                    or any(name in kwargs for name in relu_names[:len(args)])):
                return func(*args, **kwargs)
            x = args[0] if args else kwargs.get("input")
            if (isinstance(x, torch.Tensor) and x.device.type in ("cpu", "cuda") and not x.is_nested
                    and x.layout == torch.strided and x.dtype in (*_FLOAT_DTYPES, torch.float8_e4m3fn, torch.float8_e5m2)):
                from .precision.planner import functional_normalization

                adapter = functional_normalization("relu")
                inplace = func in (torch.relu_, torch.Tensor.relu_) or kwargs.get("inplace", args[1] if len(args) > 1 else False)
                if adapter is not None:
                    with _suspend_fp8_interception():
                        return adapter(x, inplace=inplace)
                plan = _get_fp8_plan(x.device.type)
                compression = _get_bf16_compression(x)
                if (plan is not None and plan is not False and _active_interception.get().get(x.device.type, True)
                        or compression is not None):
                    from .nn.functional import do_relu

                    return do_relu(x, inplace=inplace, plan=plan if plan is not False else None)
        if func is F.layer_norm:
            x = args[0] if args else kwargs.get("input")
            if (isinstance(x, torch.Tensor) and x.device.type in ("cpu", "cuda") and not x.is_nested
                    and x.layout == torch.strided and x.dtype in (*_FLOAT_DTYPES, torch.float8_e4m3fn, torch.float8_e5m2)):
                from .precision.planner import functional_normalization

                adapter = functional_normalization("layer")
                if adapter is not None:
                    with _suspend_fp8_interception():
                        return adapter(*args, **kwargs)
                plan = _get_fp8_plan(x.device.type)
                compression = _get_bf16_compression(x)
                fp8 = plan is not None and plan is not False and _active_normalization_interception.get().get(x.device.type, True)
                if fp8 or compression is not None:
                    from .nn.functional import do_layer_norm

                    selected = plan if fp8 else _native_policy_plan(compression, x.device.type)
                    return do_layer_norm(*args, **kwargs, plan=selected)
        if func in (F.batch_norm, F.group_norm):
            x = args[0] if args else kwargs.get("input")
            if (isinstance(x, torch.Tensor) and x.device.type in ("cpu", "cuda") and not x.is_nested
                    and (func is F.batch_norm and x.ndim == 4 or func is F.group_norm and x.ndim >= 2)
                    and x.layout == torch.strided and x.dtype in (*_FLOAT_DTYPES, torch.float8_e4m3fn, torch.float8_e5m2)):
                plan = _get_fp8_plan(x.device.type)
                compression = _get_bf16_compression(x)
                from .precision.planner import functional_normalization

                adapter = functional_normalization("batch" if func is F.batch_norm else "group")
                if adapter is not None:
                    with _suspend_fp8_interception():
                        return adapter(*args, **kwargs)
                fp8 = (plan is not None and plan is not False
                       and _active_normalization_interception.get().get(x.device.type, True))
                if fp8 or compression is not None:
                    from .nn.functional import do_batch_norm, do_group_norm
                    selected = plan if fp8 else _native_policy_plan(compression, x.device.type)
                    call = do_batch_norm if func is F.batch_norm else do_group_norm
                    return call(*args, **kwargs, plan=selected)
        if func in (F.conv1d, F.conv2d, F.conv3d):
            arguments = _convolution_arguments(func, args, kwargs)
            if arguments is not None:
                device_type = arguments[0].device.type
                plan = _get_fp8_plan(device_type)
                from .precision.gradient import prepared_functional_plan

                prepared = prepared_functional_plan(f"conv{(F.conv1d, F.conv2d, F.conv3d).index(func) + 1}d")
                if plan is None:
                    plan = prepared
                if prepared is not None and plan is False:
                    from .nn.functional import do_conv1d, do_conv2d, do_conv3d

                    call = (do_conv1d, do_conv2d, do_conv3d)[(F.conv1d, F.conv2d, F.conv3d).index(func)]
                    return call(*args, **kwargs, plan=prepared)
                if (plan is not None and plan is not False
                        and _active_convolution_interception.get().get(device_type, True)):
                    from deepopt.opt.nn.functional import do_conv1d, do_conv2d, do_conv3d

                    call = (do_conv1d, do_conv2d, do_conv3d)[(F.conv1d, F.conv2d, F.conv3d).index(func)]
                    return call(*args, **kwargs, plan=plan)
                compression = _get_bf16_compression(arguments[0])
                if compression is not None:
                    from deepopt.opt.nn.functional import do_conv1d, do_conv2d, do_conv3d

                    call = (do_conv1d, do_conv2d, do_conv3d)[(F.conv1d, F.conv2d, F.conv3d).index(func)]
                    policy = _native_policy_plan(compression, device_type)
                    return call(*args, **kwargs, plan=policy)
        if func in (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d):
            from .precision.planner import functional_normalization

            rank = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d).index(func) + 1
            owner = functional_normalization(f"conv_transpose{rank}d")
            if owner is not None:
                with _suspend_fp8_interception():
                    return owner(*args, **kwargs)
            arguments = _convolution_arguments(func, args, kwargs)
            if arguments is not None:
                device_type = arguments[0].device.type
                plan = _get_fp8_plan(device_type)
                from .precision.gradient import prepared_functional_plan

                prepared = prepared_functional_plan(f"conv_transpose{(F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d).index(func) + 1}d")
                if plan is None:
                    plan = prepared
                if prepared is not None and plan is False:
                    from .nn.functional import do_conv_transpose1d, do_conv_transpose2d, do_conv_transpose3d

                    call = (do_conv_transpose1d, do_conv_transpose2d, do_conv_transpose3d)[(F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d).index(func)]
                    return call(*args, **kwargs, plan=prepared)
                if (plan is not None and plan is not False
                        and _active_convolution_interception.get().get(device_type, True)):
                    from deepopt.opt.nn.functional import do_conv_transpose1d, do_conv_transpose2d, do_conv_transpose3d

                    call = (do_conv_transpose1d, do_conv_transpose2d, do_conv_transpose3d)[(F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d).index(func)]
                    return call(*args, **kwargs, plan=plan)
                compression = _get_bf16_compression(arguments[0])
                if compression is not None:
                    from deepopt.opt.nn.functional import do_conv_transpose1d, do_conv_transpose2d, do_conv_transpose3d

                    call = (do_conv_transpose1d, do_conv_transpose2d, do_conv_transpose3d)[(F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d).index(func)]
                    policy = _native_policy_plan(compression, device_type)
                    return call(*args, **kwargs, plan=policy)
        if func is F.scaled_dot_product_attention:
            from .nn.attention_backend import observing_attention

            query = args[0] if args else kwargs.get("query")
            compression = (_get_activation_compression(query.device.type)
                           if isinstance(query, torch.Tensor) else None)
            if ((compression is not None and compression["enabled"])
                    or isinstance(query, torch.Tensor) and _get_attention_backend(query.device.type) == "flash_attn_2"
                    or observing_attention()):
                from deepopt.opt.nn.functional import do_scaled_dot_product_attention

                return do_scaled_dot_product_attention(*args, **kwargs)
        if func is F.linear:
            arguments = _linear_arguments(args, kwargs)
            if arguments is not None:
                plan = _get_fp8_plan(arguments[0].device.type)
                from .precision.gradient import prepared_functional_plan

                prepared = prepared_functional_plan("linear")
                if plan is None:
                    plan = prepared
                if prepared is not None and plan is False:
                    from .nn.functional import do_linear

                    return do_linear(*arguments, plan=prepared)
                if (plan is not None and plan is not False
                        and _active_interception.get().get(arguments[0].device.type, True)):
                    from deepopt.opt.nn.functional import do_linear

                    return do_linear(*arguments, plan=plan)
                compression = _get_bf16_compression(arguments[0])
                if compression is not None:
                    from deepopt.opt.nn.functional import do_linear

                    return do_linear(*arguments, plan=_native_policy_plan(compression, arguments[0].device.type))
        return func(*args, **kwargs)


def _convolution_arguments(func, args, kwargs):
    transposed = func in (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)
    names = (("input", "weight", "bias", "stride", "padding", "output_padding", "groups", "dilation")
             if transposed else ("input", "weight", "bias", "stride", "padding", "dilation", "groups"))
    if len(args) > len(names) or set(kwargs) - set(names) or any(name in kwargs for name in names[:len(args)]):
        return None
    values = tuple(args[i] if i < len(args) else kwargs.get(name) for i, name in enumerate(names[:3]))
    x, weight, bias = values
    tensors = (x, weight) if bias is None else values
    if any(not isinstance(value, torch.Tensor) or value.is_nested or value.layout != torch.strided
           or value.dtype not in _FLOAT_DTYPES for value in tensors):
        return None
    calls = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d) if transposed else (F.conv1d, F.conv2d, F.conv3d)
    rank = calls.index(func) + 1
    if (x.device.type not in ("cpu", "cuda") or any(value.device != x.device for value in tensors)
            or x.ndim not in (rank + 1, rank + 2) or weight.ndim != rank + 2
            or not x.numel() or not weight.numel()):
        return None
    return values


@contextmanager
def _linear_mode():
    """FP8 and BF16 scopes share one dispatcher while preserving user modes."""
    if _linear_mode_active.get():
        yield
        return
    token = _linear_mode_active.set(True)
    try:
        with _FP8LinearMode():
            yield
    finally:
        _linear_mode_active.reset(token)


@contextmanager
def _suspend_fp8_interception():
    """Skip our broad dispatcher inside an already-selected FP8 operation.

    Other user modes and tensor subclass handlers remain active. If a user mode
    is above ours, leave its stack intact instead of disabling all Torch modes.
    """
    current = _get_current_function_mode()
    if getattr(current, "_deepopt_compiler_capture_mode", False):
        with _pop_mode_temporarily(), _suspend_fp8_interception():
            yield
    elif isinstance(current, _FP8LinearMode):
        with _pop_mode_temporarily(), _suspend_fp8_interception():
            yield
    else:
        yield


class _FP8Scope:
    """Resolve once; each entry owns only its context token and mode lifetime."""

    def __init__(self, device_type, fp8=_UNSET, *, config=None, _validated_snapshot=False):
        self.device_type = device_type
        self.inherit_outer = fp8 is _UNSET
        self.plan = None
        self.intercept_linear = True
        self.intercept_convolution = True
        self.intercept_normalization = True
        self.intercept_pooling = True
        if fp8 is None:
            return
        from .policy import fp8_plan, normalize_fp8_config

        if _validated_snapshot:
            # autocast owns a fully resolved snapshot. Revalidating its entire
            # experiment here (and again inside fp8_plan) costs more than small
            # GPU operations. Keep public raw-config entry points unchanged.
            from deepopt.config import fp8_config_from_dict

            settings = fp8_config_from_dict(config.get("runtime", {}).get("fp8", {}))
        else:
            settings = normalize_fp8_config(
                {"enabled": True, **fp8} if isinstance(fp8, Mapping) else fp8, config=config)
        if fp8 is _UNSET and not settings.enabled:
            return
        if settings.enabled and device_type not in ("cpu", "cuda"):
            raise ValueError("FP8 autocast supports CPU and CUDA devices")
        from deepopt.opt.nn.functional import _prepare_fp8_plan

        if settings.enabled and _validated_snapshot:
            from deepopt.config import activation_compression_config_from_dict

            compression = activation_compression_config_from_dict(config.get("runtime", {}).get("activation_compression", {}))
            self.plan = _prepare_fp8_plan(fp8_plan(settings, config=config, activation_compression=compression))
        else:
            self.plan = _prepare_fp8_plan(fp8_plan(settings, config=config)) if settings.enabled else False
        self.intercept_linear = settings.intercept_linear
        self.intercept_convolution = (settings.intercept_linear if settings.intercept_convolution is None
                                      else settings.intercept_convolution)
        self.intercept_pooling = settings.intercept_linear if settings.intercept_pooling is None else settings.intercept_pooling
        self.intercept_normalization = (settings.intercept_linear if settings.intercept_normalization is None
                                        else settings.intercept_normalization)

    @contextmanager
    def enter(self):
        previous = _active_fp8.get()
        previous_interception = _active_interception.get()
        previous_convolution = _active_convolution_interception.get()
        previous_normalization = _active_normalization_interception.get()
        previous_pooling = _active_pooling_interception.get()
        if (self.plan is None or (self.inherit_outer and previous.get(self.device_type) is not None)
                or (previous.get(self.device_type) == self.plan
                    and previous_interception.get(self.device_type) == self.intercept_linear
                    and previous_convolution.get(self.device_type) == self.intercept_convolution
                    and previous_normalization.get(self.device_type) == self.intercept_normalization
                    and previous_pooling.get(self.device_type) == self.intercept_pooling)):
            yield
            return
        token = _active_fp8.set({**previous, self.device_type: self.plan})
        interception_token = _active_interception.set({**previous_interception,
                                                      self.device_type: self.intercept_linear})
        convolution_token = _active_convolution_interception.set({**previous_convolution,
                                                                 self.device_type: self.intercept_convolution})
        normalization_token = _active_normalization_interception.set({**previous_normalization,
                                                                     self.device_type: self.intercept_normalization})
        pooling_token = _active_pooling_interception.set({**previous_pooling, self.device_type: self.intercept_pooling})
        try:
            delayed = self.plan is not False and self.plan["fp8_training"]["scaling_recipe"] == "delayed"
            scaling = nullcontext()
            if delayed:
                from deepopt.opt.scaling.delayed import fp8_scaling_scope

                scaling = fp8_scaling_scope()
            intercept = self.plan is not False and (self.intercept_linear or self.intercept_convolution
                        or self.intercept_normalization or self.intercept_pooling)
            with scaling, _linear_mode() if intercept else nullcontext():
                yield
        finally:
            _active_pooling_interception.reset(pooling_token)
            _active_normalization_interception.reset(normalization_token)
            _active_convolution_interception.reset(convolution_token)
            _active_interception.reset(interception_token)
            _active_fp8.reset(token)


def fp8_context(device_type, fp8=_UNSET, *, config=None):
    """Apply FP8 policy, preserving ambient AMP and explicit disable/inherit."""
    return _FP8Scope(device_type, fp8, config=config).enter()


class _ActivationCompressionScope:
    def __init__(self, device_type, settings=_UNSET, *, config=None, _validated_snapshot=False):
        self.device_type = device_type
        self.inherit_outer = settings is _UNSET or settings is None
        self.settings = None
        if settings is None:
            return
        snapshot = config if _validated_snapshot else resolve_config(config)
        if settings is _UNSET and "activation_compression" not in snapshot.get("runtime", {}):
            return
        from .policy import normalize_activation_compression_config

        if _validated_snapshot:
            from deepopt.config import activation_compression_config_from_dict

            self.settings = asdict(activation_compression_config_from_dict(
                snapshot.get("runtime", {}).get("activation_compression", {})))
        else:
            self.settings = asdict(normalize_activation_compression_config(settings, config=snapshot))

    @contextmanager
    def enter(self):
        previous = _active_compression.get()
        if self.settings is None or (self.inherit_outer and self.device_type in previous):
            yield
            return
        token = _active_compression.set({**previous, self.device_type: self.settings})
        try:
            with _linear_mode() if self.settings["enabled"] else nullcontext():
                yield
        finally:
            _active_compression.reset(token)


class _AttentionScope:
    def __init__(self, device_type, snapshot, inherit):
        self.device_type, self.inherit = device_type, inherit
        self.backend = snapshot.get("runtime", {}).get("attention", {}).get("backend")

    @contextmanager
    def enter(self):
        previous = _active_attention.get()
        if self.backend is None or self.inherit and self.device_type in previous:
            yield
            return
        token = _active_attention.set({**previous, self.device_type: self.backend})
        try:
            with _linear_mode() if self.backend == "flash_attn_2" else nullcontext():
                yield
        finally:
            _active_attention.reset(token)


class autocast(ContextDecorator):
    """Compose torch.autocast with optional FP8 linear execution.

    ``enabled`` controls native AMP only. ``fp8=True`` enables ordinary
    nn.Linear/F.linear calls; a dictionary or FP8Config selects its policy.
    ``fp8=False`` disables FP8 in nested scopes; ``None`` inherits it.
    Other operations keep their normal PyTorch/AMP behavior. Backward uses the
    FP8 plan saved during forward, even when executed after leaving the scope.

    ``activation_compression`` independently selects fixed two-bit saved inputs
    for FP8 and BF16 linear weight gradients. It defaults to the configuration
    snapshot; ``False`` disables it in a nested scope, and ``None`` inherits.

    ``attention_backend`` selects native SDPA or the installed ``flash_attn_2``
    kernels independently of compression and FP8. Omission inherits an outer
    backend or the owned runtime.attention configuration.

    Omitted AMP/FP8 settings use an owned configuration snapshot. An active
    native AMP dtype and outer FP8 policy are preserved when their controls are
    omitted. With no defaults loaded, native AMP keeps torch.autocast defaults.

    Like native AMP, use scopes within one thread's synchronous execution; do
    not suspend an active scope across interleaved async tasks or generators.
    """

    def __init__(self, device_type, dtype=_UNSET, enabled=_UNSET, cache_enabled=None, fp8=_UNSET,
                 *, config=None, check_finite=None, activation_compression=_UNSET, attention_backend=_UNSET,
                 bf16_policy=_UNSET, activation_memory=_UNSET):
        if check_finite is not None and type(check_finite) is not bool:
            raise TypeError("check_finite must be a boolean or None")
        self.check_finite = check_finite
        self.device_type = device_type
        self.dtype = dtype
        self.enabled = enabled
        self.cache_enabled = cache_enabled
        self.fp8 = _UNSET if fp8 is _UNSET else deepcopy(fp8)
        self.bf16_policy = bf16_policy
        self.activation_compression = (_UNSET if activation_compression is _UNSET
                                       else deepcopy(activation_compression))
        explicit_fp8 = _UNSET
        if isinstance(fp8, FP8Config):
            explicit_fp8 = asdict(fp8)
        elif isinstance(fp8, Mapping):
            explicit_fp8 = {"enabled": True, **fp8}
        elif type(fp8) is bool:
            explicit_fp8 = {"enabled": fp8}
        runtime_overrides = {} if explicit_fp8 is _UNSET else {"fp8": explicit_fp8}
        if isinstance(activation_compression, ActivationCompressionConfig):
            runtime_overrides["activation_compression"] = asdict(activation_compression)
        elif isinstance(activation_compression, Mapping):
            runtime_overrides["activation_compression"] = dict(activation_compression)
        elif type(activation_compression) is bool:
            runtime_overrides["activation_compression"] = {"enabled": activation_compression}
        if attention_backend is not _UNSET:
            runtime_overrides["attention"] = {"backend": attention_backend}
        if bf16_policy is not _UNSET:
            runtime_overrides["bf16_policy"] = resolve_bf16_policy(bf16_policy).name
        if isinstance(activation_memory, ActivationMemoryConfig):
            runtime_overrides["activation_memory"] = {
                name: getattr(activation_memory, name) for name in (
                    "multiply_backward", "pack_reuse", "reduction_scratch_bytes", "recompute_modules")}
        elif activation_memory is not _UNSET:
            runtime_overrides["activation_memory"] = activation_memory
        overrides = {"runtime": runtime_overrides} if runtime_overrides else None
        self.config = resolve_config(config, overrides=overrides)
        from deepopt.kernel_policy import resolve_kernel_policy
        runtime = self.config.get("runtime", {})
        self._bf16_policy = (resolve_bf16_policy(runtime["bf16_policy"])
                             if "bf16_policy" in runtime else None)
        self._inherit_bf16_policy = config is None and bf16_policy is _UNSET
        self.kernel_policy = resolve_kernel_policy(runtime.get("kernels"),
            fp8_backend=runtime.get("fp8", {}).get("backend") if runtime.get("fp8", {}).get("enabled") else None)
        self._activation_memory = (validate_activation_memory_execution(
            runtime["activation_memory"], kernel_policy=self.kernel_policy)
            if "activation_memory" in runtime else None)
        self._inherit_activation_memory = config is None and activation_memory is _UNSET
        self._compression_scope = _ActivationCompressionScope(
            device_type, self.activation_compression, config=self.config, _validated_snapshot=True)
        self._fp8_scope = _FP8Scope(device_type, self.fp8, config=self.config, _validated_snapshot=True)
        self._attention_scope = _AttentionScope(device_type, self.config,
                                                config is None and attention_backend is _UNSET)

    def _recreate_cm(self):
        # Entries store their state in context variables, so decorators can
        # reuse the resolved immutable policy without resolving configuration.
        return copy(self)

    def __enter__(self):
        stack = ExitStack()
        try:
            from deepopt.kernel_policy import kernel_policy_scope
            stack.enter_context(kernel_policy_scope(self.kernel_policy))
            from deepopt._activation_memory import active_activation_memory

            if self._activation_memory is not None and not (
                    self._inherit_activation_memory and active_activation_memory(self.device_type) is not None):
                stack.enter_context(activation_memory_scope(self._activation_memory, self.device_type))
            precision = self.config.get("runtime", {}).get("precision")
            ambient_amp = torch.is_autocast_enabled(self.device_type)
            enabled = self.enabled
            if enabled is _UNSET:
                enabled = ambient_amp or (precision == "bf16" if precision is not None else True)
            dtype = self.dtype
            if dtype is _UNSET:
                dtype = (torch.get_autocast_dtype(self.device_type) if ambient_amp else
                         torch.bfloat16 if precision == "bf16" else None)
            stack.enter_context(torch.autocast(self.device_type, dtype=dtype,
                                               enabled=enabled, cache_enabled=self.cache_enabled))
            ambient_policy = active_bf16_policy(self.device_type)
            selected = (ambient_policy if self._inherit_bf16_policy and ambient_policy is not None else
                        self._bf16_policy or ambient_policy or resolve_bf16_policy())
            if enabled and selected.storage_enabled:
                if torch.get_autocast_dtype(self.device_type) != torch.bfloat16:
                    raise ValueError("bf16_storage requires effective BF16 autocast")
                inherited = _get_fp8_plan(self.device_type)
                plan = (inherited if self._fp8_scope.plan is None or
                        self._fp8_scope.inherit_outer and inherited is not None
                        else self._fp8_scope.plan)
                if plan not in (None, False):
                    raise ValueError("bf16_storage requires FP8 disabled")
            stack.enter_context(bf16_rewrite_scope(bool(enabled), self.device_type))
            if selected.storage_enabled or self._bf16_policy is not None:
                stack.enter_context(bf16_policy_scope(selected, self.device_type))
            if self.check_finite is not None:
                stack.enter_context(finite_check_scope(self.check_finite))
            stack.enter_context(self._compression_scope.enter())
            stack.enter_context(self._fp8_scope.enter())
            stack.enter_context(self._attention_scope.enter())
            if enabled and selected.storage_enabled:
                stack.enter_context(_linear_mode())
        except BaseException:
            stack.close()
            raise
        _autocast_entries.set((*_autocast_entries.get(), stack))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        entries = _autocast_entries.get()
        stack = entries[-1]
        _autocast_entries.set(entries[:-1])
        return stack.__exit__(exc_type, exc_value, traceback)
