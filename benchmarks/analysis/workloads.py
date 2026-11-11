"""Construct matched native modules and explicit DO functional calls lazily."""

from copy import deepcopy
import importlib

from .catalog import ACTIVATIONS, POOL_OPS, INDEX_ALIASES


class UnsupportedWorkload(Exception):
    """A known unsupported benchmark combination, never an unexpected kernel error."""


def load_factory(reference):
    module, separator, name = reference.partition(":")
    if not separator or not module or not name:
        raise ValueError("factory must be importable_module:function")
    return getattr(importlib.import_module(module), name)


def build_source(spec, device):
    import torch
    from torch import nn
    from torch.nn import functional as F

    op, shape, options = spec["op"], tuple(spec["shape"]), deepcopy(spec["kwargs"])
    if op == "model":
        result = load_factory(spec["factory"])(device=device, **spec.get("factory_kwargs", {}))
        if not isinstance(result, dict) or not {"model", "inputs"} <= result.keys():
            raise ValueError("factory must return {'model': nn.Module, 'inputs': tuple, ...}")
        result.setdefault("kwargs", {})
        result["inputs"] = tuple(result["inputs"])
        return result
    x = torch.randn(shape, device=device, dtype=torch.bfloat16)
    if spec["layout"] == "strided":
        backing = torch.empty((*shape[:-1], shape[-1] * 2), device=device, dtype=x.dtype)
        backing[..., ::2].copy_(x)
        x = backing[..., ::2]
    elif spec["layout"] == "channels_last":
        if len(shape) not in (4, 5):
            raise UnsupportedWorkload("channels_last requires rank 4 or 5")
        x = x.contiguous(memory_format=torch.channels_last if len(shape) == 4 else torch.channels_last_3d)

    class SDPA(nn.Module):
        def __init__(self, settings):
            super().__init__()
            self.options = settings

        def forward(self, q, k, v):
            return F.scaled_dot_product_attention(q, k, v, **self.options)

    class MHA(nn.Module):
        def __init__(self, width, settings):
            super().__init__()
            settings = dict(settings)
            self.cross_length = settings.pop("cross_length", None)
            self.need_weights = settings.pop("need_weights", False)
            self.attention = nn.MultiheadAttention(width, batch_first=True, **settings)

        def forward(self, q, k, v):
            return self.attention(q, k, v, need_weights=self.need_weights)[0]

    class Pool(nn.Module):
        def __init__(self, name, settings):
            super().__init__()
            self.name, self.options = name, settings
            self.register_buffer("indices", None)
            self.register_buffer("samples", None)

        def forward(self, value):
            settings = dict(self.options)
            if self.indices is not None:
                settings["indices"] = self.indices
            if self.samples is not None:
                settings["_random_samples"] = self.samples
            # LP3d has a native expression on older Torch versions.
            fn = getattr(F, self.name, None)
            if fn is None:
                if self.name != "lp_pool3d":
                    raise UnsupportedWorkload(f"Torch does not provide {self.name}")
                from deepopt.opt.nn.pooling.spec import _lp_pool3d
                fn = _lp_pool3d
            return fn(value, **settings)

    class AttentionBlock(nn.Module):
        def __init__(self, width, settings):
            super().__init__()
            self.norm = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(width, batch_first=True, **settings)
            self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

        def forward(self, value):
            h = self.norm(value)
            value = value + self.attention(h, h, h, need_weights=False)[0]
            return value + self.mlp(value)

    inputs = (x,)
    if op in ("linear", "hf_conv1d"):
        width = options.pop("out_features", shape[-1])
        if op == "hf_conv1d":
            try:
                from transformers.pytorch_utils import Conv1D
            except ImportError as error:
                raise UnsupportedWorkload("hf_conv1d requires transformers") from error
            if options:
                raise ValueError("hf_conv1d accepts out_features only")
            model = Conv1D(width, shape[-1])
        else:
            model = nn.Linear(shape[-1], width, **options)
    elif op.startswith("conv") and op != "conv_pipeline":
        rank = int(op[-2])
        model_type = getattr(nn, ("ConvTranspose" if "transpose" in op else "Conv") + f"{rank}d")
        model = model_type(shape[1], options.pop("out_channels", shape[1]), **options)
    elif op == "batch_norm":
        if len(shape) != 4:
            raise UnsupportedWorkload("DO batch_norm currently supports BatchNorm2d")
        model = nn.BatchNorm2d(shape[1], **options)
    elif op == "group_norm":
        model = nn.GroupNorm(options.pop("num_groups", 1), shape[1], **options)
    elif op == "layer_norm":
        model = nn.LayerNorm(options.pop("normalized_shape", shape[-1]), **options)
    elif op in ACTIVATIONS:
        kind = "gelu" if op == "gelu_tanh" else op
        if op == "gelu_tanh":
            options["approximate"] = "tanh"
        classes = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "silu": nn.SiLU,
                   "gelu": nn.GELU, "tanh": nn.Tanh, "relu6": nn.ReLU6}
        if options.get("inplace"):
            raise UnsupportedWorkload("In-place leaf mutation is invalid; use a model factory with a non-leaf input")
        model = classes[kind](**options)
    elif op in POOL_OPS or op in INDEX_ALIASES:
        base = op.removesuffix("_with_indices")
        rank = int(base[-2])
        if op in INDEX_ALIASES:
            options["return_indices"] = True
        model = Pool(base, options)
        if "max_unpool" in base:
            original_shape = shape
            with torch.no_grad():
                x, indices = getattr(F, f"max_pool{rank}d")(
                    x, options["kernel_size"], options.get("stride"), options.get("padding", 0), return_indices=True)
            model.indices = indices
            model.options["output_size"] = original_shape
            inputs = (x,)
        if "fractional" in base:
            model.samples = torch.rand(shape[0], shape[1], rank, device=device, dtype=torch.bfloat16)
    elif op == "scaled_dot_product_attention":
        if len(shape) != 4:
            raise ValueError("SDPA shape must be [batch, heads, sequence, head_dim]")
        key_length = options.pop("key_length", shape[-2])
        value_heads = options.pop("kv_heads", shape[1])
        mask = options.pop("mask", "none")
        if mask != "none":
            if mask not in ("boolean", "additive"):
                raise ValueError("attention mask must be none, boolean, or additive")
            value = torch.ones(shape[-2], key_length, device=device, dtype=torch.bool)
            value[:, -1] = False
            options["attn_mask"] = value if mask == "boolean" else torch.where(value, 0.0, -10000.0).to(x.dtype)
        model = SDPA(options)
        k = torch.randn(shape[0], value_heads, key_length, shape[-1], device=device, dtype=x.dtype)
        inputs = (x, k, torch.randn_like(k))
    elif op == "multihead_attention":
        model = MHA(shape[-1], options)
        key_length = model.cross_length or shape[1]
        k = torch.randn(shape[0], key_length, shape[-1], device=device, dtype=x.dtype)
        inputs = (x, k, torch.randn_like(k))
    elif op == "linear_pipeline":
        width = options.pop("out_features", shape[-1] * 4)
        activation = options.pop("activation", "gelu")
        modules = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}
        model = nn.Sequential(nn.Linear(shape[-1], width, **options), modules[activation](), nn.Linear(width, shape[-1]))
    elif op == "conv_pipeline":
        rank = len(shape) - 2
        conv = getattr(nn, f"Conv{rank}d")
        width = options.pop("out_channels", shape[1])
        norm = nn.BatchNorm2d(width) if rank == 2 else nn.GroupNorm(1, width)
        model = nn.Sequential(conv(shape[1], width, **options), norm, nn.ReLU(),
                              getattr(nn, f"AvgPool{rank}d")(2), conv(width, width, 1))
    elif op == "attention_pipeline":
        model = AttentionBlock(shape[-1], options)
    else:
        raise UnsupportedWorkload(f"{op} is a codec diagnostic, not an autograd workload")
    model.to(device)
    if spec["layout"] == "channels_last":
        model.to(memory_format=torch.channels_last if len(shape) == 4 else torch.channels_last_3d)
    model.train(spec["training"])
    for parameter in model.parameters():
        parameter.requires_grad_(spec["parameter_requires_grad"])
    inputs = tuple(value.detach().requires_grad_(spec["input_requires_grad"]) for value in inputs)
    return {"model": model, "inputs": inputs, "kwargs": {}, "loss_fn": None}


def functional_call(source, op, inputs, plan, runtime, scaling_state=None, info=None, weight_cache=None):
    """Explicit functional boundary, with owned delayed state from a prepared adapter."""
    import torch
    from deepopt.opt.nn import functional as doF
    from deepopt.opt.quantization.activation_tensor import FP8Activation

    x = inputs[0]
    training = plan.get("fp8_training")
    controls = {"plan": plan, "_execution_info": info}
    if training and runtime["output_dtype"] != "auto":
        controls["output_dtype"] = (training["fp8_dtypes"]["activation"] if runtime["output_dtype"] == "fp8"
                                     else torch.bfloat16)
    if runtime["output_scaling"] == "provided" and training:
        controls["output_scale"] = runtime["_provided_scale"]
    if scaling_state is not None:
        controls.update(_scaling_state=scaling_state, _state_training=source.training)
    if op in ("linear", "hf_conv1d"):
        for key in ("_execution_info", "output_dtype", "output_scale"):
            controls.pop(key, None)
        controls["_weight_cache"] = weight_cache
        weight = source.weight.T if op == "hf_conv1d" else source.weight
        return doF.do_linear(x, weight, source.bias, **controls)
    if op.startswith("conv"):
        controls["_weight_cache"] = weight_cache
        controls.pop("output_dtype", None)
        controls.pop("output_scale", None)
        settings = {name: getattr(source, name) for name in ("stride", "padding", "dilation", "groups")}
        if "transpose" in op:
            settings["output_padding"] = source.output_padding
        elif source.padding_mode != "zeros":
            from torch.nn import functional as F
            x = F.pad(x, source._reversed_padding_repeated_twice, mode=source.padding_mode)
            settings["padding"] = 0
        return getattr(doF, "do_" + op)(x, source.weight, source.bias, **settings, **controls)
    if op in ACTIVATIONS:
        kind = "gelu" if op == "gelu_tanh" else op
        extra = {"approximate": source.approximate} if kind == "gelu" else {}
        return getattr(doF, "do_" + kind)(x, **extra, **controls)
    if op == "layer_norm":
        return doF.do_layer_norm(x, source.normalized_shape, source.weight, source.bias, source.eps, **controls)
    if op == "group_norm":
        requested = controls.pop("output_dtype", None)
        if controls.pop("output_scale", None) is not None:
            raise UnsupportedWorkload("GroupNorm does not expose a caller-provided output scale")
        if requested in (torch.float8_e4m3fn, torch.float8_e5m2):
            controls["_output_layout"] = "channels"
        return doF.do_group_norm(x, source.num_groups, source.weight, source.bias, source.eps, **controls)
    if op == "batch_norm":
        requested = controls.pop("output_dtype", None)
        if controls.pop("output_scale", None) is not None:
            raise UnsupportedWorkload("BatchNorm does not expose a caller-provided output scale")
        if requested in (torch.float8_e4m3fn, torch.float8_e5m2):
            controls["_output_layout"] = "channels"
        factor = source.momentum
        if source.training and source.track_running_stats:
            source.num_batches_tracked.add_(1)
            if factor is None:
                factor = source.num_batches_tracked.float().reciprocal()
        return doF.do_batch_norm(x, source.running_mean, source.running_var, source.weight, source.bias,
                                source.training or source.running_mean is None, 0.0 if factor is None else factor, source.eps, **controls)
    if op in POOL_OPS or op in INDEX_ALIASES:
        settings = dict(source.options)
        if source.indices is not None:
            settings["indices"] = source.indices
        if source.samples is not None:
            settings["_random_samples"] = source.samples
        return getattr(doF, "do_" + op)(x, **settings, **controls)
    if op == "scaled_dot_product_attention":
        # Native SDPA computes attention. FP8 affects its surrounding projections, not its math.
        if any(isinstance(value, FP8Activation) for value in inputs):
            raise UnsupportedWorkload("SDPA accepts native Q/K/V; use attention_pipeline for FP8 projections")
        return doF.do_scaled_dot_product_attention(*inputs, **source.options, plan=plan)
    raise UnsupportedWorkload(f"Functional binding is unavailable for {op}; use prepared or intercept")


def adapter_state(source, op, settings, compression, device):
    """Create owner state without converting custom functional wrappers."""
    from deepopt.opt.quantization.cache import WeightCache

    state, cache, report = None, WeightCache(), None
    if settings.enabled and settings.scaling_recipe == "delayed":
        from deepopt.opt.scaling.delayed import DelayedScalingState
        from deepopt.opt.policy import fp8_plan

        plan = fp8_plan(settings, activation_compression=compression)
        training = plan["fp8_training"]
        state = DelayedScalingState(training["delayed_scaling"], training["fp8_dtypes"], device=device)
    return state, cache, report
