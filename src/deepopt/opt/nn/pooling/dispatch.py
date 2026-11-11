"""Shared pooling frontend: native signatures, precision and compression policy."""

from math import isfinite

import torch

from .spec import SPECS
from .geometry import geometry
from ...quantization.activation_tensor import FP8Activation


FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)
FLOAT = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def pooling(kind, x, args, kwargs, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
            planned_layout=None, scaling_state=None, state_training=True, execution_info=None):
    from ..functional import _prepare_linear_plan
    from ..._autocast import _get_fp8_plan, _get_activation_compression
    from ..._execution import finite_checks_enabled
    from ..autograd.pooling import PoolFunction

    spec = SPECS[kind]
    options = spec.arguments(args, kwargs)
    if not isinstance(x, torch.Tensor):
        raise TypeError("pooling input must be a Tensor")
    if x.layout != torch.strided or x.is_nested or x.dtype not in (*FLOAT, *FP8):
        return spec.native(x, **options)
    active, compression = _get_fp8_plan(x.device.type), _get_activation_compression(x.device.type)
    if isinstance(plan, dict) and "fp8" in plan:
        selected = plan
    elif active is False:
        selected = {"fp8": False, "activation_compression": compression or {}}
    else:
        selected = active if active is not None else plan
        if selected is not None and compression is not None:
            selected = {**selected, "fp8_training": {**selected["fp8_training"], "activation_compression": compression}}
    checked = _prepare_linear_plan(selected or {"fp8": False, "activation_compression": compression or {}})
    training = checked.get("fp8_training")
    compression = (training or checked)["activation_compression"]
    if not training and not compression["enabled"] and not isinstance(x, FP8Activation) and output_dtype is None and output_scale is None:
        return spec.native(x, **options)
    if x.device.type not in ("cpu", "cuda", "meta"):
        raise ValueError("optimized pooling requires CPU, CUDA or meta input")
    input_dtype = x.dtype if x.dtype not in FP8 else torch.bfloat16
    logical_dtype = input_dtype
    if (spec.family == "power" and isfinite(float(options["norm_type"])) and x.device.type == "cuda"
            and torch.is_autocast_enabled("cuda") and logical_dtype != torch.float64):
        logical_dtype = torch.float32
    if (x.device.type == "cpu" and torch.is_autocast_enabled("cpu") and input_dtype != torch.float64
            and (spec.rank == 3 or spec.family in ("unpool", "fractional_max"))):
        logical_dtype = torch.float32
    dtype = output_dtype or (training["out_dtype"] if training else None) or (
        training["fp8_dtypes"]["activation"] if training and planned_layout is not None else logical_dtype)
    if dtype not in (*FLOAT, *FP8):
        raise ValueError("pooling output_dtype must be a supported floating-point dtype")
    native_layout = "channels" if (x.ndim == 4 and x.is_contiguous(memory_format=torch.channels_last) or
                                    x.ndim == 5 and x.is_contiguous(memory_format=torch.channels_last_3d)) else "last"
    layout = (output_layout or planned_layout or (x._activation_layout if isinstance(x, FP8Activation) else "last")) if dtype in FP8 else output_layout or native_layout
    if layout not in ("last", "channels"):
        raise ValueError("pooling layout must be last or channels")
    g = geometry(kind, x, options, layout)
    blocks = checked["block_shapes"]["activation"] if training else None
    if blocks is not None and x.device.type == "cuda" and any(v & (v - 1) for v in blocks):
        raise ValueError("CUDA pooling FP8 blocks require power-of-two dimensions")
    if output_scale is not None:
        rows, cols = g.transport.matrix_shape
        grid = (1, 1) if blocks is None else ((rows + blocks[0] - 1) // blocks[0], (cols + blocks[1] - 1) // blocks[1])
        if dtype not in FP8 or not isinstance(output_scale, torch.Tensor) or output_scale.dtype != torch.float32 or output_scale.requires_grad:
            raise TypeError("pooling output_scale requires non-differentiable FP32 multipliers and FP8 output")
        if output_scale.device != x.device or tuple(output_scale.shape) != grid or not output_scale.is_contiguous():
            raise ValueError("pooling output_scale must match its output scale grid and device")
    if dtype in FP8 and training and training["scaling_recipe"] == "delayed" and output_scale is None:
        if scaling_state is None:
            raise ValueError("delayed pooling FP8 output requires a prepared adapter or explicit scale")
        if dtype != scaling_state.dtypes["activation"]:
            raise ValueError("pooling output format differs from its prepared delayed state")
    samples = None
    if spec.family == "fractional_max":
        samples = options.get("_random_samples")
        if samples is None:
            samples = torch.rand(g.batch, g.channels, g.rank, dtype=input_dtype, device=x.device)
        if (not isinstance(samples, torch.Tensor) or tuple(samples.shape) != (g.batch, g.channels, g.rank)
                or samples.device != x.device or samples.dtype != input_dtype or samples.requires_grad):
            raise ValueError("fractional samples must have native input dtype/device and shape (N,C,rank), without gradients")
    backward = torch.is_grad_enabled() and x.requires_grad
    settings = dict(options=options, dtype=dtype, logical_dtype=logical_dtype, block_shape=blocks,
                    input_block_shape=checked["block_shapes"]["activation"] if training else None,
                    compression=compression, scale=output_scale, samples=samples, need_state=backward and bool(x.numel()),
                    backend=training["backend"] if training else "triton" if x.device.type == "cuda" else "reference",
                    check_finite=finite_checks_enabled() and (dtype in FP8 or compression["enabled"] and spec.family == "power"))
    return PoolFunction.apply(x, g, settings, scaling_state if output_scale is None else None,
                              state_training and torch.is_grad_enabled(), execution_info)
