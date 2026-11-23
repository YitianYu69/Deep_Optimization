"""Optional FlashAttention-2 kernels with explicit saved-Q/K/V compression.

The native backend never imports flash_attn. Public tensors use BHSD; the
installed FA2 kernels and saved activation groups use BSHD. The original
forward output/LSE/RNG state remain exact, matching the legacy adapter.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from functools import lru_cache
import math

import torch
from deepopt.kernel_policy import kernel_forward, kernel_backward
from torch.autograd.function import once_differentiable
from deepopt._attention_policy import effective_attention_bits

from ..quantization.activation import (
    PackedActivation, _quantize_pack_strided_activation, dequantize_unpack_activation,
    _joint_attention_pack_eligible, _quantize_pack_qkv_activation,
)
from ..backends._shared.activation_codec import attention_pack_plan, attention_pack_dispatch, attention_gradient_layout, attention_gradient_plan


_observer = ContextVar("deepopt_attention_observer", default=None)


@contextmanager
def observe_attention(callback):
    """Observe eager/capture routing and operands; callbacks never run on replay."""
    token = _observer.set(callback)
    try:
        from .._autocast import _linear_mode
        with _linear_mode():
            yield
    finally:
        _observer.reset(token)


def observing_attention():
    return _observer.get() is not None


def notify_attention(q, k, v, output, *, backend, compression, **metadata):
    callback = _observer.get()
    if callback is not None:
        callback((q, k, v), output, {"backend": backend, "compression": compression, **metadata})


@lru_cache(maxsize=1)
def _kernels():
    try:
        import flash_attn
        from flash_attn.flash_attn_interface import _wrapped_flash_attn_forward, _wrapped_flash_attn_backward
    except (ImportError, OSError) as error:
        raise RuntimeError("flash_attn_2 requires the optional flash-attn CUDA extension. Install in the active "
                           "PyTorch environment: MAX_JOBS=4 python -m pip install flash-attn==2.8.3.post2 "
                           "--no-build-isolation --no-deps") from error
    if str(flash_attn.__version__).split(".")[0] != "2":
        raise RuntimeError("flash_attn_2 requires FlashAttention version 2; validated with 2.8.3.post2")
    return _wrapped_flash_attn_forward, _wrapped_flash_attn_backward


def _validate_and_cast(q, k, v, attn_mask, is_causal, enable_gqa, dropout_p, scale):
    if attn_mask is not None or is_causal or enable_gqa:
        raise ValueError("flash_attn_2 currently supports unmasked, noncausal attention with equal Q/K/V head counts; "
                         "select runtime.attention.backend='native' for masks, causal attention or GQA")
    if any(not isinstance(x, torch.Tensor) or x.is_nested or x.layout != torch.strided or x.ndim != 4
           for x in (q, k, v)):
        raise ValueError("flash_attn_2 requires dense rank-4 Q/K/V in [batch, heads, sequence, head_dim] order")
    if q.device.type != "cuda" or torch.version.hip is not None or any(x.device != q.device for x in (k, v)):
        raise ValueError("flash_attn_2 requires Q/K/V on one NVIDIA CUDA device")
    if (any(not x.numel() for x in (q, k, v)) or q.shape[:2] != k.shape[:2] or k.shape != v.shape
            or q.shape[-1] != k.shape[-1] or not 0 < q.shape[-1] <= 256 or q.shape[-1] % 8):
        raise ValueError("flash_attn_2 requires nonempty matching batch/heads/dim, equal K/V shapes, "
                         "and head_dim a multiple of 8 up to 256")
    if isinstance(dropout_p, bool) or not isinstance(dropout_p, (int, float)) or not 0 <= dropout_p < 1:
        raise ValueError("flash_attn_2 dropout_p must be in [0, 1)")
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale)):
        raise ValueError("flash_attn_2 scale must be a finite scalar")
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        q, k, v = (x.to(dtype) if x.dtype == torch.float32 else x for x in (q, k, v))
    if q.dtype not in (torch.float16, torch.bfloat16) or any(x.dtype != q.dtype for x in (k, v)):
        raise ValueError("flash_attn_2 requires matching BF16/FP16 Q/K/V (or FP32 inputs under BF16/FP16 AMP)")
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        raise ValueError("flash_attn_2 requires an Ampere or newer NVIDIA GPU")
    return q, k, v


def _pack_operand(tensor, bits, group_size):
    if not bits:
        return (tensor,), None
    packed = _quantize_pack_strided_activation(tensor, group_size=group_size, bits=bits)
    header = (packed.shape, packed.group_size, packed.block_shape, packed.bits, tuple(tensor.shape), tensor.dtype)
    return (packed.data, packed.step, packed.minimum), header


def _pack_operands(inputs, bits, group_size):
    selection = _packing_selection(inputs, bits, group_size)
    with attention_pack_dispatch(selection):
        if bits and selection["implementation"] == "joint":
            packed = _quantize_pack_qkv_activation(inputs, group_size, bits=bits)
            return tuple(((p.data, p.step, p.minimum),
                          (p.shape, p.group_size, p.block_shape, p.bits, tuple(x.shape), x.dtype))
                         for x, p in zip(inputs, packed))
        return tuple(_pack_operand(x, bits, group_size) for x in inputs)


def _packing_selection(inputs, bits, group_size):
    x = inputs[0]
    selection = attention_pack_plan(x.shape, x.stride(), x.dtype, bits, group_size, x.device)
    if not bits or not _joint_attention_pack_eligible(inputs, group_size):
        selection = {**selection, "implementation": "separate"}
    return selection


def _restore_operand(values, header):
    if header is None:
        return values[0]
    shape, group, block, bits, original_shape, dtype = header
    packed = PackedActivation(*values, shape, group, block, bits)
    return dequantize_unpack_activation(packed, dtype=dtype).reshape(original_shape)


def _dense_gradient_layout(shape, strides):
    """Reuse a dense input permutation without allocating gaps from QKV views."""
    if strides[-1] != 1:
        return False
    expected = 1
    for stride, size in sorted((s, n) for n, s in zip(shape, strides) if n > 1):
        if stride != expected:
            return False
        expected *= size
    return True


def _gradient_buffers(values, strides, layout="auto"):
    if layout == "interleaved":
        if any(x.shape != values[0].shape for x in values):
            raise ValueError("Interleaved QKV gradients require equal Q/K/V shapes")
        b, t, h, d = values[0].shape
        return torch.empty((b, t, 3, h, d), dtype=values[0].dtype, device=values[0].device).unbind(2)
    if layout == "bhsd":
        return [torch.empty((x.shape[0], x.shape[2], x.shape[1], x.shape[3]), dtype=x.dtype,
                            device=x.device).transpose(1, 2) for x in values]
    if layout == "bshd":
        return [torch.empty(x.shape, dtype=x.dtype, device=x.device) for x in values]
    return [torch.empty_strided(x.shape, stride, dtype=x.dtype, device=x.device)
            if _dense_gradient_layout(x.shape, stride) else torch.empty(x.shape, dtype=x.dtype, device=x.device)
            for x, stride in zip(values, strides)]


class _FlashAttention2(torch.autograd.Function):
    @staticmethod
    @kernel_forward
    def forward(ctx, q, k, v, dropout_p, scale, deterministic, compression):
        ctx.gradient_layout = attention_gradient_plan(q.shape,tuple(x.stride() for x in (q,k,v)),q.dtype,q.device,compression)
        if ctx.gradient_layout == "interleaved" and (q.shape != k.shape or k.shape != v.shape):
            raise ValueError("Interleaved QKV gradients require equal Q/K/V shapes")
        forward, _ = _kernels()
        out, lse, _, rng = forward(q, k, v, dropout_p, scale, False, -1, -1, 0., None, False)
        bits = effective_attention_bits(compression)
        saved, headers = [], []
        # All inputs are BSHD here: K itself, never the manual Q @ K.T view.
        for values, header in _pack_operands((q, k, v), bits, compression["group_size"] if compression else 512):
            saved.extend(values)
            headers.append(header)
        ctx.save_for_backward(*saved, out, lse, rng)
        ctx.headers = headers
        ctx.gradient_strides = tuple(x.stride() for x in (q, k, v))
        ctx.dropout_p, ctx.scale, ctx.deterministic = dropout_p, scale, deterministic
        return out

    @staticmethod
    @once_differentiable
    @kernel_backward
    def backward(ctx, dout):
        _, backward = _kernels()
        values, offset, qkv = ctx.saved_tensors, 0, []
        for header in ctx.headers:
            count = 1 if header is None else 3
            qkv.append(_restore_operand(values[offset:offset + count], header))
            offset += count
        # The extension accepts outer strides but requires contiguous head_dim.
        # Real decoded operands already satisfy this; diagnostic substitutions
        # and nonstandard dense-save views may require materialization.
        q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in qkv)
        out, lse, rng = values[offset:]
        dq, dk, dv = _gradient_buffers((q, k, v), ctx.gradient_strides, ctx.gradient_layout)
        dout = dout if dout.stride(-1) == 1 else dout.contiguous()
        backward(dout, q, k, v, out, lse, dq, dk, dv, ctx.dropout_p,
                 ctx.scale, False, -1, -1, 0., None, ctx.deterministic, rng_state=rng)
        return (*(g if needed else None for g, needed in zip((dq, dk, dv), ctx.needs_input_grad[:3])),
                None, None, None, None)


def flash_attention_2(q, k, v, *, attn_mask=None, dropout_p=0., is_causal=False, scale=None,
                      enable_gqa=False, compression=None):
    q, k, v = _validate_and_cast(q, k, v, attn_mask, is_causal, enable_gqa, dropout_p, scale)
    compression = compression if compression and compression["enabled"] else None
    scale = float(q.shape[-1] ** -.5 if scale is None else scale)
    deterministic = torch.are_deterministic_algorithms_enabled()
    # Transpose views have unit last stride; FA2 supports their outer strides.
    inputs = tuple(x.transpose(1, 2) for x in (q, k, v))
    save_backward = torch.is_grad_enabled() and any(x.requires_grad for x in inputs)
    with torch.autocast("cuda", enabled=False):
        if save_backward:
            out = _FlashAttention2.apply(*inputs, float(dropout_p), scale, deterministic, compression)
        else:
            out = _kernels()[0](*inputs, float(dropout_p), scale, False, -1, -1, 0., None, False)[0]
    output = out.transpose(1, 2)
    selection = (_packing_selection(inputs, effective_attention_bits(compression),
                                    compression["group_size"]) if save_backward and compression else None)
    notify_attention(q, k, v, output, backend="flash_attn_2", compression=compression,
                     saved_layout="bshd", dropout_p=float(dropout_p), scale=scale,
                     packing=("none" if not save_backward else "dense" if compression is None else
                              selection["implementation"]), packing_dispatch=selection,
                     gradient_layout=(attention_gradient_plan(inputs[0].shape,tuple(x.stride() for x in inputs),
                                                              inputs[0].dtype,inputs[0].device,compression) if save_backward else "none"),
                     gradient_layout_requested=attention_gradient_layout())
    return output


def prepare_attention_model(model, *, config, compression):
    """Prepare timm/nn.MHA routing before capture while preserving parameters."""
    from deepopt._config_resolution import resolve_config
    from deepopt.config import attention_config_from_dict
    from .attention import DOMultiheadAttention

    snapshot = resolve_config(config)
    previous = getattr(model, "_deepopt_attention_config", None)
    raw = snapshot.get("runtime", {}).get("attention")
    settings = previous if raw is None and previous is not None else attention_config_from_dict(raw or {})
    if previous is not None and previous != settings:
        raise ValueError("Attention backend differs from the prepared model; reconstruct and recapture the model")
    if previous == settings:
        return model
    if settings.backend == "native":
        return model
    converted, replacements = [], {}

    def convert(module, path):
        if id(module) in replacements:
            return replacements[id(module)]
        if settings.backend == "flash_attn_2":
            if (getattr(module, "_compiled_call_impl", None) is not None
                    or getattr(module.forward, "_torchdynamo_orig_callable", None) is not None):
                raise ValueError("Select and prepare the attention backend before compilation")
            if getattr(module.forward, "_deepopt_cuda_capture", False):
                raise ValueError("Select and prepare the attention backend before CUDA Graph capture")
            if type(module) is torch.nn.MultiheadAttention:
                module_new = DOMultiheadAttention(module, plan={"fp8": False, "activation_compression": asdict(compression)})
                replacements[id(module)] = module_new
                module = module_new
                converted.append(path or "<root>")
            if isinstance(module, DOMultiheadAttention):
                module.attention_backend = settings.backend
            if type(module).__module__.startswith("timm.") and hasattr(module, "fused_attn"):
                module.fused_attn = True
                converted.append(path or "<root>")
        # The model may share modules; visit each original module only once.
        replacements.setdefault(id(module), module)
        for name, child in tuple(module._modules.items()):
            if child is not None:
                module._modules[name] = convert(child, f"{path}.{name}" if path else name)
        return module

    model = convert(model, "")
    model._deepopt_attention_config = settings
    model._deepopt_attention_report = {"backend": settings.backend, "prepared_modules": converted}
    return model
