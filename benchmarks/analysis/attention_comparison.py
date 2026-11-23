"""Controlled attention algorithms/codecs for diagnostics, not production dispatch."""

from contextlib import contextmanager, ExitStack
from unittest.mock import patch

import torch
import torch.nn.functional as F


def reconstruct(tensor, bits, group_size, *, layout="bshd", role=0, codec="current"):
    """Return BHSD reconstructed values; zero bits means dense storage.

    legacy_rules is a Torch reference for the active legacy affine rules:
    per-sample zero padding, half-up code rounding, BF16 scale/min and output.
    It is not a throughput kernel or a claim of bitwise legacy Triton parity.
    """
    if not bits:
        return tensor.detach()
    if layout == "bshd":
        value = tensor.transpose(1, 2)
    elif layout == "saved" and role == 1:
        value = tensor.transpose(-2, -1)
    elif layout in ("bhsd", "saved"):
        value = tensor
    else:
        raise ValueError("layout must be bshd, bhsd or saved")
    with torch.no_grad():
        if codec == "current":
            from deepopt.opt.quantization.activation import _quantize_pack_strided_activation, dequantize_unpack_activation
            packed = _quantize_pack_strided_activation(value, group_size=group_size, bits=bits)
            decoded = dequantize_unpack_activation(packed, dtype=value.dtype).reshape(value.shape)
        elif codec == "legacy_rules":
            flat = value.float().reshape(value.shape[0], -1)
            padded = F.pad(flat, (0, (-flat.shape[1]) % group_size))
            groups = padded.reshape(-1, group_size)
            low, high = groups.amin(-1, keepdim=True), groups.amax(-1, keepdim=True)
            step = (high - low) / (2 ** bits - 1)
            step = torch.where(high > low, step, torch.ones_like(step))
            codes = ((groups - low) * step.reciprocal() + (.5 - 1e-6)).to(torch.int32).clamp(0, 2 ** bits - 1)
            decoded = (codes.float() * step.bfloat16().float() + low.bfloat16().float()).bfloat16()
            decoded = decoded.reshape(value.shape[0], -1)[:, :flat.shape[1]].reshape(value.shape).to(value.dtype)
        else:
            raise ValueError("codec must be current or legacy_rules")
    if layout == "bshd":
        result = decoded.transpose(1, 2)
    else:
        result = decoded.transpose(-2, -1) if layout == "saved" and role == 1 else decoded
    # Preserve the chosen quantization groups/decoded values, while satisfying
    # both Flash backward APIs' unit head-dimension stride requirement.
    return result if result.stride(-1) == 1 else result.contiguous()


def describe_layout(tensors):
    """Portable physical layout/alias metadata; CPU payload clones keep values."""
    storage_ids, records = {}, []
    for value in tensors:
        key = (value.device, value.untyped_storage().data_ptr())
        alias = storage_ids.setdefault(key, len(storage_ids))
        records.append({"shape": list(value.shape), "strides": list(value.stride()),
                        "storage_offset": value.storage_offset(), "alias_group": alias})
    return records


def restore_workload(workload, device):
    """Rebuild source strides and Q/K/V aliasing outside the measured region."""
    values = workload["qkv"]
    layouts = workload.get("layout", {}).get("qkv")
    if layouts is None:
        qkv = [x.to(device).detach().clone().requires_grad_(True) for x in values]
    else:
        if len(layouts) != 3 or len(values) != 3:
            raise ValueError("Attention workloads require three Q/K/V layout records")
        groups = {}
        for value, spec in zip(values, layouts):
            shape, strides, offset = spec["shape"], spec["strides"], spec["storage_offset"]
            if list(value.shape) != shape or len(shape) != len(strides) or offset < 0 or any(s < 0 for s in strides):
                raise ValueError("Invalid saved attention layout")
            span = offset + sum((n - 1) * s for n, s in zip(shape, strides)) + 1
            alias = spec["alias_group"]
            previous = groups.get(alias, (0, value.dtype))
            if previous[1] != value.dtype:
                raise ValueError("Aliased attention operands must share dtype")
            groups[alias] = max(previous[0], span), value.dtype
        storage = {key: torch.empty(size, dtype=dtype, device=device) for key, (size, dtype) in groups.items()}
        qkv = []
        for value, spec in zip(values, layouts):
            view = storage[spec["alias_group"]].as_strided(spec["shape"], spec["strides"], spec["storage_offset"])
            view.copy_(value)
            qkv.append(view.detach().requires_grad_(True))
    upstream = workload["grad_output"]
    spec = workload.get("layout", {}).get("grad_output")
    if spec is None:
        dy = upstream.to(device)
    else:
        if list(upstream.shape) != spec["shape"] or len(spec["strides"]) != upstream.ndim or any(s < 0 for s in spec["strides"]):
            raise ValueError("Invalid saved upstream-gradient layout")
        offset = spec.get("storage_offset", 0)
        if offset < 0:
            raise ValueError("Invalid saved upstream-gradient storage offset")
        span = offset + sum((n - 1) * s for n, s in zip(upstream.shape, spec["strides"])) + 1
        dy = torch.empty(span, dtype=upstream.dtype, device=device).as_strided(upstream.shape, spec["strides"], offset)
        dy.copy_(upstream)
    return qkv, dy


def synthetic_workload(batch, heads, sequence, dim, *, dtype, layout="interleaved", device="cpu"):
    """BHSD API tensors with explicit dense or actual fused-projection storage."""
    if layout == "interleaved":
        source = torch.randn(batch, sequence, 3, heads, dim, dtype=dtype, device=device)
        qkv = tuple(x.transpose(1, 2) for x in source.unbind(2))
    elif layout == "bshd":
        qkv = tuple(torch.randn(batch, sequence, heads, dim, dtype=dtype, device=device).transpose(1, 2) for _ in range(3))
    elif layout == "bhsd":
        qkv = tuple(torch.randn(batch, heads, sequence, dim, dtype=dtype, device=device) for _ in range(3))
    else:
        raise ValueError("Source layout must be bhsd, bshd or interleaved")
    # Timm's output projection supplies a gradient in BSHD order.
    dy = torch.randn(batch, sequence, heads, dim, dtype=dtype, device=device).transpose(1, 2)
    return {"qkv": qkv, "grad_output": dy,
            "layout": {"qkv": describe_layout(qkv), "grad_output": describe_layout((dy,))[0]},
            "source": {"kind": "synthetic", "storage_layout": layout}}


class _SharedMath(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, qhat, khat, vhat, scale):
        p = (q.float() @ k.float().transpose(-2, -1) * scale).softmax(-1)
        ctx.save_for_backward(p, qhat, khat, vhat)
        ctx.scale, ctx.dtype = scale, q.dtype
        return (p @ v.float()).to(q.dtype)

    @staticmethod
    def backward(ctx, dout):
        p, q, k, v = (x.float() for x in ctx.saved_tensors)
        dp = dout.float() @ v.transpose(-2, -1)
        ds = p * (dp - (p * dp).sum(-1, keepdim=True))
        gradients = (ds @ k * ctx.scale, ds.transpose(-2, -1) @ q * ctx.scale,
                     p.transpose(-2, -1) @ dout.float())
        return (*(x.to(ctx.dtype) if needed else None for x, needed in zip(gradients, ctx.needs_input_grad)),
                None, None, None, None)


def attention_case(q, k, v, *, backend, bits=(4, 4, 4), group_size=1024, mode="production",
                   layout="bshd", codec="current", scale=None, dropout_p=0., attn_mask=None,
                   is_causal=False, enable_gqa=False):
    from deepopt.opt.nn import functional as doF
    from deepopt.opt.nn import attention_backend as installed
    from deepopt.opt._autocast import _suspend_fp8_interception

    if dropout_p or attn_mask is not None or is_causal or enable_gqa:
        raise ValueError("Alignment comparisons require dropout=0, no mask, noncausal equal-head attention")
    scale = float(q.shape[-1] ** -.5 if scale is None else scale)
    policy = {"enabled": any(bits), "group_size": group_size, "bits": 2, "attention_bits": bits[0] or 4}
    uniform = len(set(bits)) == 1
    if backend not in ("manual", "torch_flash", "flash_attn_2"):
        raise ValueError("Unknown diagnostic attention backend")
    controlled = mode == "shared" or not uniform or codec != "current"
    with _suspend_fp8_interception():
        if not controlled:
            with patch.object(doF, "_attention_compression_settings", return_value=policy if any(bits) else None):
                if backend == "manual":
                    with torch.autocast(q.device.type, enabled=False), doF._attention_compression(q, k, v):
                        p = (q.float() @ k.float().transpose(-2, -1) * scale).softmax(-1)
                        output = (p @ v.float()).to(q.dtype)
                    installed.notify_attention(q, k, v, output, backend="manual", compression=policy,
                                               saved_layout="transposed_k", scale=scale)
                    return output
                if backend == "flash_attn_2":
                    return installed.flash_attention_2(q, k, v, scale=scale, compression=policy)
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    # CPU Flash is useful for algorithm tests, not CUDA qualification.
                    return doF.do_scaled_dot_product_attention(q, k, v, scale=scale, attention_backend="native")
        # All three algorithms see exactly these BF16 reconstructed operands.
        qhat, khat, vhat = [reconstruct(x.detach(), b, group_size, layout=layout, role=i, codec=codec)
                            for i, (x, b) in enumerate(zip((q, k, v), bits))]
        if backend == "manual":
            output = _SharedMath.apply(q, k, v, qhat, khat, vhat, scale)
        elif backend == "flash_attn_2":
            from deepopt.opt.backends._shared.activation_codec import activation_codec_scope
            def save_in_bshd(tensors, unused_bits, unused_group):
                return tuple(((x.transpose(1, 2),), None) for x in (qhat, khat, vhat))

            # Exercise the actual installed custom backward; only its saved
            # operands are substituted. Forward retains the original Q/K/V.
            with patch.object(installed, "_pack_operands", side_effect=save_in_bshd), \
                    activation_codec_scope(attention_pack="separate"):
                output = installed.flash_attention_2(q, k, v, scale=scale, compression=policy)
            return output  # Installed observer has already recorded this route.
        else:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            operands = tuple(x.clone() for x in (q, k, v))
            replacements = {x.data_ptr(): y for x, y in zip(operands, (qhat, khat, vhat))}

            def save(tensor):
                # Flash output can have Q's shape; storage identity matters.
                return replacements.get(tensor.data_ptr(), tensor.detach())

            with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autograd.graph.saved_tensors_hooks(save, lambda x: x):
                output = F.scaled_dot_product_attention(*operands, scale=scale)
        installed.notify_attention(q, k, v, output, backend=backend, compression=policy,
                                   saved_layout=layout, codec=codec, shared_reconstruction=True, scale=scale)
        return output


@contextmanager
def model_attention_case(spec):
    """Route both timm SDPA and DeepOpt MHA through a diagnostic case."""
    from deepopt.opt.nn import functional as doF
    from deepopt.opt.nn import attention as mha

    def selected(q, k, v, attn_mask=None, dropout_p=0., is_causal=False, *, scale=None, enable_gqa=False, **unused):
        # Scaled dot-product attention under AMP converts FP32 projection outputs.
        if torch.is_autocast_enabled(q.device.type):
            dtype = torch.get_autocast_dtype(q.device.type)
            q, k, v = (x.to(dtype) if x.dtype == torch.float32 else x for x in (q, k, v))
        # attention_case's torch_flash path must reach the original function.
        with patch.object(doF, "do_scaled_dot_product_attention", original):
            return attention_case(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                                  is_causal=is_causal, scale=scale, enable_gqa=enable_gqa, **spec)

    original = doF.do_scaled_dot_product_attention
    with ExitStack() as stack:
        stack.enter_context(patch.object(doF, "do_scaled_dot_product_attention", side_effect=selected))
        stack.enter_context(patch.object(mha, "do_scaled_dot_product_attention", side_effect=selected))
        yield
