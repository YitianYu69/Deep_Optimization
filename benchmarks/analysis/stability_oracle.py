"""Bounded ordinary-PyTorch codec oracle; never launch a DeepOpt Triton codec."""

from contextlib import contextmanager
from functools import wraps
from importlib import import_module
import sys

import torch


MAX_ELEMENTS = 1 << 20
ORACLE_REVISION = 2


def _layout(shape, group_size, block_shape, bits):
    rows, cols = shape
    br, bc = block_shape or (0, 0)
    elements = br * bc if block_shape else group_size
    groups = ((rows + br - 1) // br) * ((cols + bc - 1) // bc) if block_shape else (
        rows * cols + elements - 1) // elements
    return groups, elements, (elements + 32 // bits - 1) // (32 // bits)


def _coordinates(shape, group_size, block_shape, start, stop, device):
    rows, cols = shape
    elements = block_shape[0] * block_shape[1] if block_shape else group_size
    group = torch.arange(start, stop, device=device)[:, None]
    offset = torch.arange(elements, device=device)[None, :]
    if block_shape:
        br, bc = block_shape
        columns = (cols + bc - 1) // bc
        r, c = group // columns * br + offset // bc, group % columns * bc + offset % bc
    else:
        index = group * elements + offset
        r, c = index // cols, index % cols
    return r, c, (r < rows) & (c < cols)


@torch.no_grad()
def pack(x, group_size=1024, block_shape=None, *, bits=2):
    """RTNE affine codec with masked tails, bounded int64 intermediates."""
    from deepopt.opt.quantization.activation import PackedActivation

    shape = tuple(x.shape)
    groups, elements, words = _layout(shape, group_size, block_shape, bits)
    payload = torch.empty((groups, words), dtype=torch.int32, device=x.device)
    steps = torch.empty(groups, dtype=torch.float32, device=x.device)
    minima = torch.empty_like(steps)
    batch = max(1, MAX_ELEMENTS // elements)
    qmax, vpw = (1 << bits) - 1, 32 // bits
    # CUDA Tensor / Python scalar folds to reciprocal multiplication. Keep
    # the divisor on-device so the oracle matches the production div.rn path
    # even when a one-ULP scale difference flips a quantization midpoint.
    divisor = torch.tensor(qmax, dtype=torch.float32, device=x.device)
    shifts = torch.arange(vpw, device=x.device, dtype=torch.int64) * bits
    for start in range(0, groups, batch):
        stop = min(groups, start + batch)
        r, c, valid = _coordinates(shape, group_size, block_shape, start, stop, x.device)
        value = x[r.clamp_max(shape[0] - 1), c.clamp_max(shape[1] - 1)].float()
        low = value.masked_fill(~valid, float("inf")).amin(1)
        high = value.masked_fill(~valid, float("-inf")).amax(1)
        span = high - low
        step = torch.where(torch.isinf(span), high / divisor - low / divisor, span / divisor)
        if bits == 1:
            step = torch.where(torch.isinf(span), -(high * .5 - low * .5), step)
        difference = value - low[:, None]
        normalized = torch.where(torch.isinf(difference),
            value / step[:, None] - low[:, None] / step[:, None], difference / step[:, None])
        if bits == 1:
            normalized = torch.where(step[:, None] < 0,
                (value * .5 - low[:, None] * .5) / -step[:, None], normalized)
        nonconstant = step[:, None] != 0 if bits == 1 else step[:, None] > 0
        codes = torch.where(nonconstant, normalized, 0.).clamp(0, qmax).round().long()
        padded = torch.zeros((stop - start, words * vpw), dtype=torch.int64, device=x.device)
        padded[:, :elements] = codes.masked_fill(~valid, 0)
        payload[start:stop] = (padded.reshape(stop - start, words, vpw) << shifts).sum(2).int()
        steps[start:stop], minima[start:stop] = step, low
    return PackedActivation(payload, steps, minima, shape, group_size, block_shape, bits)


@torch.no_grad()
def unpack(packed, dtype=torch.bfloat16, transpose=False, *, row_alignment=16):
    rows, cols = packed.shape
    groups, elements, words = _layout(packed.shape, packed.group_size, packed.block_shape, packed.bits)
    shape = (cols, rows) if transpose else (rows, cols)
    stride = ((shape[1] + row_alignment - 1) // row_alignment) * row_alignment if packed.data.is_cuda else shape[1]
    output = torch.empty_strided(shape, (stride, 1), device=packed.data.device, dtype=dtype)
    vpw, qmax = 32 // packed.bits, (1 << packed.bits) - 1
    shifts = torch.arange(vpw, device=output.device, dtype=torch.int64) * packed.bits
    for start in range(0, groups, max(1, MAX_ELEMENTS // elements)):
        stop = min(groups, start + max(1, MAX_ELEMENTS // elements))
        codes = ((packed.data[start:stop].long()[:, :, None] >> shifts) & qmax)
        codes = codes.reshape(stop - start, words * vpw)[:, :elements].float()
        step, low = packed.step[start:stop, None], packed.minimum[start:stop, None]
        product = codes * step
        value = torch.where(torch.isinf(product), (codes * (step * .5) + low * .5) * 2, product + low)
        if packed.bits == 1:
            value = torch.where(step < 0, (low * .5 + codes * -step) * 2, value)
        if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            value = value.clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max)
        r, c, valid = _coordinates(packed.shape, packed.group_size, packed.block_shape, start, stop, output.device)
        if transpose:
            output[c[valid], r[valid]] = value[valid].to(dtype)
        else:
            output[r[valid], c[valid]] = value[valid].to(dtype)
    return output


def pack_strided(x, group_size=1024, *, bits=2, source_dtype=None):
    from math import prod
    value = x.to(source_dtype or x.dtype)
    return pack(value.reshape(prod(x.shape[:-1]), x.shape[-1]), group_size, bits=bits)


def unpack_spatial(packed, shape, *, dtype, memory_format=torch.contiguous_format):
    from deepopt.opt.nn.convolution.geometry import restore_matrix
    return restore_matrix(unpack(packed, dtype=dtype, row_alignment=1), shape, memory_format)


@contextmanager
def oracle_scope():
    """Replace imported aliases too; restore everything, including on failure.

    Native GEMMs, convolutions and SDPA stay on the selected device. Only
    custom normalization/activation kernels and affine codecs use the oracle.
    """
    from deepopt.opt.quantization import activation as codec
    from deepopt.opt.backends import activation as activations
    from deepopt.opt.backends import normalization as normalization
    from deepopt.opt.nn.autograd.normalization import layer

    # Load consumers before replacing their from-import aliases.
    for name in ("deepopt.opt.nn.autograd.native_linear", "deepopt.opt.nn.autograd.convolution.native",
                 "deepopt.opt.backends.reference.normalization.layer", "deepopt.opt.nn.normalization.recenter",
                 "deepopt.opt.backends.reference.activation.elementwise", "deepopt.opt.quantization.attention"):
        import_module(name)
    replacements = {
        codec.quantize_pack_activation: pack, codec.dequantize_unpack_activation: unpack,
        codec._quantize_pack_strided_activation: pack_strided,
        codec._dequantize_unpack_spatial_activation: unpack_spatial,
        layer.executor: lambda *a, **k: import_module("deepopt.opt.backends.reference.normalization.layer"),
        activations.executor: lambda x, kind="relu": import_module(
            "deepopt.opt.backends.reference.activation." + ("relu" if kind == "relu" else "elementwise")),
        normalization.executor: lambda *a, **k: import_module("deepopt.opt.backends.reference.normalization"),
    }
    with _aliases(replacements):
        yield


@contextmanager
def _aliases(replacements):
    changes = []
    try:
        for name, module in list(sys.modules.items()):
            if name.startswith("deepopt.opt.") and module is not None:
                for attr, value in list(vars(module).items()):
                    if callable(value):
                        try:
                            replacement = replacements.get(value)
                        except TypeError:
                            continue
                        if replacement is not None:
                            changes.append((module, attr, value))
                            setattr(module, attr, replacement)
        yield
    finally:
        inverse = {value: original for original, value in replacements.items()}
        # A lazily imported consumer may have copied an oracle alias after the
        # initial scan. Restore those aliases too before the production case.
        for name, module in list(sys.modules.items()):
            if name.startswith("deepopt.opt.") and module is not None:
                for attr, value in list(vars(module).items()):
                    if callable(value):
                        try:
                            original = inverse.get(value)
                        except TypeError:
                            continue
                        if original is not None:
                            setattr(module, attr, original)
        for module, attr, value in reversed(changes):
            setattr(module, attr, value)


@contextmanager
def codec_audit_scope(records):
    """Compare exact operands/payloads and decoded values for public codecs.

    Fused LayerNorm/GELU do not call these entry points; local operator checks
    cover them. These audits do not measure throughput.
    """
    from deepopt.opt.quantization import activation as codec
    for name in ("deepopt.opt.nn.autograd.native_linear", "deepopt.opt.nn.autograd.convolution.native",
                 "deepopt.opt.quantization.attention"):
        import_module(name)
    active = False

    def audited(original, reference, label):
        @wraps(original)
        def call(*args, **kwargs):
            nonlocal active
            if active:
                return original(*args, **kwargs)
            active = True
            try:
                actual = original(*args, **kwargs)
                expected = reference(*args, **kwargs)
                finite = bool(torch.isfinite(actual.step).all() and torch.isfinite(actual.minimum).all()
                              and torch.isfinite(expected.step).all() and torch.isfinite(expected.minimum).all())
                words = int((actual.data != expected.data).sum())
                records.append({"api": label, "direction": "pack", "source_shape": list(args[0].shape),
                    "source_dtype": str(args[0].dtype), "shape": list(actual.shape),
                    "group_size": actual.group_size, "bits": actual.bits, "differing_words": words,
                    "total_words": actual.data.numel(), "metadata_finite": finite,
                    "step_max_abs": float((actual.step - expected.step).abs().max()) if actual.step.numel() and finite else None,
                    "minimum_max_abs": float((actual.minimum - expected.minimum).abs().max()) if actual.minimum.numel() and finite else None})
                return actual
            finally:
                active = False
        return call

    def audited_unpack(original, reference, label):
        @wraps(original)
        def call(*args, **kwargs):
            nonlocal active
            if active:
                return original(*args, **kwargs)
            active = True
            try:
                actual, expected = original(*args, **kwargs), reference(*args, **kwargs)
                from math import prod
                chunk = max(1, MAX_ELEMENTS // prod(actual.shape[1:]))
                different, maximum, finite = 0, 0., True
                for start in range(0, actual.shape[0], chunk):
                    a, b = actual[start:start + chunk].float(), expected[start:start + chunk].float()
                    different += int((a != b).sum())
                    finite &= bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                    if finite:
                        maximum = max(maximum, float((a - b).abs().max()))
                packed = args[0]
                records.append({"api": label, "direction": "unpack", "shape": list(actual.shape),
                    "output_dtype": str(actual.dtype), "group_size": packed.group_size, "bits": packed.bits,
                    "differing_values": different, "total_values": actual.numel(), "values_finite": finite,
                    "max_abs": maximum if finite else None})
                return actual
            finally:
                active = False
        return call

    replacements = {codec.quantize_pack_activation: audited(codec.quantize_pack_activation, pack, "affine"),
                    codec._quantize_pack_strided_activation: audited(codec._quantize_pack_strided_activation, pack_strided, "strided"),
                    codec.dequantize_unpack_activation: audited_unpack(codec.dequantize_unpack_activation, unpack, "affine_unpack"),
                    codec._dequantize_unpack_spatial_activation: audited_unpack(
                        codec._dequantize_unpack_spatial_activation, unpack_spatial, "spatial_unpack")}
    with _aliases(replacements):
        yield
