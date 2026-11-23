"""Layout-only host plumbing and TileLang casts shared by semantic executors.

These correctness kernels deliberately use scalar ownership. They provide a
portable implementation before shape-specific tiled schedules are introduced.
Pure TileLang execution keeps all arithmetic in this backend. Mixed policies
dispatch standalone codecs and quantizers to their explicitly selected family.
"""

from functools import lru_cache
from importlib import import_module
from math import isfinite
import torch

from .._runtime import runtime, compile_kernel, dtype_name, require_cuda


FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)


def auxiliary_module(family, module):
    """Honor Mixed auxiliary routes while direct low-level calls stay TileLang."""
    from deepopt.kernel_policy import current_kernel_policy
    if current_kernel_policy().backend == "mixed":
        from deepopt.opt.backends.dispatch import backend_module
        return backend_module(family, module)
    return import_module(f"deepopt.opt.backends.tilelang.{module}")


@lru_cache(maxsize=256)
def _copy_kernel(n, source, target, device):
    _, T = runtime()

    @T.prim_func
    def kernel(A: T.Tensor((n,), source), B: T.Tensor((n,), target)):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
            for i in T.Parallel(256):
                p = T.cast(bx, "int64") * 256 + i
                if p < n:
                    B[p] = T.cast(A[p], target)

    return compile_kernel(kernel, device)


def cast(value, dtype):
    """Cast in TileLang; contiguous/view operations only change host layout."""
    require_cuda(value)
    source = value.contiguous()
    result = torch.empty(value.shape, device=value.device, dtype=dtype)
    if value.numel():
        _copy_kernel(value.numel(), dtype_name(value.dtype), dtype_name(dtype), str(value.device))(
            source.view(-1), result.view(-1))
    return result


def dense(value):
    from deepopt.opt.quantization.activation_tensor import FP8Activation, restore_activation
    if isinstance(value, FP8Activation):
        dequantize = auxiliary_module("quantization", "quantization.fp8").dequantize
        matrix = dequantize(value._data, value._scale_inv, value._block_shape)
        return restore_activation(matrix, tuple(value.shape), value._activation_layout)
    return cast(value, torch.float32)


def check_finite(values, message="TileLang operation requires finite values"):
    amax_scale = auxiliary_module("scaling", "quantization.fp8").amax_scale
    if values.numel():
        try:
            amax_scale(values.reshape(1, -1), 448.0, None, True)
        except ValueError as error:
            raise ValueError(message) from error


@lru_cache(maxsize=256)
def _scale_metadata_kernel(count, device):
    _, T = runtime()

    @T.prim_func
    def kernel(S: T.Tensor((count,), "float32"), I: T.Tensor((count,), "float32"),
               Snapshot: T.Tensor((2,), "float32"), Errors: T.Tensor((count,), "int32")):
        with T.Kernel(T.ceildiv(count, 128), threads=128) as bx:
            for t in T.Parallel(128):
                p = T.cast(bx, "int64") * 128 + t
                if p < count:
                    I[p] = T.ieee_fdiv(T.float32(1.0), S[p])
                    Errors[p] = T.cast((S[p] <= 0) | (S[p] != S[p]) | (T.abs(S[p]) == T.infinity("float32")), "int32")
                    if p == 0:
                        Snapshot[0] = S[p]
                        Snapshot[1] = T.ieee_fdiv(T.float32(1.0), S[p])

    return compile_kernel(kernel, device)


def observations(values):
    implementation = auxiliary_module("scaling", "quantization.delayed")
    if ".triton." in implementation.__name__:
        _, _, _, observed, _ = implementation.convert(values, torch.ones((1, 1), device=values.device),
            torch.float8_e4m3fn, normal=True, transposed=False, bias=False, observe=True,
            check_finite=False, autotune=False)
        return observed
    from ..scaling.reduction import block_amax
    return block_amax(values, 32, 32, minimum=0).reshape(-1)


def _provided_conversion(values, dtype, blocks, scales, observe, capture_scale, checked, reciprocal=False):
    """Compose existing selected-family converters for rectangular scale grids."""
    implementation = auxiliary_module("quantization", "quantization.delayed")
    rows, cols = values.shape
    br, bc = blocks or (max(rows, 1), max(cols, 1))
    output = torch.empty_like(values, dtype=dtype)
    inverse = torch.empty_like(scales)
    snapshot = None
    if checked and any(not isfinite(v) or v <= 0 for v in scales.detach().reshape(-1).cpu().tolist()):
        raise ValueError("FP8 output requires positive finite scales")
    for r in range(scales.shape[0]):
        for c in range(scales.shape[1]):
            multiplier = scales[r:r + 1, c:c + 1]
            if reciprocal:
                # Snapshot inversion belongs to the selected quantizer too.
                _, _, metadata, _, _ = implementation.convert(torch.empty((1, 1), device=values.device),
                    multiplier, dtype, normal=True, transposed=False, bias=False, observe=False,
                    check_finite=False, autotune=False)
                multiplier = metadata[1:2].view(1, 1)
            source = values[r * br:min((r + 1) * br, rows), c * bc:min((c + 1) * bc, cols)]
            if source.numel():
                converted, _, metadata, _, _ = implementation.convert(source, multiplier, dtype,
                    normal=True, transposed=False, bias=False, observe=False, check_finite=checked,
                    autotune=False)
                output[r * br:min((r + 1) * br, rows), c * bc:min((c + 1) * bc, cols)].copy_(converted)
                inverse[r:r + 1, c:c + 1].copy_(metadata[1:2].view(1, 1))
                if snapshot is None:
                    snapshot = metadata
    return output, inverse, snapshot if capture_scale else None, observations(values) if observe else None


@lru_cache(maxsize=256)
def _inverse_cast_kernel(rows, cols, br, bc, dtype, inverse_shape, device):
    _, T = runtime()
    sr, sc = inverse_shape

    @T.prim_func
    def kernel(X: T.Tensor((rows, cols), "float32"), S: T.Tensor((sr, sc), "float32"),
               Y: T.Tensor((rows, cols), dtype)):
        with T.Kernel(T.ceildiv(rows * cols, 256), threads=256) as bx:
            for t in T.Parallel(256):
                p = T.cast(bx, "int64") * 256 + t
                if p < rows * cols:
                    r, c = p // cols, p % cols
                    Y[r, c] = T.cast(X[r, c] / S[r // br, c // bc], dtype)

    return compile_kernel(kernel, device)


def convert(values, dtype, blocks=None, scale=None, observe=False, capture_scale=False,
            checked=False, inverse=None):
    values = values.contiguous()
    if dtype not in FP8:
        return cast(values, dtype), None, None, None
    quantizer = auxiliary_module("quantization", "quantization.fp8")
    if inverse is not None:
        if ".triton." in quantizer.__name__:
            return _provided_conversion(values, dtype, blocks, inverse, observe, capture_scale, checked, reciprocal=True)
        rows, cols = values.shape
        result = torch.empty_like(values, dtype=dtype)
        br, bc = blocks or (max(rows, 1), max(cols, 1))
        if values.numel():
            _inverse_cast_kernel(rows, cols, br, bc, dtype_name(dtype), tuple(inverse.shape), str(values.device))(
                values, inverse.contiguous(), result)
        return result, inverse, None, None
    if scale is None:
        quantize = quantizer.quantize
        if not values.numel():
            rows, cols = values.shape
            br, bc = blocks or (max(rows, 1), max(cols, 1))
            shape = (1, 1) if blocks is None else ((rows + br - 1) // br, (cols + bc - 1) // bc)
            return torch.empty_like(values, dtype=dtype), torch.ones(shape, device=values.device), None, None
        data, inv = quantize(values, dtype, blocks, checked)
        return data, inv, None, None
    if ".triton." in quantizer.__name__:
        return _provided_conversion(values, dtype, blocks, scale, observe, capture_scale, checked)
    cast_scaled = quantizer.cast_scaled
    if checked:
        check_finite(values)
    inv = torch.empty_like(scale)
    snapshot = torch.empty(2, dtype=torch.float32, device=values.device)
    errors = torch.empty(scale.numel(), dtype=torch.int32, device=values.device)
    _scale_metadata_kernel(scale.numel(), str(values.device))(scale.contiguous().view(-1), inv.view(-1), snapshot, errors)
    if checked and errors.any().item():
        raise ValueError("FP8 output requires positive finite scales")
    data = cast_scaled(values, scale, dtype, blocks)
    observed = None
    if observe:
        observed = observations(values)
    return data, inv, snapshot if capture_scale else None, observed


def pack(values, group_size, block_shape, bits):
    quantize_pack_affine = auxiliary_module("activation_codec", "quantization.activation.affine").quantize_pack_affine
    from deepopt.opt.quantization.activation import PackedActivation
    data, step, minimum = quantize_pack_affine(values.contiguous(), group_size, block_shape, bits=bits)
    return PackedActivation(data, step, minimum, tuple(values.shape), group_size, block_shape, bits)


def unpack(packed):
    dequantize_unpack_affine = auxiliary_module("activation_codec", "quantization.activation.affine").dequantize_unpack_affine
    return dequantize_unpack_affine(packed.data, packed.step, packed.minimum, packed.shape,
        packed.group_size, packed.block_shape, dtype=torch.float32, bits=packed.bits, row_alignment=1)


def output_pack(values, packed):
    if packed is not None:
        encoded = pack(values, packed.group_size, packed.block_shape, packed.bits)
        packed.data.copy_(encoded.data)
        packed.step.copy_(encoded.step)
        packed.minimum.copy_(encoded.minimum)


@lru_cache(maxsize=256)
def _binary_kernel(n, operation, device):
    _, T = runtime()

    @T.prim_func
    def kernel(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"),
               Y: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
            for t in T.Parallel(256):
                p = T.cast(bx, "int64") * 256 + t
                if p < n:
                    if operation == "multiply":
                        Y[p] = A[p] * B[p]
                    else:
                        Y[p] = T.if_then_else(B[p] != 0, A[p], 0.0)

    return compile_kernel(kernel, device)


def multiply(a, b, mask=False):
    a, b = cast(a, torch.float32).contiguous(), cast(b, torch.float32).contiguous()
    result = torch.empty_like(a)
    if a.numel():
        _binary_kernel(a.numel(), "mask" if mask else "multiply", str(a.device))(a.view(-1), b.view(-1), result.view(-1))
    return result
