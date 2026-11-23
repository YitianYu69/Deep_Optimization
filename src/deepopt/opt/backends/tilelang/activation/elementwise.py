"""TileLang activation values, local derivatives and compressed saved state."""

from functools import lru_cache
import torch

from .._runtime import runtime, compile_kernel
from .common import dense, cast, convert, pack, unpack, multiply, auxiliary_module, check_finite as ensure_finite, output_pack as save_output_pack
from deepopt.opt.nn.activation.state import SavedActivation


@lru_cache(maxsize=256)
def _evaluate_kernel(n, kind, approximate, device):
    _, T = runtime()

    @T.prim_func
    def kernel(X: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32"),
               D: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
            for t in T.Parallel(256):
                p = T.cast(bx, "int64") * 256 + t
                if p < n:
                    x = X[p]
                    if kind == "relu":
                        Y[p] = T.if_then_else(x < 0, 0.0, x)
                        D[p] = T.if_then_else(x <= 0, 0.0, 1.0)
                    elif kind == "relu6":
                        Y[p] = T.if_then_else(x < 0, 0.0, T.if_then_else(x > 6, 6.0, x))
                        D[p] = T.if_then_else((x <= 0) | (x >= 6), 0.0, 1.0)
                    elif kind == "sigmoid" or kind == "silu":
                        s = 1.0 / (1.0 + T.exp(-x))
                        if kind == "sigmoid":
                            Y[p] = s
                            D[p] = s * (1.0 - s)
                        else:
                            Y[p] = x * s
                            D[p] = s * (1.0 + x * (1.0 - s))
                    elif kind == "tanh":
                        y = T.tanh(x)
                        Y[p] = y
                        D[p] = 1.0 - y * y
                    elif approximate == "tanh":
                        y = T.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))
                        Y[p] = 0.5 * x * (1.0 + y)
                        D[p] = 0.5 * (1.0 + y) + 0.5 * x * (1.0 - y * y) * 0.7978845608028654 * (1.0 + 0.134145 * x * x)
                    else:
                        cdf = 0.5 * (1.0 + T.erf(x * 0.7071067811865476))
                        Y[p] = x * cdf
                        D[p] = cdf + x * 0.3989422804014327 * T.exp(-0.5 * x * x)

    return compile_kernel(kernel, device)


def evaluate(values, kind, approximate="none"):
    values = values.contiguous()
    result, derivative = torch.empty_like(values), torch.empty_like(values)
    if values.numel():
        _evaluate_kernel(values.numel(), kind, approximate, str(values.device))(
            values.view(-1), result.view(-1), derivative.view(-1))
    return result, derivative


def forward(x, geometry, *, kind, approximate, dtype, block_shape, packed, bits, group_size,
            save_mask, inplace, reuse=False, scale=None, observe=False, check_finite=False, capture_scale=False,
            recompute=False, output_pack=None):
    source = geometry.matrix(dense(x)).contiguous()
    values, derivative = evaluate(source, kind, approximate)
    saved = None
    if save_mask:
        if recompute:
            saved = SavedActivation(source, recompute=(kind, approximate))
        elif kind == "relu6" or kind == "relu":
            if packed:
                pack_boolean_mask = auxiliary_module("activation_codec", "quantization.activation.boolean").pack_boolean_mask
                data = pack_boolean_mask(cast(derivative, torch.bool))
            else:
                data = cast(derivative, torch.bool).view(-1)
            saved = SavedActivation(data, mask=True, packed=packed)
        elif packed:
            if check_finite:
                ensure_finite(derivative, "Activation derivative compression requires finite coefficients")
            q = pack(derivative, group_size, block_shape, bits)
            saved = SavedActivation(q.data, q.step, q.minimum, group_size, block_shape, bits, packed=True)
        else:
            saved = SavedActivation(derivative)
    inverse = x._scale_inv if reuse else None
    output, out_inverse, snapshot, observations = convert(values, dtype, block_shape, scale, observe,
        capture_scale, check_finite, inverse)
    save_output_pack(output, output_pack)
    if inplace:
        storage = getattr(x, "_data", x)
        storage.copy_(output if hasattr(x, "_data") else geometry.restore(output))
        output = storage
    info = dict(backend="tilelang", activation=kind, forward_launches=int(x.numel() > 0),
        scale_launches=int(dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and not reuse and scale is None),
        coefficient_launches=int(save_mask and packed), backward_launches=int(save_mask and x.numel() > 0),
        scale_mode="reuse" if reuse else "provided" if scale is not None else "current" if out_inverse is not None else "none",
        saved_state="input_recompute" if recompute else "mask" if kind in ("relu", "relu6") else "derivative",
        producer_pack_fused=False, producer_pack_ready=output_pack is not None,
        launch_counts_complete=False)
    return output, out_inverse, saved, snapshot, observations, info


def backward(gradient, saved, geometry, *, dtype, gradient_output=None, pointwise_tile=512):
    if saved.recompute is not None:
        _, coefficient = evaluate(cast(saved.data, torch.float32), *saved.recompute)
    elif saved.mask:
        if saved.packed:
            unpack_boolean_mask = auxiliary_module("activation_codec", "quantization.activation.boolean").unpack_boolean_mask
            coefficient = unpack_boolean_mask(saved.data, geometry.matrix_shape)
        else:
            coefficient = saved.data.reshape(geometry.matrix_shape)
    elif saved.packed:
        from deepopt.opt.quantization.activation import PackedActivation
        coefficient = unpack(PackedActivation(saved.data, saved.step, saved.minimum, geometry.matrix_shape,
            saved.group_size, saved.block_shape, saved.bits))
    else:
        coefficient = saved.data
    result = multiply(geometry.matrix(dense(gradient)), coefficient, mask=saved.mask)
    return gradient_output.convert(geometry.restore(result)) if gradient_output is not None else cast(result, dtype)
