"""Scale-aware FP8 activations with ordinary floating-point autograd edges."""

from math import prod

import torch
from torch.autograd import Function
from torch.utils._pytree import tree_map
from torch._subclasses.fake_tensor import FakeTensor
from torch.nn import functional as F

from .tensor import QuantizedTensor
from .quantizer import quantize_fp8


_RELU_FUNCTIONS = (torch.relu, F.relu, torch.Tensor.relu, torch.relu_, torch.Tensor.relu_)


def activation_matrix(x, layout):
    if layout == "channels":
        return x.movedim(1, -1).reshape(-1, x.shape[1])
    if layout == "last":
        return x.reshape(-1, x.shape[-1]) if x.ndim else x.reshape(1, 1)
    raise ValueError("FP8 activation layout must be channels or last")


def restore_activation(matrix, shape, layout):
    if layout == "channels":
        return matrix.reshape(shape[0], *shape[2:], shape[1]).movedim(-1, 1)
    return matrix.reshape(shape)


class FP8Activation(torch.Tensor):
    """FP8 physical storage; logical dtype describes the gradient interface.

    This wrapper never keeps a dense shadow activation. Unsupported ordinary
    torch operations receive a reconstructed tensor through dispatch. Prepared
    precision plans put explicit boundaries before such operations.
    """

    @staticmethod
    def __new__(cls, data, scale_inv, shape, block_shape=None, layout="channels",
                logical_dtype=torch.bfloat16, strides=None):
        shape = tuple(shape)
        if layout not in ("channels", "last") or layout == "channels" and len(shape) < 2:
            raise ValueError("FP8 activation has insufficient dimensions")
        expected = ((shape[0] * prod(shape[2:]), shape[1]) if layout == "channels"
                    else (prod(shape[:-1]), shape[-1]) if shape else (1, 1))
        if data.numel():
            QuantizedTensor(data, scale_inv, block_shape)
        elif (data.ndim != 2 or data.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
              or scale_inv.dtype != torch.float32 or scale_inv.device != data.device):
            raise ValueError("Empty FP8 activation requires 2D FP8 data and matching FP32 scale metadata")
        if tuple(data.shape) != expected:
            raise ValueError(f"FP8 activation matrix must have shape {expected}")
        if logical_dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("FP8 activation logical dtype must be FP32, BF16 or FP16")
        if strides is None:
            stride, result = 1, []
            for size in reversed(shape):
                result.append(stride)
                stride *= size
            strides = tuple(reversed(result))
        output = torch.Tensor._make_wrapper_subclass(
            cls, shape, strides=strides, storage_offset=0, dtype=logical_dtype,
            device=data.device, layout=torch.strided, requires_grad=False)
        output._data, output._scale_inv = data, scale_inv
        output._block_shape, output._activation_layout = block_shape, layout
        output._logical_shape, output._logical_dtype, output._logical_strides = shape, logical_dtype, tuple(strides)
        return output

    def quantized(self, layout=None):
        if layout is not None and layout != self._activation_layout and len(self._logical_shape) != 2:
            raise ValueError("FP8 activation layout does not match the consuming operation")
        return QuantizedTensor(self._data, self._scale_inv, self._block_shape)

    def dequantize(self, dtype=None):
        if not self._data.numel():
            return torch.empty(self._logical_shape, device=self.device, dtype=dtype or self._logical_dtype)
        matrix = self.quantized().dequantize()
        return restore_activation(matrix, self._logical_shape, self._activation_layout).to(dtype or self._logical_dtype)

    def __tensor_flatten__(self):
        return ["_data", "_scale_inv"], (self._logical_shape, self._block_shape, self._activation_layout,
                                        self._logical_dtype, self._logical_strides)

    @staticmethod
    def __tensor_unflatten__(inner, metadata, outer_size, outer_stride):
        shape, block, layout, dtype, strides = metadata
        return FP8Activation(inner["_data"], inner["_scale_inv"], shape, block, layout, dtype,
                             outer_stride if outer_stride is not None else strides)

    def __repr__(self):
        return (f"FP8Activation(shape={self._logical_shape}, fp8_dtype={self._data.dtype}, "
                f"logical_dtype={self._logical_dtype}, layout={self._activation_layout!r})")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        from ..nn.pooling.spec import call_kind, FORCE_INDICES, SPECS

        pool_kind = call_kind(func)
        if pool_kind is not None:
            from ..nn import functional as doF

            if func in FORCE_INDICES:
                bound = SPECS[pool_kind].arguments(args[1:], {k: v for k, v in kwargs.items() if k != "input"})
                bound["return_indices"] = True
                return getattr(doF, "do_" + pool_kind)(args[0] if args else kwargs["input"], **bound)
            return getattr(doF, "do_" + pool_kind)(*args, **kwargs)
        if func in (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d):
            from ..nn import functional as doF

            rank = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d).index(func) + 1
            return getattr(doF, f"do_conv_transpose{rank}d")(*args, **kwargs)
        if func is F.layer_norm:
            from ..nn.functional import do_layer_norm

            return do_layer_norm(*args, **kwargs)
        from ..nn.activation.spec import arguments

        activation = arguments(func, args, kwargs)
        if activation is not None:
            from ..nn.functional import _activation
            from ..precision.gradient import functional_gradient

            kind, x, options = activation
            with functional_gradient(kind):
                return _activation(kind, x, **options)
        if func in _RELU_FUNCTIONS:
            relu_names = ("input", "inplace") if func is F.relu else ("input",)
            if (len(args) <= len(relu_names) and not set(kwargs) - set(relu_names)
                    and not any(name in kwargs for name in relu_names[:len(args)])):
                from ..nn.functional import do_relu

                inplace = func in (torch.relu_, torch.Tensor.relu_) or kwargs.get("inplace", args[1] if len(args) > 1 else False)
                return do_relu(args[0] if args else kwargs["input"], inplace=inplace)
        # Tensor.__torch_function__ would manufacture uninitialized instances
        # of this storage-owning wrapper for native dense fallback outputs.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func in (torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default):
            from math import prod

            x, requested = args[:2]
            shape = tuple(requested)
            if shape.count(-1) == 1:
                known = prod(v for v in shape if v != -1)
                if known:
                    shape = tuple(x.numel() // known if v == -1 else v for v in shape)
            forward_flatten = (x.ndim > 2 and x._activation_layout == "channels" and prod(x.shape[2:]) == 1
                               and shape == tuple(x.shape[:2]))
            restore_global = (x.ndim == 2 and len(shape) > 2 and tuple(shape[:2]) == tuple(x.shape)
                              and prod(shape[2:]) == 1)
            if shape == tuple(x.shape) or forward_flatten or restore_global:
                layout = "last" if forward_flatten else "channels" if restore_global else x._activation_layout
                return FP8Activation(x._data, x._scale_inv, shape, x._block_shape, layout, x._logical_dtype)
        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default, torch.ops.aten.clone.default):
            x = args[0]
            clone = func is torch.ops.aten.clone.default
            return FP8Activation(x._data.clone() if clone else x._data.detach(),
                                 x._scale_inv.clone() if clone else x._scale_inv.detach(),
                                 x._logical_shape, x._block_shape, x._activation_layout,
                                 x._logical_dtype, x._logical_strides)
        if func is torch.ops.aten.relu.default:
            x = args[0]
            if isinstance(x._data, FakeTensor):
                data = torch.empty_like(x._data)
            elif x.device.type == "cuda":
                from ..backends.triton.quantization.activation.relu import relu

                data = relu(x.quantized())
            else:
                q = x.quantized()
                data = torch.where(q.dequantize() > 0, q.data.float(), 0).to(q.data.dtype)
            return FP8Activation(data, x._scale_inv, x._logical_shape, x._block_shape,
                                 x._activation_layout, x._logical_dtype, x._logical_strides)

        def dequantize(value):
            return value.dequantize() if isinstance(value, FP8Activation) else value

        return func(*tree_map(dequantize, args), **tree_map(dequantize, kwargs))


class _CastActivation(Function):
    @staticmethod
    def forward(ctx, x, layout, fp8_dtype, block_shape):
        ctx.dtype = x.dtype
        q = quantize_fp8(activation_matrix(x, layout), fp8_dtype=fp8_dtype, block_shape=block_shape)
        return FP8Activation(q.data, q.scale_inv, x.shape, q.block_shape, layout, x.dtype)

    @staticmethod
    def backward(ctx, gradient):
        return gradient.to(ctx.dtype), None, None, None


class _RestoreActivation(Function):
    @staticmethod
    def forward(ctx, x):
        return x.dequantize()

    @staticmethod
    def backward(ctx, gradient):
        return gradient


def to_fp8_activation(x, *, layout="channels", fp8_dtype=torch.float8_e4m3fn, block_shape=None):
    if isinstance(x, FP8Activation):
        if x._activation_layout == layout or x.ndim == 2:
            return x
        x = dequantize_activation(x)
    return _CastActivation.apply(x, layout, fp8_dtype, block_shape)


def dequantize_activation(x):
    return _RestoreActivation.apply(x) if isinstance(x, FP8Activation) else x
