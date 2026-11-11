"""ReLU precision, mutation and exact saved-mask policy resolution."""

import torch

from .geometry import ReLUGeometry
from ...quantization.activation_tensor import FP8Activation
from ..._execution import finite_checks_enabled


FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def relu(input, inplace=False, *, plan=None, output_dtype=None, output_layout=None, output_scale=None,
         scaling_state=None, state_training=True, planned=False, execution_info=None,
         _kind="relu", approximate="none"):
    from ..functional import _prepare_linear_plan
    from ..._autocast import _get_fp8_plan, _get_activation_compression

    x = input
    if not isinstance(x, torch.Tensor) or x.layout != torch.strided or x.is_nested:
        raise TypeError("ReLU input must be a dense tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64, *FP8_DTYPES):
        raise TypeError("optimized ReLU input must have a floating point dtype")
    if x.device.type not in ("cpu", "cuda", "meta"):
        raise ValueError("optimized ReLU supports CPU, CUDA and meta")
    if type(inplace) is not bool:
        raise TypeError("inplace must be a boolean")
    if approximate not in ("none", "tanh") or (_kind != "gelu" and approximate != "none"):
        raise ValueError("approximate must be 'none' or 'tanh' for GELU")
    if _kind == "gelu" and inplace:
        raise ValueError("GELU has no in-place operation")
    active, compression = _get_fp8_plan(x.device.type), _get_activation_compression(x.device.type)
    selected = plan
    if isinstance(plan, dict) and "fp8" in plan:
        selected = plan
    elif active is False:
        selected = {"fp8": False, "activation_compression": compression or {}}
    elif not (isinstance(plan, dict) and "fp8" in plan):
        selected = active if active is not None else plan
        if (selected is not None and compression is not None
                and compression != selected["fp8_training"]["activation_compression"]):
            selected = {**selected, "fp8_training": {**selected["fp8_training"], "activation_compression": compression}}
    if selected is None:
        selected = {"fp8": False, "activation_compression": compression or {}}
    checked = _prepare_linear_plan(selected)
    training = checked.get("fp8_training")
    compression = (training or checked)["activation_compression"]
    wrapped = isinstance(x, FP8Activation)
    logical_dtype = x.dtype if x.dtype not in FP8_DTYPES else torch.bfloat16
    if output_dtype is None:
        output_dtype = (x._data.dtype if inplace and wrapped else
                        training["fp8_dtypes"]["activation"] if planned and output_layout is not None and training else
                        x._data.dtype if wrapped and active is not False and not planned else logical_dtype)
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64, *FP8_DTYPES):
        raise ValueError("ReLU output_dtype must be a supported floating point dtype")
    fp8 = output_dtype in FP8_DTYPES
    layout = output_layout or (x._activation_layout if wrapped and fp8 else "last")
    if layout not in ("channels", "last") or layout == "channels" and x.ndim < 2:
        raise ValueError("ReLU layout must be last, or channels for rank >= 2")
    geometry = ReLUGeometry(tuple(x.shape), layout)
    block_shape = checked["block_shapes"]["activation"] if training else x._block_shape if wrapped and fp8 else None
    if inplace and wrapped:
        block_shape = x._block_shape
    reuse = ((_kind == "relu" or _kind == "relu6" and inplace) and wrapped and fp8 and x._data.dtype == output_dtype and block_shape == x._block_shape
             and (layout == x._activation_layout or block_shape is None) and output_scale is None)
    if output_scale is not None:
        rows, cols = geometry.matrix_shape
        scale_shape = ((1, 1) if block_shape is None else
                       ((rows + block_shape[0] - 1) // block_shape[0], (cols + block_shape[1] - 1) // block_shape[1]))
        if not fp8 or not isinstance(output_scale, torch.Tensor) or output_scale.dtype != torch.float32:
            raise TypeError("output_scale requires FP8 output and an FP32 scale tensor")
        if output_scale.requires_grad:
            raise ValueError("output_scale is non-differentiable quantization metadata")
        if tuple(output_scale.shape) != scale_shape or output_scale.device != x.device or not output_scale.is_contiguous():
            raise ValueError("output_scale must be contiguous, on the input device and match the FP8 scale grid")
    if fp8 and block_shape is not None and x.device.type == "cuda" and any(b & (b - 1) for b in block_shape):
        raise ValueError("CUDA ReLU FP8 scale blocks must be powers of two")
    if training and training["scaling_recipe"] == "delayed" and fp8 and not reuse and output_scale is None and scaling_state is None:
        raise ValueError("Delayed FP8 ReLU output requires a prepared adapter or an explicit output_scale")
    if inplace:
        if wrapped:
            if not reuse:
                raise ValueError("In-place FP8 ReLU must preserve storage dtype, layout and scales")
        elif output_dtype != x.dtype:
            raise ValueError("In-place ReLU cannot change dtype")
        if torch.is_grad_enabled() and x.requires_grad:
            if x.is_leaf:
                raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")
            if x._base is not None and x._base.requires_grad and x._base.is_leaf:
                raise RuntimeError("a view of a leaf Variable that requires grad is being used in an in-place operation.")
        if not wrapped and any(size > 1 and stride == 0 for size, stride in zip(x.shape, x.stride())):
            raise RuntimeError("in-place ReLU does not support overlapping broadcast storage")
    from ..autograd.activation.relu import ReLUFunction

    options = {"dtype": output_dtype, "block_shape": block_shape, "packed": compression["enabled"],
               "inplace": inplace, "reuse": reuse, "scale": output_scale,
               "check_finite": finite_checks_enabled() and fp8 and not reuse}
    if _kind != "relu":
        from ..autograd.activation.elementwise import ActivationFunction

        options.update(kind=_kind, approximate=approximate, bits=compression["bits"],
                       group_size=compression["group_size"],
                       check_finite=finite_checks_enabled() and (fp8 or compression["enabled"] and _kind != "relu6"))
        if inplace and not (torch.is_grad_enabled() and x.requires_grad):
            from ...backends.activation import executor
            from torch._subclasses.fake_tensor import FakeTensor

            result = executor(x, _kind).forward(x, geometry, **options, save_mask=False)
            if wrapped or x.device.type == "cuda" or x.device.type == "meta" or isinstance(x, FakeTensor):
                torch.autograd.graph.increment_version(x)
            if execution_info is not None:
                execution_info.update(result[-1], saved_state_bytes=0, compression=False)
            return x
        return ActivationFunction.apply(x, geometry, options, scaling_state if output_scale is None else None,
                                        state_training and torch.is_grad_enabled(),
                                        torch.is_grad_enabled() and x.requires_grad, execution_info)
    backward = torch.is_grad_enabled() and x.requires_grad
    if inplace and not backward:
        # Function.apply creates an alias output when autograd is inactive.
        # Native in-place ReLU returns the original object, including leaves
        # mutated under no_grad, so execute this inference path directly.
        from ...backends.activation import executor
        from torch._subclasses.fake_tensor import FakeTensor

        result = executor(x).forward(x, geometry, **options, save_mask=False)
        info = result[-1]
        storage = x._data if wrapped else x
        if wrapped or info["backend"] == "triton" or isinstance(storage, FakeTensor) or x.device.type == "meta":
            torch.autograd.graph.increment_version(x)
        if execution_info is not None:
            execution_info.update(info, input="fp8" if wrapped else str(x.dtype),
                                  output="fp8" if fp8 else str(output_dtype), compression=False,
                                  saved_mask_bytes=0, backward_launches=0)
        return x
    return ReLUFunction.apply(x, geometry, options, scaling_state if output_scale is None else None,
                              state_training and torch.is_grad_enabled(), backward, execution_info)
