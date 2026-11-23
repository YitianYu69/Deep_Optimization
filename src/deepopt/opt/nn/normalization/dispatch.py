"""Independent normalization precision and normalized-activation storage."""

from deepopt.kernel_policy import kernel_entry, current_kernel_policy

import torch
from torch.nn import functional as F

from .geometry import NormGeometry
from ...quantization.activation_tensor import FP8Activation, dequantize_activation


@kernel_entry
def normalization(kind, x, weight=None, bias=None, *, running_mean=None, running_var=None,
                  training=True, momentum=0.1, groups=1, eps=1e-5, plan=None,
                  output_layout=None, scaling_state=None, state_training=True, execution_info=None):
    from ..functional import _prepare_linear_plan
    from ..._autocast import _get_fp8_plan, _get_activation_compression, _get_bf16_compression

    active = _get_fp8_plan(x.device.type)
    compression = _get_activation_compression(x.device.type)
    if isinstance(plan, dict) and "fp8" in plan:
        checked = _prepare_linear_plan(plan)
    elif active is False or plan is None and active is None:
        compression = _get_bf16_compression(x)
        checked = _prepare_linear_plan({"fp8": False, "activation_compression": compression or {}})
    else:
        selected = active if active is not None else plan
        if compression is not None and compression != selected["fp8_training"]["activation_compression"]:
            selected = {**selected, "fp8_training": {**selected["fp8_training"], "activation_compression": compression}}
        checked = _prepare_linear_plan(selected)
    fp8 = "fp8_training" in checked
    from .policy import capture_normalization_policy, validate_normalization_backend

    bf16_policy = capture_normalization_policy(x, checked)
    compression = checked["fp8_training"]["activation_compression"] if fp8 else checked["activation_compression"]
    if not fp8:
        output_layout, scaling_state = None, None
    def native():
        dense = dequantize_activation(x)
        if kind == "batch":
            return F.batch_norm(dense, running_mean, running_var, weight, bias, training, momentum, eps)
        return F.group_norm(dense, groups, weight, bias, eps)

    raw_fp8 = x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if kind == "batch" and x.ndim != 4:
        if bf16_policy.storage_enabled:
            raise ValueError("BF16 storage normalization currently supports BatchNorm2d geometry")
        if execution_info is not None:
            execution_info.update(output="native", backend="native", compression=False)
        return native()
    device_counter = kind == "batch" and isinstance(momentum, torch.Tensor) and x.device.type == "cuda"
    if (not bf16_policy.storage_enabled and not fp8 and not compression["enabled"] and not isinstance(x, FP8Activation) and not raw_fp8
            and not device_counter or not x.numel()):
        if execution_info is not None:
            execution_info.update(output="native", backend="native", compression=False)
        output = native()
        return output.to(torch.bfloat16) if bf16_policy.storage_enabled else output
    geometry = NormGeometry(kind, tuple(x.shape), groups, eps, training if kind == "batch" else True)
    if kind == "batch":
        if training and geometry.elements == 1:
            raise ValueError(f"Expected more than 1 value per channel when training, got input size {tuple(x.shape)}")
        if (running_mean is None) != (running_var is None) or not training and running_mean is None:
            raise ValueError("BatchNorm evaluation requires running_mean and running_var")
    else:
        F._verify_batch_size([x.shape[0] * x.shape[1] // groups, groups, *x.shape[2:]])
    for name, value in (("input", x), ("weight", weight), ("bias", bias),
                        ("running_mean", running_mean), ("running_var", running_var)):
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.is_nested:
            raise TypeError(f"normalization {name} must be a dense tensor")
        if value.dtype not in (torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2):
            raise TypeError(f"normalization {name} has unsupported dtype")
        if value.device != x.device:
            raise ValueError("normalization tensors must share a device")
        if name != "input" and tuple(value.shape) != (geometry.channels,):
            raise ValueError(f"normalization {name} must have shape (channels,)")
        if name.startswith("running_") and value.requires_grad:
            raise ValueError("normalization running statistics cannot require gradients")
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("optimized normalization supports CPU and CUDA")
    if output_layout not in (None, "channels", "last"):
        raise ValueError("normalization output layout must be channels, last, or None")
    if fp8 and checked["fp8_training"]["backend"] != "reference" and x.device.type != "cuda":
        raise ValueError(f"{checked['fp8_training']['backend'].capitalize()} normalization requires CUDA")
    if output_layout is not None and checked["fp8_training"]["scaling_recipe"] == "delayed" and scaling_state is None:
        raise ValueError("Delayed FP8 normalization outputs require a prepared adapter or FX call site")
    from ..autograd.normalization import NormalizationFunction

    backend = checked["fp8_training"]["backend"] if fp8 else current_kernel_policy().backend if x.device.type == "cuda" else "reference"
    validate_normalization_backend(bf16_policy, backend, x.device.type)
    backward = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, weight, bias))
    if execution_info is not None:
        execution_info.update(output="fp8" if output_layout is not None else "native",
                              backend=checked["fp8_training"]["backend"] if fp8 else current_kernel_policy().backend if x.device.type == "cuda" else "reference",
                              compression=compression["enabled"] and backward and (
                                  weight is not None and weight.requires_grad or x.requires_grad and geometry.batch_stats))
    return NormalizationFunction.apply(x, weight, bias, running_mean, running_var, geometry, checked,
                                        output_layout, scaling_state, state_training and torch.is_grad_enabled(),
                                        momentum, backward, execution_info, bf16_policy)
