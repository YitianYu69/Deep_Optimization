"""Per-forward output packing for a compatible scale-aware Linear consumer."""

import torch

from .activation_codec import producer_packing


def allocate_output_pack(x, geometry, *, kind, dtype, block_shape, compression, enabled, delayed=False):
    policy = producer_packing()
    if (policy == "off" or policy in ("gelu", "layer_norm") and kind != policy or not enabled or delayed
            or kind not in ("gelu", "layer_norm") or not compression.get("enabled", False)
            or dtype not in (torch.float8_e4m3fn, torch.float8_e5m2) or block_shape is not None
            or geometry.layout != "last" or not x.is_contiguous() or not x.numel()):
        return None
    from deepopt.opt.quantization.activation_tensor import FP8Activation
    if isinstance(x, FP8Activation):
        return None
    group, bits = compression["group_size"], compression.get("bits", 2)
    if not 32 <= group <= 16384 or group & (group - 1):
        return None
    shape = tuple(geometry.matrix_shape)
    if kind == "layer_norm" and (shape[0] <= 1 or shape[1] > 4096 or tuple(geometry.normalized_shape) != (shape[1],)):
        return None
    from deepopt.opt.quantization.activation import PackedActivation, _dimensions
    groups, _, words = _dimensions(shape, group, None, bits)
    return PackedActivation(torch.empty((groups, words), device=x.device, dtype=torch.int32),
                            torch.empty(groups, device=x.device, dtype=torch.float32),
                            torch.empty(groups, device=x.device, dtype=torch.float32), shape, group, None, bits)


def reference_output_pack(output, packed):
    if packed is None:
        return
    from torch._subclasses.fake_tensor import FakeTensor
    if isinstance(output, FakeTensor) or output.device.type == "meta":
        return
    from deepopt.opt.quantization.activation import quantize_pack_activation
    actual = quantize_pack_activation(output, packed.group_size, bits=packed.bits)
    packed.data.copy_(actual.data)
    packed.step.copy_(actual.step)
    packed.minimum.copy_(actual.minimum)
