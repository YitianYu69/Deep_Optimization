"""Backward capabilities are distinct from forward FP8 storage support."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BackwardCapability:
    kind: str
    layout: str
    accepts_fp8: bool
    emits_fp8: bool
    reason: str = ""


def module_capability(module):
    from ..nn.linear import DOLinear
    from ..nn.hf_conv1d import DOHFConv1D
    from ..nn.functional_linear import DOFunctionalLinear
    from ..nn.convolution.modules import DOConv1d, DOConv2d, DOConv3d, DOFunctionalConvolution
    from ..nn.normalization.modules import DOBatchNorm2d, DOGroupNorm, DOFunctionalNormalization, DOLayerNorm
    from ..nn.activation.modules import DOReLU
    from ..nn.activation.elementwise_modules import ACTIVATION_TYPES
    from ..nn.attention import DOMultiheadAttention

    from ..nn.convolution_transpose.modules import TRANSPOSE_TYPES
    from ..nn.pooling.modules import POOL_TYPES

    if type(module) in TRANSPOSE_TYPES:
        kind, layout = "convolution", "channels"
    elif type(module) in POOL_TYPES:
        kind, layout = "pooling", "channels"
    elif type(module) in (DOLinear, DOHFConv1D, DOFunctionalLinear):
        kind, layout = "linear", "last"
    elif type(module) in (DOConv1d, DOConv2d, DOConv3d, DOFunctionalConvolution):
        if getattr(module, "padding_mode", "zeros") != "zeros" or getattr(module, "padding", None) == "same":
            return BackwardCapability("convolution", "channels", False, False, "native padding boundary")
        kind, layout = "convolution", "channels"
    elif type(module) in (DOBatchNorm2d, DOGroupNorm, DOFunctionalNormalization, DOLayerNorm):
        kind = "normalization"
        layout = "last" if type(module) is DOLayerNorm or getattr(module, "kind", None) == "layer" else "channels"
    elif type(module) is DOReLU:
        kind, layout = "relu", "last"
    elif type(module) in ACTIVATION_TYPES:
        kind, layout = module.kind, "last"
    elif type(module) is DOMultiheadAttention:
        return BackwardCapability("attention", "last", False, False, "native SDPA/attention composition boundary")
    else:
        return None
    training = getattr(module, "plan", {}).get("fp8_training")
    if training is None:
        return BackwardCapability(kind, layout, False, False, "native compute plan")
    backend = training["backward_backend"]
    emits = backend == "inherit" or kind in ("relu", "normalization", "pooling") or type(module) in ACTIVATION_TYPES
    if kind == "convolution" and hasattr(module, "kernel_size"):
        padding = module.padding
        zero_padding = padding == "valid" if isinstance(padding, str) else all(p == 0 for p in padding)
        pointwise = (module.groups == 1 and all(k == 1 for k in module.kernel_size)
                     and all(s == 1 for s in module.stride) and zero_padding)
        # Spatial backward already selects the implicit Triton executor; the
        # native GEMM selector applies only to its pointwise matrix products.
        emits = emits or not pointwise
    emits = emits or type(module) in TRANSPOSE_TYPES
    return BackwardCapability(kind, layout, True, emits, "" if emits else "selected native GEMM has no FP8 output epilogue")
