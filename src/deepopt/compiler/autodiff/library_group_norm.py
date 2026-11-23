"""Exact PyTorch GroupNorm numerics for unprepared, uncompressed modules.

Prepared DOGroupNorm has a separate, intentionally different saved-xhat recipe.
This specification keeps native forward statistics instead of recomputing them
with that recipe, and makes all requested native backward operands explicit.
"""

from dataclasses import replace
from math import prod
from types import SimpleNamespace

import torch

from ..ir import CallNode, TensorMeta, TensorValue
from ..trace.metadata_utils import contiguous_stride
from .semantic import recorded, result
from .spatial import Normalization


class LibraryGroupNorm(Normalization):
    """Native GroupNorm forward/backward, sharing existing spatial VJP binding."""

    def __init__(self, node, context):
        self.module = node.module
        self.bf16_policy = node.bf16_policy
        self.module_path, self.node_name = node.module_path, node.node_name
        self.kernel_policy = context.kernel_policy
        self.profile_ranges = False
        self.precision = "native"
        self.family = "normalization"
        self.metas = tuple(value.meta if value is not None else None for value in node.args)
        self.training = node.values[0].meta.requires_grad
        self.needs = tuple(self.training and meta is not None and meta.requires_grad for meta in self.metas)
        self.public_count = len(node.values)
        self.policy = SimpleNamespace(enabled=False)
        self.groups, self.eps = self.module.num_groups, self.module.eps
        self.compute_dtype = node.values[0].meta.dtype
        xmeta = self.metas[0]
        self.n, self.channels = xmeta.shape[:2]
        self.spatial_size = prod(xmeta.shape[2:])
        self.input_format = torch.contiguous_format
        if len(xmeta.shape) == 4 and xmeta.stride is not None and all(
                size <= 1 or step == expected for size, step, expected in zip(xmeta.shape, xmeta.stride,
                    (self.channels * self.spatial_size, 1, xmeta.shape[3] * self.channels, self.channels))):
            self.input_format = torch.channels_last
        elif len(xmeta.shape) == 5 and xmeta.stride is not None and all(
                size <= 1 or step == expected for size, step, expected in zip(xmeta.shape, xmeta.stride,
                    (self.channels * self.spatial_size, 1, xmeta.shape[3] * xmeta.shape[4] * self.channels,
                     xmeta.shape[4] * self.channels, self.channels))):
            self.input_format = torch.channels_last_3d
        self.parameter_dtype = next((meta.dtype for meta in self.metas[1:3] if meta is not None), self.compute_dtype)
        device = torch.device(xmeta.device).type
        self.cast_affine = device == "cuda" and torch.is_autocast_enabled(device) and xmeta.dtype != torch.float64
        # Native mixed BF16 input/FP32 affine uses FP32 statistics. Uniform
        # BF16 and FP64 executions keep their own native statistics dtype.
        self.stats_dtype = self.compute_dtype if self.cast_affine else self.parameter_dtype
        need_values = self.needs[0] or self.needs[1]
        self.state_names = (("source", "mean", "rstd") if need_values else ()) + (
            ("weight",) if self.needs[0] and self.metas[1] is not None else ())

    def lower(self, node):
        xmeta = self.metas[0]
        stride = xmeta.stride
        if self.compute_dtype != xmeta.dtype and stride is not None:
            expected = 1
            for step, size in sorted((step, size) for step, size in zip(stride, xmeta.shape) if size > 1):
                if step != expected:
                    stride = contiguous_stride(xmeta.shape)
                    break
                expected *= size
        source_meta = replace(xmeta, dtype=self.compute_dtype, stride=stride, requires_grad=False)
        stats_meta = TensorMeta((self.n, self.groups), self.stats_dtype, xmeta.device, (self.groups, 1))
        metadata = {"source": source_meta, "mean": stats_meta, "rstd": stats_meta,
                    "weight": replace(self.metas[1], dtype=self.stats_dtype, requires_grad=False) if self.metas[1] is not None else None}
        states = tuple(TensorValue(f"{node.node_name}__native_{name}", metadata[name]) for name in self.state_names)
        if "weight" in self.state_names:
            from ..ir import inherit_model_alias

            weight = states[self.state_names.index("weight")]
            if not self.cast_affine or self.compute_dtype == self.metas[1].dtype:
                inherit_model_alias(weight, node.args[1])
            else:
                from ..ir import model_state_origin

                weight.provenance = node.args[1].provenance
                weight.model_state_copy = True
                weight.model_state_origin = model_state_origin(node.args[1])
                weight.model_state_transforms = (*node.args[1].model_state_transforms, "to")
        return CallNode(self.op, recorded(self, self.forward, "forward"), node.args, node.kwargs,
                        (*node.values, *states), self)

    def forward(self, x, weight, bias, running_mean, running_var, counter):
        # GroupNorm's public autocast kernel can cast CUDA BF16 input to FP32;
        # the lower-level native operator has no equivalent autocast dispatch.
        # Capture metadata supplies the actual public output/compute dtype.
        with torch.autocast(x.device.type, enabled=False):
            source = x.to(self.compute_dtype)
            if self.cast_affine:
                weight = weight.to(self.compute_dtype) if weight is not None else None
                bias = bias.to(self.compute_dtype) if bias is not None else None
            output, mean, rstd = torch.ops.aten.native_group_norm.default(source, weight, bias,
                self.n, self.channels, self.spatial_size, self.groups, self.eps)
        state = {"source": source, "mean": mean, "rstd": rstd, "weight": weight}
        return result([output, *(state[name] for name in self.state_names)])

    def inference(self, *inputs, **kwargs):
        values = self.forward(*inputs, **kwargs)
        return values[0] if isinstance(values, tuple) else values

    def gradients(self, gradient, *values):
        state = dict(zip(self.state_names, values))
        source = state.get("source")
        if source is None:
            # The native bias reduction is independent of source/stat values,
            # but its ABI requires correctly shaped operands. Reconstruct
            # these temporary descriptors instead of retaining an activation.
            source = torch.empty(self.metas[0].shape, dtype=self.compute_dtype,
                                 device=gradient.device, memory_format=self.input_format)
        mean = state.get("mean")
        rstd = state.get("rstd")
        if mean is None:
            mean = torch.empty((self.n, self.groups), dtype=self.stats_dtype, device=gradient.device)
            rstd = torch.empty_like(mean)
        weight = state.get("weight")
        if weight is None and self.metas[1] is not None:
            # Without dX, only affine dtype/shape are relevant to the native
            # mixed-type ABI; actual gamma values are not a dependency.
            weight = torch.empty((self.channels,), dtype=self.stats_dtype, device=gradient.device)
        with torch.autocast(gradient.device.type, enabled=False):
            gradients = torch.ops.aten.native_group_norm_backward.default(gradient, source, mean, rstd, weight,
                self.n, self.channels, self.spatial_size, self.groups, list(self.needs[:3]))
        requested = [value.to(meta.dtype) for value, meta, needed in zip(gradients, self.metas[:3], self.needs[:3]) if needed]
        return result(requested)
