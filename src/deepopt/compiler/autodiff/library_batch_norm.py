"""Native BatchNorm library state, with exactly one running-stat update."""

from dataclasses import replace
from types import SimpleNamespace

import torch

from ..ir import AuxiliaryTensorMeta, CallNode, TensorMeta, TensorValue, inherit_model_alias
from .semantic import recorded, result
from .spatial import Normalization


class LibraryBatchNorm(Normalization):
    def __init__(self, node, context):
        from deepopt.opt.nn.normalization.geometry import NormGeometry

        self.module, self.bf16_policy = node.module, node.bf16_policy
        self.module_path, self.node_name = node.module_path, node.node_name
        self.kernel_policy, self.profile_ranges = context.kernel_policy, False
        self.precision, self.family = "native", "normalization"
        self.metas = tuple(value.meta if value is not None else None for value in node.args)
        self.public_count, self.training = len(node.values), node.values[0].meta.requires_grad
        self.needs = tuple(meta is not None and meta.requires_grad and self.training for meta in self.metas)
        self.batch_training = self.module.training or node.args[3] is None
        self.geometry = NormGeometry("batch", self.metas[0].shape, eps=self.module.eps,
                                     batch_stats=self.batch_training)
        self.compression = dict(enabled=False)
        self.policy = SimpleNamespace(enabled=False)
        self.stats_dtype = torch.float64 if self.metas[0].dtype == torch.float64 else torch.float32
        self.state_names = ("source", "mean", "rstd", "weight", "running_mean", "running_var", "reserve", "implementation")

    def lower(self, node):
        source = self.metas[0]
        empty = TensorMeta((0,), source.dtype, source.device)
        stats = TensorMeta((source.shape[1],) if self.batch_training else (0,), self.stats_dtype, source.device)
        reserve = (AuxiliaryTensorMeta(dtype=torch.uint8, device=source.device)
                   if torch.device(source.device).type == "cuda" else TensorMeta((0,), torch.uint8, source.device))
        metadata = (replace(source, requires_grad=False), stats, stats,
                    self.metas[1] or empty, self.metas[3] or empty, self.metas[4] or empty,
                    reserve, TensorMeta((), torch.int64, "cpu"))
        states = tuple(TensorValue(f"{node.node_name}__native_{name}", replace(meta, requires_grad=False))
                       for name, meta in zip(self.state_names, metadata))
        for slot, argument in ((3, 1), (4, 3), (5, 4)):
            if node.args[argument] is not None:
                inherit_model_alias(states[slot], node.args[argument])
        return CallNode(self.op, recorded(self, self.forward, "forward"), node.args, {}, (*node.values, *states), self)

    def forward(self, x, weight, bias, running_mean, running_var, counter):
        factor = self.module.momentum
        if self.module.training and counter is not None:
            counter.add_(1)
            if factor is None:
                factor = 1. / float(counter)
        factor = 0. if factor is None else factor
        with torch.autocast(x.device.type, enabled=False):
            output, mean, inverse, reserve, implementation = torch.ops.aten._batch_norm_impl_index.default(
                x, weight, bias, running_mean, running_var, self.batch_training, factor,
                self.module.eps, torch.backends.cudnn.enabled)
        empty = x.new_empty((0,))
        return (output, x, mean, inverse, weight if weight is not None else empty,
                running_mean if running_mean is not None else empty,
                running_var if running_var is not None else empty,
                reserve, torch.tensor(implementation, dtype=torch.int64))

    def gradients(self, gradient, source, mean, inverse, weight, running_mean, running_var, reserve, implementation):
        with torch.autocast(gradient.device.type, enabled=False):
            values = torch.ops.aten._batch_norm_impl_index_backward.default(
                int(implementation), source, gradient, weight if self.metas[1] is not None else None,
                running_mean if self.metas[3] is not None else None,
                running_var if self.metas[4] is not None else None,
                mean, inverse, self.batch_training, self.module.eps, list(self.needs[:3]), reserve)
        return result([value.to(meta.dtype) for value, meta, needed in zip(values, self.metas[:3], self.needs[:3]) if needed])
