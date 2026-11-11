"""Record native module call shapes outside the timed/profiler passes."""

from collections import OrderedDict
from copy import deepcopy

from .catalog import default_workload
from .measurement import tensor_leaves


def record_workloads(session):
    import torch
    from torch import nn

    records, handles = OrderedDict(), []
    mapping = {nn.Linear: "linear", nn.BatchNorm2d: "batch_norm", nn.GroupNorm: "group_norm",
               nn.LayerNorm: "layer_norm", nn.ReLU: "relu", nn.ReLU6: "relu6", nn.GELU: "gelu",
               nn.SiLU: "silu", nn.Sigmoid: "sigmoid", nn.Tanh: "tanh"}
    for rank in (1, 2, 3):
        mapping[getattr(nn, f"Conv{rank}d")] = f"conv{rank}d"
        mapping[getattr(nn, f"ConvTranspose{rank}d")] = f"conv_transpose{rank}d"
    from deepopt.opt.nn.pooling.spec import MODULE_SPECS
    mapping.update({cls: item.name for cls, item in MODULE_SPECS.items()})

    def collect(module, inputs, kwargs, name, op):
        value = next(tensor_leaves((inputs, kwargs)), None)
        if value is None or not value.is_floating_point():
            return
        if op.startswith("max_unpool"):
            # Public indices cannot be reconstructed from an arbitrary unpool invocation.
            return
        key = (name, tuple(value.shape), tuple(value.stride()))
        if key in records:
            records[key]["calls_per_step"] += 1
            return
        spec = default_workload(op, small=True)
        spec.update(name=f"{session.spec['name']}::{name or '<root>'}::{len(records)}", shape=list(value.shape),
                    training=module.training, input_requires_grad=value.requires_grad,
                    parameter_requires_grad=any(p.requires_grad for p in module.parameters()), calls_per_step=1)
        options = {}
        if isinstance(module, nn.Linear):
            options = {"out_features": module.out_features, "bias": module.bias is not None}
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            options = {key: getattr(module, key) for key in ("out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "padding_mode")}
            options["bias"] = module.bias is not None
            if "transpose" in op:
                options["output_padding"] = module.output_padding
        elif isinstance(module, nn.LayerNorm):
            options = {"normalized_shape": list(module.normalized_shape), "eps": module.eps,
                       "elementwise_affine": module.elementwise_affine, "bias": module.bias is not None}
        elif isinstance(module, nn.GroupNorm):
            options = {"num_groups": module.num_groups, "eps": module.eps, "affine": module.affine}
        elif isinstance(module, nn.BatchNorm2d):
            options = {"eps": module.eps, "momentum": module.momentum, "affine": module.affine,
                       "track_running_stats": module.track_running_stats}
        elif isinstance(module, nn.GELU):
            options = {"approximate": module.approximate}
        elif type(module) in MODULE_SPECS:
            options = {key: getattr(module, key) for key, _ in MODULE_SPECS[type(module)].fields if hasattr(module, key)}
            options.pop("_random_samples", None)
        spec["kwargs"] = options
        if value.ndim in (4, 5) and value.is_contiguous(memory_format=torch.channels_last if value.ndim == 4 else torch.channels_last_3d):
            spec["layout"] = "channels_last"
        elif not value.is_contiguous():
            spec["layout"] = "strided"
        if session.spec.get("steps_per_epoch"):
            spec["steps_per_epoch"] = session.spec["steps_per_epoch"]
        records[key] = spec

    for name, module in session.model.named_modules():
        op = mapping.get(type(module))
        if op is not None:
            def callback(current, args, kwargs, name=name, op=op):
                collect(current, args, kwargs, name, op)
            handles.append(module.register_forward_pre_hook(callback, with_kwargs=True))
    session.reset()
    try:
        session.step()
    finally:
        for handle in handles:
            handle.remove()
        session.reset()
    return deepcopy(list(records.values()))
