"""Normalization adapters preserving parameter and running-buffer identities."""

import torch
from torch import nn

from ..functional import _prepare_linear_plan, do_batch_norm, do_group_norm, do_layer_norm


from ...precision.gradient import gradient_operation

class _NormalizationAdapter:
    def _prepare(self, original, plan):
        nn.Module.__init__(self)
        self.plan = _prepare_linear_plan(plan)
        self.fp8_scaling = None
        self._precision_outputs = {True: (None,), False: (None,)}
        self._precision_key = "unplanned"
        self.last_execution = None
        for name, parameter in original._parameters.items():
            self.register_parameter(name, parameter)
        for name, buffer in original._buffers.items():
            self.register_buffer(name, buffer, persistent=name not in original._non_persistent_buffers_set)
        self.train(original.training)

    def prepare_output_scaling(self, device):
        training = self.plan.get("fp8_training")
        if (training is not None and training["scaling_recipe"] == "delayed" and self.fp8_scaling is None
                and any(value is not None for outputs in self._precision_outputs.values() for value in outputs)):
            from ...scaling.delayed import DelayedScalingState

            self.fp8_scaling = DelayedScalingState(training["delayed_scaling"], training["fp8_dtypes"], device=device)

    def _output(self, x):
        from ...precision.planner import normalization_output
        from ...quantization.activation_tensor import FP8Activation

        output = normalization_output(self)
        compression = (self.plan.get("fp8_training", self.plan))["activation_compression"]
        self.last_execution = {"input": "fp8" if isinstance(x, FP8Activation) or x.dtype in (
            torch.float8_e4m3fn, torch.float8_e5m2) else str(x.dtype),
            "output": "fp8" if output is not None else "native", "compression_target": "x_hat",
            "compression": compression["enabled"], "backend": self.plan.get("fp8_training", {}).get("backend", "native")}
        return output

    @property
    def execution_info(self):
        return None if self.last_execution is None else dict(self.last_execution)


class DOBatchNorm2d(_NormalizationAdapter, nn.BatchNorm2d):
    def __init__(self, original, *, plan):
        if type(original) is not nn.BatchNorm2d:
            raise TypeError("original must be an exact torch.nn.BatchNorm2d")
        self._prepare(original, plan)
        for name in ("num_features", "eps", "momentum", "affine", "track_running_stats"):
            setattr(self, name, getattr(original, name))

    @gradient_operation("normalization")
    def forward(self, input):
        self._check_input_dim(input)
        factor = self.momentum
        if self.training and self.track_running_stats:
            self.num_batches_tracked.add_(1)
            if factor is None:
                # Device arithmetic remains valid during CUDA graph replay.
                factor = self.num_batches_tracked.to(torch.float32).reciprocal()
        if factor is None:
            factor = 0.0
        use_batch = self.training or self.running_mean is None and self.running_var is None
        return do_batch_norm(input, self.running_mean if self.track_running_stats else None,
                             self.running_var if self.track_running_stats else None, self.weight, self.bias,
                             use_batch, factor, self.eps, plan=self.plan, _output_layout=self._output(input),
                             _scaling_state=self.fp8_scaling, _state_training=self.training,
                             _execution_info=self.last_execution)


class DOGroupNorm(_NormalizationAdapter, nn.GroupNorm):
    def __init__(self, original, *, plan):
        if type(original) is not nn.GroupNorm:
            raise TypeError("original must be an exact torch.nn.GroupNorm")
        self._prepare(original, plan)
        for name in ("num_groups", "num_channels", "eps", "affine"):
            setattr(self, name, getattr(original, name))

    @gradient_operation("normalization")
    def forward(self, input):
        return do_group_norm(input, self.num_groups, self.weight, self.bias, self.eps, plan=self.plan,
                             _output_layout=self._output(input), _scaling_state=self.fp8_scaling,
                             _state_training=self.training, _execution_info=self.last_execution)


class DOFunctionalNormalization(nn.Module):
    def __init__(self, plan, kind):
        super().__init__()
        self.plan, self.kind = _prepare_linear_plan(plan), kind
        self.fp8_scaling = None
        self._precision_outputs = {True: (None,), False: (None,)}
        self._precision_key = "unplanned"
        self.last_execution = None

    prepare_output_scaling = _NormalizationAdapter.prepare_output_scaling
    execution_info = _NormalizationAdapter.execution_info

    @gradient_operation("normalization")
    def forward(self, *args, **kwargs):
        from ...precision.planner import normalization_output

        function = {"batch": do_batch_norm, "group": do_group_norm, "layer": do_layer_norm}[self.kind]
        if self.kind == "layer":
            self.last_execution = {}
            kwargs = {"_execution_info": self.last_execution, **kwargs, "_planned": True}
        return function(*args, **kwargs, plan=self.plan, _output_layout=normalization_output(self),
                        _scaling_state=self.fp8_scaling, _state_training=self.training)


class DOLayerNorm(_NormalizationAdapter, nn.LayerNorm):
    def __init__(self, original, *, plan):
        if type(original) is not nn.LayerNorm:
            raise TypeError("original must be an exact torch.nn.LayerNorm")
        self._prepare(original, plan)
        self.normalized_shape, self.eps = original.normalized_shape, original.eps
        self.elementwise_affine = original.elementwise_affine

    @gradient_operation("normalization")
    def forward(self, input):
        return do_layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps, plan=self.plan,
                             _output_layout=self._output(input), _scaling_state=self.fp8_scaling,
                             _state_training=self.training, _execution_info=self.last_execution, _planned=True)
