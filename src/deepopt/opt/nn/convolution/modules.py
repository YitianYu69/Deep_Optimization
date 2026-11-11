"""Convolution adapters preserving parameter identity, registration, and padding."""

from torch import nn
from torch.nn import functional as F

from .._cached_weights import CachedWeightModule
from ..functional import _prepare_linear_plan, do_conv1d, do_conv2d, do_conv3d


from ...precision.gradient import gradient_operation

class _DOConv(CachedWeightModule):
    rank = None
    source_type = None

    def __init__(self, original, *, plan):
        super().__init__()
        if type(original) is not self.source_type:
            raise TypeError(f"original must be an exact {self.source_type.__name__}")
        self.plan = _prepare_linear_plan(plan)
        if "fp8_training" in self.plan:
            self.configure_delayed_scaling(self.plan, original.weight.device)
        else:
            self.fp8_scaling = None
        for attribute in ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation",
                          "groups", "padding_mode"):
            setattr(self, attribute, getattr(original, attribute))
        self._reversed_padding_repeated_twice = tuple(original._reversed_padding_repeated_twice)
        for name, parameter in original._parameters.items():
            self.register_parameter(name, parameter)
        self.train(original.training)
        self.last_execution = None

    @classmethod
    def from_conv(cls, original, *, plan):
        return cls(original, plan=plan)

    @property
    def execution_info(self):
        return None if self.last_execution is None else dict(self.last_execution)

    @gradient_operation("convolution")
    def forward(self, input):
        padding = self.padding
        if self.padding_mode != "zeros":
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0,) * self.rank
        self.last_execution = {}
        function = (do_conv1d, do_conv2d, do_conv3d)[self.rank - 1]
        return function(input, self.weight, self.bias, self.stride, padding, self.dilation, self.groups,
                        plan=self.plan, _weight_cache=self._fp8_weight_cache,
                        _scaling_state=self.fp8_scaling, _state_training=self.training,
                        _execution_info=self.last_execution)

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}, groups={self.groups}, "
                f"bias={self.bias is not None}, padding_mode={self.padding_mode!r}")


class DOConv1d(_DOConv):
    rank, source_type = 1, nn.Conv1d


class DOConv2d(_DOConv):
    rank, source_type = 2, nn.Conv2d


class DOConv3d(_DOConv):
    rank, source_type = 3, nn.Conv3d


class DOFunctionalConvolution(CachedWeightModule):
    """Own delayed scales and weight-cache lifetime for an FX convolution call."""

    def __init__(self, plan, device, rank):
        super().__init__()
        self.plan, self.rank = _prepare_linear_plan(plan), rank
        self.configure_delayed_scaling(self.plan, device)

    @gradient_operation("convolution")
    def forward(self, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        function = (do_conv1d, do_conv2d, do_conv3d)[self.rank - 1]
        return function(input, weight, bias, stride, padding, dilation, groups, plan=self.plan,
                        _weight_cache=self._fp8_weight_cache, _scaling_state=self.fp8_scaling,
                        _state_training=self.training)
