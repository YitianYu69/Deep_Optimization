"""Prepared transposed modules preserving parameters and native output sizing."""

from torch import nn

from .._cached_weights import CachedWeightModule
from ..functional import _prepare_linear_plan
from ...precision.gradient import GradientEdgePlan, gradient_operation
from ...quantization.gradient import GradientSlot


class _OutputPolicy:
    def _init_output_policy(self):
        self._precision_outputs = {True: (None,), False: (None,)}
        self._precision_key = "unplanned"
        self.output_scaling = None

    def prepare_output_scaling(self, device):
        training = self.plan.get("fp8_training")
        if (training is not None and self.output_scaling is None
                and any(v is not None for values in self._precision_outputs.values() for v in values)):
            edge = GradientEdgePlan("fp8", str(training["fp8_dtypes"]["activation"]).removeprefix("torch.float8_"),
                                    "channels", self.plan["block_shapes"]["activation"],
                                    training["scaling_recipe"], "cached transposed forward output")
            self.output_scaling = GradientSlot(edge, training["delayed_scaling"], device=device)
            self.output_scaling.train(self.training)

    def _output_slot(self):
        from ...precision.planner import normalization_output

        return self.output_scaling if normalization_output(self) is not None else None


class _DOConvTranspose(_OutputPolicy, CachedWeightModule):
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
        for name in ("in_channels", "out_channels", "kernel_size", "stride", "padding",
                     "output_padding", "groups", "dilation", "padding_mode"):
            setattr(self, name, getattr(original, name))
        for name, parameter in original._parameters.items():
            self.register_parameter(name, parameter)
        self._init_output_policy()
        self.last_execution = None
        self.train(original.training)

    @gradient_operation("convolution_transpose")
    def forward(self, input, output_size=None):
        from .. import functional as doF

        extra = self.source_type._output_padding(
            self, input, output_size, self.stride, self.padding, self.kernel_size, self.rank, self.dilation)
        self.last_execution = {}
        return getattr(doF, f"do_conv_transpose{self.rank}d")(
            input, self.weight, self.bias, self.stride, self.padding, extra, self.groups, self.dilation,
            plan=self.plan, _weight_cache=self._fp8_weight_cache, _scaling_state=self.fp8_scaling,
            _state_training=self.training, _execution_info=self.last_execution, _output_slot=self._output_slot())

    @property
    def execution_info(self):
        return None if self.last_execution is None else dict(self.last_execution)

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}, output_padding={self.output_padding}, "
                f"groups={self.groups}, bias={self.bias is not None}, dilation={self.dilation}")


class DOConvTranspose1d(_DOConvTranspose):
    rank, source_type = 1, nn.ConvTranspose1d


class DOConvTranspose2d(_DOConvTranspose):
    rank, source_type = 2, nn.ConvTranspose2d


class DOConvTranspose3d(_DOConvTranspose):
    rank, source_type = 3, nn.ConvTranspose3d


class DOFunctionalConvTranspose(_OutputPolicy, CachedWeightModule):
    """Own per-call-site cache, input scales, and independent output scales."""

    def __init__(self, plan, device, rank):
        super().__init__()
        self.plan, self.rank = _prepare_linear_plan(plan), rank
        if "fp8_training" in self.plan:
            self.configure_delayed_scaling(self.plan, device)
        else:
            self.fp8_scaling = None
        self._init_output_policy()

    @gradient_operation("convolution_transpose")
    def forward(self, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        from .. import functional as doF

        return getattr(doF, f"do_conv_transpose{self.rank}d")(
            input, weight, bias, stride, padding, output_padding, groups, dilation, plan=self.plan,
            _weight_cache=self._fp8_weight_cache, _scaling_state=self.fp8_scaling,
            _state_training=self.training, _output_slot=self._output_slot())


TRANSPOSE_TYPES = (DOConvTranspose1d, DOConvTranspose2d, DOConvTranspose3d, DOFunctionalConvTranspose)
CONVERSIONS = {cls.source_type: cls for cls in TRANSPOSE_TYPES[:3]}
