"""Prepared activation adapters with per-call-site precision and scaling ownership."""

from torch import nn

from ..functional import _prepare_linear_plan
from ...precision.gradient import gradient_operation


class _DOActivation(nn.Module):
    kind = None

    def __init__(self, original=None, *, plan, inplace=False, approximate="none"):
        super().__init__()
        self.plan = _prepare_linear_plan(plan)
        self.inplace = getattr(original, "inplace", inplace)
        self.approximate = getattr(original, "approximate", approximate)
        if self.approximate not in ("none", "tanh") or self.kind != "gelu" and self.approximate != "none":
            raise ValueError("approximate must be 'none' or 'tanh' for GELU")
        self.fp8_scaling = None
        self._precision_outputs = {True: (None,), False: (None,)}
        self._precision_key = "unplanned"
        self.last_execution = None
        if original is not None:
            self.train(original.training)

    def prepare_output_scaling(self, device):
        from .modules import DOReLU

        DOReLU.prepare_output_scaling(self, device)

    @property
    def execution_info(self):
        return None if self.last_execution is None else dict(self.last_execution)

    @gradient_operation("activation")
    def forward(self, input, inplace=None, approximate=None):
        from ..functional import _activation
        from ...precision.planner import normalization_output

        self.last_execution = {}
        return _activation(self.kind, input, self.inplace if inplace is None else inplace,
                           approximate=self.approximate if approximate is None else approximate,
                           plan=self.plan, output_layout=normalization_output(self), _planned=True,
                           _scaling_state=self.fp8_scaling, _state_training=self.training,
                           _execution_info=self.last_execution)


class DOSigmoid(_DOActivation, nn.Sigmoid):
    kind = "sigmoid"


class DOSiLU(_DOActivation, nn.SiLU):
    kind = "silu"


class DOGELU(_DOActivation, nn.GELU):
    kind = "gelu"


class DOTanh(_DOActivation, nn.Tanh):
    kind = "tanh"


class DOReLU6(_DOActivation, nn.ReLU6):
    kind = "relu6"


CONVERSIONS = {nn.Sigmoid: DOSigmoid, nn.SiLU: DOSiLU, nn.GELU: DOGELU,
               nn.Tanh: DOTanh, nn.ReLU6: DOReLU6}
ACTIVATION_TYPES = tuple(CONVERSIONS.values())


def adapter(kind, **kwargs):
    return {target.kind: target for target in ACTIVATION_TYPES}[kind](**kwargs)
