"""Prepared ReLU adapter with cached output precision and owned scaling state."""

from torch import nn

from ..functional import _prepare_linear_plan, do_relu
from ...quantization.activation_tensor import FP8Activation


from ...precision.gradient import gradient_operation

class DOReLU(nn.ReLU):
    def __init__(self, original=None, *, plan, inplace=False):
        super().__init__(original.inplace if original is not None else inplace)
        self.plan = _prepare_linear_plan(plan)
        self.fp8_scaling = None
        self._precision_outputs = {True: (None,), False: (None,)}
        self._precision_key = "unplanned"
        self.last_execution = None
        if original is not None:
            self.train(original.training)

    def prepare_output_scaling(self, device):
        training = self.plan.get("fp8_training")
        if (training is not None and training["scaling_recipe"] == "delayed" and self.fp8_scaling is None
                and any(value is not None for outputs in self._precision_outputs.values() for value in outputs)):
            from ...scaling.delayed import DelayedScalingState

            self.fp8_scaling = DelayedScalingState(training["delayed_scaling"], training["fp8_dtypes"], device=device)

    @property
    def execution_info(self):
        return None if self.last_execution is None else dict(self.last_execution)

    @gradient_operation("relu")
    def forward(self, input, inplace=None):
        from ...precision.planner import normalization_output

        self.last_execution = {}
        output = normalization_output(self)
        mutation = self.inplace if inplace is None else inplace
        # In-place operations preserve their input representation and aliases.
        layout = None if mutation and isinstance(input, FP8Activation) else output
        return do_relu(input, mutation, plan=self.plan, output_layout=layout, _planned=True,
                       _scaling_state=self.fp8_scaling, _state_training=self.training,
                       _execution_info=self.last_execution)
