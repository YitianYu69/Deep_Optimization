"""Exact native pooling adapters, preserving buffers and shared identities."""

from torch import nn

from .spec import MODULE_SPECS, SPECS
from ..functional import _prepare_linear_plan
from ...precision.gradient import gradient_operation, functional_gradient


class _DOPool(nn.Module):
    kind = None

    def __init__(self, original=None, *, plan, kind=None):
        nn.Module.__init__(self)
        if kind is not None:
            self.kind = kind
        self.plan = _prepare_linear_plan(plan)
        self.fp8_scaling = None
        self._precision_outputs = {True: (None,), False: (None,)}
        self._precision_key, self.last_execution = "unplanned", None
        self._pool_fields = ()
        if original is not None:
            for name, value in original._buffers.items():
                self.register_buffer(name, value, persistent=name not in original._non_persistent_buffers_set)
            for name, _ in SPECS[self.kind].fields:
                if name != "indices" and hasattr(original, name):
                    if name not in self._buffers:
                        setattr(self, name, getattr(original, name))
                    self._pool_fields += (name,)
            self.train(original.training)

    def prepare_output_scaling(self, device):
        from ..activation.modules import DOReLU

        DOReLU.prepare_output_scaling(self, device)

    @property
    def execution_info(self):
        return None if self.last_execution is None else dict(self.last_execution)

    @gradient_operation("pooling")
    def forward(self, input, *args, **kwargs):
        from ..functional import _pool
        from ...precision.planner import normalization_output

        controls = {key: kwargs.pop(key) for key in tuple(kwargs) if key in (
            "plan", "output_dtype", "output_layout", "output_scale", "_output_layout", "_scaling_state",
            "_state_training", "_execution_info", "_planned")}
        options = {name: getattr(self, name) for name in self._pool_fields}
        if SPECS[self.kind].family == "unpool" and self._pool_fields:
            if args and "indices" in kwargs or len(args) > 1 and "output_size" in kwargs:
                raise TypeError("duplicate unpool module argument")
            if args:
                options["indices"] = args[0]
            if len(args) > 1:
                options["output_size"] = args[1]
            if len(args) > 2:
                raise TypeError("too many unpool module arguments")
            args = ()
        options.update(kwargs)
        if getattr(self, "force_indices", False):
            options = SPECS[self.kind].arguments(args, options)
            options["return_indices"] = True
            args = ()
        self.last_execution = {}
        defaults = dict(plan=self.plan, _output_layout=normalization_output(self), _scaling_state=self.fp8_scaling,
                        _state_training=self.training, _execution_info=self.last_execution)
        defaults.update({key: value for key, value in controls.items() if value is not None})
        defaults["_planned"] = True
        with functional_gradient(self.kind):
            return _pool(self.kind, input, args, options, **defaults)


class DOFunctionalPool(_DOPool):
    pass


CONVERSIONS = {}
for _source, _spec in MODULE_SPECS.items():
    _name = "DO" + _source.__name__
    _target = type(_name, (_DOPool, _source), {"kind": _spec.name, "__module__": __name__})
    globals()[_name] = _target
    CONVERSIONS[_source] = _target
POOL_TYPES = tuple(CONVERSIONS.values()) + (DOFunctionalPool,)
