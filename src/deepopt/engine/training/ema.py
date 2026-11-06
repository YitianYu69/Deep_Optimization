"""Parameter averages for validation without copying the model or its wrappers."""

from collections.abc import Mapping
from contextlib import contextmanager

import torch
from torch.nn.parameter import is_lazy
from torch._subclasses.fake_tensor import FakeTensor


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        self.active = False
        self._parameters = dict(model.named_parameters())
        self._metadata = {}
        self.shadow = {}
        for name, parameter in self._parameters.items():
            if (is_lazy(parameter) or isinstance(parameter, FakeTensor)
                    or parameter.device.type == "meta" or parameter.layout != torch.strided):
                raise ValueError("EMA requires materialized, dense model parameters")
            dtype = (torch.float32 if parameter.is_floating_point() and parameter.element_size() < 4
                     else parameter.dtype)
            self._metadata[name] = (parameter.shape, parameter.dtype, parameter.device)
            self.shadow[name] = parameter.detach().to(dtype=dtype, copy=True)

    @property
    def storage_bytes(self):
        return sum(value.numel() * value.element_size() for value in self.shadow.values())

    def _checked_parameters(self, model):
        parameters = dict(model.named_parameters())
        if (parameters.keys() != self._parameters.keys()
                or any(parameters[name] is not original for name, original in self._parameters.items())):
            raise ValueError("Model parameters changed after EMA initialization; rebuild the trainer")
        for name, parameter in parameters.items():
            if (parameter.shape, parameter.dtype, parameter.device) != self._metadata[name]:
                raise ValueError("Model parameter shapes/dtypes/devices changed after EMA initialization")
        return parameters

    @torch.no_grad()
    def update(self, model):
        if self.active:
            raise RuntimeError("Cannot update EMA while validation weights are installed")
        parameters = self._checked_parameters(model)
        for name, parameter in parameters.items():
            shadow = self.shadow[name]
            if shadow.is_floating_point() or shadow.is_complex():
                shadow.mul_(self.decay).add_(parameter.detach().to(dtype=shadow.dtype), alpha=1 - self.decay)
            else:
                shadow.copy_(parameter)
        self.updates += 1

    @contextmanager
    def average_parameters(self, model):
        """Install averages in place, retaining live parameters and current buffers."""
        if self.active:
            raise RuntimeError("EMA validation cannot be nested")
        parameters = self._checked_parameters(model)
        # Keep existing storage identities for optimizers, DDP, compilation and
        # captured model callables. Buffers start with current training values;
        # even validation-side buffer writes are rolled back on exit.
        originals = [(value, value.detach().clone()) for value in parameters.values()]
        buffers = [(value, value.detach().clone()) for value in model.buffers()]
        self.active = True
        try:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(self.shadow[name])
            yield
        finally:
            try:
                with torch.no_grad():
                    for value, original in (*originals, *buffers):
                        value.copy_(original)
            finally:
                self.active = False

    def state_dict(self):
        """Return detached views for synchronous serialization, like Module.state_dict."""
        if self.active:
            raise RuntimeError("Cannot checkpoint EMA while validation weights are installed")
        return {"format_version": 1, "decay": self.decay, "updates": self.updates,
                "shadow": {name: value.detach() for name, value in self.shadow.items()}}

    def validate_state_dict(self, state, *, completed_updates):
        if (not isinstance(state, Mapping) or type(state.get("format_version")) is not int
                or state["format_version"] != 1 or type(state.get("decay")) not in (int, float)
                or state["decay"] != self.decay):
            raise ValueError("Checkpoint EMA settings do not match this trainer")
        if (type(completed_updates) is not int or completed_updates < 0
                or type(state.get("updates")) is not int or state["updates"] != completed_updates):
            raise ValueError("Checkpoint EMA update count does not match optimizer progress")
        saved = state.get("shadow")
        if not isinstance(saved, Mapping) or saved.keys() != self.shadow.keys():
            raise ValueError("Checkpoint EMA parameter names do not match this model")
        for name, expected in self.shadow.items():
            value = saved[name]
            if (not isinstance(value, torch.Tensor) or is_lazy(value) or isinstance(value, FakeTensor)
                    or value.device.type == "meta" or value.layout != torch.strided
                    or value.shape != expected.shape or value.dtype != expected.dtype):
                raise ValueError(f"Checkpoint EMA tensor {name!r} is incompatible")

    @torch.no_grad()
    def load_state_dict(self, state, *, completed_updates):
        self.validate_state_dict(state, completed_updates=completed_updates)
        if self.active:
            raise RuntimeError("Cannot restore EMA while validation weights are installed")
        for name, value in self.shadow.items():
            value.copy_(state["shadow"][name])
        self.updates = state["updates"]
